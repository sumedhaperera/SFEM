
import argparse, json, re
from config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from db import list_recent_errors
from retrieval import search as search_playbooks
import httpx

PROMPT_TMPL = """You are a Salesforce Flow incident assistant.
Given an error context and knowledge snippets, provide:
1) A concise diagnosis (1-2 sentences)
2) 3-6 numbered next steps (actionable, specific to Salesforce).

Error Context:
{error_context}

Knowledge:
{knowledge}

Respond in plain text with a 'Diagnosis:' line and a 'Next steps:' list.
"""

def call_llm(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(timeout=30.0))
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You produce precise, actionable runbook guidance for Salesforce errors."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def local_fallback_generate(hits):
    top_text = hits[0][2] if hits else ""
    diagnosis = ""
    steps = []
    for line in top_text.splitlines():
        s = line.strip()
        if s.lower().startswith("diagnosis"):
            diagnosis = s.split(":", 1)[-1].strip()
        if re.match(r"^(\d+\.|\d+\)|[-•])\s*", s):
            steps.append(re.sub(r"^(\d+\.|\d+\)|[-•])\s*", "", s))
    if not steps:
        steps = [
            "Check duplicate/matching rules and validation rules on target fields.",
            "Verify field-level security and the running user's permissions.",
            "Correct the data or configuration and re-run the Flow.",
        ]
    return (
        f"Diagnosis: {diagnosis or 'See retrieved guidance below.'}\n"
        f"Next steps:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps[:6])])
    )

def build_query(ctx):
    sig_guess = ctx.get("error_message","").split(":",1)[0].split(" on ")[0]
    return f"{sig_guess} {ctx.get('flow_name')} v{ctx.get('flow_version')} element {ctx.get('flow_element')} :: {ctx.get('error_message')}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    rows = list_recent_errors(limit=args.limit)
    if not rows:
        print("No errors in DB. Run seed_sample_errors.py first."); return

    for r in rows:
        ctx = {
            "org_id": r["org_id"],
            "flow_name": r["flow_name"],
            "flow_version": r["flow_version"],
            "flow_element": r["flow_element"],
            "error_message": r["error_message"],
            "created_at": r["created_at"],
        }
        print("="*80)
        print(f"FLOW: {ctx['flow_name']} (v{ctx['flow_version']}) | ELEMENT: {ctx['flow_element']}")
        print(f"ERROR: {ctx['error_message']}")
        print("-"*80)

        query = build_query(ctx)
        hits = search_playbooks(query, k=TOP_K)

        knowledge = "\n\n".join([t for _,_,t in hits][:5])
        prompt = PROMPT_TMPL.format(error_context=json.dumps(ctx, indent=2), knowledge=knowledge)

        try:
            out = call_llm(prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            out = local_fallback_generate(hits)
        print(out)

if __name__ == "__main__":
    main()
