# rag_generate.py (Qdrant via retriever factory; no FAISS)
import argparse, json, re
from typing import List, Tuple, Dict, Any
from src.config.config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from src.db.db import list_recent_errors
from src.retrievers.factory import get_retriever
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

# ---- LLM call
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

# ---- Fallback using top retrieved snippet
def local_fallback_generate(hits: List[Tuple[int, float, str]]) -> str:
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

# ---- Query builder stays as-is
def build_query(ctx: Dict[str, Any]) -> str:
    sig_guess = (ctx.get("error_message") or "").split(":",1)[0].split(" on ")[0]
    return f"{sig_guess} {ctx.get('flow_name')} v{ctx.get('flow_version')} element {ctx.get('flow_element')} :: {ctx.get('error_message')}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    # NEW: accept (and effectively default to) including docs; keeps smoke test happy
    ap.add_argument("--with-docs", action="store_true", default=True,
                    help="Include retrieved doc chunks as context")
    args = ap.parse_args()

    rows = list_recent_errors(limit=args.limit)
    if not rows:
        print("No errors in DB. Run seed_sample_errors.py first."); return

    retriever = get_retriever()  # QdrantRetriever (factory)

    for r in rows:
        err_msg = r.get("error_message") or r.get("message") or ""
        ctx = {
            "org_id": r.get("org_id"),
            "flow_name": r.get("flow_name"),
            "flow_version": r.get("flow_version"),
            "flow_element": r.get("flow_element"),
            "error_message": err_msg,
            "created_at": r.get("created_at"),
        }
        print("="*80)
        print(f"FLOW: {ctx.get('flow_name')} (v{ctx.get('flow_version')}) | ELEMENT: {ctx.get('flow_element')}")
        print(f"ERROR: {ctx.get('error_message')}")
        print("-"*80)

        query = build_query(ctx)

        # Pick the best available timestamp from the DB row
        error_date = (
            r.get("occurred_at")
            or r.get("created_at")
            or r.get("updated_at")
        )

        # Build filters + pass the freshness control key
        extra = {
            # keep whatever Qdrant payload filters you already use
            "source": ["sf_help", "sf_release_notes", "pdf_manual"],
        }
        if error_date:
            # special key consumed by the retriever's freshness re-rank
            extra["_error_date"] = error_date   # can be datetime or ISO string

        # Qdrant search via factory (tenant-aware if you upserted with org_id)
        hits: List[Tuple[int, float, str]] = retriever.search(
            query=query,
            top_k=TOP_K,
            org_id=ctx.get("org_id"),
            extra_filters=extra
        )

        # Include knowledge unless you later add a --no-docs toggle
        knowledge = "\n\n".join([t for (_,_,t) in hits][:5]) if (args.with_docs and hits) else ""
        prompt = PROMPT_TMPL.format(
            # default=str handles datetime in created_at
            error_context=json.dumps(ctx, indent=2, default=str),
            knowledge=knowledge or "No snippets retrieved." if args.with_docs else "(docs not included).",
        )

        try:
            out = call_llm(prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            out = local_fallback_generate(hits)
        print(out)

if __name__ == "__main__":
    main()
