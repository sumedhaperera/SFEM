# rag_generate.py (Qdrant via retriever factory; no FAISS)
import argparse, json, re, time
from typing import List, Tuple, Dict, Any, Optional, Union

import httpx
from src.config.config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from src.db.db import list_recent_errors, insert_mitigation_atomic
from src.retrievers.factory import get_retriever

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
    steps: List[str] = []
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

# ---- Helpers
def _parse_steps_from_llm(text: str) -> Dict[str, Any]:
    """Extract a normalized steps_json from the plain-text LLM response."""
    diag = None
    m = re.search(r"(?im)^diagnosis:\s*(.+)$", text)
    if m:
        diag = m.group(1).strip()

    numbered: List[str] = []
    started = False
    for line in text.splitlines():
        if not started and re.match(r"(?i)^\s*next\s*steps\s*:", line.strip()):
            started = True
            continue
        s = line.strip()
        if re.match(r"^(\d+\.|\d+\)|[-•])\s*", s):
            s = re.sub(r"^(\d+\.|\d+\)|[-•])\s*", "", s)
            if s:
                numbered.append(s)

    if not numbered:
        for line in text.splitlines():
            s = line.strip()
            if re.match(r"^(\d+\.|\d+\)|[-•])\s*", s):
                s = re.sub(r"^(\d+\.|\d+\)|[-•])\s*", "", s)
                if s:
                    numbered.append(s)

    return {
        "version": "1.0",
        "summary": diag or (numbered[0] if numbered else None),
        "steps": [{"n": i + 1, "text": s} for i, s in enumerate(numbered[:6])],
        "raw_text": text,
    }

def _normalize_hits(hits: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize retriever hits to a common dict:
      {"text": str, "score": float|None, "payload": dict|None}
    Supports tuples (id, score, text[, payload]), dicts, or objects with attributes.
    """
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        text = None
        score = None
        payload = None
        if isinstance(h, tuple):
            if len(h) >= 3:
                score = h[1]
                text = h[2]
            if len(h) >= 4:
                payload = h[3]
        elif isinstance(h, dict):
            text = h.get("text") or h.get("chunk") or h.get("content")
            score = h.get("score")
            payload = h.get("payload")
        else:
            text = getattr(h, "text", None) or getattr(h, "content", None)
            score = getattr(h, "score", None)
            payload = getattr(h, "payload", None)
        out.append({"text": text or "", "score": float(score) if score is not None else None, "payload": payload})
    return out

def _citations_from_hits(norm_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cites = []
    for h in norm_hits:
        p = h.get("payload") or {}
        cites.append({
            "source_type": p.get("source") or p.get("product"),
            "doc_id": p.get("doc_id") or p.get("page_id") or p.get("url"),
            "chunk_id": p.get("chunk_id"),
            "score": h.get("score"),
            "url": p.get("url"),
            "title": p.get("title") or p.get("h1") or p.get("section"),
        })
    return cites

# ---- Query builder stays as-is
def build_query(ctx: Dict[str, Any]) -> str:
    sig_guess = (ctx.get("error_message") or "").split(":", 1)[0].split(" on ")[0]
    return f"{sig_guess} {ctx.get('flow_name')} v{ctx.get('flow_version')} element {ctx.get('flow_element')} :: {ctx.get('error_message')}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument(
        "--with-docs", action="store_true", default=True,
        help="Include retrieved doc chunks as context",
    )
    ap.add_argument(
        "--org-id", default=None,
        help="If set, only process errors for this org_id (useful for smoke/tenant-isolated runs)",
    )
    args = ap.parse_args()

    rows_all = list_recent_errors(limit=args.limit)
    # Optional tenant filter
    rows = [r for r in rows_all if (args.org_id is None or r.get("org_id") == args.org_id)]

    if not rows:
        if args.org_id:
            print(f"No errors found for org_id={args.org_id}. Seed or insert some first.")
        else:
            print("No errors in DB. Run seed_sample_errors.py first.")
        return

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
        print("=" * 80)
        print(f"FLOW: {ctx.get('flow_name')} (v{ctx.get('flow_version')}) | ELEMENT: {ctx.get('flow_element')}")
        print(f"ERROR: {ctx.get('error_message')}")
        print("-" * 80)

        query = build_query(ctx)

        # Pick the best available timestamp from the DB row
        error_date = (r.get("occurred_at") or r.get("created_at") or r.get("updated_at"))

        # Build filters + pass the freshness control key
        extra = {"source": ["sf_help", "sf_release_notes", "pdf_manual"]}
        if error_date:
            # special key consumed by the retriever's freshness re-rank
            extra["_error_date"] = error_date  # can be datetime or ISO string

        # Qdrant search via factory (tenant-aware if you upserted with org_id)
        hits: List[Any] = retriever.search(
            query=query,
            top_k=TOP_K,
            org_id=ctx.get("org_id"),
            extra_filters=extra,
        )

        # Normalize hits for flexible downstream handling
        norm_hits = _normalize_hits(hits)

        # Include knowledge unless you later add a --no-docs toggle
        knowledge = "\n\n".join([h["text"] for h in norm_hits][:5]) if (args.with_docs and norm_hits) else ""
        prompt = PROMPT_TMPL.format(
            error_context=json.dumps(ctx, indent=2, default=str),  # default=str handles datetimes
            knowledge=knowledge or ("No snippets retrieved." if args.with_docs else "(docs not included)."),
        )

        # Time the LLM call (for latency_ms)
        t0 = time.perf_counter()
        try:
            out = call_llm(prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            out = local_fallback_generate([(0, nh["score"] or 0.0, nh["text"]) for nh in norm_hits])
        latency_ms = int((time.perf_counter() - t0) * 1000)

        print(out)

        # ---------- Persist mitigation + citations (FAIL-FAST) ----------
        steps_json = _parse_steps_from_llm(out)
        citations = _citations_from_hits(norm_hits)

        try:
            mit_id, run_id = insert_mitigation_atomic(
                org_id=(ctx.get("org_id") or "unknown"),
                error_id=int(r["id"]),
                model_name=str(OPENAI_MODEL),
                prompt_template=PROMPT_TMPL,
                guidance_md=out,
                steps_json=steps_json,
                confidence=None,
                retrieved_json={"hits": [h for h in norm_hits]},
                latency_ms=latency_ms,
                citations=citations,
            )
            print(f"[persist] mitigation_id={mit_id} run_id={run_id}")
        except Exception as e:
            # Fail fast: surface a clear, contextual error so the process exits non-zero
            raise RuntimeError(
                f"Persistence failed for org_id={ctx.get('org_id')} error_id={r.get('id')}: {e}"
            ) from e


if __name__ == "__main__":
    main()
