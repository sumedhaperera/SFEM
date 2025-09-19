#!/usr/bin/env python3
import argparse
import os
import sys
import re
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure project root on path
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional: load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# Project imports
try:
    from retrievers.factory import get_retriever
    from db import init_schema, fetch_playbooks, insert_playbooks, list_recent_errors
except Exception as e:
    print(f"[fatal] could not import project modules: {e}", file=sys.stderr)
    sys.exit(2)

# Third-party deps (available in your repo's requirements)
from pdfminer.high_level import extract_text as pdfminer_extract_text

# Config (env-driven)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# ----------------- Helpers -----------------

SAMPLE_PLAYBOOKS = [
    dict(title="Duplicate Email on Contact", signature="DUPLICATE_VALUE Contact.Email", body="""
Diagnosis: A duplicate rule or matching rule blocked Contact creation due to an existing email.

- Verify the duplicate rule conditions for Contact.Email.
- Search for the existing Contact using the email; merge or update instead of creating.
- If the record is legit, adjust the rule or use an allowlist field.
- Re-run the Flow after resolving the conflict.
"""),
    dict(title="Field Validation: Postal Code", signature="FIELD_CUSTOM_VALIDATION_EXCEPTION Postal Code", body="""
Diagnosis: Custom validation enforces a 5-digit postal code.

- Normalize the input to 5 digits (strip non-digits).
- If international, update validation to allow country-specific formats.
- Re-run the Flow after correcting the data.
"""),
    dict(title="Picklist: Restricted Value", signature="INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST", body="""
Diagnosis: The value isn't in the restricted picklist for this field.

- Check the picklist value set for the field.
- Map incoming values to allowed values; add synonyms if needed.
- Re-run the Flow with a permitted value.
"""),
    dict(title="Row Locking", signature="UNABLE_TO_LOCK_ROW", body="""
Diagnosis: A concurrent transaction holds a lock on the same record.

- Retry with backoff; avoid bulk updates to the same parent simultaneously.
- Reduce transaction scope; update child rows outside critical section.
- Use Platform Events/Queueable for retries.
"""),
]

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
    import httpx
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
    return resp.choices[0].message.content or ""

def local_fallback_generate(top_text: str) -> str:
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


def check_qdrant():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY") or None)
        cols = client.get_collections().collections
        print(f"[ok] Qdrant reachable at {QDRANT_URL} — collections: {[c.name for c in cols]}")
    except Exception as e:
        print(f"[fatal] Qdrant not reachable: {e}", file=sys.stderr)
        sys.exit(3)

def ensure_playbooks_seeded_and_indexed():
    init_schema()
    rows = fetch_playbooks()
    if not rows:
        insert_playbooks(SAMPLE_PLAYBOOKS)
        rows = fetch_playbooks()
        print(f"[seed] inserted {len(rows)} sample playbooks")
    # Upsert to Qdrant
    from retrievers.factory import get_retriever
    r = get_retriever()
    chunks = []
    for row in rows:
        parts = []
        if row.get("title"): parts.append(f"Title: {row['title']}")
        if row.get("signature"): parts.append(f"Signature: {row['signature']}")
        if row.get("body"): parts += ["Body:", row["body"]]
        text = "\n".join(parts)
        chunks.append({
            "id": row.get("id"),
            "text": text,
            "meta": {"source_id": "kb", "chunk_id": row.get("id"), "version": "kb@v1", "org_id": "global"}
        })
    n = r.upsert(chunks)
    print(f"[index] upserted {n} KB chunks to Qdrant")

def pdf_read_pages(path: Path) -> List[Tuple[int, str]]:
    txt = pdfminer_extract_text(str(path)) or ""
    parts = [p for p in txt.split("\x0c") if p.strip()] or [txt]
    return list(enumerate(parts, start=1))

def pdf_chunk_and_upsert(pdf_path: Path, doc_id: str, org_id: str = "global", chunk_size=1200, overlap=200) -> int:
    pages = pdf_read_pages(pdf_path)
    if not any(p.strip() for _, p in pages):
        print(f"[warn] no extractable text in {pdf_path.name} — skipping")
        return 0
    chunks_raw = []
    for page_no, txt in pages:
        text = re.sub(r"[\t\r]+", " ", txt)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        buf = ""
        for para in (paragraphs or [text.strip()]):
            if len(buf) + 1 + len(para) <= chunk_size:
                buf = (buf + " " + para).strip()
            else:
                if buf:
                    chunks_raw.append({"page": page_no, "text": buf})
                buf = (buf[-overlap:] + " " + para).strip() if overlap and len(buf) > overlap else para
        if buf:
            chunks_raw.append({"page": page_no, "text": buf})
    # Upsert
    r = get_retriever()
    qdr_chunks = []
    for c in chunks_raw:
        qdr_chunks.append({
            "id": None,
            "text": c["text"],
            "meta": {
                "source_id": f"pdf:{pdf_path.name}",
                "doc_id": doc_id,
                "page": c["page"],
                "org_id": org_id,
                "version": "docs@v1"
            }
        })
    n = r.upsert(qdr_chunks, batch_size=256)
    print(f"[index] {pdf_path.name}: upserted {n} chunks")
    return n

def seed_sample_errors_if_present(root: Path):
    script = root / "seed_sample_errors.py"
    jsonl = root / "sample_errors.jsonl"
    if not script.exists() or not jsonl.exists():
        print("[warn] seed_sample_errors.py or sample_errors.jsonl not found — skipping error seeding")
        return
    import subprocess
    cmd = [sys.executable, str(script), "--from-json", str(jsonl)]
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        print("[warn] seed_sample_errors.py returned non-zero; continuing")

def build_query(ctx: Dict[str, Any]) -> str:
    sig_guess = ctx.get("error_message","").split(":",1)[0].split(" on ")[0]
    return f"{sig_guess} {ctx.get('flow_name')} v{ctx.get('flow_version')} element {ctx.get('flow_element')} :: {ctx.get('error_message')}"

def mitigate_errors(limit: int, org_id_override: str | None):
    rows = list_recent_errors(limit=limit)
    if not rows:
        print("[warn] No errors in DB. Skipping mitigation. (Run seed_sample_errors.py to add samples.)")
        return

    retriever = get_retriever()

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
        hits = retriever.search(
            query=query,
            top_k=int(os.getenv("TOP_K", "5")),
            org_id=org_id_override or ctx["org_id"],
            extra_filters=None
        )
        knowledge = "\n\n".join([t for (_,_,t) in hits][:5]) if hits else ""

        prompt = PROMPT_TMPL.format(
            error_context=json.dumps(ctx, indent=2),
            knowledge=knowledge or "No snippets retrieved."
        )
        try:
            out = call_llm(prompt)
        except Exception as e:
            print(f"[llm] call failed: {e} — using fallback")
            top_text = hits[0][2] if hits else ""
            out = local_fallback_generate(top_text)
        print(out)

def main():
    ap = argparse.ArgumentParser(description="End-to-end test: seed → index (KB/PDF) → retrieve → LLM.")
    ap.add_argument("--limit", type=int, default=3, help="How many recent errors to mitigate")
    ap.add_argument("--pdf", action="append", default=[], help="Path to a PDF to index (repeatable)")
    ap.add_argument("--pdf-dir", default=None, help="Directory to glob PDFs (*.pdf) to index")
    ap.add_argument("--org-id", default="global", help="Org/tenant id for PDF payloads and retrieval filter default")
    args = ap.parse_args()

    print(f"[info] project root: {ROOT}")
    print(f"[info] Qdrant URL: {QDRANT_URL}")

    # 1) Qdrant reachable?
    check_qdrant()

    # 2) Seed + index KB
    ensure_playbooks_seeded_and_indexed()

    # 3) Index PDFs (optional)
    pdfs: List[Path] = [Path(p) for p in args.pdf]
    if args.pdf_dir:
        d = Path(args.pdf_dir)
        if d.is_dir():
            pdfs.extend(sorted(d.glob("*.pdf")))
    if pdfs:
        print(f"[step] Indexing {len(pdfs)} PDF(s) → Qdrant")
        total = 0
        for p in pdfs:
            if not p.exists():
                print(f"[warn] PDF not found: {p}")
                continue
            total += pdf_chunk_and_upsert(p, doc_id=p.stem, org_id=args.org_id)
        print(f"[done] PDF chunks upserted: {total}")
    else:
        print("[info] No PDFs provided. Skipping doc indexing.")

    # 4) Seed sample errors (if files exist)
    seed_sample_errors_if_present(ROOT)

    # 5) Mitigate recent errors end-to-end
    mitigate_errors(limit=args.limit, org_id_override=None)

    print("[done] End-to-end test complete.")

if __name__ == "__main__":
    main()
