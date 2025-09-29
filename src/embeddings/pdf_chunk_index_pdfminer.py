#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import uuid
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

# --- PDF extraction (pdfminer.six) ---
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox, LTTextLine, LAParams

# --- Project-standard embeddings & Qdrant helpers ---
from src.embeddings.embed import get_embedder
from src.qdrant.utils import get_client, ensure_collection
from src.config.config import QDRANT_COLLECTION
from qdrant_client.http import models as qm
from tqdm import tqdm

# Deterministic namespace for UUIDv5 IDs (stable across runs)
_QDRANT_NS = uuid.UUID("800cd911-c69f-46d9-8407-2908c94a6d65")


# ------------------------
# Utilities
# ------------------------
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _iter_page_text(pdf_path: str) -> Iterable[Tuple[int, str]]:
    """
    Yield (page_number, text) for each page using pdfminer.six.
    """
    laparams = LAParams()  # default is fine for text extraction
    for i, page in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
        texts: List[str] = []
        for el in page:
            if isinstance(el, (LTTextContainer, LTTextBox, LTTextLine)):
                try:
                    texts.append(el.get_text())
                except Exception:
                    # defensive: skip malformed fragments
                    continue
        page_text = _clean_text("\n".join(texts))
        if page_text:
            yield i, page_text


def _split_chunks(text: str, max_chars: int = 1200) -> List[str]:
    """
    Split a block of text into ~max_chars chunks on paragraph boundaries when possible.
    """
    if not text:
        return []
    paras = re.split(r"\n{2,}", text)  # paragraphs by blank lines
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in paras:
        p = _clean_text(p)
        if not p:
            continue
        if cur_len + len(p) + 1 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
            if len(p) <= max_chars:
                cur = [p]
                cur_len = len(p)
            else:
                # hard split long paragraph
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks or ([text[:max_chars]] if text else [])


def _coerce_point_id(s: str | int | None) -> str | int:
    """
    Qdrant point IDs must be int or string. If not int, return a UUIDv5 string derived from s.
    """
    if isinstance(s, int):
        return s
    base = str(s) if s is not None else os.urandom(8).hex()
    try:
        return str(uuid.UUID(base))
    except Exception:
        return str(uuid.uuid5(_QDRANT_NS, base))


# ------------------------
# Embedding shims (project-standard)
# ------------------------
_EMBED = None


def load_embedder():
    """Return the project-standard embedder (callable: list[str] -> list[vec])."""
    global _EMBED
    if _EMBED is None:
        _EMBED = get_embedder()
    return _EMBED


def embed_texts(embedder, texts: List[str]) -> List[Any]:
    """
    Compatibility wrapper so existing call sites keep working.
    """
    if not texts:
        return []
    return embedder(texts)


# ------------------------
# Main pipeline
# ------------------------
def build_pdf_chunks(
    pdf_path: str, doc_id: str, file_url: str | None = None, title: str | None = None
) -> List[Dict[str, Any]]:
    """
    Produce canonical JSONL-ready records for a PDF:
      {
        id, doc_id, url, title, headings_path, anchor, text,
        source="pdf_manual", product="flow", page, chunk_index
      }
    """
    path = Path(pdf_path)
    title = title or path.name
    file_url = file_url or (f"file://{path.resolve()}")
    records: List[Dict[str, Any]] = []

    for page_no, page_text in _iter_page_text(pdf_path):
        parts = _split_chunks(page_text, max_chars=1200)
        if not parts:
            continue
        for j, text in enumerate(parts):
            raw_id = f"pdf:{doc_id}:p{page_no}:{j}"
            pid = _coerce_point_id(raw_id)
            rec = {
                "id": str(pid),  # keep as string for consistency
                "sid": raw_id,   # original string id for traceability
                "doc_id": doc_id,
                "url": file_url,
                "title": title,
                "headings_path": [f"Page {page_no}"],
                "anchor": None,
                "text": text,
                "source": "pdf_manual",
                "product": "flow",
                "page": page_no,
                "chunk_index": j,
            }
            records.append(rec)

    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def write_manifest(out_dir: str, records: List[Dict[str, Any]]) -> None:
    """
    Write a tiny manifest (no embeddings) under out_dir for debugging/inspection.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    mf = outp / "manifest.jsonl"
    write_jsonl(str(mf), records)


def upsert_records(records: List[Dict[str, Any]], batch: int = 256) -> int:
    """
    Embed via project embedder and upsert to Qdrant using project config.
    """
    if not records:
        return 0

    EMBED = load_embedder()
    texts = [r["text"] for r in records]
    vectors = EMBED(texts)
    # find dim
    probe = vectors[0]
    dim = getattr(probe, "shape", [len(probe)])[-1] if hasattr(probe, "shape") else len(probe)
    ensure_collection(dim)

    points: List[qm.PointStruct] = []
    for r, vec in zip(records, vectors):
        pid = _coerce_point_id(r.get("id") or r.get("sid") or r.get("url") or r.get("text"))
        payload = dict(r)
        payload.pop("html", None)
        points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))

    client = get_client()
    for i in tqdm(range(0, len(points), batch), desc=f"Upserting → {QDRANT_COLLECTION}"):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i : i + batch])

    return len(points)


def main():
    ap = argparse.ArgumentParser(description="Index a PDF into chunks and (optionally) upsert to Qdrant.")
    ap.add_argument("--pdf", required=True, help="Path to PDF file")
    ap.add_argument("--doc-id", required=True, help="Logical document id (stable)")
    ap.add_argument("--out", required=True, help="Output directory for local artifacts (manifest)")
    ap.add_argument("--title", default=None, help="Optional display title")
    ap.add_argument(
        "--jsonl-out",
        dest="jsonl_out",
        default=None,
        help="Write chunks to JSONL (no embeddings/upsert); use with --no-upsert.",
    )
    ap.add_argument(
        "--no-upsert",
        dest="no_upsert",
        action="store_true",
        default=False,
        help="Skip embedding/upsert (use when generating JSONL only).",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=256,
        help="Upsert batch size (only used when not --no-upsert).",
    )
    args = ap.parse_args()

    pdf_path = args.pdf
    doc_id = args.doc_id
    out_dir = args.out
    title = args.title
    jsonl_out = getattr(args, "jsonl_out", None)
    no_upsert = getattr(args, "no_upsert", False)
    batch = getattr(args, "batch", 256)

    if not os.path.exists(pdf_path):
        raise SystemExit(f"[pdf] not found: {pdf_path}")

    # 1) Build JSONL-ready records from the PDF
    records = build_pdf_chunks(pdf_path, doc_id, file_url=None, title=title)
    if not records:
        print("[pdf] no text extracted from PDF; exiting")
        return

    # 2) Always write a small manifest to --out for inspection
    write_manifest(out_dir, records)
    print(f"[pdf] manifest written → {Path(out_dir) / 'manifest.jsonl'} ({len(records)} chunks)")

    # 3) Optional: write a JSONL file if requested
    if jsonl_out:
        n = write_jsonl(jsonl_out, records)
        print(f"[pdf] JSONL written → {jsonl_out} ({n} chunks)")
        if no_upsert:
            return  # JSONL-only mode

    # 4) Default behavior: embed + upsert directly (backwards compatible)
    if not no_upsert:
        total = upsert_records(records, batch=batch)
        print(f"[done] Upserted {total} → {QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()
