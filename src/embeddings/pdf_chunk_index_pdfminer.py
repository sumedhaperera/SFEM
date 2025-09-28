#!/usr/bin/env python3
"""
PDF indexer (pdfminer) -> chunk -> embed -> Qdrant upsert

CLI:
  --pdf PATH               (required)
  --doc-id ID              (required; used for idempotency and filtering)
  --out PREFIX             (required; writes <PREFIX>.json manifest)
  --chunk-size N           (optional; default 1200 characters)
  --overlap N              (optional; default 100 characters)
  --org-id ORG             (optional; tenant label)
  --force                  (optional; delete previous doc points before reindex)

Env (from .env):
  QDRANT_URL (default http://localhost:6333)
  QDRANT_API_KEY (blank locally; set on cloud)
  QDRANT_COLLECTION (default sfem_kb)
  QDRANT_DISTANCE (Cosine | Dot | Euclid)  default Cosine
  EMBED_MODEL (default sentence-transformers/all-MiniLM-L6-v2)
"""

import os, re
import sys
import json
import argparse
from typing import List, Dict, Any, Iterable, Tuple

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


_NAMESPACE = uuid.UUID("800cd911-c69f-46d9-8407-2908c94a6d65")  # any fixed UUID
USE_BACKEND = os.getenv("PDF_BACKEND", "pdfminer").lower()
# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, help="Path to PDF")
    p.add_argument("--doc-id", required=True, help="Doc identifier (e.g., flow_manual)")
    p.add_argument("--out", required=True, help="Manifest output prefix, e.g., index/flow_docs")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--org-id", default=None, help="Optional tenant/org for payload filtering")
    p.add_argument("--force", action="store_true", help="Re-index even if doc already present (deletes old points first)")
    return p.parse_args()

# --- sentence-aware chunking (library-backed) -------------------------------
import os, re

# Choose how to chunk:
#   CHUNK_MODE=sentence | paragraph | char   (default: char)
CHUNK_MODE = os.getenv("CHUNK_MODE", "char").lower()

# Which sentence splitter to use:
#   SENT_SPLITTER=blingfire|pysbd|nltk|spacy|naive  (default: blingfire→pysbd→naive)
SENT_SPLITTER = os.getenv("SENT_SPLITTER", "blingfire").lower()


def split_into_sentences(text: str) -> list[str]:
    """Return list of sentences using the requested backend, with graceful fallback."""
    t = (text or "").strip()
    if not t:
        return []

    if SENT_SPLITTER in ("blingfire", "auto"):
        try:
            from blingfire import text_to_sentences
            # returns newline-separated sentences
            return [s.strip() for s in text_to_sentences(t).splitlines() if s.strip()]
        except Exception:
            # fall through to next choice
            pass

    if SENT_SPLITTER in ("pysbd", "auto", "blingfire"):  # allow fallback chain
        try:
            import pysbd
            seg = pysbd.Segmenter(language="en", clean=False)
            return [s.strip() for s in seg.segment(t) if s.strip()]
        except Exception:
            pass

    if SENT_SPLITTER in ("nltk",):
        try:
            import nltk
            try:
                _ = nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(t) if s.strip()]
        except Exception:
            pass

    if SENT_SPLITTER in ("spacy",):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["tagger","parser","ner","lemmatizer"])
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            doc = nlp(t)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            pass

    # Naive fallback
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]


def chunk_text_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Pack sentences up to ~chunk_size characters.
    Overlap = last sentence of previous chunk is carried to the next chunk if overlap > 0.
    If a single sentence > chunk_size, char-slice it with a tiny internal overlap.
    """
    sents = split_into_sentences(text)
    if not sents:
        return []

    chunks: list[str] = []
    cur: list[str] = []

    def cur_len_with(s: str) -> int:
        return len(" ".join(cur)) + (1 if cur else 0) + len(s)

    i = 0
    while i < len(sents):
        s = sents[i]
        if len(s) <= chunk_size:
            if not cur:
                cur = [s]
            elif cur_len_with(s) <= chunk_size:
                cur.append(s)
            else:
                chunks.append(" ".join(cur))
                cur = ([cur[-1]] if overlap > 0 else []) + [s]
            i += 1
            continue

        # sentence too long -> char-slice with small internal overlap
        rest = s
        tail_ov = min(overlap, max(0, chunk_size // 10))  # small internal overlap
        while len(rest) > chunk_size:
            part = rest[:chunk_size]
            if not cur:
                chunks.append(part)
            elif cur_len_with(part) <= chunk_size:
                cur.append(part); chunks.append(" ".join(cur)); cur = ([cur[-1]] if overlap > 0 else [])
            else:
                chunks.append(" ".join(cur)); cur = ([cur[-1]] if overlap > 0 else []) + [part]
            cut = chunk_size - tail_ov if tail_ov < chunk_size else chunk_size
            rest = rest[cut:]
        # leftover goes into current buffer
        if rest:
            if not cur:
                cur = [rest]
            elif cur_len_with(rest) <= chunk_size:
                cur.append(rest)
            else:
                chunks.append(" ".join(cur))
                cur = ([cur[-1]] if overlap > 0 else []) + [rest]
        i += 1

    if cur:
        chunks.append(" ".join(cur))
    return chunks


# (optional) paragraph-first that uses sentences for long paragraphs
def _normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s or "").strip()

def chunk_text_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    raw_paras = [p for p in re.split(r"\n{2,}", text or "") if p.strip()]
    paras = [_normalize_ws(p) for p in raw_paras]
    out: list[str] = []
    cur: list[str] = []

    def cur_len_with(p: str) -> int:
        return len("\n\n".join(cur)) + (2 if cur else 0) + len(p)

    for p in paras:
        if len(p) <= chunk_size:
            if not cur:
                cur = [p]
            elif cur_len_with(p) <= chunk_size:
                cur.append(p)
            else:
                out.append("\n\n".join(cur))
                cur = ([cur[-1]] if overlap > 0 else []) + [p]
            continue
        # paragraph too long -> sentence-pack inside
        for piece in chunk_text_sentence(p, chunk_size, max(0, overlap // 2)):
            if not cur:
                cur = [piece]
            elif cur_len_with(piece) <= chunk_size:
                cur.append(piece)
            else:
                out.append("\n\n".join(cur))
                cur = ([cur[-1]] if overlap > 0 else []) + [piece]

    if cur:
        out.append("\n\n".join(cur))
    return out


def make_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    mode = CHUNK_MODE
    if mode == "sentence":
        return chunk_text_sentence(text, chunk_size, overlap)
    if mode == "paragraph":
        return chunk_text_paragraph(text, chunk_size, overlap)
    # default: your existing char-based splitter
    return chunk_text(text, chunk_size, overlap)

# -----------------------------
# PDF → pages → [text per page]
# -----------------------------
# at top

def extract_text_by_page(pdf_path: str) -> list[str]:
    if USE_BACKEND == "pymupdf":
        # pip install pymupdf
        import fitz
        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # text from blocks preserves reading order better than raw
                pages.append(page.get_text("text"))
        return pages
    elif USE_BACKEND == "pdfplumber":
        # pip install pdfplumber
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        return pages
    else:
        # default: pdfminer (what you have today)
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        out = []
        for layout in extract_pages(pdf_path):
            lines = []
            for el in layout:
                if isinstance(el, LTTextContainer):
                    lines.append(el.get_text())
            out.append("".join(lines).strip())
        return out



# -----------------------------
# Chunking (character-based)
# -----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks


# -----------------------------
# Embedding
# -----------------------------
def load_embedder() -> SentenceTransformer:
    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)


# -----------------------------
# Qdrant helpers
# -----------------------------
def _distance_from_env() -> Distance:
    name = os.getenv("QDRANT_DISTANCE", "Cosine").strip().lower()
    if name == "dot":
        return Distance.DOT
    if name == "euclid":
        return Distance.EUCLID
    return Distance.COSINE


def _qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", "")
    # Avoid warning: only send API key over HTTPS
    use_key = url.startswith("https://") and bool(api_key)
    return QdrantClient(url=url, api_key=(api_key if use_key else None))


def ensure_collection(client: QdrantClient, name: str, size: int, distance: Distance) -> None:
    # Create collection if not exists; else assume it's compatible
    try:
        client.get_collection(name)
    except Exception:
        client.recreate_collection(name=name, vectors_config=VectorParams(size=size, distance=distance))


def _doc_exists(client: QdrantClient, collection: str, doc_id: str, org_id: str | None) -> bool:
    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if org_id:
        must.append(FieldCondition(key="org_id", match=MatchValue(value=org_id)))
    res = client.count(collection_name=collection, count_filter=Filter(must=must), exact=True)
    return (res.count or 0) > 0


def _delete_doc_points(client: QdrantClient, collection: str, doc_id: str, org_id: str | None) -> None:
    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if org_id:
        must.append(FieldCondition(key="org_id", match=MatchValue(value=org_id)))
    client.delete(collection_name=collection, points_selector=Filter(must=must))


def _attach_doc_tags(payload: Dict[str, Any], doc_id: str, org_id: str | None) -> Dict[str, Any]:
    p = dict(payload)
    p["doc_id"] = doc_id
    if org_id:
        p["org_id"] = org_id
    return p


                        
def _point_id(doc_id: str, page: int, idx: int) -> str:
    # stable per doc/page/chunk across runs
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{page}:{idx}"))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    # Early exit if PDF is missing (good for CI/smoke)
    if not os.path.isfile(args.pdf):
        print(f"[info] No PDF at {args.pdf}. Skipping doc indexing.", file=sys.stderr)
        sys.exit(0)

    # Extract text per page
    pages_text = extract_text_by_page(args.pdf)
    num_pages = len(pages_text)

    # Build chunks and remember (page, chunk_idx, text)
    chunk_triplets: List[Tuple[int, int, str]] = []
    for page_idx, text in enumerate(pages_text, start=1):
        chunks = make_chunks(text, args.chunk_size, args.overlap)
        for ci, ch in enumerate(chunks):
            chunk_triplets.append((page_idx, ci, ch))

    num_chunks = len(chunk_triplets)
    print(f"[info] pages={num_pages} chunks={num_chunks}")

    if num_chunks == 0:
        # Nothing to embed/push
        manifest_path = args.out if args.out.endswith(".json") else f"{args.out}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"doc_id": args.doc_id, "org_id": args.org_id, "pdf": args.pdf, "pages": num_pages, "chunks": 0}, f)
        print(f"[done] wrote manifest {manifest_path}")
        return

    # Load embedder & embed all chunks
    embedder = load_embedder()
    vectors = embed_texts(embedder, [t for (_, _, t) in chunk_triplets])
    dim = int(vectors.shape[1])

    # Qdrant client & collection
    client = _qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", "sfem_kb")
    ensure_collection(client, collection, size=dim, distance=_distance_from_env())

    # Idempotency: skip if already indexed; or delete old then reinsert if --force
    if _doc_exists(client, collection, args.doc_id, args.org_id):
        if not args.force:
            print(f"[info] '{args.doc_id}' already indexed; use --force to reindex.")
            # still write manifest so downstream steps can rely on it
            manifest_path = args.out if args.out.endswith(".json") else f"{args.out}.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "doc_id": args.doc_id,
                        "org_id": args.org_id,
                        "pdf": args.pdf,
                        "pages": num_pages,
                        "chunks": num_chunks,
                        "skipped": True,
                    },
                    f,
                )
            print(f"[done] wrote manifest {manifest_path}")
            return
        # force: clean existing first
        _delete_doc_points(client, collection, args.doc_id, args.org_id)

    # Build points
    points: List[PointStruct] = []
    for i, (page, cidx, txt) in enumerate(chunk_triplets):
        payload = {
            "page": page,
            "chunk_index": cidx,
            "text": txt,
        }
        payload = _attach_doc_tags(payload, args.doc_id, args.org_id)
        pid = _point_id(args.doc_id, page, cidx)
        points.append(PointStruct(id=pid, vector=vectors[i].tolist(), payload=payload))

    # Upsert (deterministic IDs -> no duplicates; force path already deleted prior)
    client.upsert(collection_name=collection, points=points)
    print(f"[done] upserted {len(points)} chunks into Qdrant collection (see QDRANT_COLLECTION).")

    # Manifest
    manifest = {
        "doc_id": args.doc_id,
        "org_id": args.org_id,
        "pdf": args.pdf,
        "pages": num_pages,
        "chunks": len(points),
        "collection": collection,
    }
    out_path = args.out if args.out.endswith(".json") else f"{args.out}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"[done] wrote manifest {out_path}")


if __name__ == "__main__":
    main()
