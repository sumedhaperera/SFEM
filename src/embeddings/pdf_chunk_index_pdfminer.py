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

import os
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


# -----------------------------
# PDF → pages → [text per page]
# -----------------------------
def extract_text_by_page(pdf_path: str) -> List[str]:
    pages_text: List[str] = []
    for page_layout in extract_pages(pdf_path):
        lines: List[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                lines.append(element.get_text())
        pages_text.append("".join(lines).strip())
    return pages_text


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
        chunks = chunk_text(text, args.chunk_size, args.overlap)
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
