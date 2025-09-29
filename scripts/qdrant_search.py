#!/usr/bin/env python3


import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from src.embeddings.embed import get_embedder





def main():
    coll = os.getenv("QDRANT_COLLECTION", "sfem_kb")
    url  = os.getenv("QDRANT_URL", "http://localhost:6333")
    key  = os.getenv("QDRANT_API_KEY")

    query = "record-triggered flow error handling subflow"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    embed = get_embedder()
    vec = embed([query])[0]

    client = QdrantClient(url=url, api_key=key)

    # Optional source filter; remove if you want everything
    src_filter = qm.Filter(should=[
        qm.FieldCondition(key="source", match=qm.MatchValue(value="sf_help")),
        qm.FieldCondition(key="source", match=qm.MatchValue(value="sf_release_notes")),
        qm.FieldCondition(key="source", match=qm.MatchValue(value="pdf_manual")),
    ])

    # Try the new API first
    try:
        out = client.query_points(
            collection_name=coll,
            query=qm.Query(
                vector=vec,
                limit=5,
                filter=src_filter,
                with_payload=True,
                with_vectors=False,
            ),
        )
        points = out.points  # new API shape
    except Exception:
        # Fallback to legacy API (deprecated but works)
        points = client.search(
            collection_name=coll,
            query_vector=vec,           # 'vector' in older clients, 'query_vector' in newer
            limit=5,
            with_payload=True,
            query_filter=src_filter,
        )

    for r in points:
        payload = r.payload or {}
        print(f"{getattr(r,'score',getattr(r,'scored_point',None)) or r.score:.3f}  "
              f"[{payload.get('source')}] {payload.get('title') or payload.get('doc_id')}  {payload.get('url')}")

    def count(src):
        return client.count(
            collection_name=coll,
            count_filter=qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=src))]),
            exact=True
        ).count


    print("sf_help:", count("sf_help"))
    print("sf_release_notes:", count("sf_release_notes"))
    print("pdf_manual:", count("pdf_manual"))
if __name__ == "__main__":
    main()
