from __future__ import annotations

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

try:
    from src.config.config import (
        QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_DISTANCE
    )
except Exception:
    QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sfem_kb")
    QDRANT_DISTANCE   = (os.getenv("QDRANT_DISTANCE", "Cosine") or "Cosine").lower()

_DISTANCE_MAP = {
    "cosine": qm.Distance.COSINE,
    "cos": qm.Distance.COSINE,
    "dot": qm.Distance.DOT,
    "ip": qm.Distance.DOT,
    "euclid": qm.Distance.EUCLID,
    "l2": qm.Distance.EUCLID,
}

def distance_enum(name: str) -> qm.Distance:
    key = (name or "cosine").strip().lower()
    return _DISTANCE_MAP.get(key, qm.Distance.COSINE)

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection(vector_size: int) -> None:
    client = get_client()
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        # Validate dimension
        try:
            current_size = info.config.params.vectors.size
        except Exception:
            current_size = None
        if current_size is not None and current_size != vector_size:
            raise RuntimeError(
                f"Collection '{QDRANT_COLLECTION}' exists with size={current_size} "
                f"but you're trying to upsert dim={vector_size}. Choose a different collection or re-create."
            )
        return
    except Exception:
        pass
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=distance_enum(QDRANT_DISTANCE)),
    )
