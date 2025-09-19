# retrievers/qdrant_retriever.py  — Qdrant-only retriever (IDs = UUIDv5)
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional, Tuple
import os, uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .base import Retriever
from embed import get_embedder

# --- robust config import ---
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sfem_kb")
QDRANT_DISTANCE   = (os.getenv("QDRANT_DISTANCE", "Cosine") or "Cosine").lower()

def _distance():
    if QDRANT_DISTANCE.startswith("cos"): return qm.Distance.COSINE
    if QDRANT_DISTANCE.startswith("dot"): return qm.Distance.DOT
    return qm.Distance.EUCLID

def _stable_uuid(text: str, meta: Dict[str, Any]) -> str:
    """Deterministic UUIDv5 from text + key metadata (Qdrant-acceptable)."""
    parts = [text]
    for k in ("chunk_id","source_id","org_id","version","doc_id","page"):
        v = meta.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    seed = "|".join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

def _normalize_point_id(pid: Any, text: str, meta: Dict[str, Any]) -> Any:
    """Return uint/int as-is, UUID string if valid, else deterministic UUIDv5."""
    if isinstance(pid, int):      # ok (uint64)
        return pid
    if isinstance(pid, str):
        try:
            return str(uuid.UUID(pid))  # validate UUID
        except Exception:
            pass
    return _stable_uuid(text, meta)

class QdrantRetriever(Retriever):
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embed = get_embedder()
        self._ensure_collection()

    def _ensure_collection(self):
        dim = self.embed(["probe"]).shape[1]
        try:
            self.client.get_collection(QDRANT_COLLECTION)
        except Exception:
            self.client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qm.VectorParams(size=int(dim), distance=_distance()),
                hnsw_config=qm.HnswConfigDiff(m=32, ef_construct=200),
                optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
            )
        for fld in ("org_id","source_id","version","object","field","chunk_id","doc_id","page"):
            try:
                self.client.create_payload_index(QDRANT_COLLECTION, field_name=fld, field_schema="keyword")
            except Exception:
                pass

    def upsert(self, chunks: Iterable[Dict[str, Any]], batch_size: int = 256) -> int:
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids:   List[Any] = []
        total = 0

        def _flush():
            nonlocal texts, metas, ids, total
            if not texts:
                return
            vecs = self.embed(texts).astype(np.float32)
            self.client.upsert(
                QDRANT_COLLECTION,
                qm.Batch(ids=ids, vectors=vecs, payloads=metas),
                wait=True
            )
            total += len(ids)
            texts.clear(); metas.clear(); ids.clear()

        for ch in chunks:
            t = ch["text"]
            m = (ch.get("meta") or {}).copy()
            m.setdefault("text", t)  # keep text in payload for one-hop reads

            # ✅ THIS is the important line: always normalize to an int or UUID
            ids.append(_normalize_point_id(ch.get("id"), t, m))

            texts.append(t)
            metas.append(m)
            if len(texts) >= batch_size:
                _flush()
        _flush()
        return total

    def search(self, query: str, top_k: int = 5, org_id: Optional[str] = None, extra_filters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float, str]]:
        qv = self.embed([query])[0].astype(np.float32)
        must: list[qm.FieldCondition] = []
        if org_id:
            must.append(qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id)))
        if extra_filters:
            for k, v in extra_filters.items():
                if v is None: continue
                if isinstance(v, (list, tuple, set)):
                    must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=list(v))))
                else:
                    must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        flt = qm.Filter(must=must) if must else None

        res = self.client.search(QDRANT_COLLECTION, query_vector=qv, limit=int(top_k), query_filter=flt, with_payload=True, with_vectors=False)
        dist = _distance()
        out: List[Tuple[int, float, str]] = []
        for pt in res:
            score = float(pt.score)
            if dist == qm.Distance.EUCLID:
                score = -score
            payload = pt.payload or {}
            text = payload.get("text", "")
            # return original chunk_id if numeric; else -1
            cid = payload.get("chunk_id", -1)
            try: idx = int(cid) if cid is not None else -1
            except Exception: idx = -1
            out.append((idx, score, text))
        return out
