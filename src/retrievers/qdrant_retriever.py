# src/retrievers/qdrant_retriever.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import math, os, uuid
from datetime import datetime, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.qdrant.utils import get_client, ensure_collection
from src.embeddings.embed import get_embedder
from src.config.config import QDRANT_COLLECTION

# -----------------------
# Distance helper (matches your config/env)
# -----------------------
try:
    from src.config.config import QDRANT_DISTANCE
except Exception:
    QDRANT_DISTANCE = "cosine"

def _distance():
    d = (QDRANT_DISTANCE or "cosine").lower()
    if d in ("cosine", "cos"):
        return qm.Distance.COSINE
    if d in ("euclid", "euclidean", "l2"):
        return qm.Distance.EUCLID
    if d in ("dot", "ip", "inner", "innerproduct"):
        return qm.Distance.DOT
    return qm.Distance.COSINE

# -----------------------
# Freshness helpers (optional)
# -----------------------
try:
    from src.config.config import (
        FRESH_TIME_DECAY_DAYS,
        FRESH_BOOST_ALPHA,
        FRESH_RN_MULTIPLIER,
    )
except Exception:
    FRESH_TIME_DECAY_DAYS = 60
    FRESH_BOOST_ALPHA = 0.25
    FRESH_RN_MULTIPLIER = 1.25

_QDRANT_NS = uuid.UUID("800cd911-c69f-46d9-8407-2908c94a6d65")

def _coerce_point_id(raw) -> str | int:
    if isinstance(raw, int):
        return raw
    s = str(raw) if raw is not None else None
    if not s:
        return str(uuid.uuid5(_QDRANT_NS, "auto:" + os.urandom(8).hex()))
    try:
        return str(uuid.UUID(s))
    except Exception:
        return str(uuid.uuid5(_QDRANT_NS, s))

def _as_ts(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.timestamp()
    return None

def _time_boost(error_ts: float | None, doc_ts: float | None, tau_days: float) -> float:
    if error_ts is None or doc_ts is None:
        return 0.0
    delta_days = abs(error_ts - doc_ts) / 86400.0
    return math.exp(-delta_days / max(tau_days, 1e-6))

def _fresh_rescore(points, error_ts: float | None):
    if error_ts is None:
        return points
    tau = float(FRESH_TIME_DECAY_DAYS)
    alpha = float(FRESH_BOOST_ALPHA)
    rn_mult = float(FRESH_RN_MULTIPLIER)

    rescored = []
    for p in points:
        pl = p.payload or {}
        doc_ts = pl.get("doc_ts")
        ts = _as_ts(doc_ts) if isinstance(doc_ts, (str, int, float)) else None
        tb = _time_boost(error_ts, ts, tau)
        src_mult = rn_mult if pl.get("source") == "sf_release_notes" else 1.0
        new_score = (p.score or 0.0) + alpha * tb * src_mult
        pl.setdefault("_time_boost", tb)
        pl.setdefault("_rescored", True)
        rescored.append((new_score, p))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in rescored]

# -----------------------
# Retriever
# -----------------------
class QdrantRetriever:
    """Concrete retriever that satisfies BaseRetriever.search(...)."""

    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or get_client()
        self.collection = QDRANT_COLLECTION
        self.embedder = get_embedder()

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder(texts)
        return np.asarray(vecs, dtype="float32")

    def upsert(self, chunks: list[dict] | list, batch: int = 256) -> int:
        """
        Embed and upsert a list of chunk dicts (must include 'text').
        Returns the number of points upserted.

        NOTE: If no 'org_id' is provided, we default to 'global' so global
        corpora (Flow help / release notes / manuals / KB) are shared.
        """
        records = []
        for ch in chunks or []:
            if isinstance(ch, dict):
                text = ch.get("text") or ch.get("content") or ch.get("body")
                if not text:
                    continue
                rec = dict(ch)
            else:
                try:
                    text = ch[-1]
                except Exception:
                    continue
                if not isinstance(text, str) or not text.strip():
                    continue
                rec = {"text": text}

            rec.setdefault("source", rec.get("source") or "kb_seed")
            rec.setdefault("org_id", "global")  # << default for shared corpora
    
            raw_id = rec.get("id") or rec.get("sid") or rec.get("url") \
                  or rec.get("doc_id") or rec.get("chunk_id") or rec["text"][:64]
            rec["id"] = _coerce_point_id(raw_id)
            if raw_id is not None:
                rec.setdefault("sid", str(raw_id))
            rec.pop("html", None)
            records.append(rec)

        if not records:
            return 0

        texts = [r["text"] for r in records]
        vecs = self.embedder(texts)

        probe = vecs[0]
        dim = probe.shape[-1] if hasattr(probe, "shape") else len(probe)
        ensure_collection(dim)

        points: list[qm.PointStruct] = []
        for r, v in zip(records, vecs):
            points.append(qm.PointStruct(id=r["id"], vector=v, payload=r))

        for i in range(0, len(points), batch):
            self.client.upsert(collection_name=self.collection, points=points[i:i+batch])

        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        org_id: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        qv = self.embed([query])[0].astype(np.float32)

        xfilters = dict(extra_filters or {})
        error_date = xfilters.pop("_error_date", None)

        must: List[qm.FieldCondition] = []
        for k, v in xfilters.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=list(v))))
            else:
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))

        should: List[qm.Condition] = []
        if org_id:
            should.extend([
                qm.FieldCondition(key="org_id", match=qm.MatchValue(value=org_id)),
                qm.FieldCondition(key="org_id", match=qm.MatchValue(value="global")),
            ])
            try:
                should.append(qm.IsNullCondition(is_null=qm.PayloadField(key="org_id")))
            except AttributeError:
                pass

        flt = qm.Filter(must=must, should=should) if (must or should) else None

        pool = max(int(top_k) * 4, 20)
        try:
            res = self.client.search(
                self.collection,
                query_vector=qv,
                limit=pool,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            )
        except TypeError:
            res = self.client.search(
                self.collection,
                vector=qv,
                limit=pool,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            )

        error_ts = _as_ts(error_date)
        ranked_pts = _fresh_rescore(res, error_ts) if error_ts is not None else res

        dist = _distance()
        out: List[Tuple[int, float, str, Dict[str, Any]]] = []
        for pt in ranked_pts[: int(top_k)]:
            score = float(pt.score)
            if dist == qm.Distance.EUCLID:
                score = -score
            payload = pt.payload or {}
            text = payload.get("text", "")
            cid = payload.get("chunk_id", -1)
            try:
                idx = int(cid) if cid is not None else -1
            except Exception:
                idx = -1
            out.append((idx, score, text, payload))
        return out
