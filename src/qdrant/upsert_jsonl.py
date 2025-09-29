# src/qdrant/upsert_jsonl.py
from __future__ import annotations
import argparse, json, os, uuid
from typing import Any, Dict, List

from src.embeddings.embed import get_embedder
from src.qdrant.utils import get_client, ensure_collection
from src.config.config import QDRANT_COLLECTION
from qdrant_client.http import models as qm
from tqdm import tqdm

EMBED = get_embedder()

# Use the same namespace across the project so IDs are stable across runs
_QDRANT_NS = uuid.UUID("800cd911-c69f-46d9-8407-2908c94a6d65")

def _coerce_point_id(raw) -> str | int:
    # keep integers as-is
    if isinstance(raw, int):
        return raw
    s = str(raw) if raw is not None else None
    if not s:
        # stable-but-random fallback
        return str(uuid.uuid5(_QDRANT_NS, "auto:" + os.urandom(8).hex()))
    # if it's already a UUID string, keep it
    try:
        u = uuid.UUID(s)
        return str(u)                 # ← return string, not UUID object
    except Exception:
        # deterministically derive UUIDv5 from the original string id
        return str(uuid.uuid5(_QDRANT_NS, s))   # ← return string

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def _to_point(rec: Dict[str, Any], vec) -> qm.PointStruct:
    raw = rec.get("id") or rec.get("url") or rec.get("doc_id") or rec.get("text")
    pid = _coerce_point_id(raw)   # ← now str or int
    payload = {k: v for k, v in rec.items() if k != "html"}
    if raw is not None:
        payload.setdefault("sid", str(raw))  # keep original id for traceability
    return qm.PointStruct(id=pid, vector=vec, payload=payload)

def upsert_jsonls(inputs: List[str], batch: int = 512) -> int:
    records: List[Dict[str, Any]] = []
    for p in inputs:
        if not os.path.exists(p):
            print(f"[warn] not found: {p}")
            continue
        records.extend([r for r in _load_jsonl(p) if r.get("text")])
    if not records:
        print("[exit] no records")
        return 0

    probe = EMBED([records[0]["text"]])[0]
    dim = probe.shape[-1] if hasattr(probe, "shape") else len(probe)
    ensure_collection(dim)

    vectors = EMBED([r["text"] for r in records])
    points = [_to_point(r, v) for r, v in zip(records, vectors)]

    client = get_client()
    for i in tqdm(range(0, len(points), batch), desc=f"Upserting → {QDRANT_COLLECTION}"):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+batch])

    print(f"[done] Upserted {len(points)} → {QDRANT_COLLECTION}")
    return len(points)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Upsert JSONL → Qdrant (project-standard)")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    upsert_jsonls(args.inputs, args.batch)
