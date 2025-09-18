# retrieval.py
import os, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from db import fetch_playbooks  # assumes you have this helper returning rows

INDEX_DIR = os.getenv("INDEX_DIR", "index")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype("float32", copy=False)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _encode_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    return _normalize(X)

def _to_text(pb: Dict) -> str:
    # Flatten title + signature + body for embedding
    title = pb.get("title", "").strip()
    sig   = pb.get("signature", "").strip()
    body  = (pb.get("body", "") or "").strip()
    parts = []
    if title: parts.append(f"Title: {title}")
    if sig:   parts.append(f"Signature: {sig}")
    if body:
        # ensure steps are kept as numbered bullets if present
        parts.append("Body:")
        parts.append(body)
    return "\n".join(parts)

def rebuild_index() -> Tuple[str, str]:
    """Rebuild HNSW-IP index for KB with cosine-normalized vectors."""
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    base = Path(INDEX_DIR) / "kb"

    # 1) Load KB rows
    rows = fetch_playbooks()  # [{id, title, signature, body, ...}, ...]
    if not rows:
        raise RuntimeError("No playbooks found; seed first (insert_playbooks).")

    texts = [_to_text(r) for r in rows]
    X = _encode_texts(texts, MODEL_NAME)
    d = X.shape[1]

    # 2) Build HNSW (IP on unit vectors ⇒ cosine)
    ix = faiss.IndexHNSWFlat(d, 32)
    ix.hnsw.efConstruction = 80
    ix.hnsw.efSearch = max(getattr(ix.hnsw, "efSearch", 64), 96)
    ix.add(X)

    # 3) Persist
    faiss.write_index(ix, str(base) + ".index")
    # map json: include normalized flag for downstream searchers
    chunks = []
    for r, t in zip(rows, texts):
        chunks.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "signature": r.get("signature"),
            "text": t,
        })
    with open(str(base) + "_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "embedding_dim": int(d),
            "normalized": True,
            "chunks": chunks
        }, f, ensure_ascii=False)

    # optional: numpy fallback
    np.save(str(base) + ".embeddings.npy", X)

    return str(base) + ".index", str(base) + "_map.json"

def search(query: str, k: int = 5) -> List[Tuple[int, float, str]]:
    """
    Returns [(idx, score, text), ...] where score is cosine similarity (0..1).
    """
    base = Path(INDEX_DIR) / "kb"
    idx_path = str(base) + ".index"
    map_path = str(base) + "_map.json"

    ix = faiss.read_index(idx_path)
    meta = json.load(open(map_path, "r", encoding="utf-8"))
    model_name = meta.get("model", MODEL_NAME)
    normalized = bool(meta.get("normalized", False))

    model = SentenceTransformer(model_name)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    if normalized:
        qv = _normalize(qv)

    D, I = ix.search(qv, k)
    D, I = D[0], I[0]
    chunks = meta.get("chunks", [])
    out = []
    for s, idx in zip(D, I):
        if idx == -1: 
            continue
        text = chunks[idx].get("text", "") if idx < len(chunks) else ""
        out.append((int(idx), float(s), text))
    # already sorted by FAISS (desc for IP)
    return out
