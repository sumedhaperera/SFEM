# docs_search.py  (drop-in replacement)
import argparse, json, os, glob
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# safer threading on macOS/arm (harmless elsewhere)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

def load_index(prefix: Path):
    # accept either base ("index/flow_docs") or full file ("index/flow_docs.index")
    p = prefix
    if p.suffix == ".index":
        base = p.with_suffix("")
        idx_path = str(p)
        map_path = str(base) + ".json"
    else:
        idx_path = str(p.with_suffix(".index"))
        map_path = str(p.with_suffix(".json"))
    if not os.path.exists(idx_path):
        # allow folder form: index/flow_docs/flow_docs.index
        if p.is_dir():
            cands = [q for q in glob.glob(str(p / "*.index"))]
            if not cands:
                raise SystemExit(f"[error] no .index in {p}")
            idx_path = cands[0]
            map_path = os.path.splitext(idx_path)[0] + ".json"
        else:
            raise SystemExit(f"[error] missing index: {idx_path}")
    if not os.path.exists(map_path):
        alt = os.path.splitext(idx_path)[0] + ".json"
        if os.path.exists(alt):
            map_path = alt
        else:
            raise SystemExit(f"[error] missing map json near: {idx_path}")
    ix = faiss.read_index(idx_path)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return ix, mapping, idx_path, map_path

def encode_query(q: str, model_name: str, normalized: bool):
    model = SentenceTransformer(model_name)
    qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    if normalized:
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    return qv

def search(prefix: Path, query: str, k: int = 5):
    index, mapping, idx_path, map_path = load_index(prefix)
    try:
        index.hnsw.efSearch = max(getattr(index.hnsw, "efSearch", 64), 96)
    except AttributeError:
        pass
    model_name = mapping.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    normalized = bool(mapping.get("normalized", False))
    q = encode_query(query, model_name, normalized)
    D, I = index.search(q, k)
    return D[0], I[0], mapping, idx_path, map_path, normalized, model_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Prefix used when building index (e.g., index/flow_docs)")
    ap.add_argument("--q", required=True, help="Search query")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    D, I, mapping, idx_path, map_path, normalized, model_name = search(Path(args.index), args.q, args.k)
    print(f"# index: {idx_path}")
    print(f"# map  : {map_path}")
    print(f"# model: {model_name} | normalized: {normalized}")
    print(f"# query: {args.q}\n")

    chunks = mapping.get("chunks", [])
    label = "cosine_sim" if normalized else "inner_product"
    for i, (s, idx) in enumerate(zip(D, I), 1):
        if idx == -1: 
            continue
        ch = chunks[idx]
        sec = ch.get("section") or "Unknown section"
        page = ch.get("page", "?")
        text = ch.get("text", "")
        print(f"{i:>2}. {label}={s:.3f} | p.{page} | {sec}")
        print((text[:400].replace("\n", " ") + ("..." if len(text)>400 else "")))
        print("-"*80)

if __name__ == "__main__":
    main()
