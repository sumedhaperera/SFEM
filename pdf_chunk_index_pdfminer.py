import argparse, json, os, re, sys
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdfminer_extract_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

def read_pdf_text_pages(path: Path) -> List[Tuple[int, str]]:
    txt = pdfminer_extract_text(str(path)) or ""
    parts = [p for p in txt.split("\x0c") if p.strip()] or [txt]
    return list(enumerate(parts, start=1))

def chunk_paragraphs(pairs, doc_id: str, chunk_size=1200, overlap=200):
    chunks = []
    for page_no, txt in pairs:
        text = re.sub(r"[\t\r]+", " ", txt)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        buf = ""
        for para in paragraphs or [text.strip()]:
            if len(buf) + 1 + len(para) <= chunk_size:
                buf = (buf + " " + para).strip()
            else:
                if buf:
                    chunks.append({"doc_id": doc_id, "section": None, "page": page_no, "text": buf})
                buf = (buf[-overlap:] + " " + para).strip() if overlap and len(buf) > overlap else para
        if buf:
            chunks.append({"doc_id": doc_id, "section": None, "page": page_no, "text": buf})
    return chunks

def embed_texts(texts: List[str]):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # unit vectors
    return X, model_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--doc-id", required=True, help="Doc identifier (e.g., flow_manual)")
    ap.add_argument("--out", required=True, help="Output prefix, e.g., index/flow_docs")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    pdf_path = Path(args.pdf); out_prefix = Path(args.out)
    if not pdf_path.is_file():
        print(f"[error] PDF not found: {pdf_path}", file=sys.stderr); sys.exit(2)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pages = read_pdf_text_pages(pdf_path)
    if not any(p.strip() for _, p in pages):
        print("[error] PDF appears to have no extractable text (might be scanned).", file=sys.stderr); sys.exit(3)

    chunks = chunk_paragraphs(pages, args.doc_id, args.chunk_size, args.overlap)
    texts = [c["text"] for c in chunks if c["text"].strip()]
    print(f"[info] pages={len(pages)} chunks={len(texts)}")

    # Embeddings (unit vectors)
    X, model_name = embed_texts(texts)
    d = X.shape[1]

    # === HNSW: prefer INNER_PRODUCT (so scores are true cosine on unit vectors) ===
    metric = "ip"
    try:
        # FAISS ≥1.8 usually supports passing metric_type here
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    except TypeError:
        # Fallback: older builds only create L2 HNSW
        index = faiss.IndexHNSWFlat(d, 32)
        metric = "l2"  # we’ll convert to cosine at read time in the searcher

    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(X)

    # Persist artifacts
    faiss.write_index(index, str(out_prefix.with_suffix(".index")))
    meta = {
        "model": model_name,
        "embedding_dim": int(d),
        "normalized": True,
        "metric": metric,              # <-- record the metric we actually used
        "chunks": chunks
    }
    with open(out_prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # Optional no-FAISS fallback
    np.save(str(out_prefix) + ".embeddings.npy", X)

    print(f"[done] wrote {out_prefix}.index and {out_prefix}.json (normalized={True}, metric={metric})")

if __name__ == "__main__":
    main()

