# pdf_chunk_index_qdrant.py  — Qdrant-only (via retriever factory), no FAISS
import argparse, json, os, re, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from pdfminer.high_level import extract_text as pdfminer_extract_text
from src.retrievers.factory import get_retriever

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
        for para in (paragraphs or [text.strip()]):
            if len(buf) + 1 + len(para) <= chunk_size:
                buf = (buf + " " + para).strip()
            else:
                if buf:
                    chunks.append({"doc_id": doc_id, "section": None, "page": page_no, "text": buf})
                buf = (buf[-overlap:] + " " + para).strip() if overlap and len(buf) > overlap else para
        if buf:
            chunks.append({"doc_id": doc_id, "section": None, "page": page_no, "text": buf})
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--doc-id", required=True, help="Doc identifier (e.g., flow_manual)")
    ap.add_argument("--out", required=True, help="Manifest output prefix, e.g., index/flow_docs")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--org-id", default="global", help="Optional tenant/org for payload filtering")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_prefix = Path(args.out)
    if not pdf_path.is_file():
        print(f"[error] PDF not found: {pdf_path}", file=sys.stderr); sys.exit(2)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pages = read_pdf_text_pages(pdf_path)
    if not any(p.strip() for _, p in pages):
        print("[error] PDF appears to have no extractable text (maybe scanned).", file=sys.stderr); sys.exit(3)

    chunks_raw = chunk_paragraphs(pages, args.doc_id, args.chunk_size, args.overlap)
    texts = [c["text"] for c in chunks_raw if c["text"].strip()]
    print(f"[info] pages={len(pages)} chunks={len(texts)}")

    # Prepare Qdrant payload chunks
    retriever = get_retriever()  # QdrantRetriever via factory
    qdr_chunks: List[Dict[str, Any]] = []
    pdf_name = pdf_path.name
    for c in chunks_raw:
        qdr_chunks.append({
            # optional stable id: combine doc_id + page + hash if you want; leaving None lets retriever hash
            "id": None,
            "text": c["text"],
            "meta": {
                "source_id": f"pdf:{pdf_name}",
                "doc_id": args.doc_id,
                "page": c["page"],
                "section": c.get("section"),
                "org_id": args.org_id,
                "version": "docs@v1"
            }
        })

    # Upsert to Qdrant (embedding done inside the retriever)
    n = retriever.upsert(qdr_chunks, batch_size=256)
    print(f"[done] upserted {n} chunks into Qdrant collection (see QDRANT_COLLECTION).")

    # Write a tiny manifest (for your own bookkeeping/tools)
    manifest = {
        "backend": "qdrant",
        "pdf": str(pdf_path),
        "doc_id": args.doc_id,
        "source_id": f"pdf:{pdf_name}",
        "chunks": len(qdr_chunks),
        "org_id": args.org_id,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "version": "docs@v1"
    }
    with open(out_prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[done] wrote manifest {out_prefix.with_suffix('.json')}")

if __name__ == "__main__":
    main()
