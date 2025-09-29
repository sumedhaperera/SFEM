# V8 Patch — PDF JSONL Mode

Adds to `src/embeddings/pdf_chunk_index_pdfminer.py`:
- `--jsonl-out FILE`: write chunks to a standard JSONL schema
- `--no-upsert`: skip embedding/upsert (use with `--jsonl-out`)

Also swaps embedding to use your project embedder:
`from src.embeddings.embed import get_embedder`

## Recommended pipeline
```bash
# 1) Generate JSONL from PDF
python src/embeddings/pdf_chunk_index_pdfminer.py   --pdf source_data/automate_your_business_processes_9-14-2025.pdf   --doc-id flow_manual   --out data/flow_pdf_manifest   --jsonl-out data/pdf_chunks.jsonl   --no-upsert

# 2) Upsert JSONL with Phase‑1 upserter (same embedder + config)
export PYTHONPATH=.
python phase1_ingestion/jsonl_to_qdrant.py --in data/pdf_chunks.jsonl data/sf_flow_chunks.jsonl data/flow_releasenotes.jsonl
```
