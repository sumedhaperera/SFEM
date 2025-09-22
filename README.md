
# SFEM (Salesforce Error Mitigator) — Clean Starter

# 1) Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Ensure the repo root is on PYTHONPATH (helps scripts & tests)
export PYTHONPATH=.

# 3) Seed sample errors (JSONL -> DB)
python scripts/seed_sample_errors.py --from-json test//data/test/data/sample_errors.jsonl

# 4) Build/seed KB embeddings
python -m src.embeddings.kb_seed

# 5) (Optional) Index a PDF manual (pdfminer-only)
mkdir -p index
python -m src.embeddings.pdf_chunk_index_pdfminer \
  --pdf docs/automate_your_business_processes_9-14-2025.pdf \
  --doc-id flow_manual \
  --out index/flow_docs_safe

# 6) (Optional) Run a basic RAG generation
cp .env.example .env   # add your OpenAI key if you want LLM calls
python -m src.llm.rag_generate --limit 5 --with-docs
```
