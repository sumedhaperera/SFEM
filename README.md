
# SFEM (Salesforce Error Mitigator) — Clean Starter

# 1) Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Ensure the repo root is on PYTHONPATH (helps scripts & tests)
export PYTHONPATH=.
# 3) Run docker compose to bring up the services.
# start both services in the background
docker compose up -d

# point your app at them (from your host)
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/sfem"
export QDRANT_URL="http://localhost:6333"

# create tables
python -c "from src.db.db import init_schema; init_schema()"

# when done
docker compose down               # stop containers
docker compose down -v            # stop + remove volumes (wipes data)

# 4) Seed sample errors (JSONL -> DB)
python scripts/seed_sample_errors.py --from-json test//data/test/data/sample_errors.jsonl

# 5) Build/seed KB embeddings
python -m src.embeddings.kb_seed

# 6) (Optional) Index a PDF manual (pdfminer-only)
mkdir -p index
python -m src.embeddings.pdf_chunk_index_pdfminer \
  --pdf docs/automate_your_business_processes_9-14-2025.pdf \
  --doc-id flow_manual \
  --out index/flow_docs_safe

# 7) (Optional) Run a basic RAG generation
cp .env.example .env   # add your OpenAI key if you want LLM calls
python -m src.llm.rag_generate --limit 5 --with-docs
```
