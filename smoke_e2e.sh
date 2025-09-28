#!/usr/bin/env bash
set -euo pipefail

# --- 0) Load .env first (if present), then set defaults safely (works with set -u)
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
: "${DATABASE_URL:=postgresql://postgres:postgres@localhost:5432/sfem}"
: "${QDRANT_URL:=http://localhost:6333}"
: "${QDRANT_COLLECTION:=sfem_kb}"
export DATABASE_URL QDRANT_URL QDRANT_COLLECTION

echo "[env] DATABASE_URL=$DATABASE_URL"
echo "[env] QDRANT_URL=$QDRANT_URL"
echo "[env] QDRANT_COLLECTION=$QDRANT_COLLECTION"

# --- 1) tiny helpers
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }
need docker
need python

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# make repo root importable so "from src..." works when running files under scripts/
export PYTHONPATH="$ROOT_DIR"
echo "[env] PYTHONPATH=$PYTHONPATH"


# --- 2) Start Postgres (once)
if ! docker ps --format '{{.Names}}' | grep -q '^sfem-pg$'; then
  if ! docker ps -a --format '{{.Names}}' | grep -q '^sfem-pg$'; then
    echo "[pg] starting postgres:16"
    docker run --name sfem-pg \
      -e POSTGRES_PASSWORD=postgres \
      -e POSTGRES_DB=sfem \
      -p 5432:5432 -d postgres:16
  else
    echo "[pg] container exists, starting it"
    docker start sfem-pg >/dev/null
  fi
else
  echo "[pg] already running"
fi

# --- 3) Start Qdrant (once)
if ! docker ps --format '{{.Names}}' | grep -q '^sfem-qdrant$'; then
  if ! docker ps -a --format '{{.Names}}' | grep -q '^sfem-qdrant$'; then
    echo "[qdrant] starting qdrant"
    docker run --name sfem-qdrant \
      -p 6333:6333 -p 6334:6334 -d qdrant/qdrant:latest
  else
    echo "[qdrant] container exists, starting it"
    docker start sfem-qdrant >/dev/null
  fi
else
  echo "[qdrant] already running"
fi

# --- 3.5) Fresh reset (DEFAULT; set NO_FRESH=1 to skip)
if [ "${NO_FRESH:-0}" != "1" ]; then
  echo "[fresh] resetting data stores"

  # Wait for Postgres to be ready (quick loop)
  for i in {1..30}; do
    if docker exec sfem-pg pg_isready -U postgres -d sfem >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done

  echo "[fresh][pg] drop tables errors, playbooks"
  docker exec -i sfem-pg psql -U postgres -d sfem -v ON_ERROR_STOP=0 \
    -c "DROP TABLE IF EXISTS errors, playbooks;"

  echo "[fresh][qdrant] drop collection: $QDRANT_COLLECTION"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -X DELETE "$QDRANT_URL/collections/$QDRANT_COLLECTION" >/dev/null || true
  else
    echo "[fresh][qdrant] 'curl' not found; skipping collection drop"
  fi

  echo "[fresh] clear local index/"
  rm -rf index/* || true
else
  echo "[fresh] NO_FRESH=1 set; skipping resets"
fi

# --- 4) Python deps
if [ -f "requirements.txt" ]; then
  echo "[pip] installing requirements"
  python -m pip install -r requirements.txt
fi

# --- 5) Create DB schema
echo "[db] init schema"
python - <<'PY'
from src.db.db import init_schema
init_schema()
print("schema ok")
PY

# --- 6) Seed sample errors
SAMPLE_JSONL="test/data/sample_errors.jsonl"
if [ ! -f "$SAMPLE_JSONL" ]; then
  echo "[seed] ERROR: $SAMPLE_JSONL not found"; exit 1
fi
echo "[seed] seeding errors from $SAMPLE_JSONL"
python scripts/seed_sample_errors.py --from-json "$SAMPLE_JSONL"

# --- 7) Build embeddings / index PDF
PDF_PATH="docs/automate_your_business_processes_9-14-2025.pdf"
OUT_DIR="index/flow_docs_safe"
mkdir -p index
echo "[emb] building KB"
python -m src.embeddings.kb_seed
echo "[emb] indexing PDF: $PDF_PATH"
python -m src.embeddings.pdf_chunk_index_pdfminer \
  --pdf "$PDF_PATH" \
  --doc-id flow_manual \
  --out "$OUT_DIR"

# --- 8) End-to-end test script
echo "[test] running test/run_e2e_full.py"
export PYTHONPATH=.
python test/run_e2e_full.py

# --- 9) Optional LLM smoke (only if key is present)
if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "[llm] OPENAI_API_KEY detected; running small RAG generate"
  python -m src.llm.rag_generate --limit 3 --with-docs
else
  echo "[llm] OPENAI_API_KEY not set; skipping LLM call (RAG generate)"
fi

echo "✅ Smoke test completed."
