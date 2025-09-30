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

# --- 1.5) Find repo root (works if this file is at repo root OR under scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer a directory that actually contains src/
if [ -d "$SCRIPT_DIR/src" ]; then
  REPO_ROOT="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../src" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -d "$SCRIPT_DIR/../../src" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
  # As a last resort, try git (if available)
  if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
  else
    echo "[env] ERROR: could not locate repo root (no src/ found near $SCRIPT_DIR)."
    echo "       Move this script into the repo or adjust PYTHONPATH manually."
    exit 1
  fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[env] REPO_ROOT=$REPO_ROOT"
echo "[env] PYTHONPATH=$PYTHONPATH"

section(){ echo; echo "=== $*"; }

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
# Phase-1 minimal deps (crawler / upserter)
if [ -f "requirements.phase1.txt" ]; then
  echo "[pip] installing requirements.phase1.txt"
  python -m pip install -r requirements.phase1.txt
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

# --- 7) Build embeddings / index PDF (existing path retained)
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

# --- 7.5) Phase-1 Canonical + Fresh (Salesforce Help + Release Notes) → JSONL → Qdrant
section "Phase-1 Canonical + Fresh"
FLOW_JSONL="data/sf_flow_chunks.smoke.jsonl"
RN_JSONL="data/flow_releasenotes.smoke.jsonl"
mkdir -p data

echo "[phase1] crawling Flow docs (seeded, small for smoke)"
FLOW_SEEDS=(
  'https://developer.salesforce.com/docs/atlas.en-us.flow.meta/flow/flow_overview.htm'
  'https://developer.salesforce.com/docs/atlas.en-us.flow.meta/flow/flow_build.htm'
)
python phase1_ingestion/crawl_sf_flow.py \
  --seeds "${FLOW_SEEDS[@]}" \
  --path-prefix "/docs/atlas.en-us.flow.meta/flow/" \
  --max-pages 5 \
  --out "$FLOW_JSONL"

echo "[phase1] ingesting Flow release notes (seeded)"
RN_SEEDS=(
  'https://help.salesforce.com/s/articleView?id=release-notes.rn_automate_flow_builder.htm&language=en_US&release=258&type=5'
  'https://help.salesforce.com/s/articleView?id=release-notes.rn_automate_flow_builder_screen_flows.htm&language=en_US&release=258&type=5'
)
python phase1_ingestion/release_notes_ingest.py \
  --seeds "${RN_SEEDS[@]}" \
  --out "$RN_JSONL" \
  --debug-dump 1

echo "[phase1] line counts"
wc -l "$FLOW_JSONL" || true
wc -l "$RN_JSONL"   || true

echo "[phase1] upserting JSONL to Qdrant (project embedder + config)"
python -m src.qdrant.upsert_jsonl --in "$FLOW_JSONL" "$RN_JSONL" || {
  echo "[phase1] WARN: upsert failed or empty; continuing smoke"
}

# --- 8) End-to-end test script
echo "[test] running test/run_e2e_full.py"
export PYTHONPATH=.
python test/run_e2e_full.py

# --- 9) Optional LLM smoke (only if key is present)
#if [ -n "${OPENAI_API_KEY:-}" ]; then
#  echo "[llm] OPENAI_API_KEY detected; running small RAG generate"
#  python -m src.llm.rag_generate --limit 3 --with-docs
#else
#  echo "[llm] OPENAI_API_KEY not set; skipping LLM call (RAG generate)"
#fi

echo "✅ Smoke test completed."
