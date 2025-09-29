# V8 Patch — Phase 1 (Canonical + Fresh)

This patch is **additive**. It adds:
- `phase1_ingestion/` with crawlers + upserter
- `src/qdrant/utils.py` for standardized Qdrant config
- `requirements.phase1.txt` with minimal deps

## Apply
Unzip at repo root so the new files land in place. Ensure `export PYTHONPATH=.` when running.

## Run
```bash
# install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.phase1.txt

# generate JSONL
python phase1_ingestion/crawl_sf_flow.py --base https://help.salesforce.com/s/articleView   --path-prefix /docs/atlas.en-us.flow.meta/flow/   --max-pages 50   --out data/sf_flow_chunks.jsonl

python phase1_ingestion/release_notes_ingest.py   --index https://help.salesforce.com/s/articleView   --out data/flow_releasenotes.jsonl

# upsert using project-standard embeddings + config
export PYTHONPATH=.
python phase1_ingestion/jsonl_to_qdrant.py --in data/sf_flow_chunks.jsonl data/flow_releasenotes.jsonl
```


# Flow docs
mapfile -t FLOW_SEEDS < seeds/seeds.flow.txt
python phase1_ingestion/crawl_sf_flow.py \
  --seeds "${FLOW_SEEDS[@]}" \
  --path-prefix "/docs/atlas.en-us.flow.meta/flow/" \
  --max-pages 100 \
  --out data/sf_flow_chunks.jsonl

# Release notes (quote not needed here since mapfile preserves each URL as one item)
mapfile -t RN_SEEDS < seeds/seeds.release_notes.txt
python phase1_ingestion/release_notes_ingest.py \
  --seeds "${RN_SEEDS[@]}" \
  --out data/flow_releasenotes.jsonl

# Upsert both
python -m src.qdrant.upsert_jsonl --in data/sf_flow_chunks.jsonl data/flow_releasenotes.jsonl

# Generate seeds automatically
python seeds/generate_release_notes_seeds.py 258 256 252 > seeds/seeds.release_notes.txt

