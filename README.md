
# SFEM (Salesforce Error Mitigator) — Clean Starter

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python seed_sample_errors.py --from-json sample_errors.jsonl
python kb_seed.py

# optional PDF manual index (pdfminer-only)
mkdir -p index
python pdf_chunk_index_pdfminer.py   --pdf "source_data/automate_your_business_processes_9-14-2025.pdf"   --doc-id flow_manual   --out index/flow_docs_safe

cp .env.example .env   # add your OpenAI key (optional)
python rag_generate.py --limit 5 --with-docs
```
