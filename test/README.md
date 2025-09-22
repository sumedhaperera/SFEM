# End-to-End Runner (Full)

Run the whole flow locally with Qdrant:

```bash
python test/run_e2e_full.py --limit 3 --pdf-dir ./docs   # or --pdf path/to/file.pdf (repeatable)
```

What it does:
1. Checks Qdrant connectivity
2. Seeds playbooks into the DB and upserts them to Qdrant
3. Optionally indexes your PDFs into Qdrant (embedding done inside the retriever)
4. Seeds sample errors from `test/data/test/data/test/data/sample_errors.jsonl` if present
5. Retrieves context and calls the LLM (falls back to local template if no API key)

Env you should set:
- `QDRANT_URL` (e.g., `http://localhost:6333`), `QDRANT_COLLECTION`
- `EMBED_MODEL` (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- `OPENAI_API_KEY`, `OPENAI_MODEL`
