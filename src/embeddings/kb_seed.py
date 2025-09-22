# kb_seed.py  (Qdrant-only retriever via factory)
from typing import List, Dict
from src.db.db import init_schema, insert_playbooks, fetch_playbooks
from src.retrievers.factory import get_retriever

# Your sample playbooks (unchanged)
SAMPLE_PLAYBOOKS = [
    dict(title="Duplicate Email on Contact", signature="DUPLICATE_VALUE Contact.Email", body="""
Diagnosis: A duplicate rule or matching rule blocked Contact creation due to an existing email.

- Verify the duplicate rule conditions for Contact.Email.
- Search for the existing Contact using the email; merge or update instead of creating.
- If the record is legit, adjust the rule or use an allowlist field.
- Re-run the Flow after resolving the conflict.
"""),
    dict(title="Field Validation: Postal Code", signature="FIELD_CUSTOM_VALIDATION_EXCEPTION Postal Code", body="""
Diagnosis: Custom validation enforces a 5-digit postal code.

- Normalize the input to 5 digits (strip non-digits).
- If international, update validation to allow country-specific formats.
- Re-run the Flow after correcting the data.
"""),
    dict(title="Picklist: Restricted Value", signature="INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST", body="""
Diagnosis: The value isn't in the restricted picklist for this field.

- Check the picklist value set for the field.
- Map incoming values to allowed values; add synonyms if needed.
- Re-run the Flow with a permitted value.
"""),
    dict(title="Row Locking", signature="UNABLE_TO_LOCK_ROW", body="""
Diagnosis: A concurrent transaction holds a lock on the same record.

- Retry with backoff; avoid bulk updates to the same parent simultaneously.
- Reduce transaction scope; update child rows outside critical section.
- Use Platform Events/Queueable for retries.
"""),
]

def _to_text(row: Dict) -> str:
    """Flatten title/signature/body into a single passage for embedding."""
    parts: List[str] = []
    t = (row.get("title") or "").strip()
    s = (row.get("signature") or "").strip()
    b = (row.get("body") or "").strip()
    if t: parts.append(f"Title: {t}")
    if s: parts.append(f"Signature: {s}")
    if b: parts += ["Body:", b]
    return "\n".join(parts)

if __name__ == "__main__":
    # 1) seed DB
    init_schema()
    insert_playbooks(SAMPLE_PLAYBOOKS)

    # 2) read rows back and upsert into Qdrant via the retriever factory
    rows = fetch_playbooks()  # [{id, title, signature, body, ...}]
    if not rows:
        raise SystemExit("No playbooks found after insert_playbooks().")

    retriever = get_retriever()  # currently returns QdrantRetriever
    chunks = []
    for r in rows:
        text = _to_text(r)
        chunks.append({
            "id": r.get("id"),  # optional stable id for idempotency
            "text": text,
            "meta": {
                "source_id": "kb",
                "chunk_id": r.get("id"),
                "version": "kb@v1",
                # add org-scoping later if needed: "org_id": "global"
            }
        })

    n = retriever.upsert(chunks)
    print(f"Seeded {len(rows)} playbooks and upserted {n} chunks into Qdrant collection.")

