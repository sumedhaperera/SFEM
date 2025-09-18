
from db import init_schema, insert_playbooks
from retrieval import rebuild_index

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

if __name__ == "__main__":
    init_schema()
    insert_playbooks(SAMPLE_PLAYBOOKS)
    rebuild_index()
    print("Seeded playbooks and built HNSW index at index/kb.index")
