from __future__ import annotations
import uuid, hashlib

_NAMESPACE = uuid.UUID("800cd911-c69f-46d9-8407-2908c94a6d65")

def pdf_chunk_id(doc_id: str, page: int, idx: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{page}:{idx}"))

def html_chunk_id(url: str, anchor: str|None) -> str:
    h = hashlib.md5((url or '').encode('utf-8')).hexdigest()[:10]
    return f"{h}#{anchor}" if anchor else h
