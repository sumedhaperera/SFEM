from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    source: str
    product: str = "flow"
    url: Optional[str] = None
    title: Optional[str] = None
    headings_path: Optional[List[str]] = None
    anchor: Optional[str] = None
    release_train: Optional[str] = None
    lastmod: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    org_id: Optional[str] = None
    crawl_ts: Optional[str] = None
