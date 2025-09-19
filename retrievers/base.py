# retrievers/base.py
from typing import List, Dict, Any, Tuple, Iterable, Optional
from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def upsert(self, chunks: Iterable[Dict[str, Any]], batch_size: int = 256) -> int:
        """Upsert chunks: each = {'text': str, 'meta': { ... }}"""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5, org_id: Optional[str] = None, extra_filters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float, str]]:
        """Return [(idx, score, text), ...]; idx can be a chunk_id or -1."""
        raise NotImplementedError

