# embed.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.config import EMBED_MODEL

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    def _encode(texts: List[str]) -> np.ndarray:
        X = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        X = X.astype("float32", copy=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms
    return _encode
