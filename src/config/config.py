
from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class Settings:
    # Heroku sets DATABASE_URL automatically. Local dev default below.
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/sfem"
    )

settings = Settings()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
DATA_DB = Path(os.getenv("DATA_DB", "data/errors.db"))
TOP_K = int(os.getenv("TOP_K", "3"))
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")  # "faiss" | "qdrant"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sfem_kb")
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")

os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DB.parent.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")