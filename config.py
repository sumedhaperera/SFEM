
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
DATA_DB = Path(os.getenv("DATA_DB", "data/errors.db"))
TOP_K = int(os.getenv("TOP_K", "3"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DB.parent.mkdir(parents=True, exist_ok=True)
