#!/usr/bin/env python3
"""
Minimal, targeted import updates for the new src/ layout.
- Updates simple 'import X' and 'from X import Y' where X moved under src.
- Leaves stdlib/third-party imports alone.

Rules:
  db            -> src.db.db
  config        -> src.config.config
  embed         -> src.embeddings.embed
  kb_seed       -> src.embeddings.kb_seed
  pdf_chunk_index_pdfminer -> src.embeddings.pdf_chunk_index_pdfminer
  docs_search   -> src.llm.docs_search
  rag_generate  -> src.llm.rag_generate
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent

# Skip editing this fixer itself
SKIP_FILES = {"fix_imports_src.py"}

# Compile rewrite rules
rules = [
    # db
    (re.compile(r'(^|\n)\s*import\s+db(\s|$)', re.M), r'\1
from src.db import db as db\2'),
    (re.compile(r'(^|\n)\s*from\s+db\s+import\s+', re.M), r'from src.db.db import '),

    # config
    (re.compile(r'(^|\n)\s*import\s+config(\s|$)', re.M), r'\1
from src.config import config as config\2'),
    (re.compile(r'(^|\n)\s*from\s+config\s+import\s+', re.M), r'from src.config.config import '),

    # embeddings
    (re.compile(r'(^|\n)\s*import\s+embed(\s|$)', re.M), r'\1
from src.embeddings import embed\2'),
    (re.compile(r'(^|\n)\s*from\s+embed\s+import\s+', re.M), r'from src.embeddings.embed import '),

    (re.compile(r'(^|\n)\s*import\s+kb_seed(\s|$)', re.M), r'\1
from src.embeddings import kb_seed\2'),
    (re.compile(r'(^|\n)\s*from\s+kb_seed\s+import\s+', re.M), r'from src.embeddings.kb_seed import '),

    (re.compile(r'(^|\n)\s*import\s+pdf_chunk_index_pdfminer(\s|$)', re.M), r'\1
from src.embeddings import pdf_chunk_index_pdfminer\2'),
    (re.compile(r'(^|\n)\s*from\s+pdf_chunk_index_pdfminer\s+import\s+', re.M), r'from src.embeddings.pdf_chunk_index_pdfminer import '),

    # llm
    (re.compile(r'(^|\n)\s*import\s+docs_search(\s|$)', re.M), r'\1
from src.llm import docs_search\2'),
    (re.compile(r'(^|\n)\s*from\s+docs_search\s+import\s+', re.M), r'from src.llm.docs_search import '),

    (re.compile(r'(^|\n)\s*import\s+rag_generate(\s|$)', re.M), r'\1
from src.llm import rag_generate\2'),
    (re.compile(r'(^|\n)\s*from\s+rag_generate\s+import\s+', re.M), r'from src.llm.rag_generate import '),
]

# Heuristic skip list for well-known libs (avoid accidental rewrites)
SKIP_BASES = {
    "os","sys","re","json","pathlib","typing","logging","datetime","time","subprocess","itertools",
    "collections","functools","math","random","argparse","pandas","numpy","torch","transformers",
    "sentence_transformers","openai","anthropic","qdrant_client","faiss","sklearn","pytest","dotenv"
}

def should_skip_import(name: str) -> bool:
    base = name.split(".")[0]
    return base in SKIP_BASES

# Lightweight pass: update only when patterns match clearly
changed = 0
for py in ROOT.rglob("*.py"):
    if py.name in SKIP_FILES:
        continue
    text = py.read_text(encoding="utf-8", errors="ignore")

    # Fast skip: if file doesn't reference any of our known modules, move on
    if not re.search(r'\b(db|config|embed|kb_seed|pdf_chunk_index_pdfminer|docs_search|rag_generate)\b', text):
        continue

    new = text
    for pat, repl in rules:
        new = pat.sub(repl, new)

    if new != text:
        py.write_text(new, encoding="utf-8")
        changed += 1

print(f"Updated imports in {changed} files.")

