#!/usr/bin/env bash
set -euo pipefail

# Creates src/ layout and moves files using git mv (preserving history).
# Safe to run once on your old structure. It only moves files that actually exist.

need_git() {
  if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required (not found in PATH)"; exit 1
  fi
}

mv_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    if [ "$src" != "$dst" ]; then
      echo "git mv \"$src\" \"$dst\""
      git mv "$src" "$dst"
    fi
  fi
}

need_git

branch="refactor/src-layout"
current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
if [ -n "$current_branch" ] && [ "$current_branch" != "$branch" ]; then
  git checkout -b "$branch"
fi

# Ensure target dirs
mkdir -p src/config src/db src/embeddings src/llm

# Move from old flat files → src/
mv_if_exists "config.py" "src/config/config.py"
mv_if_exists "db.py" "src/db/db.py"
mv_if_exists "docs_search.py" "src/llm/docs_search.py"
mv_if_exists "rag_generate.py" "src/llm/rag_generate.py"
mv_if_exists "embed.py" "src/embeddings/embed.py"
mv_if_exists "kb_seed.py" "src/embeddings/kb_seed.py"
mv_if_exists "pdf_chunk_index_pdfminer.py" "src/embeddings/pdf_chunk_index_pdfminer.py"

# If you already had these grouped but not under src/, move them under src/
mv_if_exists "config/config.py" "src/config/config.py"
mv_if_exists "db/db.py" "src/db/db.py"
mv_if_exists "embeddings/embed.py" "src/embeddings/embed.py"
mv_if_exists "embeddings/kb_seed.py" "src/embeddings/kb_seed.py"
mv_if_exists "embeddings/pdf_chunk_index_pdfminer.py" "src/embeddings/pdf_chunk_index_pdfminer.py"
mv_if_exists "llm/docs_search.py" "src/llm/docs_search.py"
mv_if_exists "llm/rag_generate.py" "src/llm/rag_generate.py"

# Move retriever(s) directory if present
if [ -d "retrievers" ]; then
  mkdir -p src
  echo "git mv retrievers src/retrievers"
  git mv "retrievers" "src/retrievers"
elif [ -d "retriever" ]; then
  mkdir -p src
  echo "git mv retriever src/retriever"
  git mv "retriever" "src/retriever"
fi

# Optional: keep CLI/dev helpers outside src/
mv_if_exists "seed_sample_errors.py" "scripts/seed_sample_errors.py"

echo
echo "Move complete. Next, run:  python fix_imports_src.py"

