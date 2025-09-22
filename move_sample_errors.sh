#!/usr/bin/env bash
set -euo pipefail

need_git() { command -v git >/dev/null || { echo "git is required"; exit 1; }; }
need_git

branch="refactor/move-sample-errors"
current="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
[ -n "$current" ] && [ "$current" != "$branch" ] && git checkout -b "$branch"

# 1) Move the file (preserves history)
mkdir -p test/data
if [ -e "sample_errors.jsonl" ]; then
  git mv "sample_errors.jsonl" "test/data/sample_errors.jsonl"
fi

# 2) Update references across the repo (skip .sh files to avoid self-modification)
python - <<'PY'
import pathlib
root = pathlib.Path('.')
old = 'sample_errors.jsonl'
new = 'test/data/sample_errors.jsonl'

SCAN_EXTS = {'.py','.md','.txt','.yml','.yaml','.ini','.cfg'}  # intentionally exclude .sh
for p in root.rglob('*'):
    if not p.is_file(): 
        continue
    if p.suffix.lower() not in SCAN_EXTS:
        continue
    s = p.read_text(encoding='utf-8', errors='ignore')
    if old in s:
        p.write_text(s.replace(old, new), encoding='utf-8')
        print('updated', p)
PY

echo "Done. Review with: git status && git diff"

