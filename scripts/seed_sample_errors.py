#!/usr/bin/env python
import argparse, json, sys
from typing import Dict, Any, List, Optional
from src.db.db import insert_errors, init_schema

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True, help="Path to JSONL of errors")
    ap.add_argument("--org-id", default=None,
                    help="If set, override/set raw.org_id for every seeded error")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on rows")
    args = ap.parse_args()

    init_schema()  # idempotent, safe

    rows: List[Dict[str, Any]] = []
    with open(args.from_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[seed][WARN] bad JSON line skipped: {e}", file=sys.stderr)
                continue

            # We only *need* raw; signature/message can be backfilled downstream
            raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else dict(rec)
            if args.org_id:
                raw["org_id"] = args.org_id

            rows.append({
                "signature": rec.get("signature"),
                "message": rec.get("message") or rec.get("error_message"),
                "raw": raw,
            })

            if args.limit and len(rows) >= args.limit:
                break

    if not rows:
        print("[seed] no rows to insert"); return

    n = insert_errors(rows)
    print(f"[seed] inserted {n} errors (org={'as-is' if not args.org_id else args.org_id})")

if __name__ == "__main__":
    main()
