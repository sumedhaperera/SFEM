#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from src.db.db import init_schema, insert_error  # Postgres-backed

def seed_from_json(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Let insert_error backfill from raw
            insert_error({"signature": obj.get("signature"), "message": obj.get("message"), "raw": obj})
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True)
    args = ap.parse_args()
    init_schema()
    n = seed_from_json(args.from_json)
    print(f"Seeded {n} error rows.")

if __name__ == "__main__":
    main()
