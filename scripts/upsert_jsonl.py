#!/usr/bin/env python3
from __future__ import annotations
import argparse
from src.qdrant.upsert_jsonl import upsert_jsonls

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Upsert JSONL files into Qdrant")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    upsert_jsonls(args.inputs, args.batch)
