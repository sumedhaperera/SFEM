
import argparse, json
from db import init_schema, insert_error

def seed_from_json(path: str):
    init_schema()
    n = 0
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            row = {
                "org_id": data["org_id"],
                "flow_name": data["flow_name"],
                "flow_version": int(data.get("flow_version") or 0),
                "flow_element": data.get("flow_element") or "Unknown",
                "error_message": data.get("error_message") or "Unknown",
                "created_at": data["created_at"],
                "record_ids": json.dumps(data.get("record_ids") or []),
                "user_id": data.get("user_id"),
                "raw": json.dumps(data.get("raw") or {}),
            }
            insert_error(row); n += 1
    print(f"Seeded {n} error rows into SQLite at data/errors.db")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True, help="Path to sample_errors.jsonl")
    args = ap.parse_args()
    seed_from_json(args.from_json)
