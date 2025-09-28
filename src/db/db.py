# src/db/db.py
from __future__ import annotations
from typing import Iterable, Dict, Any, List, Optional
from psycopg_pool import ConnectionPool
from src.config.config import settings
from psycopg.types.json import Json

# aliases to backfill missing fields from raw JSON
_SIGNATURE_KEYS = ("signature","error_signature","errorSignature","statusCode","errorCode","exceptionType","code")
_MESSAGE_KEYS   = ("message","error","text","detail","details","errorMessage","error_description")
_ORG_KEYS       = ("org_id","org","OrgId","organization_id","organizationId")
_FLOW_KEYS      = ("flow_name","flow","FlowName","flowLabel","flow_label","flowApiName","flowName","Flow API Name")

def _first_from(d, keys):
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
    return None


# Small pooled connector (works locally and on Heroku)
POOL = ConnectionPool(
    settings.database_url,
    min_size=1, max_size=5,
    kwargs={"autocommit": True},
)

def init_schema() -> None:
    """Create minimal tables if they don't exist."""
    with POOL.connection() as conn, conn.cursor() as cur:
        # Playbooks used for next-step guidance
        cur.execute("""
        CREATE TABLE IF NOT EXISTS playbooks (
            id BIGSERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            signature TEXT NOT NULL UNIQUE,
            body TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """)
        # Recent SF Flow errors (lightweight log)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id BIGSERIAL PRIMARY KEY,
            signature TEXT,
            message   TEXT,
            raw       JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_errors_created_at ON errors (created_at DESC);")

def insert_playbooks(items: Iterable[Dict[str, Any]]) -> int:
    """
    Upsert playbooks by signature (title/body updated on conflict).
    Shape: [{"title": str, "signature": str, "body": str}, ...]
    """
    rows = [(i.get("title",""), i.get("signature",""), i.get("body","")) for i in items]
    if not rows:
        return 0
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO playbooks (title, signature, body)
            VALUES (%s, %s, %s)
            ON CONFLICT (signature) DO UPDATE
              SET title = EXCLUDED.title,
                  body  = EXCLUDED.body,
                  updated_at = now();
            """,
            rows,
        )
        # psycopg rowcount with executemany may be -1; return number attempted
        return len(rows)

def fetch_playbooks(signature: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch playbooks; optionally filter by exact signature."""
    with POOL.connection() as conn, conn.cursor() as cur:
        if signature:
            cur.execute(
                "SELECT id, title, signature, body FROM playbooks WHERE signature = %s",
                (signature,),
            )
        else:
            cur.execute("SELECT id, title, signature, body FROM playbooks")
        return [
            {"id": r[0], "title": r[1], "signature": r[2], "body": r[3]}
            for r in cur.fetchall()
        ]


def insert_error(item: Union[Dict[str, Any], str, None],
                 message: Optional[str] = None,
                 raw: Optional[dict] = None) -> int:
    """
    Flexible insert:
      - insert_error({"signature": "...", "message": "...", "raw": {...}})
      - insert_error("SIG", "Message", {"k": "v"})
    Returns new row id.
    """
    if isinstance(item, dict):
        signature = item.get("signature")
        message = item.get("message")
        raw = item.get("raw")
    else:
        signature = item

    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO errors (signature, message, raw) VALUES (%s, %s, %s) RETURNING id",
            (signature, message, Json(raw) if raw is not None else None),
        )
        return cur.fetchone()[0]

def insert_errors(items: Iterable[Dict[str, Any]]) -> int:
    """
    Bulk insert for list of dicts with keys: signature, message, raw (optional).
    """
    rows = []
    for it in items:
        rows.append((
            it.get("signature"),
            it.get("message"),
            Json(it.get("raw")) if it.get("raw") is not None else None
        ))
    if not rows:
        return 0
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO errors (signature, message, raw) VALUES (%s, %s, %s)",
            rows,
        )
    return len(rows)

def list_recent_errors(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Returns recent errors; flattens `raw`, guarantees keys, and
    backfills signature/message/org_id/flow_name from `raw` if missing.
    """
    with POOL.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, signature, message, raw, created_at "
            "FROM errors ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        out: List[Dict[str, Any]] = []
        for rid, sig, msg, raw, ts in cur.fetchall():
            row: Dict[str, Any] = {
                "id": rid,
                "signature": sig,
                "message": msg,
                "created_at": ts,
                "raw": raw,
                "org_id": None,
                "flow_name": None,
            }
            # flatten raw without overwriting already present keys
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if k not in row:
                        row[k] = v

                # backfill normalized fields from aliases
                if row["org_id"] is None:
                    row["org_id"] = _first_from(raw, _ORG_KEYS)
                if row["flow_name"] is None:
                    row["flow_name"] = _first_from(raw, _FLOW_KEYS)
                if row["signature"] is None:
                    row["signature"] = _first_from(raw, _SIGNATURE_KEYS)
                if row["message"] is None:
                    row["message"] = _first_from(raw, _MESSAGE_KEYS)

            # ensure keys exist
            row.setdefault("org_id", None)
            row.setdefault("flow_name", None)
            out.append(row)
        return out

