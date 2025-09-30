# src/db/db.py
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Iterable, Dict, Any, List, Optional, Union, Tuple

from psycopg_pool import ConnectionPool
from psycopg.types.json import Json
from psycopg import sql
from psycopg import errors as pg_errors
#from psycopg.extras import execute_values  # batch insert helper (psycopg3)

from src.config.config import settings

# aliases to backfill missing fields from raw JSON
_SIGNATURE_KEYS = ("signature","error_signature","errorSignature","statusCode","errorCode","exceptionType","code")
_MESSAGE_KEYS   = ("message","error","text","detail","details","errorMessage","error_description")
_ORG_KEYS       = ("org_id","org","OrgId","organization_id","organizationId")
_FLOW_KEYS      = ("flow_name","flow","FlowName","flowLabel","flow_label","flowApiName","flowName","Flow API Name")

# ---- Adaptive batch insert for citations (scales from tiny to huge) ----
_HAS_EXECUTE_VALUES = False

try:
    from psycopg.extras import execute_values  # optional extra
    _HAS_EXECUTE_VALUES = True
except Exception:
    execute_values = None  # type: ignore

def _insert_citations_batch(cur, rows):
    """
    Insert rows into mitigation_citations using the best available path:
      - COPY for very large batches (> 5000)
      - execute_values if available (51..5000)
      - executemany for small batches (<= 50)
    Each row must be a 7-tuple matching the insert order.
    """
    n = len(rows)
    if n == 0:
        return

    if n > 5000:
        # COPY is fastest for very large loads
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        for r in rows:
            writer.writerow(r)
        buf.seek(0)
        cur.copy(
            "COPY mitigation_citations (mitigation_id, source_type, doc_id, chunk_id, score, url, title) "
            "FROM STDIN WITH (FORMAT CSV)",
            buf,
        )
        return

    if _HAS_EXECUTE_VALUES and n > 50:
        sql = """
        INSERT INTO mitigation_citations(
          mitigation_id, source_type, doc_id, chunk_id, score, url, title
        ) VALUES %s
        """
        execute_values(cur, sql, rows, page_size=1000)
        return

    # tiny batches: simple and reliable
    cur.executemany(
        """
        INSERT INTO mitigation_citations(
          mitigation_id, source_type, doc_id, chunk_id, score, url, title
        ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        rows,
    )



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

# --------------------------------------------------------------------------------------
# Schema init (idempotent)  — includes scalable, multi-tenant mitigations persistence
# --------------------------------------------------------------------------------------
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

        # -------------------------
        # Mitigations (multi-tenant)
        # -------------------------
        # Note: we avoid DEFAULT gen_random_uuid() to prevent pgcrypto permission issues.
        #       We generate run_id in application code.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mitigations (
            id               BIGSERIAL PRIMARY KEY,
            org_id           TEXT NOT NULL,           -- tenant scope
            error_id         BIGINT NOT NULL,         -- references errors(id)
            model_name       TEXT,
            prompt_template  TEXT,
            guidance_md      TEXT NOT NULL,           -- full markdown guidance
            steps_json       JSONB NOT NULL,          -- structured steps (+ raw)
            confidence       NUMERIC,                 -- 0..1
            retrieved_json   JSONB,                   -- raw retriever payloads (TOASTed)
            run_id           UUID NOT NULL,           -- generated in app code
            latency_ms       INT,
            dedupe_hash      TEXT NOT NULL,           -- tenant+error+content hash
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """)

        # Foreign key to errors table (add only if not present)
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM   information_schema.table_constraints
            WHERE  constraint_type = 'FOREIGN KEY'
              AND  table_name = 'mitigations'
              AND  constraint_name = 'fk_mitigations_errors'
          ) THEN
            ALTER TABLE mitigations
              ADD CONSTRAINT fk_mitigations_errors
              FOREIGN KEY (error_id) REFERENCES errors(id) ON DELETE CASCADE;
          END IF;
        END$$;
        """)

        # Tenant-scoped uniqueness improves planner behavior and prevents cross-tenant collisions
        cur.execute("DROP INDEX IF EXISTS ux_mitigations_dedupe;")
        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_mitigations_tenant_dedupe
          ON mitigations (org_id, error_id, dedupe_hash);
        """)

        # Hot read path: list-by-error for a tenant (newest first)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_mitigations_org_error_created
          ON mitigations (org_id, error_id, created_at DESC);
        """)

        # Optional: full-text search over guidance (generated column + GIN)
        cur.execute("""
        ALTER TABLE mitigations
          ADD COLUMN IF NOT EXISTS guidance_tsv tsvector
          GENERATED ALWAYS AS (to_tsvector('english', coalesce(guidance_md, ''))) STORED;
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_mitigations_guidance_tsv
          ON mitigations USING GIN (guidance_tsv);
        """)

        # Per-snippet provenance for a mitigation (for explainability)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mitigation_citations (
            id             BIGSERIAL PRIMARY KEY,
            mitigation_id  BIGINT NOT NULL REFERENCES mitigations(id) ON DELETE CASCADE,
            source_type    TEXT,
            doc_id         TEXT,
            chunk_id       TEXT,
            score          REAL,
            url            TEXT,
            title          TEXT
        );
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_mit_citations_mid
          ON mitigation_citations(mitigation_id);
        """)

# --------------------------------------------------------------------------------------
# Playbooks (unchanged)
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Errors (unchanged public surface)
# --------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------
# Mitigations (multi-tenant) — scalable, atomic insert with batched citations
# --------------------------------------------------------------------------------------
def _dedupe_hash(*, org_id: str, error_id: int, model_name: str,
                 guidance_md: str, steps_json: Dict[str, Any]) -> str:
    """
    Content-based hash scoped by tenant+error so identical content
    in another tenant doesn't collide.
    """
    m = hashlib.sha256()
    m.update(org_id.encode())
    m.update(str(error_id).encode())
    m.update((model_name or "").encode())
    m.update(guidance_md.encode())
    m.update(json.dumps(steps_json, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return m.hexdigest()

def insert_mitigation_atomic(
    *,
    org_id: str,
    error_id: int,
    model_name: str,
    prompt_template: Optional[str],
    guidance_md: str,
    steps_json: Dict[str, Any],
    confidence: Optional[float] = None,
    retrieved_json: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[int] = None,
    citations: Iterable[Dict[str, Any]] = (),
) -> Tuple[int, str]:
    # Compute tenant-scoped de-dupe and generate run_id here
    dedupe = _dedupe_hash(
        org_id=org_id,
        error_id=error_id,
        model_name=model_name or "",
        guidance_md=guidance_md,
        steps_json=steps_json,
    )
    run_id = uuid.uuid4()

    # Parameterized INSERT with tenant-scoped ON CONFLICT
    sql_mit = """
    INSERT INTO mitigations(
      org_id, error_id, model_name, prompt_template,
      guidance_md, steps_json, confidence, retrieved_json,
      run_id, latency_ms, dedupe_hash
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (org_id, error_id, dedupe_hash) DO UPDATE
      SET created_at = NOW()
    RETURNING id, run_id::text;
    """

    with POOL.connection() as conn:
        # Even with pool autocommit=True, this creates an explicit transaction
        with conn.transaction():
            with conn.cursor() as cur:
                # Insert mitigation row (or noop-update if duplicate)
                cur.execute(
                    sql_mit,
                    (
                        org_id,
                        error_id,
                        model_name,
                        prompt_template,
                        guidance_md,
                        Json(steps_json),
                        confidence,
                        Json(retrieved_json) if retrieved_json is not None else None,
                        run_id,
                        latency_ms,
                        dedupe,
                    ),
                )
                mitigation_id, returned_run_id = cur.fetchone()

                # Batch-insert citations using adaptive path
                rows = []
                for c in citations or ():
                    rows.append(
                        (
                            mitigation_id,
                            c.get("source_type"),
                            c.get("doc_id"),
                            c.get("chunk_id"),
                            c.get("score"),
                            c.get("url"),
                            c.get("title"),
                        )
                    )
                _insert_citations_batch(cur, rows)

            # Transaction commits here
            return int(mitigation_id), str(returned_run_id)


