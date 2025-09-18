
import sqlite3, os
from typing import Iterable, Dict, Any, List
from config import DATA_DB
from typing import List, Dict
def _conn():
    con = sqlite3.connect(DATA_DB)
    con.row_factory = sqlite3.Row
    return con

def init_schema():
    con = _conn(); cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id TEXT,
        flow_name TEXT,
        flow_version INTEGER,
        flow_element TEXT,
        error_message TEXT,
        created_at TEXT,
        record_ids TEXT,
        user_id TEXT,
        raw TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_errors_created ON errors(created_at);
    CREATE INDEX IF NOT EXISTS idx_errors_flow ON errors(flow_name, flow_version);

    CREATE TABLE IF NOT EXISTS playbooks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        signature TEXT,
        body TEXT
    );
    """)
    con.commit(); con.close()

def insert_error(row: Dict[str, Any]):
    con = _conn(); cur = con.cursor()
    cur.execute("""
        INSERT INTO errors (org_id, flow_name, flow_version, flow_element, error_message,
                            created_at, record_ids, user_id, raw)
        VALUES (:org_id, :flow_name, :flow_version, :flow_element, :error_message,
                :created_at, :record_ids, :user_id, :raw)
    """, row)
    con.commit(); con.close()

def list_recent_errors(limit: int = 5) -> List[sqlite3.Row]:
    con = _conn(); cur = con.cursor()
    cur.execute("SELECT * FROM errors ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall(); con.close()
    return rows

def list_playbooks() -> List[sqlite3.Row]:
    con = _conn(); cur = con.cursor()
    cur.execute("SELECT id, title, signature, body FROM playbooks ORDER BY id ASC");
    rows = cur.fetchall(); con.close()
    return rows

def insert_playbooks(rows: Iterable[Dict[str, str]]):
    con = _conn(); cur = con.cursor()
    cur.executemany("""
        INSERT INTO playbooks (title, signature, body) VALUES (:title, :signature, :body)
    """, list(rows))
    con.commit(); con.close()


def fetch_playbooks() -> List[Dict]:
    with _conn() as cx:
        cx.row_factory = sqlite3.Row
        cur = cx.execute("SELECT id, title, signature, body FROM playbooks ORDER BY id ASC")
        return [dict(r) for r in cur.fetchall()]

def upsert_playbooks(rows):
    con = _conn(); cur = con.cursor()
    cur.executemany("""
        INSERT INTO playbooks (title, signature, body)
        VALUES (:title, :signature, :body)
        ON CONFLICT(signature) DO UPDATE SET
            title=excluded.title,
            body =excluded.body
    """, list(rows))
    con.commit(); con.close()
