"""SQLite helpers with WAL mode and a single-writer pattern.

Why these settings:
- WAL gives readers (Streamlit dashboard) concurrent access while one writer
  (the daily pipeline / scheduler) appends. Without WAL we get
  "database is locked" failures under load.
- foreign_keys=ON: enforce relational integrity (off by default in SQLite).
- synchronous=NORMAL with WAL: durable across crashes for our use case while
  ~10x faster than FULL.
- busy_timeout: lets concurrent writers wait briefly instead of failing fast.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.utils.logger import get_logger
from src.utils.secrets import get_settings, project_root

log = get_logger("utils.db")


def resolve_db_path(db_path: str | None = None) -> Path:
    """Resolve the configured DB path to an absolute Path inside the project."""
    raw = db_path or get_settings().db_path
    p = Path(raw)
    if not p.is_absolute():
        p = project_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA busy_timeout=5000;")  # ms
    cur.execute("PRAGMA cache_size=-20000;")  # ~20MB cache
    cur.close()


def connect(db_path: str | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    p = resolve_db_path(db_path)
    if read_only:
        uri = f"file:{p.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None)
    else:
        conn = sqlite3.connect(str(p), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


@contextmanager
def transaction(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Explicit transaction context. Commits on success, rolls back on error."""
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


def execute_script(script_sql: str, db_path: str | None = None) -> None:
    """Apply a multi-statement SQL script (e.g. schema.sql)."""
    conn = connect(db_path)
    try:
        conn.executescript(script_sql)
    finally:
        conn.close()


def fetch_one(query: str, params: tuple = (), db_path: str | None = None) -> sqlite3.Row | None:
    conn = connect(db_path, read_only=True)
    try:
        cur = conn.execute(query, params)
        return cur.fetchone()
    finally:
        conn.close()


def fetch_all(query: str, params: tuple = (), db_path: str | None = None) -> list[sqlite3.Row]:
    conn = connect(db_path, read_only=True)
    try:
        cur = conn.execute(query, params)
        return list(cur.fetchall())
    finally:
        conn.close()


def execute(query: str, params: tuple = (), db_path: str | None = None) -> int:
    """Run a single write statement inside an IMMEDIATE transaction.

    Returns rowcount. For batch writes prefer `executemany` or use the
    `transaction()` context directly so all statements share one txn.
    """
    with transaction(db_path) as conn:
        cur = conn.execute(query, params)
        return cur.rowcount


def executemany(query: str, params_seq, db_path: str | None = None) -> int:
    """Bulk-write helper. Wraps everything in a single IMMEDIATE transaction
    so partial failures do not leave the DB in an inconsistent state.
    """
    with transaction(db_path) as conn:
        cur = conn.executemany(query, params_seq)
        return cur.rowcount
