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

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.utils.logger import get_logger
from src.utils.secrets import get_settings, project_root

log = get_logger("utils.db")


# ---------------------------------------------------------------------------
# Optional Postgres (Neon) read backend for the cloud dashboard.
#
# The local pipeline always uses SQLite (full dataset). When the dashboard is
# deployed to the cloud it reads a small *mirror* of the dashboard-relevant
# tables from a managed Postgres (Neon). To keep behaviour identical to SQLite
# we only route the READ helpers (fetch_one/fetch_all) to Postgres, and only
# when explicitly opted in via WEB_DB_BACKEND=postgres (so local runs never
# accidentally read the smaller cloud mirror). Writes always go to SQLite.
#
# The connection string comes from the DATABASE_URL env var -- never
# hardcoded -- and Neon mandates TLS (sslmode=require) by default.
# ---------------------------------------------------------------------------

_pg_tls = threading.local()


def _pg_url() -> str | None:
    url = (os.getenv("DATABASE_URL") or "").strip()
    return url or None


def use_postgres() -> bool:
    """True when reads should be served from the Postgres mirror.

    Deterministic and opt-in: requires WEB_DB_BACKEND=postgres *and* a
    DATABASE_URL. Local development and the test suite never set the flag, so
    they always use SQLite.
    """
    backend = (os.getenv("WEB_DB_BACKEND") or "").strip().lower()
    return backend == "postgres" and _pg_url() is not None


def _to_pg(query: str) -> str:
    """Translate SQLite ``?`` placeholders to psycopg ``%s``.

    Our read queries never contain a literal ``?`` or ``%`` inside string
    literals, so this straight substitution is safe. (Date/`rowid` dialect
    differences are handled in the queries themselves, not here.)
    """
    return query.replace("?", "%s")


def _pg_connect():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(
        _pg_url(),
        autocommit=True,            # read-only SELECTs; avoid idle-in-txn
        connect_timeout=15,
        row_factory=dict_row,       # rows behave like dicts (row["col"])
    )


def _pg_conn():
    """A per-thread cached Postgres connection (Starlette runs sync DB work in
    a bounded threadpool, so one connection per worker thread is plenty)."""
    conn = getattr(_pg_tls, "conn", None)
    if conn is not None and not conn.closed:
        return conn
    conn = _pg_connect()
    _pg_tls.conn = conn
    return conn


def _pg_reset() -> None:
    conn = getattr(_pg_tls, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    _pg_tls.conn = None


def _pg_query(query: str, params: tuple, *, one: bool):
    import psycopg

    sql = _to_pg(query)
    try:
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if one else list(cur.fetchall())
    except psycopg.Error as exc:
        # Stale pooled connection (Neon idles them out) -> reconnect once.
        log.warning("postgres read failed, retrying once: {}", exc)
        _pg_reset()
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone() if one else list(cur.fetchall())


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


def fetch_one(query: str, params: tuple = (), db_path: str | None = None) -> Any | None:
    # Route to the Postgres mirror only for the default DB (no explicit
    # db_path) and only when the cloud backend is opted in. Returns a
    # dict-like row in both backends (sqlite3.Row and psycopg dict_row both
    # support row["col"] and dict(row)).
    if db_path is None and use_postgres():
        return _pg_query(query, params, one=True)
    conn = connect(db_path, read_only=True)
    try:
        cur = conn.execute(query, params)
        return cur.fetchone()
    finally:
        conn.close()


def fetch_all(query: str, params: tuple = (), db_path: str | None = None) -> list[Any]:
    if db_path is None and use_postgres():
        return _pg_query(query, params, one=False)
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
