"""Apply schema.sql idempotently and record the schema version.

For fresh databases, ``schema.sql`` creates everything in one go. For
existing databases that pre-date a column or index addition, we use
``ALTER TABLE``-based migrations guarded by ``schema_version`` so the
upgrade is non-destructive and re-entrant.

Bump ``SCHEMA_VERSION`` AND register a callable in ``_VERSIONED_MIGRATIONS``
when adding a column / index / table that needs to land on already-deployed
DBs (i.e. anything that ``CREATE TABLE IF NOT EXISTS`` cannot patch in).
"""

from __future__ import annotations

from pathlib import Path

from src.utils.db import connect, execute_script
from src.utils.logger import get_logger

log = get_logger("db.migrate")

SCHEMA_VERSION = 4  # v4: paper_trades enrichment + outbox unique index (Week 5)
_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


# ---------------------------------------------------------------------------
# Versioned migrations -- run for DBs that started on an older version.
#
# Each handler receives an open sqlite3.Connection and must be idempotent
# (use IF NOT EXISTS / PRAGMA table_info checks). NEVER drop or rename
# columns here -- that breaks downgrade and audit replay.
# ---------------------------------------------------------------------------


def _table_columns(conn, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_to_v4(conn) -> None:
    """v4: enrich paper_trades with lifecycle + risk columns; ensure
    the unique index on signal_outbox(symbol, signal_date) exists.

    Existing DBs created on v3 only have the original 13 columns, so we
    add the new ones via ALTER TABLE. SQLite supports adding nullable
    columns and columns with constant defaults; that's all we need.
    """
    have = _table_columns(conn, "paper_trades")
    additions: list[tuple[str, str]] = [
        ("sector",         "TEXT"),
        ("status",         "TEXT NOT NULL DEFAULT 'open'"),
        ("stop_loss",      "REAL"),
        ("take_profit",    "REAL"),
        ("trailing_stop",  "REAL"),
        ("entry_atr",      "REAL"),
        ("high_watermark", "REAL"),
        ("exit_reason",    "TEXT"),
        ("entry_prob",     "REAL"),
        ("threshold",      "REAL"),
        ("run_id",         "TEXT"),
        # NOTE: SQLite ALTER TABLE rejects non-constant defaults, so for
        # the timestamp columns we use a fixed sentinel and rely on app
        # writes to keep them current.
        ("created_at",     "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z'"),
        ("updated_at",     "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z'"),
    ]
    for col, ddl in additions:
        if col not in have:
            log.info("v4 migration: adding paper_trades.{}", col)
            conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {ddl}")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_status ON paper_trades(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_symbol ON paper_trades(symbol)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_outbox_uq_symbol_date "
        "ON signal_outbox(symbol, signal_date)"
    )


_VERSIONED_MIGRATIONS = {
    4: _migrate_to_v4,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _current_version(conn) -> int:
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


def apply_schema(db_path: str | None = None) -> int:
    """Apply the schema (CREATE IF NOT EXISTS) then versioned ALTERs.

    Returns the post-apply version.
    """
    sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    execute_script(sql, db_path=db_path)

    conn = connect(db_path)
    try:
        # The schema_version table is created by schema.sql. Read what's there
        # before recording our new version so we know which migrations to run.
        prev = _current_version(conn)
        if prev < SCHEMA_VERSION:
            for v in range(prev + 1, SCHEMA_VERSION + 1):
                handler = _VERSIONED_MIGRATIONS.get(v)
                if handler is not None:
                    log.info("Running migration to v{}", v)
                    handler(conn)
        cur = conn.execute(
            "INSERT OR IGNORE INTO schema_version(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()
        if cur.rowcount:
            log.info("Schema version recorded: {}", SCHEMA_VERSION)
        cur.close()
        return _current_version(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    v = apply_schema()
    log.info("Schema is at version {}", v)
