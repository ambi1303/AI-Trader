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

SCHEMA_VERSION = 7  # v7: price_forecasts provenance (method, model_run_id)
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


def _migrate_to_v5(conn) -> None:
    """v5: fundamentals support + tri-class / price-target prediction columns.

    ``fundamental_data`` itself is created by schema.sql (CREATE IF NOT
    EXISTS) which runs before these handlers, so here we only patch the
    pre-existing wide tables that ALTER TABLE must touch:

    * ``feature_data`` gains nullable fundamental feature columns.
    * ``predictions_log`` gains the verdict / class-probability /
      predicted-return / target / stop columns the tri-class model emits.

    All additions are nullable so old rows remain valid.
    """
    feat_cols = _table_columns(conn, "feature_data")
    feat_additions: list[tuple[str, str]] = [
        ("pe_ttm",          "REAL"),
        ("pb",              "REAL"),
        ("roe",             "REAL"),
        ("debt_to_equity",  "REAL"),
        ("profit_margin",   "REAL"),
        ("revenue_growth",  "REAL"),
        ("earnings_growth", "REAL"),
        ("dividend_yield",  "REAL"),
        ("log_market_cap",  "REAL"),
    ]
    for col, ddl in feat_additions:
        if col not in feat_cols:
            log.info("v5 migration: adding feature_data.{}", col)
            conn.execute(f"ALTER TABLE feature_data ADD COLUMN {col} {ddl}")

    pred_cols = _table_columns(conn, "predictions_log")
    pred_additions: list[tuple[str, str]] = [
        ("verdict",          "TEXT"),
        ("prob_buy",         "REAL"),
        ("prob_hold",        "REAL"),
        ("prob_sell",        "REAL"),
        ("predicted_return", "REAL"),
        ("target_price",     "REAL"),
        ("stop_price",       "REAL"),
    ]
    for col, ddl in pred_additions:
        if col not in pred_cols:
            log.info("v5 migration: adding predictions_log.{}", col)
            conn.execute(f"ALTER TABLE predictions_log ADD COLUMN {col} {ddl}")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_fundamental_symbol_date "
        "ON fundamental_data(symbol, as_of_date)"
    )


def _migrate_to_v6(conn) -> None:
    """v6: per-regime backtest tagging.

    ``backtest_trades`` gains a nullable ``entry_regime`` column so each trade
    records the market regime active at entry. The ``market_regime`` table
    itself is a brand-new standalone table created by schema.sql's
    CREATE IF NOT EXISTS (which runs before these handlers), so no ALTER is
    needed for it here.
    """
    bt_cols = _table_columns(conn, "backtest_trades")
    if "entry_regime" not in bt_cols:
        log.info("v6 migration: adding backtest_trades.entry_regime")
        conn.execute("ALTER TABLE backtest_trades ADD COLUMN entry_regime TEXT")


def _migrate_to_v7(conn) -> None:
    """v7: provenance on price forecasts.

    ``price_forecasts`` gains ``method`` ('ml' vs analytic 'drift') and
    ``model_run_id`` (the horizon-bundle run that produced an ML row) so the UI
    and audits can tell learned targets apart from the transparent projection.
    Both are nullable; existing rows predate the learned model and read as the
    analytic projection.
    """
    fc_cols = _table_columns(conn, "price_forecasts")
    for col, ddl in (("method", "TEXT"), ("model_run_id", "TEXT")):
        if col not in fc_cols:
            log.info("v7 migration: adding price_forecasts.{}", col)
            conn.execute(f"ALTER TABLE price_forecasts ADD COLUMN {col} {ddl}")


_VERSIONED_MIGRATIONS = {
    4: _migrate_to_v4,
    5: _migrate_to_v5,
    6: _migrate_to_v6,
    7: _migrate_to_v7,
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
