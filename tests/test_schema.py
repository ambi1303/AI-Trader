"""Verify schema applies cleanly and key tables exist."""

from __future__ import annotations

from src.db.migrate import apply_schema
from src.utils.db import fetch_all, fetch_one


EXPECTED_TABLES = {
    "schema_version",
    "nifty_constituents",
    "trading_calendar",
    "price_data",
    "corporate_actions",
    "circuit_flags",
    "news_headlines",
    "model_runs",
    "predictions_log",
    "signal_outbox",
    "paper_trades",
    "validation_failures",
    # Week 2
    "index_data",
    "stock_sectors",
    "feature_data",
}


def test_apply_schema_idempotent() -> None:
    v1 = apply_schema()
    v2 = apply_schema()
    assert v1 == v2 >= 1


def test_expected_tables_exist() -> None:
    apply_schema()
    rows = fetch_all(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {r["name"] for r in rows}
    missing = EXPECTED_TABLES - names
    assert not missing, f"Missing tables: {missing}"


def test_wal_mode_active() -> None:
    apply_schema()
    row = fetch_one("PRAGMA journal_mode;")
    assert row is not None
    # SQLite returns the current mode in column "journal_mode"
    val = row["journal_mode"] if "journal_mode" in row.keys() else row[0]
    assert str(val).lower() == "wal"


def test_foreign_keys_enabled() -> None:
    apply_schema()
    row = fetch_one("PRAGMA foreign_keys;")
    assert row is not None
    val = row["foreign_keys"] if "foreign_keys" in row.keys() else row[0]
    assert int(val) == 1


def test_price_data_check_constraints_reject_bad_rows() -> None:
    apply_schema()
    from src.utils.db import transaction
    import sqlite3

    # high < low must fail
    raised = False
    try:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO price_data "
                "(symbol, bar_date, open, high, low, close, volume, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("RELIANCE", "2024-01-02", 100, 99, 100, 100, 1000, "yfinance"),
            )
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "CHECK (high>=low) should have rejected the row"
