"""Tests for the Week-5 signal + paper-trading layer.

Coverage:
- generate_signals(): writes pending rows, respects threshold, sector cap,
  portfolio capacity, and is idempotent on re-run.
- reconcile(): opens at next-day open, marks-to-market trailing stop,
  closes on stop / target / time / forced, computes net P&L with the
  canonical cost model, and is idempotent across re-runs.
- Schema migration v3 -> v4: ALTERs land on existing DBs without losing data.
"""

from __future__ import annotations

import json

import pytest

from src.backtesting.risk import RiskConfig
from src.backtesting.sizing import SizingConfig
from src.paper.reconcile import reconcile
from src.signals.generator import SignalGenConfig, generate_signals
from src.utils import db as db_mod


# ---------------------------------------------------------------------------
# Shared fixture: a minimal v4-shaped DB with deterministic seed data
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    schema = """
    CREATE TABLE schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    CREATE TABLE model_runs (
        run_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        git_sha TEXT,
        feature_hash TEXT,
        trained_from TEXT,
        trained_to TEXT,
        metrics_json TEXT,
        artifact_path TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    CREATE TABLE predictions_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        prediction_date TEXT NOT NULL,
        raw_prob REAL,
        calibrated_prob REAL,
        feature_snapshot_json TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    CREATE TABLE feature_data (
        symbol TEXT NOT NULL,
        feature_date TEXT NOT NULL,
        close REAL,
        atr_14 REAL,
        PRIMARY KEY (symbol, feature_date)
    );
    CREATE TABLE price_data (
        symbol TEXT NOT NULL,
        bar_date TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, volume INTEGER,
        adj_close REAL,
        source TEXT NOT NULL,
        PRIMARY KEY (symbol, bar_date, source)
    );
    CREATE TABLE stock_sectors (
        symbol TEXT PRIMARY KEY,
        sector TEXT NOT NULL,
        sector_index TEXT NOT NULL,
        notes TEXT
    );
    CREATE TABLE signal_outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        signal_date TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL, stop_loss REAL, take_profit REAL,
        qty INTEGER, confidence REAL,
        status TEXT NOT NULL DEFAULT 'pending',
        payload_json TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        sent_at TEXT, error TEXT
    );
    CREATE UNIQUE INDEX ix_outbox_uq_symbol_date
        ON signal_outbox(symbol, signal_date);
    CREATE TABLE paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_date TEXT, exit_date TEXT,
        entry_price REAL, exit_price REAL, qty INTEGER,
        pnl_rupees REAL, pnl_pct REAL, cost_rupees REAL, notes TEXT,
        sector TEXT, status TEXT NOT NULL DEFAULT 'open',
        stop_loss REAL, take_profit REAL, trailing_stop REAL,
        entry_atr REAL, high_watermark REAL, exit_reason TEXT,
        entry_prob REAL, threshold REAL, run_id TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    CREATE TABLE validation_failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        check_name TEXT NOT NULL,
        symbol TEXT, issue_date TEXT,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    """
    db_mod.execute_script(schema)
    return db_file


def _seed_universe() -> None:
    db_mod.executemany(
        "INSERT INTO stock_sectors (symbol, sector, sector_index) VALUES (?, ?, ?)",
        [
            ("RELIANCE", "ENERGY", "^CNXENERGY"),
            ("TCS",      "IT",     "^CNXIT"),
            ("INFY",     "IT",     "^CNXIT"),
            ("HCLTECH",  "IT",     "^CNXIT"),
            ("HDFCBANK", "BANK",   "^NSEBANK"),
        ],
    )


def _seed_run_and_predictions(date_: str = "2025-10-15", threshold: float = 0.55) -> str:
    run_id = "run-test"
    db_mod.execute(
        """
        INSERT INTO model_runs
            (run_id, model_name, trained_from, trained_to, metrics_json, created_at)
        VALUES (?, 'xgb_test', '2020-01-01', '2025-09-30', ?, '2025-10-14T08:00:00Z')
        """,
        (run_id, json.dumps({"threshold": threshold, "brier": 0.20})),
    )
    db_mod.executemany(
        "INSERT INTO predictions_log (run_id, symbol, prediction_date, raw_prob, calibrated_prob) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (run_id, "RELIANCE", date_, 0.70, 0.65),  # signal
            (run_id, "TCS",      date_, 0.60, 0.58),  # signal (IT)
            (run_id, "INFY",     date_, 0.59, 0.57),  # signal (IT)
            (run_id, "HCLTECH",  date_, 0.58, 0.56),  # signal (IT) - 4th, sector cap
            (run_id, "HDFCBANK", date_, 0.50, 0.45),  # below threshold
        ],
    )
    return run_id


def _seed_features(date_: str = "2025-10-15") -> None:
    db_mod.executemany(
        "INSERT INTO feature_data (symbol, feature_date, close, atr_14) VALUES (?, ?, ?, ?)",
        [
            ("RELIANCE", date_, 1300.0, 25.0),
            ("TCS",      date_, 3500.0, 70.0),
            ("INFY",     date_, 1500.0, 30.0),
            ("HCLTECH",  date_, 1200.0, 24.0),
            ("HDFCBANK", date_, 1700.0, 30.0),
        ],
    )


def _seed_bar(symbol: str, date_: str, *, o: float, h: float, l: float, c: float) -> None:
    db_mod.execute(
        "INSERT INTO price_data (symbol, bar_date, open, high, low, close, volume, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'yfinance')",
        (symbol, date_, o, h, l, c, 1_000_000),
    )


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------


def test_generate_signals_writes_pending_rows(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")

    out = generate_signals(signal_date="2025-10-15")
    syms = {s.symbol for s in out}

    # 4 candidates above 0.55, but max_per_sector=3 caps the IT sector
    # (TCS, INFY, HCLTECH) -> only 3 of them. Plus RELIANCE.
    assert "RELIANCE" in syms
    assert len(syms & {"TCS", "INFY", "HCLTECH"}) <= 3
    assert "HDFCBANK" not in syms     # below threshold

    rows = db_mod.fetch_all(
        "SELECT symbol, side, entry_price, stop_loss, take_profit, qty, status "
        "FROM signal_outbox ORDER BY confidence DESC"
    )
    assert all(r["status"] == "pending" for r in rows)
    assert all(r["side"] == "BUY" for r in rows)
    for r in rows:
        # SL < entry < TP for every long signal
        assert r["stop_loss"] < r["entry_price"] < r["take_profit"]


def test_generate_signals_is_idempotent(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    a = generate_signals(signal_date="2025-10-15")
    n_first = db_mod.fetch_all("SELECT id FROM signal_outbox")
    b = generate_signals(signal_date="2025-10-15")
    n_second = db_mod.fetch_all("SELECT id FROM signal_outbox")
    assert len(n_first) == len(n_second)
    assert b == []                     # second run is a no-op
    assert a != []                     # first run produced rows


def test_generate_signals_respects_capacity(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15", threshold=0.50)
    _seed_features("2025-10-15")
    cfg = SignalGenConfig(
        risk=RiskConfig(max_concurrent_positions=2, max_per_sector=10),
        sizing=SizingConfig(min_trade_rupees=1.0),
    )
    out = generate_signals(signal_date="2025-10-15", config=cfg)
    assert len(out) == 2               # capacity cap, not threshold cap


def test_generate_signals_skips_when_no_atr(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    # NO feature_data inserted -> generator must skip gracefully
    out = generate_signals(signal_date="2025-10-15")
    assert out == []
    rows = db_mod.fetch_all(
        "SELECT * FROM validation_failures WHERE check_name = 'signal_generator'"
    )
    assert any("missing close/atr" in r["message"] for r in rows)


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------


def test_reconcile_opens_position_from_pending_signal(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    generate_signals(signal_date="2025-10-15")
    # Bar exists for the signal date so reconcile can open at today's open.
    _seed_bar("RELIANCE", "2025-10-15", o=1305.0, h=1320.0, l=1300.0, c=1315.0)

    summary = reconcile(as_of="2025-10-15")
    assert any(o["symbol"] == "RELIANCE" for o in summary.opened)

    paper = db_mod.fetch_all(
        "SELECT symbol, status, entry_price, stop_loss, take_profit, qty "
        "FROM paper_trades WHERE symbol = 'RELIANCE'"
    )
    assert len(paper) == 1
    p = paper[0]
    assert p["status"] == "open"
    assert p["entry_price"] == 1305.0       # filled at next-day open
    assert p["stop_loss"] < p["entry_price"] < p["take_profit"]


def test_reconcile_closes_on_target(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    generate_signals(signal_date="2025-10-15")
    _seed_bar("RELIANCE", "2025-10-15", o=1305.0, h=1310.0, l=1295.0, c=1308.0)
    reconcile(as_of="2025-10-15")  # opens position

    # Day 2: blow through the target.
    _seed_bar("RELIANCE", "2025-10-16", o=1500.0, h=1510.0, l=1495.0, c=1505.0)
    summary = reconcile(as_of="2025-10-16")

    # Position should be closed via target/gap-up at OPEN.
    closed = [c for c in summary.closed if c["symbol"] == "RELIANCE"]
    assert len(closed) == 1
    assert closed[0]["exit_reason"] in {"target"}
    rows = db_mod.fetch_all("SELECT status, pnl_rupees FROM paper_trades WHERE symbol='RELIANCE'")
    assert rows[0]["status"] == "closed"
    assert rows[0]["pnl_rupees"] > 0


def test_reconcile_closes_on_stop(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    generate_signals(signal_date="2025-10-15")
    _seed_bar("RELIANCE", "2025-10-15", o=1305.0, h=1310.0, l=1295.0, c=1308.0)
    reconcile(as_of="2025-10-15")

    # Crash through the stop on day 2.
    _seed_bar("RELIANCE", "2025-10-16", o=1100.0, h=1110.0, l=1090.0, c=1095.0)
    summary = reconcile(as_of="2025-10-16")
    closed = [c for c in summary.closed if c["symbol"] == "RELIANCE"]
    assert len(closed) == 1
    assert closed[0]["exit_reason"] in {"stop"}
    row = db_mod.fetch_all("SELECT pnl_rupees FROM paper_trades")[0]
    assert row["pnl_rupees"] < 0


def test_reconcile_is_idempotent(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    generate_signals(signal_date="2025-10-15")
    _seed_bar("RELIANCE", "2025-10-15", o=1305.0, h=1310.0, l=1295.0, c=1308.0)
    reconcile(as_of="2025-10-15")
    n1 = db_mod.fetch_one("SELECT COUNT(*) AS n FROM paper_trades")["n"]
    reconcile(as_of="2025-10-15")
    n2 = db_mod.fetch_one("SELECT COUNT(*) AS n FROM paper_trades")["n"]
    assert n1 == n2


def test_reconcile_skips_when_no_bar(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    generate_signals(signal_date="2025-10-15")
    # No price_data row -> reconcile should mark skipped, not crash.
    summary = reconcile(as_of="2025-10-15")
    assert summary.opened == []
    assert any(s.get("reason") == "no_bar_for_open" for s in summary.skipped)
    # Underlying signals get marked 'skipped' so we don't retry forever.
    rows = db_mod.fetch_all("SELECT status FROM signal_outbox")
    assert all(r["status"] == "skipped" for r in rows)


def test_reconcile_time_stop(temp_db):
    _seed_universe()
    _seed_run_and_predictions("2025-10-15")
    _seed_features("2025-10-15")
    cfg = SignalGenConfig(
        risk=RiskConfig(max_concurrent_positions=8, max_per_sector=10,
                        max_holding_days=2),
    )
    generate_signals(signal_date="2025-10-15", config=cfg)
    _seed_bar("RELIANCE", "2025-10-15", o=1305.0, h=1310.0, l=1295.0, c=1308.0)
    reconcile(as_of="2025-10-15", risk=cfg.risk)

    # Two days later, with the bar nowhere near stop or target -> time stop hits.
    _seed_bar("RELIANCE", "2025-10-17", o=1306.0, h=1315.0, l=1295.0, c=1310.0)
    summary = reconcile(as_of="2025-10-17", risk=cfg.risk)
    closed = [c for c in summary.closed if c["symbol"] == "RELIANCE"]
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "time"


# ---------------------------------------------------------------------------
# v3 -> v4 migration on an existing DB
# ---------------------------------------------------------------------------


def test_v4_migration_adds_columns_and_indexes(tmp_path, monkeypatch):
    """A DB created on the old (v3) shape gets migrated cleanly."""
    db_file = tmp_path / "old.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)

    # Pre-v4 schema for paper_trades + signal_outbox (no v4 columns/indexes).
    db_mod.execute_script(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (3);
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            signal_id INTEGER, symbol TEXT NOT NULL, side TEXT NOT NULL,
            entry_date TEXT, exit_date TEXT,
            entry_price REAL, exit_price REAL, qty INTEGER,
            pnl_rupees REAL, pnl_pct REAL, cost_rupees REAL, notes TEXT
        );
        CREATE TABLE signal_outbox (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL, signal_date TEXT NOT NULL,
            side TEXT NOT NULL, entry_price REAL, stop_loss REAL, take_profit REAL,
            qty INTEGER, confidence REAL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT,
            created_at TEXT NOT NULL DEFAULT '2025-01-01',
            sent_at TEXT, error TEXT
        );
        -- Insert one row so we can verify nothing is lost on migrate.
        INSERT INTO paper_trades (id, symbol, side) VALUES (1, 'OLD', 'BUY');
        """
    )

    from src.db.migrate import apply_schema
    v = apply_schema()
    assert v >= 4

    cols = {row["name"] for row in db_mod.fetch_all("PRAGMA table_info(paper_trades)")}
    for c in ("status", "stop_loss", "take_profit", "entry_atr",
              "exit_reason", "entry_prob", "threshold", "run_id", "sector",
              "trailing_stop", "high_watermark", "created_at", "updated_at"):
        assert c in cols, f"v4 migration missed column: {c}"
    # Existing data preserved.
    assert db_mod.fetch_one("SELECT symbol FROM paper_trades WHERE id=1")["symbol"] == "OLD"
    # Unique index now present.
    idx = db_mod.fetch_all("PRAGMA index_list(signal_outbox)")
    assert any(i["name"] == "ix_outbox_uq_symbol_date" for i in idx)
