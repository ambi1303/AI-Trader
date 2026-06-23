"""Integration tests for the regime store against a temp SQLite DB."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.regime import BULL_TREND, CRISIS
from src.regime.store import latest_regime, previous_regime, store_regime
from src.utils import db as db_mod

AS_OF = "2026-06-17"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "regime.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    db_mod.execute_script(
        """
        CREATE TABLE feature_data (
            symbol TEXT NOT NULL, feature_date TEXT NOT NULL,
            dist_ema_50_pct REAL, dist_ema_200_pct REAL,
            dd_from_high_252d REAL, ret_1d REAL,
            PRIMARY KEY (symbol, feature_date)
        );
        CREATE TABLE index_data (
            index_symbol TEXT NOT NULL, bar_date TEXT NOT NULL, close REAL NOT NULL,
            PRIMARY KEY (index_symbol, bar_date)
        );
        CREATE TABLE market_regime (
            as_of_date TEXT PRIMARY KEY, regime TEXT NOT NULL,
            nifty_above_ma200 INTEGER, nifty_ma50_gt_ma200 INTEGER, vix REAL,
            pct_above_50dma REAL, pct_above_200dma REAL, adv_decl_ratio REAL,
            breadth_score REAL, scores_json TEXT,
            created_at TEXT NOT NULL DEFAULT '2026-01-01'
        );
        """
    )
    return db_file


def _seed_nifty_uptrend(end=AS_OF, n=210):
    """n ascending closes so MA50 > MA200 and last close > MA200."""
    end_d = date.fromisoformat(end)
    rows = []
    for i in range(n):
        d = end_d - timedelta(days=(n - 1 - i))
        rows.append(("^NSEI", d.isoformat(), 100.0 + i))   # strictly rising
    db_mod.executemany(
        "INSERT INTO index_data (index_symbol, bar_date, close) VALUES (?, ?, ?)",
        rows,
    )


def _seed_vix(level, end=AS_OF):
    db_mod.execute(
        "INSERT INTO index_data (index_symbol, bar_date, close) VALUES (?, ?, ?)",
        ("^INDIAVIX", end, level),
    )


def _seed_breadth(pct_above, n=20, date_=AS_OF):
    """n symbols; `pct_above` fraction above both MAs (rest below)."""
    above = int(round(pct_above * n))
    rows = []
    for i in range(n):
        d = 0.1 if i < above else -0.1
        rows.append((f"S{i}", date_, d, d, -0.02, 0.01 if i < above else -0.01))
    db_mod.executemany(
        "INSERT INTO feature_data "
        "(symbol, feature_date, dist_ema_50_pct, dist_ema_200_pct, "
        " dd_from_high_252d, ret_1d) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def test_store_bull_trend_and_fields(temp_db):
    _seed_nifty_uptrend()
    _seed_vix(15.0)
    _seed_breadth(0.8)                       # 80% participation

    payload = store_regime(AS_OF)
    assert payload["regime"] == BULL_TREND

    row = db_mod.fetch_one(
        "SELECT * FROM market_regime WHERE as_of_date = ?", (AS_OF,)
    )
    assert row["regime"] == BULL_TREND
    assert row["nifty_above_ma200"] == 1
    assert row["nifty_ma50_gt_ma200"] == 1
    assert row["vix"] == 15.0
    assert row["breadth_score"] == 80.0
    assert row["scores_json"]                # diagnostic payload persisted


def test_store_crisis_on_vix_spike(temp_db):
    _seed_nifty_uptrend()
    _seed_vix(36.0)                          # spike overrides the uptrend
    _seed_breadth(0.8)

    payload = store_regime(AS_OF)
    assert payload["regime"] == CRISIS
    assert latest_regime(AS_OF) == CRISIS


def test_upsert_is_idempotent(temp_db):
    _seed_nifty_uptrend()
    _seed_vix(15.0)
    _seed_breadth(0.8)

    store_regime(AS_OF)
    store_regime(AS_OF)                      # second run must not duplicate
    n = db_mod.fetch_one(
        "SELECT COUNT(*) AS c FROM market_regime WHERE as_of_date = ?", (AS_OF,)
    )["c"]
    assert n == 1


def test_previous_regime_reads_prior_day(temp_db):
    db_mod.execute(
        "INSERT INTO market_regime (as_of_date, regime) VALUES (?, ?)",
        ("2026-06-16", "RANGE"),
    )
    assert previous_regime(AS_OF) == "RANGE"
    assert previous_regime("2026-06-16") is None
