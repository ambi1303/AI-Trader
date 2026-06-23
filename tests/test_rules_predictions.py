"""Integration tests for the rules-strategy backtest predictions generator.

Verifies the bridge that replays the scorers + regime router over history:
  - routing routes by regime and BLOCKS new entries on defensive (CRISIS/BEAR)
    days, while emitting momentum signals on BULL days;
  - calibrated_prob is the production Kelly mapping of the score;
  - the no-routing baseline scores momentum every day regardless of regime.
"""

from __future__ import annotations

import pytest

from src.backtesting.rules_predictions import build_rules_predictions
from src.signals.strategy import _score_to_prob
from src.utils import db as db_mod

BULL_DAY = "2026-06-10"
CRISIS_DAY = "2026-06-11"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "rules.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    db_mod.execute_script(
        """
        CREATE TABLE feature_data (
            symbol TEXT NOT NULL, feature_date TEXT NOT NULL,
            close REAL, dist_ema_200_pct REAL, dist_ema_50_pct REAL,
            rsi_14 REAL, mom_60d REAL, ret_20d REAL, dd_from_high_252d REAL,
            macd_hist REAL, bb_pct_b REAL, vol_ratio_20d REAL,
            roe REAL, debt_to_equity REAL, profit_margin REAL, pe_ttm REAL,
            PRIMARY KEY (symbol, feature_date)
        );
        CREATE TABLE stock_sectors (symbol TEXT PRIMARY KEY, sector TEXT);
        CREATE TABLE market_regime (
            as_of_date TEXT PRIMARY KEY, regime TEXT NOT NULL
        );
        """
    )
    db_mod.executemany(
        "INSERT INTO stock_sectors (symbol, sector) VALUES (?, ?)",
        [("AAA", "IT"), ("BBB", "BANK")],
    )
    db_mod.executemany(
        "INSERT INTO market_regime (as_of_date, regime) VALUES (?, ?)",
        [(BULL_DAY, "BULL_TREND"), (CRISIS_DAY, "CRISIS")],
    )
    return db_file


def _seed_strong_momentum(symbol: str, day: str, mom: float = 0.20) -> None:
    """A clean momentum pass: above both EMAs, healthy RSI, near highs."""
    db_mod.execute(
        "INSERT INTO feature_data "
        "(symbol, feature_date, close, dist_ema_200_pct, dist_ema_50_pct, "
        " rsi_14, mom_60d, ret_20d, dd_from_high_252d, macd_hist, bb_pct_b, "
        " vol_ratio_20d, roe, debt_to_equity, profit_margin, pe_ttm) "
        "VALUES (?, ?, 500, 0.15, 0.06, 62, ?, 0.08, -0.02, 0.5, 0.7, "
        " 1.2, 0.22, 0.3, 0.18, 18)",
        (symbol, day, mom),
    )


def test_routing_blocks_new_entries_on_crisis_day(temp_db):
    for sym in ("AAA", "BBB"):
        _seed_strong_momentum(sym, BULL_DAY)
        _seed_strong_momentum(sym, CRISIS_DAY)

    rp = build_rules_predictions(start=BULL_DAY, end=CRISIS_DAY)

    days = set(rp.predictions["feature_date"])
    assert BULL_DAY in days                      # BULL -> momentum entries
    assert CRISIS_DAY not in days                # CRISIS -> defensive, no entries
    # Every emitted signal on the bull day uses the momentum engine.
    bull_audit = rp.audit[rp.audit["feature_date"] == BULL_DAY]
    assert set(bull_audit["engine"]) == {"momentum"}
    assert set(bull_audit["regime"]) == {"BULL_TREND"}


def test_calibrated_prob_is_score_to_prob(temp_db):
    _seed_strong_momentum("AAA", BULL_DAY)

    rp = build_rules_predictions(start=BULL_DAY, end=BULL_DAY)
    merged = rp.predictions.merge(rp.audit, on=["symbol", "feature_date"])
    assert not merged.empty
    for r in merged.itertuples(index=False):
        assert r.calibrated_prob == pytest.approx(_score_to_prob(r.score))


def test_baseline_ignores_regime_and_trades_every_day(temp_db):
    for sym in ("AAA", "BBB"):
        _seed_strong_momentum(sym, BULL_DAY)
        _seed_strong_momentum(sym, CRISIS_DAY)

    rp = build_rules_predictions(
        start=BULL_DAY, end=CRISIS_DAY, regime_routing=False)

    days = set(rp.predictions["feature_date"])
    assert {BULL_DAY, CRISIS_DAY} <= days        # momentum runs even on crisis
    assert set(rp.audit["engine"]) == {"momentum"}
