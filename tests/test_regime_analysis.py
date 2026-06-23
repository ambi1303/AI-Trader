"""Tests for the pure per-regime backtest summarizer."""

from __future__ import annotations

import pandas as pd

from src.backtesting.regime_analysis import summarize_by_regime


def _trades(rows):
    return pd.DataFrame(
        rows, columns=["net_pnl", "holding_days", "entry_regime"]
    )


def test_empty_trades_and_no_equity_returns_empty():
    assert summarize_by_regime(pd.DataFrame(), None) == {}


def test_groups_trades_by_entry_regime():
    df = _trades([
        (100.0, 5, "BULL_TREND"),
        (-40.0, 3, "BULL_TREND"),
        (-20.0, 2, "RANGE"),
    ])
    out = summarize_by_regime(df)
    assert out["BULL_TREND"]["n_trades"] == 2
    assert out["BULL_TREND"]["win_rate_pct"] == 50.0
    assert out["BULL_TREND"]["net_pnl"] == 60.0
    # profit factor = gross win 100 / gross loss 40 = 2.5
    assert out["BULL_TREND"]["profit_factor"] == 2.5
    assert out["RANGE"]["n_trades"] == 1
    assert out["RANGE"]["net_pnl"] == -20.0


def test_profit_factor_none_when_no_losers():
    df = _trades([(50.0, 2, "BULL_TREND"), (10.0, 1, "BULL_TREND")])
    out = summarize_by_regime(df)
    assert out["BULL_TREND"]["profit_factor"] is None


def test_none_entry_regime_buckets_as_unknown():
    df = _trades([(10.0, 1, None), (-5.0, 1, None)])
    out = summarize_by_regime(df)
    assert "UNKNOWN" in out
    assert out["UNKNOWN"]["n_trades"] == 2


def test_equity_adds_daily_sharpe_and_days():
    trades = _trades([(100.0, 5, "BULL_TREND")])
    equity = pd.DataFrame({
        "equity": [100.0, 101.0, 102.0, 103.0],
        "regime": ["BULL_TREND"] * 4,
    })
    out = summarize_by_regime(trades, equity)
    assert out["BULL_TREND"]["days_in_regime"] == 4
    assert "daily_sharpe" in out["BULL_TREND"]
    # steadily rising equity -> positive Sharpe
    assert out["BULL_TREND"]["daily_sharpe"] > 0


def test_equity_only_still_reports_regime():
    equity = pd.DataFrame({
        "equity": [100.0, 99.0, 101.0],
        "regime": ["RANGE", "RANGE", "RANGE"],
    })
    out = summarize_by_regime(None, equity)
    assert out["RANGE"]["days_in_regime"] == 3
