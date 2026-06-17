"""Unit tests for the pure stock-analysis engine (no DB, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import stock_analysis as sa


def _series(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    close = pd.Series(closes, dtype="float64")
    return pd.DataFrame({
        "bar_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series(np.full(n, 100000.0)),
    })


def test_uptrend_scores_bullish() -> None:
    closes = [100.0 * (1.004 ** i) for i in range(260)]
    df = _series(closes)
    fund = {"roe": 0.20, "profit_margin": 0.15, "revenue_growth": 0.20,
            "debt_to_equity": 0.3, "pe_ttm": 18.0, "pb": 2.0}
    res = sa.analyze(df, fund)

    t = res["technicals"]
    assert t["available"] is True
    assert t["trend_tone"] == "good"
    assert t["rsi"] is not None

    conv = res["conviction"]
    assert conv["overall"] >= 60
    assert {f["name"] for f in conv["factors"]} == {
        "Fundamentals", "Valuation", "Technicals", "Momentum", "Risk"
    }

    assert res["rule_verdict"]["verdict"] == "BUY"

    z = res["zones"]
    assert z["available"] is True
    # Targets are R-multiples of risk -> R:R is 1.5 by construction.
    assert z["risk_reward"] == 1.5
    assert z["stop"] < z["current"] < z["target1"] < z["target2"]


def test_downtrend_scores_bearish() -> None:
    closes = [200.0 * (0.996 ** i) for i in range(260)]
    df = _series(closes)
    res = sa.analyze(df, fundamentals=None)
    assert res["technicals"]["trend_tone"] in ("bad", "warn")
    # No fundamentals -> those factors are simply absent, weights renormalise.
    names = {f["name"] for f in res["conviction"]["factors"]}
    assert "Fundamentals" not in names
    assert res["rule_verdict"]["verdict"] in ("SELL", "HOLD")


def test_insufficient_history_is_graceful() -> None:
    df = _series([100.0, 101.0, 102.0])  # < 20 rows
    res = sa.analyze(df, fundamentals=None)
    assert res["technicals"]["available"] is False
    assert res["conviction"]["overall"] is None
    assert res["zones"]["available"] is False
    assert res["rule_verdict"]["verdict"] is None


def test_empty_frame_does_not_raise() -> None:
    res = sa.analyze(pd.DataFrame(), fundamentals=None)
    assert res["technicals"]["available"] is False
    assert res["zones"]["available"] is False


def test_conviction_renormalises_over_available_factors() -> None:
    # Only fundamentals available (no technicals): score must still be valid.
    fund = {"roe": 0.25, "profit_margin": 0.20, "pe_ttm": 12.0, "pb": 1.2}
    conv = sa.conviction_score(fund, {"available": False})
    assert conv["overall"] is not None
    names = {f["name"] for f in conv["factors"]}
    assert names == {"Fundamentals", "Valuation"}
