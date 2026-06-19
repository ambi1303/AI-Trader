"""Unit tests for the AI discovery / screener strategy logic."""

from __future__ import annotations

import pytest

from src.analysis import discovery as dsc


@pytest.fixture()
def synthetic_rows(monkeypatch: pytest.MonkeyPatch):
    rows = [
        {  # deep value, cheap, decent quality, no momentum
            "symbol": "CHEAP", "sector": "ENERGY", "close": 100.0,
            "pe_ttm": 6.0, "pb": 0.8, "roe": 0.25, "revenue_growth": 0.05,
            "earnings_growth": 0.10, "debt_to_equity": 0.3, "profit_margin": 0.12,
            "mom_60d": -0.02, "mom_20d": -0.01, "ret_20d": -0.01, "rsi_14": 48.0,
            "dd_from_high_252d": -0.20, "dist_ema_50_pct": -0.03,
            "dist_ema_200_pct": -0.05, "calibrated_prob": 0.55, "verdict": "BUY",
        },
        {  # high-growth, momentum, near 52w high, expensive
            "symbol": "GROW", "sector": "IT", "close": 500.0,
            "pe_ttm": 40.0, "pb": 8.0, "roe": 0.22, "revenue_growth": 0.35,
            "earnings_growth": 0.20, "debt_to_equity": 0.10, "profit_margin": 0.20,
            "mom_60d": 0.30, "mom_20d": 0.10, "ret_20d": 0.08, "rsi_14": 62.0,
            "dd_from_high_252d": -0.02, "dist_ema_50_pct": 0.05,
            "dist_ema_200_pct": 0.10, "calibrated_prob": 0.60, "verdict": "BUY",
        },
        {  # oversold but above the 200-EMA
            "symbol": "OVSOLD", "sector": "METAL", "close": 200.0,
            "pe_ttm": 18.0, "pb": 2.0, "roe": 0.11, "revenue_growth": 0.10,
            "earnings_growth": 0.0, "debt_to_equity": 0.5, "profit_margin": 0.09,
            "mom_60d": 0.02, "mom_20d": -0.05, "ret_20d": -0.06, "rsi_14": 30.0,
            "dd_from_high_252d": -0.12, "dist_ema_50_pct": -0.04,
            "dist_ema_200_pct": 0.03, "calibrated_prob": 0.49, "verdict": "HOLD",
        },
    ]
    monkeypatch.setattr(dsc, "_rows", lambda force=False: rows)
    return rows


def _syms(results):
    return [r["symbol"] for r in results]


def test_value_ranks_cheapest_first(synthetic_rows) -> None:
    res = dsc.scan("value")
    syms = _syms(res)
    assert "CHEAP" in syms and "GROW" not in syms  # GROW PE 40 excluded
    assert syms[0] == "CHEAP"


def test_growth_filters_to_high_growth(synthetic_rows) -> None:
    assert _syms(dsc.scan("growth")) == ["GROW"]


def test_momentum_requires_uptrend(synthetic_rows) -> None:
    assert _syms(dsc.scan("momentum")) == ["GROW"]


def test_oversold_requires_above_200ema(synthetic_rows) -> None:
    assert _syms(dsc.scan("oversold")) == ["OVSOLD"]


def test_breakout_near_52w_high(synthetic_rows) -> None:
    assert _syms(dsc.scan("breakout")) == ["GROW"]


def test_top_conviction_sorted_by_prob(synthetic_rows) -> None:
    assert _syms(dsc.scan("top_conviction")) == ["GROW", "CHEAP", "OVSOLD"]


def test_quality_excludes_low_roe(synthetic_rows) -> None:
    syms = _syms(dsc.scan("quality"))
    assert "OVSOLD" not in syms          # ROE 11% < 15%
    assert "CHEAP" in syms and "GROW" in syms


def test_unknown_strategy_returns_empty(synthetic_rows) -> None:
    assert dsc.scan("does_not_exist") == []


def test_limit_is_respected(synthetic_rows) -> None:
    assert len(dsc.scan("top_conviction", limit=1)) == 1
