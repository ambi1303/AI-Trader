"""Tests for the regime -> strategy router (pure)."""

from __future__ import annotations

from src.regime import (
    BEAR_TREND,
    BULL_TREND,
    CRISIS,
    HIGH_VOLATILITY,
    RANGE,
)
from src.regime.router import select_strategy


def test_bull_trend_full_momentum_book():
    plan = select_strategy(BULL_TREND)
    assert plan.engine == "momentum"
    assert plan.allow_new_entries is True
    assert plan.config.target_holdings == 22


def test_crisis_is_defensive_no_new_entries():
    plan = select_strategy(CRISIS)
    assert plan.engine == "defensive"
    assert plan.allow_new_entries is False


def test_bear_trend_is_defensive_no_new_entries():
    plan = select_strategy(BEAR_TREND)
    assert plan.allow_new_entries is False


def test_high_volatility_uses_breakout_smaller_and_pickier():
    plan = select_strategy(HIGH_VOLATILITY)
    assert plan.engine == "breakout"
    assert plan.allow_new_entries is True
    assert plan.config.target_holdings < 22
    assert plan.config.min_score > 55.0
    assert plan.config.max_per_sector < 6


def test_range_uses_mean_reversion():
    plan = select_strategy(RANGE)
    assert plan.engine == "mean_reversion"
    assert plan.allow_new_entries is True


def test_none_fails_open_to_momentum():
    plan = select_strategy(None)
    assert plan.engine == "momentum"
    assert plan.allow_new_entries is True


def test_every_regime_maps_to_a_plan():
    for regime in (BULL_TREND, BEAR_TREND, RANGE, HIGH_VOLATILITY, CRISIS, None):
        plan = select_strategy(regime)
        assert plan.engine in ("momentum", "mean_reversion", "breakout", "defensive")
        assert isinstance(plan.allow_new_entries, bool)
