"""Tests for the rule-based regime classifier (pure)."""

from __future__ import annotations

from src.regime import (
    BEAR_TREND,
    BULL_TREND,
    CRISIS,
    HIGH_VOLATILITY,
    RANGE,
)
from src.regime.classifier import RegimeInputs, classify_regime


def _inp(**over):
    base = dict(
        nifty_ma50_gt_ma200=True,
        nifty_above_ma200=True,
        vix=15.0,
        breadth_score=70.0,
        pct_above_200dma=70.0,
    )
    base.update(over)
    return RegimeInputs(**base)


def test_clear_bull_trend():
    assert classify_regime(_inp()).regime == BULL_TREND


def test_crisis_on_vix_spike_overrides_everything():
    # Even a textbook uptrend flips to CRISIS when VIX spikes.
    r = classify_regime(_inp(vix=35.0))
    assert r.regime == CRISIS


def test_crisis_on_breadth_collapse():
    r = classify_regime(_inp(
        nifty_ma50_gt_ma200=False, nifty_above_ma200=False,
        vix=20.0, breadth_score=10.0, pct_above_200dma=15.0,
    ))
    assert r.regime == CRISIS


def test_high_volatility_band():
    r = classify_regime(_inp(vix=25.0))
    assert r.regime == HIGH_VOLATILITY


def test_bear_trend():
    r = classify_regime(_inp(
        nifty_ma50_gt_ma200=False, nifty_above_ma200=False,
        vix=18.0, breadth_score=30.0, pct_above_200dma=30.0,
    ))
    assert r.regime == BEAR_TREND


def test_range_when_trend_up_but_breadth_weak():
    # Uptrend MAs but participation below the bull threshold -> ranging.
    r = classify_regime(_inp(breadth_score=48.0))
    assert r.regime == RANGE


def test_hysteresis_keeps_bull_on_borderline_breadth():
    # breadth 45 is below the 50 bull threshold, but with prev=BULL the
    # threshold relaxes to 42, so we stay BULL instead of flipping to RANGE.
    held = classify_regime(_inp(breadth_score=45.0), prev=BULL_TREND)
    fresh = classify_regime(_inp(breadth_score=45.0), prev=None)
    assert held.regime == BULL_TREND
    assert fresh.regime == RANGE


def test_missing_vix_does_not_crash_and_uses_trend():
    r = classify_regime(_inp(vix=None))
    assert r.regime in (BULL_TREND, RANGE, BEAR_TREND)


def test_result_carries_reasons():
    r = classify_regime(_inp())
    assert isinstance(r.reasons, list) and r.reasons
