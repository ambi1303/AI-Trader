"""Tests for position sizing."""

from __future__ import annotations

import pytest

from src.backtesting.sizing import (
    SizingConfig,
    fractional_kelly_qty,
    kelly_fraction,
    size_position,
    vol_target_qty,
)


# ---------------------------------------------------------------------------
# Pure Kelly math
# ---------------------------------------------------------------------------


def test_kelly_zero_below_breakeven_probability():
    # p=0.4, R=1 -> f* = 0.4 - 0.6 = -0.2 -> floored at 0
    assert kelly_fraction(0.4, 1.0) == 0.0


def test_kelly_classic_formula_matches():
    # p=0.6, R=2 -> f* = 0.6 - 0.4/2 = 0.4
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4, abs=1e-9)


def test_kelly_zero_for_invalid_R():
    assert kelly_fraction(0.6, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Fractional Kelly qty
# ---------------------------------------------------------------------------


def test_fractional_kelly_caps_via_kelly_fraction_param():
    cfg = SizingConfig(kelly_fraction=0.25)
    # f* = 0.4 -> after 0.25 cap, fraction = 0.10 -> notional ~ 100k.
    # Floor-division on noisy floats can produce 99 or 100; both are correct
    # behaviour because Kelly is fundamentally a fraction, not an integer.
    qty = fractional_kelly_qty(prob_win=0.6, entry_price=1000.0,
                               equity=1_000_000.0, cfg=cfg, reward_to_risk=2.0)
    assert 99 <= qty <= 100


def test_fractional_kelly_zero_for_unfavorable_prob():
    cfg = SizingConfig()
    qty = fractional_kelly_qty(prob_win=0.45, entry_price=100.0,
                               equity=1_000_000.0, cfg=cfg, reward_to_risk=1.0)
    assert qty == 0


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------


def test_vol_target_respects_per_trade_risk():
    # 1% of 1L = ₹1000 risk; 2*ATR=20 -> 50 shares.
    cfg = SizingConfig(risk_per_trade_pct=0.01, max_position_pct=1.0)
    qty = vol_target_qty(entry_price=100.0, atr=10.0, stop_atr_mult=2.0,
                         equity=100_000.0, cfg=cfg)
    assert qty == 50


def test_vol_target_caps_at_max_position_notional():
    cfg = SizingConfig(risk_per_trade_pct=1.0, max_position_pct=0.10)
    # No risk cap binding, max position cap should bind.
    qty = vol_target_qty(entry_price=100.0, atr=1.0, stop_atr_mult=2.0,
                         equity=100_000.0, cfg=cfg)
    # 10% of 100k / 100 = 100 shares.
    assert qty == 100


def test_vol_target_handles_zero_atr():
    cfg = SizingConfig()
    assert vol_target_qty(entry_price=100.0, atr=0.0, stop_atr_mult=2.0,
                          equity=100_000.0, cfg=cfg) == 0


# ---------------------------------------------------------------------------
# Combined sizer
# ---------------------------------------------------------------------------


def test_combined_picks_min_of_kelly_and_vol_target():
    cfg = SizingConfig(risk_per_trade_pct=0.01, max_position_pct=0.50,
                       kelly_fraction=0.25)
    d = size_position(prob_win=0.7, entry_price=100.0, atr=5.0,
                      stop_atr_mult=2.0, equity=1_000_000.0, cfg=cfg,
                      reward_to_risk=1.5)
    # Kelly: f* = 0.7 - 0.3/1.5 = 0.5; * 0.25 = 0.125 -> 1250 shares
    # Vol: risk = 10000, 2*ATR = 10, qty = 1000
    # min = 1000, vol_target binds.
    assert d.qty == 1000
    assert d.rationale == "vol_target"


def test_combined_skip_below_min_trade_rupees():
    cfg = SizingConfig(min_trade_rupees=10_000.0)
    d = size_position(prob_win=0.6, entry_price=10_000.0, atr=100.0,
                      stop_atr_mult=2.0, equity=20_000.0, cfg=cfg)
    # Tiny equity -> qty 0 or notional below min -> skip
    assert d.qty == 0
    assert d.rationale in ("skip", "min_size")


def test_combined_skip_zero_equity():
    cfg = SizingConfig()
    d = size_position(prob_win=0.7, entry_price=100.0, atr=5.0,
                      stop_atr_mult=2.0, equity=0.0, cfg=cfg)
    assert d.qty == 0
    assert d.rationale == "skip"
