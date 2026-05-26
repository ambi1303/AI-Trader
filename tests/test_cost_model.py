"""Tests for the Indian equity cost model.

Reference numbers cross-checked against Zerodha's published brokerage
calculator for EQ_DELIVERY in 2025: a ₹1L round-trip on RELIANCE comes to
roughly ₹110-130 in costs ex-slippage, plus slippage. The ranges below are
sufficiently wide to absorb minor STT rate updates without becoming brittle.
"""

from __future__ import annotations

import pytest

from src.backtesting.cost_model import (
    CostConfig,
    LegCost,
    compute_leg_cost,
    compute_round_trip,
    load_cost_config,
    round_trip_pct,
)


@pytest.fixture()
def delivery_cfg():
    return load_cost_config(intraday=False)


@pytest.fixture()
def intraday_cfg():
    return load_cost_config(intraday=True)


# ---------------------------------------------------------------------------
# Per-leg
# ---------------------------------------------------------------------------


def test_buy_leg_includes_stamp_duty_excludes_stt(delivery_cfg):
    leg = compute_leg_cost("BUY", price=100.0, qty=1000,
                           symbol="RELIANCE", cfg=delivery_cfg)
    assert leg.stamp_duty > 0
    assert leg.stt == 0.0


def test_sell_leg_includes_stt_excludes_stamp_duty(delivery_cfg):
    leg = compute_leg_cost("SELL", price=100.0, qty=1000,
                           symbol="RELIANCE", cfg=delivery_cfg)
    assert leg.stt > 0
    assert leg.stamp_duty == 0.0


def test_gst_is_18pct_of_brokerage_plus_txn(delivery_cfg):
    leg = compute_leg_cost("BUY", price=200.0, qty=500,
                           symbol="RELIANCE", cfg=delivery_cfg)
    expected_gst = 0.18 * (leg.brokerage + leg.exchange_txn)
    assert leg.gst == pytest.approx(expected_gst, rel=1e-9)


def test_per_stock_slippage_overrides_default(delivery_cfg):
    cheap = compute_leg_cost("BUY", price=100.0, qty=1000,
                             symbol="RELIANCE", cfg=delivery_cfg)
    expensive = compute_leg_cost("BUY", price=100.0, qty=1000,
                                 symbol="UNKNOWN_TICKER", cfg=delivery_cfg)
    # RELIANCE override should be cheaper than the default fallback.
    assert cheap.slippage < expensive.slippage


def test_invalid_inputs_raise(delivery_cfg):
    with pytest.raises(ValueError):
        compute_leg_cost("BUY", price=100.0, qty=0,
                         symbol="X", cfg=delivery_cfg)
    with pytest.raises(ValueError):
        compute_leg_cost("BUY", price=0.0, qty=1, symbol="X", cfg=delivery_cfg)
    with pytest.raises(ValueError):
        compute_leg_cost("HOLD", price=10.0, qty=1, symbol="X", cfg=delivery_cfg)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_total_is_legs_plus_extra(delivery_cfg):
    rt = compute_round_trip(symbol="TCS", entry_price=3500.0, exit_price=3550.0,
                            qty=10, cfg=delivery_cfg, extra_per_exit_rupees=15.0)
    expected = rt.buy.total + rt.sell.total + 15.0
    assert rt.total_rupees == pytest.approx(expected, rel=1e-9)


def test_round_trip_pct_is_in_plausible_range(delivery_cfg):
    pct = round_trip_pct("RELIANCE", delivery_cfg)
    # Expect roughly 0.1% - 0.4% for a liquid name with 3 bps slippage override.
    assert 0.0008 < pct < 0.0050, f"got {pct:.4%}"


def test_intraday_brokerage_caps_at_flat(intraday_cfg):
    # ₹2L notional: flat ₹20 should be lower than 0.03% = ₹60. Cap should bind.
    leg = compute_leg_cost("BUY", price=200.0, qty=1000,
                           symbol="RELIANCE", cfg=intraday_cfg)
    assert leg.brokerage == pytest.approx(20.0, abs=0.01)


def test_intraday_stt_is_lower_than_delivery(delivery_cfg, intraday_cfg):
    d = compute_leg_cost("SELL", price=1000.0, qty=100,
                         symbol="X", cfg=delivery_cfg)
    i = compute_leg_cost("SELL", price=1000.0, qty=100,
                         symbol="X", cfg=intraday_cfg)
    assert i.stt < d.stt
