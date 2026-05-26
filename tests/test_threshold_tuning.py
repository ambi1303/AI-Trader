"""Tests for expected-utility threshold tuning."""

from __future__ import annotations

import numpy as np

from src.models.threshold_tuning import _round_trip_cost_pct, tune_threshold


def test_obvious_separability_picks_high_threshold():
    # Probability strongly correlated with positive returns above 0.7;
    # below 0.7 returns are negative on average. The optimiser should
    # pick a threshold around 0.7.
    rng = np.random.default_rng(0)
    n = 1000
    p = rng.uniform(size=n)
    returns = np.where(p > 0.7, rng.normal(0.02, 0.005, size=n),
                       rng.normal(-0.005, 0.005, size=n))
    res = tune_threshold(p, returns, cost_pct=0.001, min_signals=10)
    assert res.threshold >= 0.65
    assert res.expected_return_per_trade > 0


def test_no_signal_above_cost_falls_back_gracefully():
    # Every threshold yields negative expected return after cost.
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(size=n)
    returns = -np.abs(rng.normal(size=n)) * 0.01  # all negative
    res = tune_threshold(p, returns, cost_pct=0.005, min_signals=5)
    # Falls back to least-negative; we just check it does not crash and is
    # within the grid.
    assert 0.5 <= res.threshold <= 0.95


def test_min_signals_constraint_is_respected():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=100)
    r = rng.normal(0.0, 0.01, size=100)
    res = tune_threshold(p, r, cost_pct=0.0, min_signals=20)
    assert res.n_signals >= 20 or res.curve["n_signals"].max() < 20


def test_cost_pct_default_round_trip_is_reasonable():
    # Delegates to the canonical Week-4 cost model loaded from
    # config/cost_model.yaml. We just check the result is plausible
    # for an Indian discount broker on EQ_DELIVERY (between ~10 bps and 80 bps
    # round-trip including slippage).
    c = _round_trip_cost_pct(None)
    assert 0.0010 < c < 0.0080, f"unexpected round-trip cost: {c:.4%}"


def test_explicit_cost_overrides_cost_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    r = rng.normal(0.0, 0.01, size=200)
    res_a = tune_threshold(p, r, cost_pct=0.001)
    res_b = tune_threshold(p, r, cost_pct=0.020)
    assert res_a.cost_pct == 0.001
    assert res_b.cost_pct == 0.020


def test_curve_contains_full_grid():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=300)
    r = rng.normal(0.0, 0.005, size=300)
    res = tune_threshold(
        p, r, cost_pct=0.001, grid_min=0.5, grid_max=0.95, grid_step=0.01
    )
    # 0.50 .. 0.95 inclusive in 0.01 steps -> 46 rows
    assert len(res.curve) == 46
