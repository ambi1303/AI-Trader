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


def test_compressed_scores_trigger_adaptive_grid():
    # Simulate isotonic-compressed probabilities: every score is BELOW the
    # fixed grid floor of 0.50 (max ~0.43, like the real model). The tuner
    # must fall back to a data-adaptive grid and return a threshold that
    # actually lives inside the observed range -- NOT 0.50.
    rng = np.random.default_rng(7)
    n = 600
    p = rng.uniform(0.05, 0.43, size=n)
    # Make higher scores genuinely predictive so a meaningful threshold exists.
    returns = np.where(p > 0.30,
                       rng.normal(0.015, 0.005, size=n),
                       rng.normal(-0.004, 0.005, size=n))
    res = tune_threshold(p, returns, cost_pct=0.001, min_signals=10)
    # Threshold must be inside the compressed range, not the fixed-grid floor.
    assert res.threshold < 0.50
    assert res.threshold <= p.max()
    assert res.n_signals >= 10
    assert res.expected_return_per_trade > 0


def test_compressed_scores_disable_adaptive_returns_zero_signal_threshold():
    # With adaptive_fallback=False we keep the OLD behaviour: a fixed grid
    # that never fires returns a 0-signal threshold. This guards the toggle.
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.43, size=300)
    r = rng.normal(0.0, 0.005, size=300)
    res = tune_threshold(p, r, cost_pct=0.001, min_signals=10,
                         adaptive_fallback=False)
    assert res.n_signals == 0
    assert res.threshold == 0.50


def test_adaptive_mode_searches_observed_range():
    # adaptive=True should pick a threshold from the score distribution even
    # when scores comfortably span 0.50 -- the grid is data-derived, not fixed.
    rng = np.random.default_rng(3)
    n = 800
    p = rng.uniform(0.10, 0.46, size=n)  # compressed, like isotonic output
    returns = np.where(p > 0.32,
                       rng.normal(0.012, 0.004, size=n),
                       rng.normal(-0.003, 0.004, size=n))
    res = tune_threshold(p, returns, cost_pct=0.001, min_signals=10,
                         adaptive=True)
    assert res.threshold <= p.max()
    assert res.n_signals >= 10
    # grid floor is the median, so the chosen threshold is never below it
    assert res.threshold >= float(np.quantile(p, 0.50)) - 1e-6


def test_adaptive_grid_threshold_actually_fires_on_holdout():
    # End-to-end sanity: a threshold chosen by the adaptive path should,
    # when applied as `prob >= threshold`, select a non-empty subset.
    rng = np.random.default_rng(11)
    p = rng.uniform(0.10, 0.40, size=500)
    r = rng.normal(0.002, 0.01, size=500)
    res = tune_threshold(p, r, cost_pct=0.001, min_signals=5)
    fired = (p >= res.threshold).sum()
    assert fired >= 5
