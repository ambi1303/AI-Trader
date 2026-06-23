"""Unit tests for the pure spread / z-score / signal logic."""

from __future__ import annotations

import numpy as np

from src.pairs.spread import (
    EXIT,
    FLAT,
    HOLD,
    LONG_SPREAD,
    SHORT_SPREAD,
    classify,
    compute_spread,
    latest_zscore,
    signal_for,
)


def test_compute_spread():
    y = np.array([10.0, 12.0, 14.0])
    x = np.array([2.0, 3.0, 4.0])
    s = compute_spread(y, x, beta=2.0, alpha=1.0)
    assert np.allclose(s, y - 1.0 - 2.0 * x)


def test_latest_zscore_matches_manual():
    spread = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, mean, std = latest_zscore(spread, window=5)
    assert np.isclose(mean, 3.0)
    assert np.isclose(std, np.std(spread, ddof=1))
    assert np.isclose(z, (5.0 - 3.0) / np.std(spread, ddof=1))


def test_latest_zscore_zero_when_flat():
    z, mean, std = latest_zscore(np.array([7.0, 7.0, 7.0]), window=3)
    assert z == 0.0 and std == 0.0 and mean == 7.0


def test_classify_thresholds():
    assert classify(2.5) == SHORT_SPREAD       # rich spread
    assert classify(-2.5) == LONG_SPREAD        # cheap spread
    assert classify(0.3) == EXIT                # reverted
    assert classify(4.0) == FLAT                # diverged / stop
    assert classify(-4.0) == FLAT
    assert classify(1.0) == HOLD                # in the band


def test_signal_for_end_to_end():
    # A dispersed (std ~ 1) spread that ends ~2.5 std above its mean ->
    # SHORT_SPREAD. (A lone spike would instead give z ~ sqrt(N-1) -> FLAT.)
    rng = np.random.default_rng(0)
    spread = rng.normal(0.0, 1.0, 60)
    spread[-1] = 2.6
    x = np.zeros(60)
    sig = signal_for(spread, x, beta=0.0, alpha=0.0, window=60)
    assert sig.signal == SHORT_SPREAD
    assert 2.0 < sig.zscore < 3.5
