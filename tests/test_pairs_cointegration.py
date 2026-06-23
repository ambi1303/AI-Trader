"""Unit tests for the pure cointegration statistics."""

from __future__ import annotations

import numpy as np

from src.pairs.cointegration import (
    adf_tstat,
    engle_granger,
    half_life,
    hedge_ratio,
)


def _ar1(n, phi, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, sigma)
    return s


def _random_walk(n, sigma=0.5, base=500.0, seed=1):
    rng = np.random.default_rng(seed)
    return base + np.cumsum(rng.normal(0, sigma, n))


def test_hedge_ratio_recovers_beta_alpha():
    x = _random_walk(200)
    y = 3.0 + 2.0 * x                     # exact linear relationship
    alpha, beta, spread = hedge_ratio(y, x)
    assert np.isclose(beta, 2.0, atol=1e-6)
    assert np.isclose(alpha, 3.0, atol=1e-4)
    assert np.allclose(spread, 0.0, atol=1e-6)


def test_adf_rejects_unit_root_for_stationary_series():
    s = _ar1(400, phi=0.5)                # strongly mean-reverting
    assert adf_tstat(s) < -3.34


def test_adf_does_not_reject_for_random_walk():
    s = _random_walk(400, sigma=1.0)
    assert adf_tstat(s) > -3.34           # can't reject a unit root


def test_half_life_positive_for_mean_reverting():
    s = _ar1(2000, phi=0.9)               # half-life ~ ln2 / 0.1 ~ 6.9
    hl = half_life(s)
    assert hl is not None
    assert 3.0 < hl < 12.0


def test_half_life_none_when_not_reverting():
    s = np.arange(100, dtype=float)       # trending, b ~ 0 -> not reverting
    assert half_life(s) is None


def test_engle_granger_flags_cointegrated_pair():
    # x drives the relationship (large sigma) so beta is well-identified; the
    # spread is a persistent-but-stationary AR(1).
    x = _random_walk(300, sigma=2.0, seed=2)
    spread = _ar1(300, phi=0.9, seed=3)
    y = 50.0 + 1.5 * x + spread
    res = engle_granger(y, x)
    assert res.cointegrated is True
    assert np.isclose(res.beta, 1.5, atol=0.15)
    assert res.adf_tstat < -3.34
    assert res.half_life is not None and 1.0 <= res.half_life <= 120.0


def test_engle_granger_rejects_independent_walks():
    x = _random_walk(300, seed=4)
    y = _random_walk(300, seed=5)         # independent -> spread non-stationary
    res = engle_granger(y, x)
    assert res.cointegrated is False


def test_engle_granger_too_few_obs():
    x = _random_walk(20)
    y = 2 * x
    res = engle_granger(y, x)
    assert res.cointegrated is False
    assert res.n_obs == 20
