"""Correctness tests for hand-rolled indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import technical_indicators as ti


def _series(values: list[float], name: str = "x") -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, name=name, dtype="float64")


def test_ema_matches_recursive_definition() -> None:
    s = _series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    out = ti.ema(s, span=3)
    # alpha = 2/(span+1) = 0.5; first valid value at index span-1 (= 2)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    # Manually compute
    alpha = 2 / (3 + 1)
    expected = s.iloc[0]
    for i in range(1, len(s)):
        expected = alpha * s.iloc[i] + (1 - alpha) * expected
        if i >= 2:
            assert out.iloc[i] == pytest.approx(expected, rel=1e-9)


def test_rsi_constant_series_is_nan() -> None:
    """Flat series: both avg_up and avg_down are 0 -> RSI undefined -> NaN."""
    s = _series([100.0] * 30)
    r = ti.rsi(s, 14)
    assert pd.isna(r.iloc[-1])


def test_rsi_strictly_rising_yields_high() -> None:
    s = _series([100 + i for i in range(30)])
    r = ti.rsi(s, 14)
    # Strictly rising -> avg_down=0 -> RSI = 100
    assert r.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_strictly_falling_yields_low() -> None:
    s = _series([200 - i for i in range(30)])
    r = ti.rsi(s, 14)
    # Strictly falling -> avg_up=0 -> RSI = 0
    assert r.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_macd_components_consistent() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.normal(size=400)) + 1000.0)
    out = ti.macd(s)
    # macd_hist == macd - macd_signal everywhere they're both defined
    both = out.dropna()
    assert (both["macd_hist"] - (both["macd"] - both["macd_signal"])).abs().max() < 1e-10


def test_bollinger_pct_b_at_mid_is_half() -> None:
    s = _series([100.0] * 25)
    bb = ti.bollinger(s, window=20, num_std=2.0)
    # Constant series -> std=0 -> band collapses; pct_b = NaN by guard
    assert pd.isna(bb["bb_pct_b"].iloc[-1])


def test_atr_positive() -> None:
    rng = np.random.default_rng(0)
    n = 100
    close = 1000 + np.cumsum(rng.normal(size=n))
    high = close + rng.uniform(1, 5, n)
    low = close - rng.uniform(1, 5, n)
    a = ti.atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    a = a.dropna()
    assert (a > 0).all()


def test_adx_in_zero_to_hundred() -> None:
    rng = np.random.default_rng(0)
    n = 200
    close = 1000 + np.cumsum(rng.normal(size=n))
    high = close + rng.uniform(1, 5, n)
    low = close - rng.uniform(1, 5, n)
    out = ti.adx(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    a = out["adx_14"].dropna()
    assert (a >= 0).all()
    assert (a <= 100).all()


def test_stochastic_in_zero_to_hundred() -> None:
    rng = np.random.default_rng(0)
    n = 100
    close = 1000 + np.cumsum(rng.normal(size=n))
    high = close + rng.uniform(1, 5, n)
    low = close - rng.uniform(1, 5, n)
    out = ti.stochastic(pd.Series(high), pd.Series(low), pd.Series(close), 14, 3)
    k = out["stoch_k"].dropna()
    assert (k >= 0).all() and (k <= 100).all()
