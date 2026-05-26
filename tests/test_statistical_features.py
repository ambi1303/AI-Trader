"""Correctness for returns / volatility / momentum / drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import statistical_features as sf


def _series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype="float64")


def test_simple_return_one_period() -> None:
    s = _series([100, 110, 121])
    r = sf.simple_return(s, 1)
    assert pd.isna(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(0.10)


def test_log_return_consistency_with_simple_return_for_small_moves() -> None:
    s = _series([100 * (1.001 ** i) for i in range(50)])
    r_log = sf.log_return(s, 1).dropna()
    r_simple = sf.simple_return(s, 1).dropna()
    assert (r_log - np.log1p(r_simple)).abs().max() < 1e-12


def test_realized_vol_constant_series_zero() -> None:
    s = _series([100.0] * 40)
    v = sf.realized_vol(s, 20).dropna()
    assert (v.abs() < 1e-12).all()


def test_drawdown_from_high_at_new_high_is_zero() -> None:
    s = _series([100, 105, 110, 120])
    dd = sf.drawdown_from_high(s, 20)
    assert dd.iloc[-1] == pytest.approx(0.0)


def test_drawdown_from_high_after_decline_negative() -> None:
    s = _series([100, 110, 120, 90])
    dd = sf.drawdown_from_high(s, 20)
    assert dd.iloc[-1] == pytest.approx((90 - 120) / 120)


def test_overnight_gap_pct() -> None:
    open_ = _series([100, 102, 99])
    close = _series([100, 100, 101])
    g = sf.overnight_gap_pct(open_, close.shift(1))
    assert pd.isna(g.iloc[0])
    assert g.iloc[1] == pytest.approx((102 - 100) / 100)
    assert g.iloc[2] == pytest.approx((99 - 100) / 100)


def test_volume_zscore_constant_volume_zero_or_nan() -> None:
    v = _series([1000.0] * 30)
    z = sf.volume_zscore(v, 20).dropna()
    # std=0 -> NaN by guard, so dropna -> empty
    assert z.empty
