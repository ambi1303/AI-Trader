"""The leakage audit must:
1) PASS our real, leakage-safe features.
2) FAIL a deliberately-leaky feature (close.shift(-1)).

If either property breaks, the audit is not protecting us.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import statistical_features as sf
from src.features import technical_indicators as ti
from src.features.leakage_audit import (
    assert_no_future_dependence,
    audit_feature,
)


# ---------- positive cases: real features must pass ----------

def test_rsi_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: ti.rsi(df["close"], 14),
        feature_name="rsi_14",
    )


def test_macd_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: ti.macd(df["close"]),
        feature_name="macd",
    )


def test_bollinger_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: ti.bollinger(df["close"], 20, 2.0),
        feature_name="bollinger",
    )


def test_atr_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: ti.atr(df["high"], df["low"], df["close"], 14),
        feature_name="atr_14",
    )


def test_adx_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: ti.adx(df["high"], df["low"], df["close"], 14),
        feature_name="adx_14",
    )


def test_stochastic_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: ti.stochastic(df["high"], df["low"], df["close"], 14, 3),
        feature_name="stochastic",
    )


def test_returns_are_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: sf.simple_return(df["close"], 5),
        feature_name="ret_5d",
    )


def test_realized_vol_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: sf.realized_vol(df["close"], 20),
        feature_name="vol_20d",
    )


def test_drawdown_is_leakage_free() -> None:
    assert_no_future_dependence(
        lambda df: sf.drawdown_from_high(df["close"], 60),
        feature_name="dd_60d",
    )


# ---------- negative cases: known-leaky must fail ----------

def _leaky_next_close(df: pd.DataFrame) -> pd.Series:
    """Classic mistake: today's feature uses tomorrow's close."""
    return df["close"].shift(-1)


def _leaky_centered_window(df: pd.DataFrame) -> pd.Series:
    """Subtler mistake: window centered on row i sees future."""
    return df["close"].rolling(window=11, center=True, min_periods=11).mean()


def test_audit_detects_shift_minus_one() -> None:
    rep = audit_feature(
        _leaky_next_close, feature_name="leaky_shift_minus_one"
    )
    assert not rep.passed
    assert rep.leak_rows > 0


def test_assert_helper_raises_on_leak() -> None:
    with pytest.raises(AssertionError) as exc:
        assert_no_future_dependence(
            _leaky_next_close, feature_name="leaky_shift_minus_one"
        )
    msg = str(exc.value)
    assert "LEAKAGE" in msg
    assert "leaky_shift_minus_one" in msg


def test_audit_detects_centered_window() -> None:
    rep = audit_feature(
        _leaky_centered_window, feature_name="leaky_centered_mean"
    )
    assert not rep.passed
    assert rep.leak_rows > 0
