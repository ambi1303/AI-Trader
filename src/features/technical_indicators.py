"""Technical indicators in pure pandas / numpy.

We deliberately do not depend on `pandas-ta` (its 0.3.14b release breaks
against numpy 2.x) or TA-Lib (C compilation pain on Windows + ARM). All
indicators here are auditable, leakage-safe (only use t and earlier),
and unit-tested against either known reference values or invariants.

All functions take a DataFrame indexed by `bar_date` containing at minimum
columns: `open`, `high`, `low`, `close`, `volume`. They return either a
Series (single-output indicators) or a DataFrame with the indicator's
component columns.

Convention: NO function reads `df.close.shift(-k)` or any future-looking
slice. The accompanying leakage_audit catches violations automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. `adjust=False` to match the standard
    recursive EMA formula used by trading platforms."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# RSI (Wilder's smoothing)
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/period
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_up / avg_down
        out = 100.0 - (100.0 / (1.0 + rs))
    # Edge cases:
    # - constant series: avg_up == 0 and avg_down == 0 -> RSI undefined
    # - strictly rising:  avg_down == 0 and avg_up > 0  -> RSI = 100
    # - strictly falling: avg_up == 0  and avg_down > 0  -> RSI = 0
    out = out.where(~((avg_up == 0) & (avg_down == 0)))
    out = out.mask((avg_down == 0) & (avg_up > 0), 100.0)
    out = out.mask((avg_up == 0) & (avg_down > 0), 0.0)
    return out


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        }
    )


# ---------------------------------------------------------------------------
# True Range / ATR
# ---------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# ADX (Wilder)
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    tr = true_range(high, low, close)

    atr_n = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_n.replace(0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_n.replace(0, np.nan)
    )
    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame(
        {"adx_14": adx_val, "plus_di_14": plus_di, "minus_di_14": minus_di}
    )


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    pct_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": pct_k, "stoch_d": pct_d})
