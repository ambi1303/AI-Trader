"""Returns, volatility, momentum, drawdowns, gaps.

All functions are leakage-safe by construction: every rolling/diff operation
looks only at past values relative to the row index. The leakage_audit
verifies this empirically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def simple_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return close.pct_change(periods=periods)


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close).diff(periods)


# ---------------------------------------------------------------------------
# Volatility (annualised? -- we keep it raw daily here; consumers can scale)
# ---------------------------------------------------------------------------


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of 1-day log returns. Daily, not annualised."""
    r = log_return(close, 1)
    return r.rolling(window=window, min_periods=window).std(ddof=0)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def momentum(close: pd.Series, window: int) -> pd.Series:
    """`close[t] / close[t-window] - 1`. Same as simple_return(window)."""
    return close.pct_change(periods=window)


# ---------------------------------------------------------------------------
# Drawdown from rolling high
# ---------------------------------------------------------------------------


def drawdown_from_high(close: pd.Series, window: int) -> pd.Series:
    """Negative or zero. (close - rolling_max) / rolling_max."""
    high = close.rolling(window=window, min_periods=1).max()
    return (close - high) / high.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def overnight_gap_pct(open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Percent gap between previous-day close and today's open."""
    return (open_ - prev_close) / prev_close.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volume z-score and ratio
# ---------------------------------------------------------------------------


def volume_avg(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window=window, min_periods=window).mean()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window=window, min_periods=window).mean()
    std = volume.rolling(window=window, min_periods=window).std(ddof=0)
    return (volume - mean) / std.replace(0, np.nan)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today / rolling mean. > 1 means above-average volume."""
    avg = volume.rolling(window=window, min_periods=window).mean()
    return volume / avg.replace(0, np.nan)
