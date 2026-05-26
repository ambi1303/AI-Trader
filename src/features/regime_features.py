"""Market-regime features.

These attach a market-context view to each (symbol, date) row:
- Nifty 50 (^NSEI) trend: distance from 50d / 200d MA.
- India VIX (^INDIAVIX): level and 5d % change.
- Stock's beta and correlation to Nifty over a 60d rolling window.
- Sector relative strength: stock's 20d return minus its sector index's 20d return.

All inputs come from `index_data` and `price_data` already in the DB.
The functions are pure: given price/index DataFrames they produce features;
no IO inside the math. The IO lives in `feature_builder`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def index_trend_features(index_close: pd.Series) -> pd.DataFrame:
    ma50 = index_close.rolling(window=50, min_periods=50).mean()
    ma200 = index_close.rolling(window=200, min_periods=200).mean()
    return pd.DataFrame(
        {
            "nifty_dist_ma50_pct": (index_close - ma50) / ma50.replace(0, np.nan),
            "nifty_dist_ma200_pct": (index_close - ma200) / ma200.replace(0, np.nan),
        }
    )


def vix_features(vix_close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vix_level": vix_close,
            "vix_chg_5d_pct": vix_close.pct_change(5),
        }
    )


def beta_corr(
    stock_log_ret: pd.Series, index_log_ret: pd.Series, window: int = 60
) -> pd.DataFrame:
    """Rolling beta and correlation of stock to a benchmark, on log returns."""
    cov = stock_log_ret.rolling(window=window, min_periods=window).cov(index_log_ret)
    var = index_log_ret.rolling(window=window, min_periods=window).var(ddof=0)
    beta = cov / var.replace(0, np.nan)
    corr = stock_log_ret.rolling(window=window, min_periods=window).corr(index_log_ret)
    return pd.DataFrame({"beta_60d": beta, "corr_60d": corr})


def sector_relative_strength(
    stock_close: pd.Series, sector_close: pd.Series, window: int = 20
) -> pd.Series:
    """Stock's window-day return minus the sector index's window-day return."""
    s_ret = stock_close.pct_change(window)
    sec_ret = sector_close.pct_change(window)
    return s_ret - sec_ret
