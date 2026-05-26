"""Feature builder: per-symbol feature matrix -> feature_data table.

Layout:
1. Load BhavCopy bars for the symbol over the requested date range.
   (We use BhavCopy because it's the unadjusted ground-truth close that
   matches NSE; yfinance Close is split-adjusted retroactively, which
   distorts technical indicators around old corporate actions.)
2. Compute technical, statistical, regime, circuit feature columns.
3. Align all features on the symbol's trading-day index.
4. Upsert into feature_data.

Feature-set version is recorded in `feature_set_version` so we can detect
later when a model was trained on an older feature definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.features import (
    circuit_features as cf,
)
from src.features import (
    regime_features as rf,
)
from src.features import (
    statistical_features as sf,
)
from src.features import (
    technical_indicators as ti,
)
from src.utils.db import fetch_all, transaction
from src.utils.logger import get_logger

log = get_logger("features.builder")

FEATURE_SET_VERSION = 1

# Columns of feature_data that we populate. Keep in sync with schema.sql.
_FEATURE_COLUMNS: tuple[str, ...] = (
    "close",
    "volume",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "log_ret_1d",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "dd_from_high_20d",
    "dd_from_high_60d",
    "dd_from_high_252d",
    "gap_pct",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_20",
    "ema_50",
    "ema_200",
    "dist_ema_20_pct",
    "dist_ema_50_pct",
    "dist_ema_200_pct",
    "bb_upper",
    "bb_lower",
    "bb_pct_b",
    "bb_bandwidth",
    "atr_14",
    "atr_pct",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "obv",
    "stoch_k",
    "stoch_d",
    "vol_avg_20d",
    "vol_z_20d",
    "vol_ratio_20d",
    "nifty_dist_ma50_pct",
    "nifty_dist_ma200_pct",
    "vix_level",
    "vix_chg_5d_pct",
    "beta_60d",
    "corr_60d",
    "sector_rs_20d",
    "hit_upper_circuit",
    "hit_lower_circuit",
    "days_since_circuit",
    "low_volume_flag",
)


@dataclass
class BuildSummary:
    symbol: str
    rows_in: int
    rows_out: int


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _load_price_df(symbol: str, source: str = "bhavcopy") -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT bar_date, open, high, low, close, volume
        FROM   price_data
        WHERE  symbol = ? AND source = ?
        ORDER BY bar_date
        """,
        (symbol, source),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    df = df.set_index("bar_date").sort_index()
    return df


def _load_index_close(symbol: str) -> pd.Series:
    rows = fetch_all(
        "SELECT bar_date, close FROM index_data WHERE index_symbol = ? ORDER BY bar_date",
        (symbol,),
    )
    if not rows:
        return pd.Series(dtype="float64", name=symbol)
    s = pd.Series(
        [r["close"] for r in rows],
        index=[datetime.fromisoformat(r["bar_date"]).date() for r in rows],
        name=symbol,
        dtype="float64",
    )
    return s.sort_index()


def _load_circuit_flags(symbol: str) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT bar_date, hit_upper, hit_lower
        FROM   circuit_flags
        WHERE  symbol = ?
        ORDER BY bar_date
        """,
        (symbol,),
    )
    if not rows:
        return pd.DataFrame(columns=["bar_date", "hit_upper", "hit_lower"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    return df


def _load_sector_index(symbol: str) -> str:
    rows = fetch_all(
        "SELECT sector_index FROM stock_sectors WHERE symbol = ?", (symbol,)
    )
    return rows[0]["sector_index"] if rows else "^NSEI"


# ---------------------------------------------------------------------------
# Feature construction (pure)
# ---------------------------------------------------------------------------


def compute_features_for_symbol(
    price_df: pd.DataFrame,
    *,
    nifty_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    sector_close: pd.Series | None = None,
    circuit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pure: takes already-loaded DataFrames and returns a feature DataFrame."""
    if price_df.empty:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)

    out = pd.DataFrame(index=price_df.index)
    close = price_df["close"]
    high = price_df["high"]
    low = price_df["low"]
    open_ = price_df["open"]
    volume = price_df["volume"]

    out["close"] = close
    out["volume"] = volume

    # Returns
    out["ret_1d"] = sf.simple_return(close, 1)
    out["ret_5d"] = sf.simple_return(close, 5)
    out["ret_10d"] = sf.simple_return(close, 10)
    out["ret_20d"] = sf.simple_return(close, 20)
    out["log_ret_1d"] = sf.log_return(close, 1)

    # Volatility
    out["vol_5d"] = sf.realized_vol(close, 5)
    out["vol_20d"] = sf.realized_vol(close, 20)
    out["vol_60d"] = sf.realized_vol(close, 60)

    # Momentum
    out["mom_5d"] = sf.momentum(close, 5)
    out["mom_20d"] = sf.momentum(close, 20)
    out["mom_60d"] = sf.momentum(close, 60)

    # Drawdown
    out["dd_from_high_20d"] = sf.drawdown_from_high(close, 20)
    out["dd_from_high_60d"] = sf.drawdown_from_high(close, 60)
    out["dd_from_high_252d"] = sf.drawdown_from_high(close, 252)

    # Gap
    out["gap_pct"] = sf.overnight_gap_pct(open_, close.shift(1))

    # Technicals
    out["rsi_14"] = ti.rsi(close, 14)
    macd_df = ti.macd(close)
    out[["macd", "macd_signal", "macd_hist"]] = macd_df

    e20 = ti.ema(close, 20)
    e50 = ti.ema(close, 50)
    e200 = ti.ema(close, 200)
    out["ema_20"] = e20
    out["ema_50"] = e50
    out["ema_200"] = e200
    out["dist_ema_20_pct"] = (close - e20) / e20.replace(0, np.nan)
    out["dist_ema_50_pct"] = (close - e50) / e50.replace(0, np.nan)
    out["dist_ema_200_pct"] = (close - e200) / e200.replace(0, np.nan)

    bb = ti.bollinger(close, 20, 2.0)
    out[["bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth"]] = bb

    out["atr_14"] = ti.atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)

    adx_df = ti.adx(high, low, close, 14)
    out[["adx_14", "plus_di_14", "minus_di_14"]] = adx_df

    out["obv"] = ti.obv(close, volume)
    stoch = ti.stochastic(high, low, close, 14, 3)
    out[["stoch_k", "stoch_d"]] = stoch

    # Volume features
    out["vol_avg_20d"] = sf.volume_avg(volume, 20)
    out["vol_z_20d"] = sf.volume_zscore(volume, 20)
    out["vol_ratio_20d"] = sf.volume_ratio(volume, 20)

    # Regime: nifty trend
    if nifty_close is not None and not nifty_close.empty:
        nifty_aligned = nifty_close.reindex(out.index).ffill()
        nifty_trend = rf.index_trend_features(nifty_aligned)
        out[["nifty_dist_ma50_pct", "nifty_dist_ma200_pct"]] = nifty_trend
        # beta / corr
        stock_lr = sf.log_return(close, 1)
        idx_lr = sf.log_return(nifty_aligned, 1)
        bc = rf.beta_corr(stock_lr, idx_lr, window=60)
        out[["beta_60d", "corr_60d"]] = bc
    else:
        out["nifty_dist_ma50_pct"] = np.nan
        out["nifty_dist_ma200_pct"] = np.nan
        out["beta_60d"] = np.nan
        out["corr_60d"] = np.nan

    # VIX
    if vix_close is not None and not vix_close.empty:
        vix_aligned = vix_close.reindex(out.index).ffill()
        vix_df = rf.vix_features(vix_aligned)
        out[["vix_level", "vix_chg_5d_pct"]] = vix_df
    else:
        out["vix_level"] = np.nan
        out["vix_chg_5d_pct"] = np.nan

    # Sector RS
    if sector_close is not None and not sector_close.empty:
        sec_aligned = sector_close.reindex(out.index).ffill()
        out["sector_rs_20d"] = rf.sector_relative_strength(close, sec_aligned, 20)
    else:
        out["sector_rs_20d"] = np.nan

    # Circuit / liquidity
    out = cf.attach_circuit_flags(out, circuit_df)
    out["low_volume_flag"] = cf.low_volume_flag(volume, 20, 0.20)

    return out[list(_FEATURE_COLUMNS)]


# ---------------------------------------------------------------------------
# Public entry point: build + persist for a symbol
# ---------------------------------------------------------------------------


def _to_python(x):
    """Convert pandas/numpy scalars to plain Python (None for NaN)."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (np.integer, np.int64, np.int32)):
        return int(x)
    if isinstance(x, (np.floating, np.float64, np.float32)):
        return None if np.isnan(x) else float(x)
    return x


def _persist(symbol: str, feat_df: pd.DataFrame) -> int:
    if feat_df.empty:
        return 0
    cols = list(_FEATURE_COLUMNS)
    placeholders = ",".join(["?"] * (1 + 1 + len(cols) + 1))
    insert_cols = ["symbol", "feature_date", *cols, "feature_set_version"]
    update_set = ", ".join(f"{c}=excluded.{c}" for c in [*cols, "feature_set_version"])
    sql = (
        f"INSERT INTO feature_data ({', '.join(insert_cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(symbol, feature_date) DO UPDATE SET {update_set}"
    )
    rows = []
    for d, row in feat_df.iterrows():
        if isinstance(d, (datetime,)):
            d = d.date()
        if hasattr(d, "isoformat"):
            d_str = d.isoformat()
        else:
            d_str = str(d)
        rows.append(
            (
                symbol,
                d_str,
                *[_to_python(row[c]) for c in cols],
                FEATURE_SET_VERSION,
            )
        )
    with transaction() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def build_for_symbol(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    nifty_symbol: str = "^NSEI",
    vix_symbol: str = "^INDIAVIX",
) -> BuildSummary:
    price_df = _load_price_df(symbol)
    if start is not None:
        price_df = price_df[price_df.index >= start]
    if end is not None:
        price_df = price_df[price_df.index <= end]

    if price_df.empty:
        log.warning("No bhavcopy price data for {} in the given range", symbol)
        return BuildSummary(symbol=symbol, rows_in=0, rows_out=0)

    nifty_close = _load_index_close(nifty_symbol)
    vix_close = _load_index_close(vix_symbol)
    sector_idx = _load_sector_index(symbol)
    sector_close = (
        _load_index_close(sector_idx) if sector_idx and sector_idx != nifty_symbol
        else nifty_close
    )
    circuit_df = _load_circuit_flags(symbol)

    feat_df = compute_features_for_symbol(
        price_df,
        nifty_close=nifty_close,
        vix_close=vix_close,
        sector_close=sector_close,
        circuit_df=circuit_df,
    )
    rows_out = _persist(symbol, feat_df)
    log.info("Built {} feature rows for {}", rows_out, symbol)
    return BuildSummary(symbol=symbol, rows_in=len(price_df), rows_out=rows_out)
