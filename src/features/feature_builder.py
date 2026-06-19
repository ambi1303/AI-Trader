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
from src.utils.db import fetch_all, fetch_one, transaction
from src.utils.logger import get_logger

log = get_logger("features.builder")

FEATURE_SET_VERSION = 2  # v2: added fundamental features (as-of joined)

# Fundamental feature columns (as-of joined from fundamental_data, no
# look-ahead). market_cap is stored as its natural log so the size factor is
# well-behaved for tree splits and cross-sectional ranking.
_FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "pe_ttm",
    "pb",
    "roe",
    "debt_to_equity",
    "profit_margin",
    "revenue_growth",
    "earnings_growth",
    "dividend_yield",
    "log_market_cap",
)

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
    *_FUNDAMENTAL_COLUMNS,
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
    """Load OHLCV for a symbol, preferring BhavCopy but merging yfinance for
    more recent dates not yet covered by BhavCopy.

    This keeps the system current between BhavCopy ingests (yfinance updates
    faster) while still using the authoritative NSE data where available.
    """
    rows = fetch_all(
        """
        SELECT bar_date, open, high, low, close, volume
        FROM   price_data
        WHERE  symbol = ? AND source = ?
        ORDER BY bar_date
        """,
        (symbol, source),
    )
    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    if not df.empty:
        df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
        df = df.set_index("bar_date").sort_index()

    # Supplement with yfinance rows that are NEWER than the bhavcopy max date.
    yf_rows = fetch_all(
        """
        SELECT bar_date, open, high, low, close, volume
        FROM   price_data
        WHERE  symbol = ? AND source = 'yfinance'
              AND bar_date > COALESCE(
                  (SELECT MAX(bar_date) FROM price_data
                   WHERE symbol = ? AND source = ?), '1900-01-01')
        ORDER BY bar_date
        """,
        (symbol, symbol, source),
    )
    if yf_rows:
        yf = pd.DataFrame([dict(r) for r in yf_rows])
        yf["bar_date"] = pd.to_datetime(yf["bar_date"]).dt.date
        yf = yf.set_index("bar_date").sort_index()
        df = pd.concat([df, yf]) if not df.empty else yf

    return df


_PRICE_SOURCES_IN = "('bhavcopy','yfinance')"


def _last_feature_date(symbol: str) -> date | None:
    """Latest feature_date already stored for the *current* feature version.

    Restricting to the current version means a ``FEATURE_SET_VERSION`` bump
    transparently forces a full rebuild (old rows don't count as 'fresh').
    """
    row = fetch_one(
        "SELECT MAX(feature_date) AS d FROM feature_data "
        "WHERE symbol = ? AND feature_set_version = ?",
        (symbol, FEATURE_SET_VERSION),
    )
    d = row["d"] if row else None
    return date.fromisoformat(d) if d else None


def _max_price_date(symbol: str) -> date | None:
    row = fetch_one(
        f"SELECT MAX(bar_date) AS d FROM price_data "
        f"WHERE symbol = ? AND source IN {_PRICE_SOURCES_IN}",
        (symbol,),
    )
    d = row["d"] if row else None
    return date.fromisoformat(d) if d else None


def _warmup_start(symbol: str, last_fd: date, lookback_bars: int) -> date | None:
    """The bar_date ``lookback_bars`` trading days before ``last_fd``.

    Recomputing the tail over this window gives the recursive indicators
    (EMA/RSI/ATR/ADX) enough warm-up to converge to within a negligible
    error of a full-history build, while loading/processing far less data.
    """
    rows = fetch_all(
        f"""
        SELECT bar_date FROM (
            SELECT DISTINCT bar_date FROM price_data
            WHERE symbol = ? AND bar_date <= ? AND source IN {_PRICE_SOURCES_IN}
            ORDER BY bar_date DESC LIMIT ?
        ) ORDER BY bar_date ASC LIMIT 1
        """,
        (symbol, last_fd.isoformat(), int(lookback_bars)),
    )
    if not rows:
        return None
    d = rows[0]["bar_date"]
    return date.fromisoformat(d) if isinstance(d, str) else d


def load_market_context(
    nifty_symbol: str = "^NSEI", vix_symbol: str = "^INDIAVIX"
) -> tuple[pd.Series, pd.Series]:
    """Load Nifty + VIX close series once so a batch build can reuse them
    instead of re-querying the full index history for every symbol."""
    return _load_index_close(nifty_symbol), _load_index_close(vix_symbol)


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


def _load_fundamentals(symbol: str) -> pd.DataFrame:
    """Load all fundamental rows for a symbol, oldest first.

    Returns an empty frame if none exist (the symbol simply gets NaN
    fundamental features, which XGBoost handles natively).
    """
    rows = fetch_all(
        """
        SELECT as_of_date, pe_ttm, pb, roe, debt_to_equity, profit_margin,
               revenue_growth, earnings_growth, dividend_yield, market_cap
        FROM   fundamental_data
        WHERE  symbol = ?
        ORDER BY as_of_date
        """,
        (symbol,),
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def _attach_fundamentals(
    out: pd.DataFrame, fundamentals_df: pd.DataFrame | None
) -> pd.DataFrame:
    """As-of (backward) join fundamentals onto each feature_date.

    For every feature row dated T we take the most recent fundamental row
    whose ``as_of_date`` is <= T. This guarantees no look-ahead: a feature
    date can only see fundamentals that were already published by then.
    Values persist (forward-fill) until the next quarterly report.
    """
    for c in _FUNDAMENTAL_COLUMNS:
        out[c] = np.nan
    if fundamentals_df is None or fundamentals_df.empty:
        return out

    right = fundamentals_df.copy()
    right["d"] = pd.to_datetime(right["as_of_date"], errors="coerce")
    right = right.dropna(subset=["d"]).sort_values("d")
    if right.empty:
        return out

    mc = pd.to_numeric(right.get("market_cap"), errors="coerce")
    right["log_market_cap"] = np.log(mc.where(mc > 0))

    cols_present = [c for c in _FUNDAMENTAL_COLUMNS if c in right.columns]
    left = pd.DataFrame({"d": pd.to_datetime(pd.Index(out.index))})
    left = left.sort_values("d").reset_index(drop=True)
    merged = pd.merge_asof(
        left, right[["d", *cols_present]], on="d", direction="backward"
    )
    merged["date_key"] = merged["d"].dt.date
    for c in cols_present:
        mapping = dict(zip(merged["date_key"], merged[c]))
        out[c] = [mapping.get(d, np.nan) for d in out.index]
    return out


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
    fundamentals_df: pd.DataFrame | None = None,
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

    # Fundamentals (as-of joined, forward-filled, no look-ahead)
    out = _attach_fundamentals(out, fundamentals_df)

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
    incremental: bool = False,
    lookback_bars: int = 600,
    nifty_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    sector_close_cache: dict[str, pd.Series] | None = None,
) -> BuildSummary:
    """Build (and upsert) features for one symbol.

    Incremental mode (``incremental=True``): skip the symbol entirely if its
    stored features are already current, otherwise recompute only the tail
    (warmed up by ``lookback_bars`` prior bars) and persist just the new dates.
    This turns the daily run from a full-history rebuild into a cheap append.

    Pre-loaded ``nifty_close`` / ``vix_close`` and a shared
    ``sector_close_cache`` let a batch caller load the index series once
    instead of re-querying them for every symbol.
    """
    persist_after: date | None = None
    if incremental:
        last_fd = _last_feature_date(symbol)
        max_pd = _max_price_date(symbol)
        if last_fd is not None and max_pd is not None and last_fd >= max_pd:
            return BuildSummary(symbol=symbol, rows_in=0, rows_out=0)  # up to date
        if last_fd is not None:
            persist_after = last_fd
            ws = _warmup_start(symbol, last_fd, lookback_bars)
            if ws is not None and (start is None or ws > start):
                start = ws

    price_df = _load_price_df(symbol)
    if start is not None:
        price_df = price_df[price_df.index >= start]
    if end is not None:
        price_df = price_df[price_df.index <= end]

    if price_df.empty:
        log.warning("No bhavcopy price data for {} in the given range", symbol)
        return BuildSummary(symbol=symbol, rows_in=0, rows_out=0)

    if nifty_close is None:
        nifty_close = _load_index_close(nifty_symbol)
    if vix_close is None:
        vix_close = _load_index_close(vix_symbol)

    sector_idx = _load_sector_index(symbol)
    if sector_close_cache is not None:
        if sector_idx not in sector_close_cache:
            sector_close_cache[sector_idx] = (
                _load_index_close(sector_idx)
                if sector_idx and sector_idx != nifty_symbol else nifty_close
            )
        sector_close = sector_close_cache[sector_idx]
    else:
        sector_close = (
            _load_index_close(sector_idx)
            if sector_idx and sector_idx != nifty_symbol else nifty_close
        )
    circuit_df = _load_circuit_flags(symbol)
    fundamentals_df = _load_fundamentals(symbol)

    feat_df = compute_features_for_symbol(
        price_df,
        nifty_close=nifty_close,
        vix_close=vix_close,
        sector_close=sector_close,
        circuit_df=circuit_df,
        fundamentals_df=fundamentals_df,
    )
    if persist_after is not None and not feat_df.empty:
        feat_df = feat_df[feat_df.index > persist_after]
    rows_out = _persist(symbol, feat_df)
    log.info("Built {} feature rows for {}{}", rows_out, symbol,
             " (incremental)" if incremental else "")
    return BuildSummary(symbol=symbol, rows_in=len(price_df), rows_out=rows_out)
