"""Data sourcing for the Complete Analysis page.

Decides where a symbol's price history + fundamentals come from:

* **Universe stocks** (the daily-pipeline names) already have ``price_data`` and
  ``fundamental_data`` in the DB -> read those (instant, no network).
* **Any other NSE stock** -> fetch on demand from yfinance (daily history +
  ``Ticker.info`` snapshot), cached briefly so repeated visits/polls don't
  re-hit Yahoo.

The on-demand fetch reuses :func:`fundamentals.fetch_snapshot`, so the (fixed)
dividend-yield scaling and all other ratio handling stay consistent with the
daily pipeline. Everything is read-only; nothing here writes to the DB.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

log = get_logger("analysis.datasource")

# Symbols whose on-demand fetch we cache: symbol -> (epoch, (df, fundamentals)).
_CACHE_TTL_S = 600.0  # 10 minutes; fundamentals/daily bars don't move faster
_cache: dict[str, tuple[float, tuple[pd.DataFrame, dict[str, Any] | None]]] = {}
_lock = threading.Lock()

_FUND_KEYS = (
    "pe_ttm", "pb", "roe", "debt_to_equity", "profit_margin",
    "revenue_growth", "earnings_growth", "dividend_yield", "market_cap",
    "eps_ttm", "book_value", "as_of_date", "source",
)


def load_history_db(symbol: str) -> pd.DataFrame:
    """Daily OHLCV for a symbol from ``price_data`` (dedup by date)."""
    from src.utils.db import fetch_all
    rows = fetch_all(
        """
        SELECT bar_date, open, high, low, close, volume, source
        FROM   price_data
        WHERE  symbol = ? AND source IN ('angelone','bhavcopy','yfinance')
        ORDER  BY bar_date
        """,
        (symbol.upper(),),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    rank = {"angelone": 0, "bhavcopy": 1, "yfinance": 2}
    df["_r"] = df["source"].map(lambda s: rank.get(s, 9))
    df = (df.sort_values(["bar_date", "_r"])
            .drop_duplicates("bar_date", keep="first")
            .sort_values("bar_date"))
    return df[["bar_date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _fetch_ondemand(symbol: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Fetch ~2y daily history + a fundamentals snapshot from yfinance.

    Cached for ``_CACHE_TTL_S``. Returns (empty df, None) on failure so the
    caller can show a graceful "couldn't fetch" state instead of crashing.
    """
    sym = symbol.upper().strip()
    now = time.monotonic()
    with _lock:
        hit = _cache.get(sym)
        if hit and (now - hit[0]) < _CACHE_TTL_S:
            return hit[1]

    df = pd.DataFrame()
    fundamentals: dict[str, Any] | None = None

    # Never make a live network call inside the test suite (keeps tests fast,
    # offline and deterministic).
    import os
    if os.getenv("PYTEST_CURRENT_TEST"):
        return df, fundamentals
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import yfinance as yf
        from src.data_ingestion.fundamentals import _build_session, fetch_snapshot

        session = _build_session()
        tk = yf.Ticker(f"{sym}.NS", session=session)
        hist = tk.history(period="2y", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]
            df = pd.DataFrame({
                "bar_date": pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d"),
                "open": pd.to_numeric(hist.get("Open"), errors="coerce"),
                "high": pd.to_numeric(hist.get("High"), errors="coerce"),
                "low": pd.to_numeric(hist.get("Low"), errors="coerce"),
                "close": pd.to_numeric(hist.get("Close"), errors="coerce"),
                "volume": pd.to_numeric(hist.get("Volume"), errors="coerce"),
            })
        try:
            info = tk.info or {}
        except Exception:  # noqa: BLE001
            info = {}
        if info:
            snap = fetch_snapshot(sym, info)
            d = asdict(snap)
            fundamentals = {k: d.get(k) for k in _FUND_KEYS}
    except Exception as exc:  # noqa: BLE001
        log.warning("on-demand fetch failed for {}: {}", sym, exc)

    result = (df, fundamentals)
    with _lock:
        _cache[sym] = (now, result)
    return result


def get_price_and_fundamentals(
    symbol: str,
    fundamentals_db: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Return (ohlcv_df, fundamentals, source) for analysis.

    Price history comes from the DB when available (BhavCopy covers ~all NSE);
    fundamentals come from the DB for universe names and are fetched live for
    everything else. ``source`` is "db" or "live" so the UI can label it.
    """
    df = load_history_db(symbol)
    source = "db"
    fundamentals = fundamentals_db

    # Fetch live only for what's missing: price (rare) and/or fundamentals.
    if len(df) < 20 or fundamentals is None:
        live_df, live_fund = _fetch_ondemand(symbol)
        if len(df) < 20:
            df = live_df
            source = "live"
        if fundamentals is None and live_fund is not None:
            fundamentals = live_fund
    return df, fundamentals, source
