"""Daily OHLCV via yfinance with adjusted closes.

We fetch with auto_adjust=True so OHLC are split/dividend-adjusted; the raw
close is reconstructed from the relative adjustment factor in adj_close /
close ratio if needed by the validator. yfinance occasionally returns gaps
or zero-volume bars; we skip rows that fail Bar validation rather than
crashing the whole batch.

We use curl_cffi's browser-impersonating session because Yahoo Finance now
fingerprints stock-requests clients and returns empty / "possibly delisted"
responses to plain requests sessions. With impersonation we look like
Chrome and the API behaves normally.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import yfinance as yf

from src.contracts import Bar, BarSource
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("ingest.yfinance")


def _build_session():
    """A curl_cffi session impersonating Chrome. Falls back gracefully."""
    try:
        from curl_cffi import requests as crequests

        return crequests.Session(impersonate="chrome")
    except Exception as e:  # noqa: BLE001
        log.warning(
            "curl_cffi session unavailable ({}); falling back to default yfinance",
            str(e),
        )
        return None


def _to_yf_symbol(symbol: str) -> str:
    # NSE tickers on Yahoo use a .NS suffix.
    # M&M, BAJAJ-AUTO etc. are passed through unchanged.
    return f"{symbol}.NS"


def fetch_history(
    symbol: str,
    start: date,
    end: date,
) -> list[Bar]:
    """Fetch [start, end) daily bars for a single symbol.

    We deliberately use auto_adjust=False so the stored `close` is the *raw*
    market close (matching BhavCopy), and `adj_close` carries Yahoo's
    split-and-dividend-adjusted close for downstream feature engineering.
    Mixing the two on one column produces persistent multi-percent
    cross-source mismatches whenever there's a recent dividend.
    """
    yf_sym = _to_yf_symbol(symbol)
    log.debug("yfinance fetch {} {} -> {}", yf_sym, start, end)
    session = _build_session()
    df = yf.download(
        yf_sym,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        session=session,
    )
    if df is None or df.empty:
        log.warning("yfinance returned empty for {}", yf_sym)
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    cache_dir = project_root() / "data" / "raw" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{symbol}_{start}_{end}.parquet", index=False)

    bars: list[Bar] = []
    for _, row in df.iterrows():
        try:
            d = row["Date"]
            if hasattr(d, "date"):
                d = d.date()
            adj_close_val = row["Adj Close"] if "Adj Close" in row.index else row["Close"]
            bars.append(
                Bar(
                    symbol=symbol,
                    bar_date=d,
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    adj_close=Decimal(str(adj_close_val))
                    if not pd.isna(adj_close_val)
                    else None,
                    source=BarSource.YFINANCE,
                )
            )
        except Exception as e:
            log.warning("Skipping yf row {} {}: {}", symbol, row.get("Date"), str(e))
    return bars
