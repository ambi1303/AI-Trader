"""Daily history for indices (^NSEI, ^INDIAVIX, sector indices).

Same yfinance + curl_cffi session pattern as the equity loader, written
into the `index_data` table. Indices have no PK collision with equities,
and we use auto_adjust=False because Yahoo doesn't auto-adjust indices anyway.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.data_ingestion.yfinance_loader import _build_session
from src.utils.db import transaction
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("ingest.indices")


def fetch_index(symbol: str, start: date, end: date) -> pd.DataFrame:
    session = _build_session()
    df = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        session=session,
    )
    if df is None or df.empty:
        log.warning("yfinance returned empty for index {}", symbol)
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    cache_dir = project_root() / "data" / "raw" / "indices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        cache_dir / f"{symbol.replace('^', '_')}_{start}_{end}.parquet", index=False
    )
    return df


def upsert_index_data(symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    for _, r in df.iterrows():
        d = r["Date"]
        if hasattr(d, "date"):
            d = d.date()
        close = r.get("Close")
        if pd.isna(close) or close is None or float(close) <= 0:
            continue
        rows.append(
            (
                symbol,
                d.isoformat(),
                float(r["Open"]) if not pd.isna(r.get("Open")) else None,
                float(r["High"]) if not pd.isna(r.get("High")) else None,
                float(r["Low"]) if not pd.isna(r.get("Low")) else None,
                float(close),
                int(r["Volume"]) if not pd.isna(r.get("Volume")) else None,
                "yfinance",
            )
        )
    if not rows:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO index_data
              (index_symbol, bar_date, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(index_symbol, bar_date, source) DO UPDATE SET
              open  = excluded.open,
              high  = excluded.high,
              low   = excluded.low,
              close = excluded.close,
              volume= excluded.volume
            """,
            rows,
        )
    log.info("Upserted {} index_data rows for {}", len(rows), symbol)
    return len(rows)


def ingest_indices(symbols: Iterable[str], start: date, end: date) -> int:
    total = 0
    for s in symbols:
        df = fetch_index(s, start, end)
        total += upsert_index_data(s, df)
    return total
