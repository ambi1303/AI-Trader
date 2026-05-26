"""Persist Bar contracts to price_data with idempotent upsert."""

from __future__ import annotations

from src.contracts import Bar
from src.utils.db import transaction
from src.utils.logger import get_logger

log = get_logger("ingest.bar_writer")


def upsert_bars(bars: list[Bar]) -> int:
    if not bars:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO price_data
                (symbol, bar_date, open, high, low, close, volume, adj_close, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, bar_date, source) DO UPDATE SET
                open      = excluded.open,
                high      = excluded.high,
                low       = excluded.low,
                close     = excluded.close,
                volume    = excluded.volume,
                adj_close = excluded.adj_close
            """,
            [
                (
                    b.symbol,
                    b.bar_date.isoformat(),
                    float(b.open),
                    float(b.high),
                    float(b.low),
                    float(b.close),
                    int(b.volume),
                    float(b.adj_close) if b.adj_close is not None else None,
                    b.source.value,
                )
                for b in bars
            ],
        )
    log.debug("Wrote {} bars to price_data", len(bars))
    return len(bars)
