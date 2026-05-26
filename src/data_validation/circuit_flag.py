"""Heuristic circuit-limit detector.

NSE assigns price bands per stock (typically 5%, 10%, 20% or "no limit").
Without a per-stock band table, we approximate: when a daily move equals
high == low (no intraday range) and the move is at a common circuit %
band relative to previous close, treat as a circuit hit. This is a
v1 approximation; replace with NSE's published price band file in v2.
"""

from __future__ import annotations

from datetime import date

from src.utils.db import fetch_all, transaction
from src.utils.logger import get_logger

log = get_logger("validation.circuit_flag")

_BAND_CANDIDATES = (0.05, 0.10, 0.20)
_BAND_TOLERANCE = 0.005  # 0.5% tolerance around the band


def detect_and_persist(symbol: str, start: date | None = None, end: date | None = None) -> int:
    """Scan price_data for stale-range bars at common circuit bands.
    Returns count of flags written.
    """
    where = ["symbol = ?", "source = 'bhavcopy'"]
    params: list = [symbol]
    if start:
        where.append("bar_date >= ?")
        params.append(start.isoformat())
    if end:
        where.append("bar_date <= ?")
        params.append(end.isoformat())

    rows = fetch_all(
        f"""
        SELECT bar_date, open, high, low, close,
               LAG(close) OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_close
        FROM   price_data
        WHERE  {' AND '.join(where)}
        ORDER BY bar_date
        """,  # noqa: S608 (placeholders bound)
        tuple(params),
    )

    flags: list[tuple] = []
    for r in rows:
        prev_close = r["prev_close"]
        if prev_close is None or prev_close == 0:
            continue
        # No intraday range: stale, possibly at circuit
        if r["high"] != r["low"]:
            continue
        move = (r["close"] - prev_close) / prev_close
        for band in _BAND_CANDIDATES:
            if abs(abs(move) - band) <= _BAND_TOLERANCE:
                flags.append(
                    (
                        symbol,
                        r["bar_date"],
                        1 if move > 0 else 0,
                        1 if move < 0 else 0,
                        band,
                    )
                )
                break

    if not flags:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO circuit_flags
                (symbol, bar_date, hit_upper, hit_lower, band_pct)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, bar_date) DO UPDATE SET
                hit_upper = excluded.hit_upper,
                hit_lower = excluded.hit_lower,
                band_pct  = excluded.band_pct
            """,
            flags,
        )
    log.info("Wrote {} circuit flags for {}", len(flags), symbol)
    return len(flags)
