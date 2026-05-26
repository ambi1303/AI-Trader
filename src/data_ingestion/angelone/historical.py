"""Historical EOD candles via Angel One SmartAPI ``getCandleData``.

Why this exists alongside yfinance:
* Angel One pulls directly from NSE so adj-close / split handling is
  consistent with what your broker actually books. Useful as a third
  cross-source check (yfinance + bhavcopy + angelone).
* It's the only one of the three that has an LTP endpoint we can use
  for the "data freshness" check in the daily report.

Rate limits (per Angel One docs):
* getCandleData: 3 req/sec, 180/min, 5000/hour, 50000/day.
* We rate-limit conservatively to ~2 req/sec to leave headroom and we
  retry on 429 with exponential backoff via ``tenacity``.

The returned bars go into ``price_data`` with ``source='angelone'``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from decimal import Decimal
from typing import Iterable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.contracts.models import Bar, BarSource
from src.data_ingestion.angelone.instrument_master import (
    Instrument,
    download_instrument_master,
    resolve_token,
)
from src.data_ingestion.angelone.session import (
    AngelOneAPIError,
    AngelOneSession,
)
from src.data_ingestion.bar_writer import upsert_bars
from src.utils.logger import get_logger

log = get_logger("angelone.historical")

_CANDLE_PATH = "/rest/secure/angelbroking/historical/v1/getCandleData"

# Conservative client-side rate limit (2/s -> 120/min, well under the
# 180/min server cap so we have headroom for retries).
_RATE_LIMIT_INTERVAL_S = 0.5

# NSE EOD bars are stamped at 09:15 IST. We request a wide window
# (09:00..15:45) so the server returns the full trading day even on
# half-sessions.
_SESSION_START = dtime(9, 0)
_SESSION_END = dtime(15, 45)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandleFetchSummary:
    """Tally returned by :func:`fetch_daily_candles` so the orchestrator
    can log and surface in the daily report."""
    requested: int
    fetched_bars: int
    upserted_bars: int
    failed_symbols: list[str]


# ---------------------------------------------------------------------------
# Internal: request a single symbol
# ---------------------------------------------------------------------------


def _format_dt(d: date, t: dtime) -> str:
    """Angel One wants 'YYYY-MM-DD HH:MM' (IST, no seconds)."""
    return f"{d.isoformat()} {t.hour:02d}:{t.minute:02d}"


@retry(
    retry=retry_if_exception_type(AngelOneAPIError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _request_candles(
    session: AngelOneSession,
    instrument: Instrument,
    *,
    fromdate: date,
    todate: date,
    interval: str = "ONE_DAY",
) -> list[list]:
    payload = {
        "exchange": instrument.exchange,
        "symboltoken": instrument.token,
        "interval": interval,
        "fromdate": _format_dt(fromdate, _SESSION_START),
        "todate":   _format_dt(todate,   _SESSION_END),
    }
    body = session.request("POST", _CANDLE_PATH, json_body=payload)
    if not body or not body.get("status"):
        msg = (body or {}).get("message", "")
        # Angel One returns status=False for "no data in range" too.
        # We treat empty data as success-with-zero-bars.
        if "no data" in msg.lower():
            return []
        raise AngelOneAPIError(
            f"getCandleData failed for {instrument.symbol}: {msg!r}"
        )
    return body.get("data") or []


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def _to_bar(symbol: str, row: list) -> Bar | None:
    """Each row is ``[timestamp, open, high, low, close, volume]``.

    ``timestamp`` is an ISO8601 IST string like '2025-09-15T09:15:00+05:30'.
    We keep only the date component for daily bars.
    """
    if not row or len(row) < 6:
        return None
    try:
        ts = datetime.fromisoformat(str(row[0]))
        bar_date = ts.date()
        o = Decimal(str(row[1]))
        h = Decimal(str(row[2]))
        l = Decimal(str(row[3]))
        c = Decimal(str(row[4]))
        v = int(row[5])
    except (ValueError, TypeError, IndexError) as exc:
        log.warning("Discarding malformed angelone row for {}: {}", symbol, exc)
        return None
    if min(o, h, l, c) <= 0 or v < 0:
        return None
    return Bar(
        symbol=symbol, bar_date=bar_date,
        open=o, high=h, low=l, close=c, volume=v,
        adj_close=c,        # Angel One returns adjusted close as `close`.
        source=BarSource.ANGELONE,
    )


def fetch_daily_candles(
    session: AngelOneSession,
    symbols: Iterable[str],
    *,
    start: date,
    end: date | None = None,
    persist: bool = True,
    instruments: list[Instrument] | None = None,
) -> CandleFetchSummary:
    """Fetch daily bars for ``symbols`` between ``start`` and ``end``.

    Parameters
    ----------
    session : an authenticated :class:`AngelOneSession`. Will auto-login
        on first request if not yet authenticated.
    symbols : canonical NSE symbols (e.g. ``RELIANCE``, ``INFY``).
    start, end : inclusive date range. ``end`` defaults to today.
    persist : if True, upsert into ``price_data``; else just return counts.
    instruments : optional pre-loaded instrument master (saves one HTTP
        call when ingesting the universe).

    Notes
    -----
    Angel One caps the per-request range at 30 days for ONE_DAY interval.
    We chunk longer ranges automatically.
    """
    end = end or date.today()
    if start > end:
        raise ValueError(f"start={start} must be <= end={end}")

    insts = instruments if instruments is not None else download_instrument_master()
    by_symbol = {i.symbol: i for i in insts}

    requested = 0
    fetched = 0
    upserted = 0
    failed: list[str] = []
    last_call = 0.0

    for sym in symbols:
        sym = sym.upper().strip()
        requested += 1
        instrument = by_symbol.get(sym) or resolve_token(sym, insts)
        if instrument is None:
            log.warning("No Angel One token for symbol={}", sym)
            failed.append(sym)
            continue

        bars: list[Bar] = []
        # Chunk into <=30-day windows; Angel One rejects wider windows
        # with a vague error otherwise.
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + timedelta(days=29))
            elapsed = time.monotonic() - last_call
            if elapsed < _RATE_LIMIT_INTERVAL_S:
                time.sleep(_RATE_LIMIT_INTERVAL_S - elapsed)
            try:
                rows = _request_candles(
                    session, instrument,
                    fromdate=chunk_start, todate=chunk_end,
                )
            except AngelOneAPIError as exc:
                log.error(
                    "AngelOne candles failed | sym={} window={}->{} err={}",
                    sym, chunk_start, chunk_end, exc,
                )
                failed.append(sym)
                bars = []
                break
            finally:
                last_call = time.monotonic()
            for row in rows:
                bar = _to_bar(sym, row)
                if bar is not None:
                    bars.append(bar)
            chunk_start = chunk_end + timedelta(days=1)

        fetched += len(bars)
        if persist and bars:
            upserted += upsert_bars(bars)

    log.info(
        "Angel One historical | requested={} fetched={} upserted={} failed={}",
        requested, fetched, upserted, len(failed),
    )
    return CandleFetchSummary(
        requested=requested,
        fetched_bars=fetched,
        upserted_bars=upserted,
        failed_symbols=failed,
    )
