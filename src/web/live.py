"""Near-real-time last-traded-price (LTP) service for the dashboard.

Wraps the read-only Angel One LTP endpoint (``src/data_ingestion/angelone``)
behind a tiny, dashboard-friendly service that:

* gates fetching to NSE market hours (IST, weekdays, non-holidays) so we
  don't spend API calls when nothing is moving;
* re-uses a single logged-in Angel One session across requests (login is
  expensive and rate-limited) and transparently re-logs in if it drops;
* caches each symbol's quote for a few seconds so repeated polls from the
  browser collapse into at most one upstream call per ``_CACHE_TTL_S``.

It is READ-ONLY and never places orders. When the market is closed, or
credentials/instruments are unavailable, it returns ``market_open=False``
(or ``ltp=None``) and the UI falls back to the last EOD close from the DB.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any

from src.utils.db import fetch_one
from src.utils.logger import get_logger

log = get_logger("web.live")

# India has no DST, so a fixed +5:30 offset is correct and dependency-free
# (avoids needing the tzdata package for zoneinfo on Windows).
_IST = timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)

# Browser polls every ~10s; cache slightly under that so each symbol hits the
# upstream API at most once per poll cycle even with several open tabs.
_CACHE_TTL_S = 8.0

# Module-level singletons guarded by a lock. uvicorn runs sync work in a
# threadpool, so concurrent requests are possible.
_lock = threading.Lock()
_session: Any = None
_instruments: Any = None
_quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}  # symbol -> (ts, payload)


def ist_now() -> datetime:
    return datetime.now(_IST)


def is_market_open(now: datetime | None = None) -> bool:
    """True when NSE cash market is in its continuous session right now.

    Mon-Fri, 09:15-15:30 IST, excluding days flagged in ``trading_calendar``.
    The pre-open auction (09:00-09:15) is treated as closed for LTP purposes.
    """
    n = now or ist_now()
    if not (_MARKET_OPEN <= n.time() <= _MARKET_CLOSE):
        return False
    # Canonical NSE calendar (weekends + declared holidays) from
    # config/nse_holidays.yaml -- the same source the validator uses.
    try:
        from src.data_validation import calendar_check
        return calendar_check.is_trading_day(n.date())
    except Exception:  # noqa: BLE001 -- never let a calendar hiccup break the gate
        return n.weekday() < 5


def _get_session():
    """Return a cached Angel One session, logging in lazily. Caller holds _lock."""
    global _session, _instruments
    if _session is None:
        from src.data_ingestion.angelone import (
            download_instrument_master,
            load_session_from_env,
        )
        _session = load_session_from_env()
        if _session is None:
            return None
        if _instruments is None:
            _instruments = download_instrument_master()
    return _session


def _drop_session() -> None:
    """Forget the cached session so the next call re-logs in. Caller holds _lock."""
    global _session
    try:
        if _session is not None:
            _session.logout()
    except Exception:  # noqa: BLE001
        pass
    _session = None


def _last_eod_close(symbol: str) -> tuple[float | None, str | None]:
    """Most recent stored close + its date, used as the closed-market fallback."""
    row = fetch_one(
        "SELECT close, bar_date FROM price_data "
        "WHERE symbol = ? AND source IN ('angelone','bhavcopy','yfinance') "
        "ORDER BY bar_date DESC, "
        "CASE source WHEN 'angelone' THEN 0 WHEN 'bhavcopy' THEN 1 ELSE 2 END "
        "LIMIT 1",
        (symbol,),
    )
    if row is None or row["close"] is None:
        return None, None
    return float(row["close"]), row["bar_date"]


def _payload(
    symbol: str,
    *,
    market_open: bool,
    ltp: float | None,
    prev_close: float | None,
    as_of: str | None,
    source: str,
) -> dict[str, Any]:
    change = change_pct = None
    if ltp is not None and prev_close not in (None, 0):
        change = ltp - prev_close
        change_pct = (ltp / prev_close - 1.0) * 100.0
    return {
        "symbol": symbol,
        "market_open": market_open,
        "ltp": ltp,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "as_of": as_of,
        "source": source,
        "ts": ist_now().isoformat(timespec="seconds"),
    }


def get_live_quote(symbol: str) -> dict[str, Any]:
    """Live LTP for one symbol, with market-hours gating + caching + fallback.

    Always returns a dict the UI can render; ``ltp`` may be ``None`` when the
    market is closed or the quote is unavailable (then the UI shows the last
    EOD close from ``as_of``/``prev_close``).
    """
    sym = symbol.upper().strip()
    prev_close, eod_date = _last_eod_close(sym)

    if not is_market_open():
        return _payload(sym, market_open=False, ltp=None,
                        prev_close=prev_close, as_of=eod_date, source="eod")

    now = time.monotonic()
    with _lock:
        cached = _quote_cache.get(sym)
        if cached and (now - cached[0]) < _CACHE_TTL_S:
            return cached[1]

        try:
            from src.data_ingestion.angelone import fetch_ltp
            session = _get_session()
            if session is None:
                # No creds -> behave like closed market (fall back to EOD).
                payload = _payload(sym, market_open=False, ltp=None,
                                   prev_close=prev_close, as_of=eod_date,
                                   source="eod")
                _quote_cache[sym] = (now, payload)
                return payload

            quotes = fetch_ltp(session, [sym], instruments=_instruments)
        except Exception as exc:  # noqa: BLE001
            log.warning("live LTP fetch failed for {}: {}", sym, exc)
            _drop_session()  # force a fresh login next time
            payload = _payload(sym, market_open=True, ltp=None,
                               prev_close=prev_close, as_of=eod_date,
                               source="error")
            _quote_cache[sym] = (now, payload)
            return payload

        if quotes:
            qt = quotes[0]
            # LtpQuote.close is the previous close per Angel One; prefer it.
            pc = qt.close if qt.close not in (None, 0) else prev_close
            payload = _payload(sym, market_open=True, ltp=qt.ltp,
                               prev_close=pc, as_of=ist_now().date().isoformat(),
                               source="angelone")
        else:
            payload = _payload(sym, market_open=True, ltp=None,
                               prev_close=prev_close, as_of=eod_date,
                               source="unavailable")
        _quote_cache[sym] = (now, payload)
        return payload


def get_live_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Batch helper for the dashboard. Returns {symbol: payload}."""
    out: dict[str, dict[str, Any]] = {}
    for s in symbols:
        sym = (s or "").upper().strip()
        if sym:
            out[sym] = get_live_quote(sym)
    return out
