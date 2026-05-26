"""Live LTP / quote fetch via Angel One SmartAPI.

We use this to:
* Mark open paper positions to market in near-real-time during the daily
  report rendering (right now we use last EOD close from yfinance).
* Provide a freshness probe: if Angel One says LTP=0 or returns an error
  for a symbol, the daily report flags it.

Endpoint (read-only): POST /rest/secure/angelbroking/order/v1/getLtpData
Rate limit: 50 req/sec; we call it at most once per universe per run so
this is irrelevant in practice.

The endpoint name lives under /order/v1/ in Angel's URL space but is a
GET-of-LTP, NOT an order placement. We mark ``is_order_endpoint=False``
in the session call accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data_ingestion.angelone.instrument_master import (
    Instrument,
    download_instrument_master,
)
from src.data_ingestion.angelone.session import (
    AngelOneAPIError,
    AngelOneSession,
)
from src.utils.logger import get_logger

log = get_logger("angelone.quote")

_LTP_PATH = "/rest/secure/angelbroking/order/v1/getLtpData"


@dataclass(frozen=True)
class LtpQuote:
    symbol: str
    ltp: float
    open: float | None
    high: float | None
    low: float | None
    close: float | None        # previous close


def _request_one(
    session: AngelOneSession,
    instrument: Instrument,
) -> LtpQuote | None:
    payload = {
        "exchange": instrument.exchange,
        "tradingsymbol": instrument.angel_symbol,
        "symboltoken": instrument.token,
    }
    try:
        body = session.request("POST", _LTP_PATH, json_body=payload)
    except AngelOneAPIError as exc:
        log.warning("LTP fetch failed for {}: {}", instrument.symbol, exc)
        return None

    if not body or not body.get("status"):
        log.warning("LTP non-OK for {}: {}", instrument.symbol,
                    (body or {}).get("message"))
        return None

    data = body.get("data") or {}
    try:
        ltp = float(data.get("ltp") or 0.0)
    except (TypeError, ValueError):
        return None
    if ltp <= 0:
        return None

    def _opt_float(key: str) -> float | None:
        v = data.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return LtpQuote(
        symbol=instrument.symbol,
        ltp=ltp,
        open=_opt_float("open"),
        high=_opt_float("high"),
        low=_opt_float("low"),
        close=_opt_float("close"),
    )


def fetch_ltp(
    session: AngelOneSession,
    symbols: Iterable[str],
    *,
    instruments: list[Instrument] | None = None,
) -> list[LtpQuote]:
    """Fetch LTP for each requested symbol; missing symbols are skipped."""
    insts = instruments if instruments is not None else download_instrument_master()
    by_sym = {i.symbol: i for i in insts}
    out: list[LtpQuote] = []
    for sym in symbols:
        sym = sym.upper().strip()
        instrument = by_sym.get(sym)
        if instrument is None:
            log.debug("No instrument for {}", sym)
            continue
        q = _request_one(session, instrument)
        if q is not None:
            out.append(q)
    return out
