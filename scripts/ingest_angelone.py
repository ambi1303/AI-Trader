"""Ingest daily OHLCV bars from Angel One SmartAPI into ``price_data``.

Read-only: this script never places an order. It only calls
``getCandleData`` (and optionally ``getLtpData``) and writes results
under ``source='angelone'`` so the validator can cross-check it against
yfinance and BhavCopy.

Examples
--------
    # Backfill the last 60 days for the whole NIFTY 50 universe
    python -m scripts.ingest_angelone --days 60

    # Refresh today's bars only
    python -m scripts.ingest_angelone --days 1

    # Specific symbols
    python -m scripts.ingest_angelone --symbols RELIANCE,TCS --days 30

    # Smoke test (3 symbols, 5 days, no DB writes)
    python -m scripts.ingest_angelone --limit 3 --days 5 --no-persist
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from src.data_ingestion.angelone import (
    download_instrument_master,
    fetch_daily_candles,
    load_session_from_env,
)
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("scripts.ingest_angelone")


def _read_universe() -> list[str]:
    path: Path = project_root() / "config" / "stocks_universe.txt"
    syms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        syms.append(line.upper())
    return syms


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=5,
                   help="Backfill window in calendar days, ending today.")
    p.add_argument("--start", type=str, default=None,
                   help="ISO YYYY-MM-DD start date (overrides --days).")
    p.add_argument("--end", type=str, default=None,
                   help="ISO YYYY-MM-DD end date (default: today).")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols; default = stocks_universe.txt.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N symbols (smoke testing).")
    p.add_argument("--no-persist", action="store_true",
                   help="Fetch but do not write price_data (read-only smoke).")
    p.add_argument("--refresh-instruments", action="store_true",
                   help="Force re-download of the instrument master cache.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    session = load_session_from_env()
    if session is None:
        log.error(
            "Angel One credentials not configured. Add ANGEL_API_KEY / "
            "ANGEL_CLIENT_CODE / ANGEL_MPIN / ANGEL_TOTP_SECRET to .env "
            "(see .env.example) and rerun."
        )
        return 2

    end = date.fromisoformat(args.end) if args.end else date.today()
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        start = end - timedelta(days=int(args.days))

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _read_universe()
    if args.limit:
        symbols = symbols[: args.limit]

    log.info(
        "Angel One ingest start | symbols={} window={}..{} persist={}",
        len(symbols), start, end, not args.no_persist,
    )

    instruments = download_instrument_master(force=args.refresh_instruments)
    summary = fetch_daily_candles(
        session, symbols,
        start=start, end=end,
        persist=not args.no_persist,
        instruments=instruments,
    )

    log.info(
        "Angel One ingest done | requested={} fetched={} upserted={} failed={}",
        summary.requested, summary.fetched_bars,
        summary.upserted_bars, len(summary.failed_symbols),
    )
    if summary.failed_symbols:
        log.warning("Failed symbols (first 10): {}",
                    summary.failed_symbols[:10])
    session.logout()                       # best-effort, never raises
    return 0 if summary.upserted_bars > 0 or args.no_persist else 1


if __name__ == "__main__":
    sys.exit(main())
