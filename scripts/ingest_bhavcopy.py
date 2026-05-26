"""Bulk ingest BhavCopy for a date range. Skips weekends & holidays.

Usage:
  python scripts/ingest_bhavcopy.py --start 2024-12-01 --end 2024-12-31
  python scripts/ingest_bhavcopy.py --years 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from tqdm import tqdm

from src.data_ingestion.bar_writer import upsert_bars
from src.data_ingestion.bhavcopy_loader import fetch_bhavcopy
from src.data_validation.calendar_check import trading_days_between
from src.utils.logger import get_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated; if set, only these symbols are persisted",
    )
    p.add_argument("--limit-days", type=int, help="Limit number of days (smoke test)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.ingest_bhavcopy")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=int(args.years * 365.25))

    days = trading_days_between(start, end)
    if args.limit_days:
        days = days[: args.limit_days]

    symbol_filter: set[str] | None = None
    if args.symbols:
        symbol_filter = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    log.info("BhavCopy ingest: {} trading days {} -> {}", len(days), start, end)

    total = 0
    for d in tqdm(days, desc="bhavcopy"):
        bars = fetch_bhavcopy(d)
        if symbol_filter:
            bars = [b for b in bars if b.symbol in symbol_filter]
        wrote = upsert_bars(bars)
        total += wrote
        log.debug("{}: {} bars written", d, wrote)
    log.info("Done. Total bars written: {}", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
