"""Bulk ingest yfinance daily history for a list of symbols.

Usage:
  python scripts/ingest_yfinance.py --years 5
  python scripts/ingest_yfinance.py --start 2020-01-01 --end 2025-01-01 --symbols RELIANCE,TCS
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

from src.data_ingestion.bar_writer import upsert_bars
from src.data_ingestion.yfinance_loader import fetch_history
from src.utils.logger import get_logger
from src.utils.secrets import project_root


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
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--symbols", type=str, help="Comma-separated; default = stocks_universe.txt")
    p.add_argument("--limit", type=int, help="Limit to first N symbols (for smoke tests)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.ingest_yfinance")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    else:
        end = date.today()
        start = end - timedelta(days=int(args.years * 365.25))

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _read_universe()
    if args.limit:
        symbols = symbols[: args.limit]

    log.info(
        "yfinance ingest: {} symbols, {} -> {}",
        len(symbols),
        start,
        end,
    )

    total = 0
    for sym in tqdm(symbols, desc="yfinance"):
        bars = fetch_history(sym, start, end)
        wrote = upsert_bars(bars)
        total += wrote
        log.debug("{}: {} bars written", sym, wrote)
    log.info("Done. Total bars written: {}", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
