"""Ingest Nifty 50, India VIX, and NSE sectoral indices."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from src.data_ingestion.index_loader import ingest_indices
from src.utils.logger import get_logger

DEFAULT_INDICES = (
    "^NSEI",         # Nifty 50
    "^INDIAVIX",     # India VIX
    "^NSEBANK",      # Nifty Bank
    "^CNXIT",        # Nifty IT
    "^CNXFMCG",      # Nifty FMCG
    "^CNXAUTO",      # Nifty Auto
    "^CNXPHARMA",    # Nifty Pharma
    "^CNXMETAL",     # Nifty Metal
    "^CNXENERGY",    # Nifty Energy
    "^CNXFIN",       # Nifty Fin Services (may also be ^CNXFINANCE on Yahoo)
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--symbols", type=str, help="Comma-separated; default = all")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.ingest_indices")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=int(args.years * 365.25))

    syms = (
        tuple(s.strip() for s in args.symbols.split(",") if s.strip())
        if args.symbols
        else DEFAULT_INDICES
    )
    log.info("Index ingest: {} indices, {} -> {}", len(syms), start, end)
    n = ingest_indices(syms, start, end)
    log.info("Done. Total rows written: {}", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
