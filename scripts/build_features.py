"""Build the feature matrix for one or more symbols."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from tqdm import tqdm

from src.features.feature_builder import build_for_symbol
from src.utils.db import fetch_all
from src.utils.logger import get_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, help="Comma-separated; default = all in v_universe_today")
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--limit", type=int)
    return p.parse_args(argv)


def _resolve_symbols(arg: str | None) -> list[str]:
    if arg:
        return [s.strip().upper() for s in arg.split(",") if s.strip()]
    rows = fetch_all("SELECT symbol FROM v_universe_today ORDER BY symbol")
    return [r["symbol"] for r in rows]


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.build_features")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    symbols = _resolve_symbols(args.symbols)
    if args.limit:
        symbols = symbols[: args.limit]

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    log.info("Feature build: {} symbols", len(symbols))
    total = 0
    for sym in tqdm(symbols, desc="features"):
        s = build_for_symbol(sym, start=start, end=end)
        total += s.rows_out
    log.info("Done. Total feature rows written: {}", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
