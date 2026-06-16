"""Ingest fundamental ratios (valuation / quality / growth) via yfinance.

Read-only: pulls ``Ticker.info`` for a live snapshot and quarterly
statements to reconstruct point-in-time history, writing both into
``fundamental_data``.

Examples
--------
    # Whole universe (snapshot + reconstructed history)
    python -m scripts.ingest_fundamentals

    # Specific symbols
    python -m scripts.ingest_fundamentals --symbols RELIANCE,TCS

    # Smoke test (3 symbols, no DB writes)
    python -m scripts.ingest_fundamentals --limit 3 --no-persist

    # Snapshot only (skip the slower quarterly reconstruction)
    python -m scripts.ingest_fundamentals --no-reconstruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_ingestion.fundamentals import ingest_fundamentals
from src.utils.db import fetch_all
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("scripts.ingest_fundamentals")


def _read_universe() -> list[str]:
    """Default universe = symbols that already have feature/price data.

    We prefer the symbols present in feature_data (the model universe) so
    fundamentals line up with what we actually train/predict on. Fall back
    to the static universe file if feature_data is empty.
    """
    rows = fetch_all("SELECT DISTINCT symbol FROM feature_data ORDER BY symbol")
    if rows:
        return [r["symbol"] for r in rows]
    rows = fetch_all(
        "SELECT DISTINCT symbol FROM price_data WHERE source='bhavcopy' ORDER BY symbol"
    )
    if rows:
        return [r["symbol"] for r in rows]
    path: Path = project_root() / "config" / "stocks_universe.txt"
    syms: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                syms.append(line.upper())
    return syms


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols; default = feature_data universe.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N symbols (smoke testing).")
    p.add_argument("--no-persist", action="store_true",
                   help="Fetch but do not write fundamental_data (read-only smoke).")
    p.add_argument("--no-reconstruct", action="store_true",
                   help="Skip quarterly history reconstruction (snapshot only).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _read_universe()
    if args.limit:
        symbols = symbols[: args.limit]

    if not symbols:
        log.error("No symbols to ingest. Build features or pass --symbols.")
        return 2

    log.info(
        "Fundamentals ingest start | symbols={} persist={} reconstruct={}",
        len(symbols), not args.no_persist, not args.no_reconstruct,
    )

    summary = ingest_fundamentals(
        symbols,
        persist=not args.no_persist,
        reconstruct=not args.no_reconstruct,
    )

    log.info(
        "Fundamentals ingest done | requested={} snapshots={} "
        "reconstructed={} upserted={} failed={}",
        summary.requested, summary.snapshots, summary.reconstructed,
        summary.upserted, len(summary.failed_symbols),
    )
    if summary.failed_symbols:
        log.warning("Failed symbols (first 10): {}", summary.failed_symbols[:10])

    return 0 if (summary.upserted > 0 or args.no_persist) else 1


if __name__ == "__main__":
    sys.exit(main())
