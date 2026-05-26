"""End-to-end Week-2 smoke pipeline.

Assumes Week-1 has been run (DB initialised, constituents/calendar seeded,
some price_data ingested). Then:

1. Migrate schema to v2.
2. Seed stock_sectors.
3. Ingest indices (^NSEI, ^INDIAVIX, sector indices).
4. Build features for the requested symbols.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from scripts import (
    build_features,
    init_db,
    ingest_bhavcopy,
    ingest_indices,
    seed_data,
    seed_sectors,
)
from src.utils.logger import get_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny slice: 3 symbols x ~1 year of bhavcopy + index data",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="RELIANCE,TCS,INFY",
        help="Comma-separated symbols (smoke only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.week2_pipeline")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    log.info("==> Step 1/5: ensure DB schema is at v2 + reference data")
    init_db.main()
    seed_data.main()
    seed_sectors.main()

    if args.smoke:
        end = date.today() - timedelta(days=2)
        start = end - timedelta(days=365)  # 1 year of context for features
        symbols = args.symbols
        log.info(
            "==> Step 2/5: ingest bhavcopy for {} from {} to {}",
            symbols,
            start,
            end,
        )
        ingest_bhavcopy.main(
            ["--start", start.isoformat(), "--end", end.isoformat(), "--symbols", symbols]
        )
        log.info("==> Step 3/5: ingest indices ^NSEI, ^INDIAVIX (smoke set)")
        ingest_indices.main(
            [
                "--start",
                start.isoformat(),
                "--end",
                end.isoformat(),
                "--symbols",
                "^NSEI,^INDIAVIX,^CNXIT,^CNXENERGY",
            ]
        )
        log.info("==> Step 4/5: build features for {}", symbols)
        build_features.main(
            ["--symbols", symbols, "--start", start.isoformat(), "--end", end.isoformat()]
        )
    else:
        log.info("==> Step 2/5: ingest indices (5y)")
        ingest_indices.main([])
        log.info("==> Step 3/5: build features for full universe")
        build_features.main([])

    log.info("==> Step 5/5: complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
