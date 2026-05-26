"""End-to-end Week-1 smoke pipeline.

1. init DB (apply schema)
2. seed constituency, corp actions, calendar
3. ingest a small slice from yfinance and bhavcopy
4. cross-source validate
5. split-audit known events that fall in the slice

Designed to run in < 2 minutes for a small `--smoke` slice. Use for daily
sanity checks before scaling up.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from scripts import (
    audit_splits,
    ingest_bhavcopy,
    ingest_yfinance,
    init_db,
    seed_data,
    validate_data,
)
from src.utils.logger import get_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny slice: 3 symbols x ~10 trading days",
    )
    p.add_argument("--years", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.week1_pipeline")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    log.info("==> Step 1/5: init DB")
    rc = init_db.main()
    if rc:
        return rc

    log.info("==> Step 2/5: seed reference data")
    rc = seed_data.main()
    if rc:
        return rc

    if args.smoke:
        end = date.today() - timedelta(days=2)
        start = end - timedelta(days=14)
        symbols = "RELIANCE,TCS,INFY"
        log.info(
            "==> Step 3/5: ingest yfinance + bhavcopy ({} -> {}, {})",
            start,
            end,
            symbols,
        )
        ingest_yfinance.main(
            ["--start", start.isoformat(), "--end", end.isoformat(), "--symbols", symbols]
        )
        ingest_bhavcopy.main(
            ["--start", start.isoformat(), "--end", end.isoformat(), "--symbols", symbols]
        )
    else:
        end = date.today() - timedelta(days=2)
        start = end - timedelta(days=int(args.years * 365.25))
        log.info("==> Step 3/5: ingest yfinance + bhavcopy {} -> {}", start, end)
        ingest_yfinance.main(["--start", start.isoformat(), "--end", end.isoformat()])
        ingest_bhavcopy.main(["--start", start.isoformat(), "--end", end.isoformat()])

    log.info("==> Step 4/5: cross-source validate")
    validate_data.main([])

    log.info("==> Step 5/5: split audit")
    audit_splits.main([])

    log.info("Week-1 pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
