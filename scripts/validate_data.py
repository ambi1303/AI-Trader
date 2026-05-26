"""Run cross-source validator and persist any issues."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date

from src.data_validation.cross_source_check import compare_sources
from src.data_validation.issue_writer import write_issues
from src.utils.logger import get_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=0.5,
        help="Fail (exit 2) if mismatched rows > this percent of compared rows",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.validate_data")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    run_id = f"validate-{uuid.uuid4().hex[:12]}"
    rep = compare_sources(run_id, symbols=symbols, start=start, end=end)
    write_issues(rep.issues)

    miss_pct = (
        100.0 * rep.rows_mismatched / rep.rows_compared if rep.rows_compared else 0.0
    )
    log.info(
        "run_id={} compared={} mismatched={} ({:.3f}%) match_rate={:.4f}",
        run_id,
        rep.rows_compared,
        rep.rows_mismatched,
        miss_pct,
        rep.match_rate,
    )

    if miss_pct > args.fail_threshold_pct:
        log.error(
            "Validation gate FAILED: mismatched={:.3f}% > threshold={}%",
            miss_pct,
            args.fail_threshold_pct,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
