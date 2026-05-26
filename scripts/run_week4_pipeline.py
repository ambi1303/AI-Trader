"""End-to-end Week-4 smoke pipeline.

Steps:
  1. Apply DB migrations (idempotent; v3 adds backtest_runs/equity/trades).
  2. Run a fresh walk-forward (or load saved predictions) to get OOS probs.
  3. Run the full-period backtest on whatever symbols are in feature_data.
  4. Run the stress-window backtests (best-effort: skips empty windows).

Usage:
    python -m scripts.run_week4_pipeline
    python -m scripts.run_week4_pipeline --symbols TCS,INFY,RELIANCE
    python -m scripts.run_week4_pipeline --skip-stress
"""

from __future__ import annotations

import argparse
import sys

from scripts import run_backtest, run_stress_tests
from src.db.migrate import apply_schema
from src.utils.logger import get_logger

log = get_logger("script.week4_pipeline")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None)
    p.add_argument("--model-name", default="xgb_v1")
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold (else taken from registered model)")
    p.add_argument("--skip-stress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    log.info("=== Step 1/3: migrations ===")
    apply_schema()

    log.info("=== Step 2/3: full-period backtest (regenerating predictions) ===")
    bt_argv = [
        "--regen",
        "--model-name", args.model_name,
        "--initial-capital", str(args.initial_capital),
        "--name", "full_smoke",
    ]
    if args.symbols:
        bt_argv += ["--symbols", args.symbols]
    if args.threshold is not None:
        bt_argv += ["--threshold", str(args.threshold)]
    rc = run_backtest.main(bt_argv)
    if rc != 0:
        log.error("Full backtest failed")
        return rc

    if args.skip_stress:
        log.info("=== Step 3/3: stress tests skipped ===")
    else:
        log.info("=== Step 3/3: stress windows ===")
        st_argv = [
            "--regen",
            "--model-name", args.model_name,
            "--initial-capital", str(args.initial_capital),
            "--persist",
        ]
        if args.symbols:
            st_argv += ["--symbols", args.symbols]
        if args.threshold is not None:
            st_argv += ["--threshold", str(args.threshold)]
        rc = run_stress_tests.main(st_argv)
        if rc != 0:
            log.warning("Stress step returned rc={}", rc)

    log.success("Week-4 pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
