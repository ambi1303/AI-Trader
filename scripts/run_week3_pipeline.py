"""End-to-end Week-3 smoke pipeline.

Steps:
  1. Apply DB migrations (idempotent).
  2. Train a model on whatever is in feature_data.
  3. Run walk-forward eval on the same data.
  4. Predict for today using the freshly-trained model.

This is intentionally conservative on CPU: walk-forward uses small windows
sized for the smoke dataset. For real evaluation use scripts.walk_forward_eval
with default windows after the full universe is ingested.

Usage:
    python -m scripts.run_week3_pipeline
    python -m scripts.run_week3_pipeline --symbols TCS,INFY,RELIANCE
"""

from __future__ import annotations

import argparse
import sys

from scripts import predict_today, train_model, walk_forward_eval
from src.db.migrate import apply_schema
from src.utils.logger import get_logger

log = get_logger("script.week3_pipeline")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None)
    p.add_argument("--model-name", default="xgb_v1")
    p.add_argument("--target", type=float, default=0.005)
    p.add_argument("--skip-walk-forward", action="store_true",
                   help="Skip the walk-forward eval (use on tiny smoke data).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    log.info("=== Step 1/4: migrations ===")
    apply_schema()

    log.info("=== Step 2/4: training model '{}' ===", args.model_name)
    train_argv = ["--target", str(args.target), "--model-name", args.model_name]
    if args.symbols:
        train_argv += ["--symbols", args.symbols]
    rc = train_model.main(train_argv)
    if rc != 0:
        log.error("Training failed with rc={}", rc)
        return rc

    if not args.skip_walk_forward:
        log.info("=== Step 3/4: walk-forward eval (small windows for smoke) ===")
        wf_argv = [
            "--target", str(args.target),
            "--initial-train-days", "120",
            "--calib-days", "30",
            "--test-days", "30",
            "--step-days", "30",
            "--min-train-rows", "60",
        ]
        if args.symbols:
            wf_argv += ["--symbols", args.symbols]
        rc = walk_forward_eval.main(wf_argv)
        if rc != 0:
            log.warning("Walk-forward step returned rc={} (continuing)", rc)
    else:
        log.info("=== Step 3/4: skipped per --skip-walk-forward ===")

    log.info("=== Step 4/4: predict for latest feature dates ===")
    pred_argv = ["--model-name", args.model_name]
    if args.symbols:
        pred_argv += ["--symbols", args.symbols]
    rc = predict_today.main(pred_argv)
    if rc != 0:
        log.error("Prediction failed with rc={}", rc)
        return rc

    log.success("Week-3 pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
