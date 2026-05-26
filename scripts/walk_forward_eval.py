"""Walk-forward evaluation across the full feature_data history.

Reports per-fold and aggregate metrics, optionally writes the
concatenated test predictions to data/reports/wf_<timestamp>.parquet
so they can be feed into the Week-4 backtester.

Usage:
    python -m scripts.walk_forward_eval
    python -m scripts.walk_forward_eval --symbols TCS,INFY,RELIANCE
    python -m scripts.walk_forward_eval --initial-train-days 365 --test-days 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.models.dataset import build_training_matrix
from src.models.walk_forward import (
    WalkForwardConfig,
    aggregate_walk_forward,
    run_walk_forward,
)
from src.models.xgboost_classifier import XGBParams
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("script.wf_eval")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None)
    p.add_argument("--target", type=float, default=0.005)
    p.add_argument("--initial-train-days", type=int, default=365 * 2)
    p.add_argument("--calib-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--step-days", type=int, default=90)
    p.add_argument("--min-train-rows", type=int, default=200)
    p.add_argument("--save", action="store_true",
                   help="Persist test predictions and per-fold report to disk")
    return p.parse_args(argv)


def _load_cost_model() -> dict:
    cm_path = project_root() / "config" / "cost_model.yaml"
    if not cm_path.exists():
        return {}
    with cm_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else None
    )

    matrix = build_training_matrix(symbols, target_return_threshold=args.target)
    if matrix.X.empty:
        log.error("Empty training matrix.")
        return 1

    cfg = WalkForwardConfig(
        initial_train_days=args.initial_train_days,
        calib_days=args.calib_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_train_rows=args.min_train_rows,
    )

    cost_model = _load_cost_model()
    results = run_walk_forward(matrix, cfg=cfg, params=XGBParams(),
                               cost_model=cost_model)
    agg = aggregate_walk_forward(results)
    log.info("Walk-forward aggregate: {}", json.dumps({
        "folds_completed": agg.get("folds_completed", 0),
        "n_predictions": agg.get("n_test_predictions", 0),
        "raw": agg.get("raw"),
        "calibrated": agg.get("calibrated"),
        "mean_threshold": agg.get("mean_threshold"),
    }, indent=2, default=str))

    if args.save and agg.get("folds_completed", 0) > 0:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = project_root() / "data" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_path = out_dir / f"wf_predictions_{ts}.parquet"
        meta_path = out_dir / f"wf_meta_{ts}.json"
        agg["test_predictions"].to_parquet(preds_path, index=False)
        meta_safe = {k: v for k, v in agg.items() if k != "test_predictions"}
        meta_path.write_text(json.dumps(meta_safe, indent=2, default=str),
                             encoding="utf-8")
        log.success("Saved {} predictions -> {} | meta -> {}",
                    len(agg["test_predictions"]), preds_path, meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
