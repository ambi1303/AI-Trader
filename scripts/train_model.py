"""Train a single XGBoost model on the full feature_data history.

Pipeline:
  1. Build training matrix from feature_data + price_data.
  2. Time-based 70/15/15 split (train/calib/test). NO shuffle.
  3. Fit DeterministicXGBClassifier on train.
  4. Fit isotonic calibrator on calib.
  5. Tune decision threshold on calib by expected utility (uses cost_model).
  6. Evaluate on test (raw + calibrated).
  7. Register model in data/models/<run_id>/ + model_runs table.

Usage:
    python -m scripts.train_model                 # all symbols
    python -m scripts.train_model --symbols TCS,INFY,RELIANCE
    python -m scripts.train_model --target 0.005  # next-day return threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.models.calibration import CalibratedXGB, fit_isotonic_calibrator
from src.models.dataset import boundary_dates, build_training_matrix, time_based_split
from src.models.metrics import evaluate_classifier
from src.models.registry import save_model
from src.models.threshold_tuning import tune_threshold
from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("script.train_model")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols. Default: all in feature_data.")
    p.add_argument("--target", type=float, default=0.005,
                   help="Next-day return threshold for positive label (default 0.005 = 0.5%%).")
    p.add_argument("--model-name", default="xgb_v1",
                   help="Logical model name (used in registry + filenames).")
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

    log.info("Building training matrix (target threshold = {:.3%})", args.target)
    matrix = build_training_matrix(symbols, target_return_threshold=args.target)
    if matrix.X.empty:
        log.error("Empty training matrix. Run scripts.build_features first.")
        return 1

    train_idx, val_idx, test_idx = time_based_split(matrix.meta)
    log.info(
        "Split sizes: train={} val={} test={}",
        len(train_idx), len(val_idx), len(test_idx),
    )
    if min(len(train_idx), len(val_idx), len(test_idx)) < 50:
        log.error("Too few rows in one of the splits; need more feature_data.")
        return 1

    X_tr, y_tr = matrix.X.iloc[train_idx], matrix.y.iloc[train_idx]
    X_val, y_val = matrix.X.iloc[val_idx], matrix.y.iloc[val_idx]
    X_te, y_te = matrix.X.iloc[test_idx], matrix.y.iloc[test_idx]
    fwd_val = matrix.forward_return.iloc[val_idx].to_numpy()

    log.info("Training XGBoost...")
    base = DeterministicXGBClassifier(params=XGBParams())
    base.fit(X_tr, y_tr)

    log.info("Calibrating with isotonic regression on val slice...")
    cal_obj = fit_isotonic_calibrator(base, X_val, y_val)
    cxgb = CalibratedXGB(base, cal_obj)

    log.info("Tuning decision threshold on calibration slice...")
    cost_model = _load_cost_model()
    val_cal_probs = cxgb.predict_calibrated(X_val)
    thr = tune_threshold(val_cal_probs, fwd_val, cost_model=cost_model)
    log.info(
        "Chosen threshold = {:.2f} | n_signals(val)={} | expected_per_trade={:.4%} | "
        "round-trip cost = {:.4%}",
        thr.threshold, thr.n_signals, thr.expected_return_per_trade, thr.cost_pct,
    )

    log.info("Evaluating on held-out test slice...")
    raw_te = base.predict_proba(X_te)[:, 1]
    cal_te = cxgb.predict_calibrated(X_te)
    rep_raw = evaluate_classifier(y_te.to_numpy(), raw_te)
    rep_cal = evaluate_classifier(y_te.to_numpy(), cal_te)
    log.info("Test (raw)        : {}", rep_raw.as_dict())
    log.info("Test (calibrated) : {}", rep_cal.as_dict())

    metrics = {
        "test_raw": rep_raw.as_dict(),
        "test_calibrated": rep_cal.as_dict(),
        "threshold": {
            "value": thr.threshold,
            "n_signals_val": thr.n_signals,
            "expected_return_per_trade": thr.expected_return_per_trade,
            "expected_total_utility": thr.expected_total_utility,
            "round_trip_cost_pct": thr.cost_pct,
        },
        "split_sizes": {"train": int(len(X_tr)), "val": int(len(X_val)),
                        "test": int(len(X_te))},
    }

    train_lo, train_hi = boundary_dates(matrix.meta, train_idx)
    meta = save_model(
        cxgb,
        model_name=args.model_name,
        feature_columns=matrix.feature_columns,
        trained_from=train_lo.isoformat() if train_lo else None,
        trained_to=train_hi.isoformat() if train_hi else None,
        metrics=metrics,
        threshold=thr.threshold,
        target_return_threshold=args.target,
    )
    log.success("Model saved | run_id={} | path={}", meta.run_id,
                Path("data/models") / meta.run_id)

    # Gate-3 self check: calibrated Brier < 0.22 on test.
    if rep_cal.brier > 0.22:
        log.warning(
            "Gate 3 NOT met yet: calibrated Brier on test = {:.4f} (target < 0.22). "
            "This is expected on small datasets; revisit after full universe ingest.",
            rep_cal.brier,
        )
    else:
        log.success("Gate 3 satisfied: calibrated Brier = {:.4f} (< 0.22)",
                    rep_cal.brier)

    return 0


if __name__ == "__main__":
    sys.exit(main())
