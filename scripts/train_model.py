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

import numpy as np
import pandas as pd
import yaml

from src.models.calibration import (
    CalibratedMulticlass,
    CalibratedXGB,
    fit_isotonic_calibrator,
    fit_isotonic_multiclass,
)
from src.models.dataset import (
    TRICLASS_LABELS,
    boundary_dates,
    build_training_matrix,
    time_based_split,
)
from src.models.metrics import evaluate_classifier
from src.models.registry import save_model
from src.models.return_regressor import DeterministicXGBRegressor
from src.models.threshold_tuning import tune_threshold
from src.models.xgboost_classifier import (
    DeterministicXGBClassifier,
    DeterministicXGBMulticlass,
    XGBParams,
)
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("script.train_model")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols. Default: all in feature_data.")
    p.add_argument("--target", type=float, default=0.005,
                   help="Return threshold for positive label when "
                        "--label-mode=absolute (default 0.005 = 0.5%%).")
    p.add_argument("--horizon", type=int, default=20,
                   help="Forward-return horizon in trading days. Default 20 "
                        "(~1 month): the validated edge lives at the monthly "
                        "horizon, not next-day. Use 1 for the legacy "
                        "next-day model.")
    p.add_argument("--label-mode", choices=["absolute", "cross_sectional"],
                   default="absolute",
                   help="absolute = up > target over horizon; cross_sectional "
                        "= beats peers that day (top 1-quantile).")
    p.add_argument("--label-quantile", type=float, default=0.50,
                   help="Cross-sectional cutoff when --label-mode="
                        "cross_sectional (0.70 = top 30%%).")
    p.add_argument("--cross-sectional-features",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Add per-day universe rank features (xs_*). On by "
                        "default; --no-cross-sectional-features to disable.")
    p.add_argument("--model-name", default="xgb_v1",
                   help="Logical model name (used in registry + filenames).")
    p.add_argument("--triclass-mode", choices=["cross_sectional", "absolute"],
                   default="cross_sectional",
                   help="How the SELL/HOLD/BUY verdict buckets are defined. "
                        "cross_sectional = per-day terciles of forward return; "
                        "absolute = fixed +/- return bands.")
    p.add_argument("--triclass-buy", type=float, default=0.03,
                   help="Forward return above which a row is BUY when "
                        "--triclass-mode=absolute (default +3%%).")
    p.add_argument("--triclass-sell", type=float, default=-0.03,
                   help="Forward return below which a row is SELL when "
                        "--triclass-mode=absolute (default -3%%).")
    return p.parse_args(argv)


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank correlation between predicted and realised returns (the IC).

    Implemented without scipy: Pearson correlation of the rank transforms.
    """
    if len(y_true) < 3:
        return float("nan")
    rt = pd.Series(y_true).rank().to_numpy()
    rp = pd.Series(y_pred).rank().to_numpy()
    if rt.std() == 0 or rp.std() == 0:
        return float("nan")
    return float(np.corrcoef(rt, rp)[0, 1])


def _evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "ic_spearman": _spearman_ic(y_true, y_pred),
    }


def _evaluate_triclass(y_true: np.ndarray, proba: np.ndarray, labels) -> dict:
    pred = proba.argmax(axis=1)
    acc = float((pred == y_true).mean()) if len(y_true) else float("nan")
    per_class = {}
    for i, lab in enumerate(labels):
        mask = y_true == i
        support = int(mask.sum())
        recall = float((pred[mask] == i).mean()) if support else float("nan")
        sel = pred == i
        precision = float((y_true[sel] == i).mean()) if sel.sum() else float("nan")
        per_class[lab] = {
            "support": support, "precision": precision, "recall": recall,
        }
    return {"accuracy": acc, "per_class": per_class}


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

    log.info(
        "Building training matrix | label_mode={} horizon={} xsec={} "
        "(target={:.3%}, quantile={})",
        args.label_mode, args.horizon, args.cross_sectional_features,
        args.target, args.label_quantile,
    )
    matrix = build_training_matrix(
        symbols,
        target_return_threshold=args.target,
        label_mode=args.label_mode,
        label_quantile=args.label_quantile,
        horizon=args.horizon,
        cross_sectional_features=args.cross_sectional_features,
        triclass_mode=args.triclass_mode,
        triclass_buy=args.triclass_buy,
        triclass_sell=args.triclass_sell,
    )
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
    # adaptive=True: search the *observed* calibrated-score distribution rather
    # than a fixed 0.50 floor. Isotonic calibration compresses probabilities
    # toward the base rate (~0.35 here, max ~0.45), so a 0.50 floor would only
    # fire on rare historical outliers and the live universe would produce
    # ~zero signals/day. The expected-utility + min_signals guards still keep
    # the chosen operating point honest.
    thr = tune_threshold(val_cal_probs, fwd_val, cost_model=cost_model,
                         adaptive=True)
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

    # ----------------------------------------------------------------------
    # Auxiliary heads: return regressor (price target) + tri-class verdict.
    # Trained on the SAME splits so they're consistent with the binary model.
    # ----------------------------------------------------------------------
    yret_tr = matrix.y_return.iloc[train_idx]
    yret_te = matrix.y_return.iloc[test_idx].to_numpy()
    ytri_tr = matrix.y_triclass.iloc[train_idx]
    ytri_val = matrix.y_triclass.iloc[val_idx]
    ytri_te = matrix.y_triclass.iloc[test_idx].to_numpy()

    log.info("Training return regressor (price-target head)...")
    regressor = DeterministicXGBRegressor()
    regressor.fit(X_tr, yret_tr)
    reg_pred_te = regressor.predict(X_te)
    reg_metrics = _evaluate_regressor(yret_te, reg_pred_te)
    log.info("Regressor test: {}", reg_metrics)

    log.info("Training tri-class verdict head (SELL/HOLD/BUY)...")
    tri_base = DeterministicXGBMulticlass(params=XGBParams(), num_class=3)
    tri_base.fit(X_tr, ytri_tr)
    triclass = None
    tri_metrics: dict = {}
    try:
        tri_cal = fit_isotonic_multiclass(tri_base, X_val, ytri_val)
        triclass = CalibratedMulticlass(tri_base, tri_cal, TRICLASS_LABELS)
        tri_proba_te = triclass.predict_proba(X_te)
        tri_metrics = _evaluate_triclass(ytri_te, tri_proba_te, TRICLASS_LABELS)
        log.info("Tri-class test: {}", tri_metrics)
    except ValueError as exc:
        log.warning(
            "Tri-class calibration skipped ({}); verdict head not saved.", exc
        )

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
        "regressor": reg_metrics,
        "triclass": tri_metrics,
        "config": {
            "label_mode": args.label_mode,
            "label_quantile": args.label_quantile,
            "horizon": args.horizon,
            "cross_sectional_features": args.cross_sectional_features,
            "triclass_mode": args.triclass_mode,
            "n_features": int(matrix.X.shape[1]),
            "n_symbols": int(matrix.meta["symbol"].nunique()),
        },
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
        horizon=args.horizon,
        cross_sectional_features=args.cross_sectional_features,
        regressor=regressor,
        triclass=triclass,
        triclass_labels=list(TRICLASS_LABELS),
        triclass_mode=args.triclass_mode,
        class_thresholds={"buy": args.triclass_buy, "sell": args.triclass_sell},
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
