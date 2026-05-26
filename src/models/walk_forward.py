"""Expanding-window walk-forward evaluation.

For each fold:
  TRAIN window = [data_start, train_end]
  CALIB window = (train_end, calib_end]      (used for isotonic + threshold)
  TEST  window = (calib_end, test_end]       (out-of-sample evaluation)

Successive folds advance `train_end` by `step_days` so test windows are
non-overlapping. We aggregate test-set predictions across folds to get a
true out-of-sample equity / probability curve.

Why expanding (not rolling): we want the model to learn from the longest
possible history at every retraining point. Rolling can be added later if we
want to enforce concept-drift adaptation.

Why retrain-and-recalibrate every fold: regimes change. The threshold tuned
on Q4-2021 calibration data is not the threshold we should use to trade
Q1-2022. We retrain and re-tune at every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.models.calibration import CalibratedXGB, fit_isotonic_calibrator
from src.models.dataset import TrainingMatrix
from src.models.metrics import ClassificationReport, evaluate_classifier
from src.models.threshold_tuning import ThresholdResult, tune_threshold
from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams
from src.utils.logger import get_logger

log = get_logger("models.wf")


@dataclass
class WalkForwardConfig:
    initial_train_days: int = 365 * 2  # 2 years initial train
    calib_days: int = 90              # 1 quarter calibration
    test_days: int = 90               # 1 quarter test
    step_days: int = 90               # advance by 1 quarter each fold
    min_train_rows: int = 200         # below this we skip the fold


@dataclass
class FoldResult:
    fold_index: int
    train_window: tuple[date, date]
    calib_window: tuple[date, date]
    test_window: tuple[date, date]
    n_train: int
    n_calib: int
    n_test: int
    test_report_raw: ClassificationReport | None
    test_report_calibrated: ClassificationReport | None
    threshold: ThresholdResult | None
    test_signals: int = 0
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def _date_mask(meta: pd.DataFrame, start: date, end: date) -> np.ndarray:
    d = pd.to_datetime(meta["feature_date"]).dt.date.to_numpy()
    return (d >= start) & (d <= end)


def generate_folds(
    meta: pd.DataFrame, cfg: WalkForwardConfig
) -> list[tuple[date, date, date, date]]:
    """Return [(train_end, calib_end, test_start, test_end), ...]."""
    if meta.empty:
        return []
    d = pd.to_datetime(meta["feature_date"]).dt.date
    start = d.min()
    end = d.max()

    folds = []
    train_end = start + timedelta(days=cfg.initial_train_days)
    while True:
        calib_end = train_end + timedelta(days=cfg.calib_days)
        test_end = calib_end + timedelta(days=cfg.test_days)
        if test_end > end:
            break
        folds.append((train_end, calib_end, calib_end + timedelta(days=1), test_end))
        train_end = train_end + timedelta(days=cfg.step_days)
    return folds


def run_walk_forward(
    matrix: TrainingMatrix,
    cfg: WalkForwardConfig | None = None,
    params: XGBParams | None = None,
    *,
    cost_model: dict | None = None,
) -> list[FoldResult]:
    cfg = cfg or WalkForwardConfig()
    params = params or XGBParams()

    if matrix.X.empty:
        log.warning("Empty training matrix -- skipping walk-forward")
        return []

    folds = generate_folds(matrix.meta, cfg)
    if not folds:
        log.warning(
            "Not enough date range for walk-forward "
            "(initial_train_days={} + calib={} + test={} > available)",
            cfg.initial_train_days,
            cfg.calib_days,
            cfg.test_days,
        )
        return []

    log.info("Walk-forward: {} folds will be evaluated", len(folds))

    meta_dates = pd.to_datetime(matrix.meta["feature_date"]).dt.date
    data_start = meta_dates.min()

    results: list[FoldResult] = []
    for i, (train_end, calib_end, test_start, test_end) in enumerate(folds):
        train_mask = _date_mask(matrix.meta, data_start, train_end)
        calib_mask = _date_mask(
            matrix.meta, train_end + timedelta(days=1), calib_end
        )
        test_mask = _date_mask(matrix.meta, test_start, test_end)

        n_tr = int(train_mask.sum())
        n_cal = int(calib_mask.sum())
        n_te = int(test_mask.sum())

        if n_tr < cfg.min_train_rows or n_cal < 30 or n_te < 10:
            log.warning(
                "Skipping fold {}: insufficient rows (train={}, calib={}, test={})",
                i, n_tr, n_cal, n_te,
            )
            results.append(
                FoldResult(
                    fold_index=i,
                    train_window=(data_start, train_end),
                    calib_window=(train_end + timedelta(days=1), calib_end),
                    test_window=(test_start, test_end),
                    n_train=n_tr, n_calib=n_cal, n_test=n_te,
                    test_report_raw=None,
                    test_report_calibrated=None,
                    threshold=None,
                )
            )
            continue

        X_tr = matrix.X.loc[train_mask].reset_index(drop=True)
        y_tr = matrix.y.loc[train_mask].reset_index(drop=True)
        X_cal = matrix.X.loc[calib_mask].reset_index(drop=True)
        y_cal = matrix.y.loc[calib_mask].reset_index(drop=True)
        X_te = matrix.X.loc[test_mask].reset_index(drop=True)
        y_te = matrix.y.loc[test_mask].reset_index(drop=True)
        fwd_cal = matrix.forward_return.loc[calib_mask].to_numpy()
        meta_te = matrix.meta.loc[test_mask].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_cal)) < 2:
            log.warning(
                "Skipping fold {}: single-class slice in train or calib", i
            )
            continue

        base = DeterministicXGBClassifier(params=params)
        base.fit(X_tr, y_tr)

        try:
            cal = fit_isotonic_calibrator(base, X_cal, y_cal)
        except ValueError as e:
            log.warning("Skipping fold {}: calibrator fit failed ({})", i, e)
            continue
        cxgb = CalibratedXGB(base, cal)

        # Threshold tuned on the same calibration slice (using calibrated probs)
        cal_probs = cxgb.predict_calibrated(X_cal)
        thr = tune_threshold(
            cal_probs,
            fwd_cal,
            cost_model=cost_model,
        )

        # Evaluate on the held-out test set.
        raw_te = base.predict_proba(X_te)[:, 1]
        cal_te = cxgb.predict_calibrated(X_te)

        results.append(
            FoldResult(
                fold_index=i,
                train_window=(data_start, train_end),
                calib_window=(train_end + timedelta(days=1), calib_end),
                test_window=(test_start, test_end),
                n_train=n_tr, n_calib=n_cal, n_test=n_te,
                test_report_raw=evaluate_classifier(y_te.to_numpy(), raw_te),
                test_report_calibrated=evaluate_classifier(y_te.to_numpy(), cal_te),
                threshold=thr,
                test_signals=int((cal_te >= thr.threshold).sum()),
                test_predictions=pd.DataFrame(
                    {
                        "symbol": meta_te["symbol"].to_numpy(),
                        "feature_date": meta_te["feature_date"].to_numpy(),
                        "raw_prob": raw_te,
                        "calibrated_prob": cal_te,
                        "y_true": y_te.to_numpy(),
                        "fwd_return": matrix.forward_return.loc[test_mask].to_numpy(),
                    }
                ),
            )
        )
        log.info(
            "Fold {} OK | train={} calib={} test={} | "
            "Brier raw={:.4f} cal={:.4f} | thr={:.2f} signals={}",
            i, n_tr, n_cal, n_te,
            results[-1].test_report_raw.brier,
            results[-1].test_report_calibrated.brier,
            thr.threshold,
            results[-1].test_signals,
        )

    return results


def aggregate_walk_forward(results: list[FoldResult]) -> dict:
    """Concatenate per-fold test predictions and compute aggregate metrics."""
    completed = [r for r in results if r.test_report_calibrated is not None]
    if not completed:
        return {"folds_completed": 0}

    all_preds = pd.concat(
        [r.test_predictions for r in completed], ignore_index=True
    )
    raw_rep = evaluate_classifier(
        all_preds["y_true"].to_numpy(), all_preds["raw_prob"].to_numpy()
    )
    cal_rep = evaluate_classifier(
        all_preds["y_true"].to_numpy(), all_preds["calibrated_prob"].to_numpy()
    )

    return {
        "folds_completed": len(completed),
        "folds_skipped": len(results) - len(completed),
        "n_test_predictions": int(len(all_preds)),
        "raw": raw_rep.as_dict(),
        "calibrated": cal_rep.as_dict(),
        "mean_threshold": float(
            np.mean([r.threshold.threshold for r in completed])
        ),
        "test_predictions": all_preds,
    }
