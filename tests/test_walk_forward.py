"""Tests for walk-forward fold generation and the end-to-end runner.

Critical properties:
  - Train end < calib start; calib end < test start; test windows are
    contiguous and non-overlapping.
  - No fold uses test data dated <= its calibration end (no leakage).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.models.dataset import TrainingMatrix
from src.models.walk_forward import (
    WalkForwardConfig,
    aggregate_walk_forward,
    generate_folds,
    run_walk_forward,
)


def _make_meta(start: date, n_days: int, n_symbols: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        for s in range(n_symbols):
            rows.append({"symbol": f"S{s}", "feature_date": d})
    return pd.DataFrame(rows)


def test_fold_windows_are_strictly_ordered():
    meta = _make_meta(date(2020, 1, 1), n_days=365 * 4)
    cfg = WalkForwardConfig(initial_train_days=365, calib_days=60,
                            test_days=60, step_days=60)
    folds = generate_folds(meta, cfg)
    assert len(folds) > 0
    for train_end, calib_end, test_start, test_end in folds:
        assert train_end < calib_end
        assert calib_end < test_start
        assert test_start <= test_end


def test_fold_test_windows_advance_monotonically():
    meta = _make_meta(date(2020, 1, 1), n_days=365 * 4)
    cfg = WalkForwardConfig(initial_train_days=365, calib_days=60,
                            test_days=60, step_days=60)
    folds = generate_folds(meta, cfg)
    last_test_end = date.min
    for _, _, test_start, test_end in folds:
        assert test_start > last_test_end or last_test_end == date.min
        last_test_end = test_end


def test_no_folds_when_data_too_short():
    meta = _make_meta(date(2024, 1, 1), n_days=30)
    cfg = WalkForwardConfig(initial_train_days=365, calib_days=90, test_days=90)
    assert generate_folds(meta, cfg) == []


def _synthetic_matrix(n_days: int = 800, n_symbols: int = 4, n_features: int = 8):
    rng = np.random.default_rng(0)
    rows = []
    feat_names = [f"f{i}" for i in range(n_features)]
    for s in range(n_symbols):
        sym = f"S{s}"
        # symbol-specific feature drift
        for i in range(n_days):
            d = date(2020, 1, 1) + timedelta(days=i)
            rows.append({
                "symbol": sym,
                "feature_date": d,
                **{c: rng.standard_normal() for c in feat_names},
                # forward return correlated with f0 + noise
            })
    df = pd.DataFrame(rows)
    df["fwd_return"] = (df["f0"] * 0.01) + 0.001 * np.random.default_rng(1).standard_normal(len(df))
    df["y"] = (df["fwd_return"] > 0.005).astype(int)

    meta = df[["symbol", "feature_date"]].reset_index(drop=True)
    X = df[feat_names].reset_index(drop=True)
    y = df["y"].reset_index(drop=True)
    fwd = df["fwd_return"].reset_index(drop=True)
    return TrainingMatrix(
        X=X, y=y, forward_return=fwd, meta=meta, feature_columns=feat_names
    )


def test_walk_forward_runs_end_to_end_and_aggregates():
    matrix = _synthetic_matrix(n_days=600, n_symbols=3, n_features=6)
    cfg = WalkForwardConfig(
        initial_train_days=300, calib_days=60, test_days=60, step_days=120,
        min_train_rows=100,
    )
    results = run_walk_forward(matrix, cfg=cfg)
    completed = [r for r in results if r.test_report_calibrated is not None]
    assert len(completed) >= 1
    agg = aggregate_walk_forward(results)
    assert agg["folds_completed"] == len(completed)
    assert "raw" in agg and "calibrated" in agg
    assert agg["raw"]["brier"] >= 0
    assert agg["calibrated"]["brier"] >= 0


def test_walk_forward_no_leakage_in_test_predictions():
    """Every test prediction must be produced by a model whose training data
    ended before the prediction date."""
    matrix = _synthetic_matrix(n_days=600, n_symbols=3, n_features=6)
    cfg = WalkForwardConfig(
        initial_train_days=300, calib_days=60, test_days=60, step_days=120,
        min_train_rows=100,
    )
    results = run_walk_forward(matrix, cfg=cfg)
    for r in results:
        if r.test_predictions.empty:
            continue
        test_dates = pd.to_datetime(r.test_predictions["feature_date"]).dt.date
        # Every test date must be > calib_end (which is > train_end by construction)
        assert (test_dates > r.calib_window[1]).all()
