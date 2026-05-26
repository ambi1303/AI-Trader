"""Tests for isotonic calibration.

Critical property: calibration must NOT increase Brier score on the
held-out slice in a typical (mildly miscalibrated) setting. We construct
a deliberately-overconfident base model by training XGBoost without
sufficient regularisation on a moderately-noisy dataset, then verify
that isotonic on top reduces Brier on the calibration set itself
(in-sample on the calib set is the best-case lower bound for isotonic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from src.models.calibration import CalibratedXGB, fit_isotonic_calibrator
from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams


@pytest.fixture()
def overconfident_setup():
    rng = np.random.default_rng(7)
    n = 1000
    X = rng.standard_normal((n, 8))
    coefs = rng.standard_normal(8)
    logits = X @ coefs
    # Add heavy noise so the *truth* isn't deterministic; XGB will be overconfident.
    p_true = 1.0 / (1.0 + np.exp(-logits * 0.5))
    y = (rng.uniform(size=n) < p_true).astype(int)

    cols = [f"f_{i}" for i in range(8)]
    Xdf = pd.DataFrame(X, columns=cols)
    n_train = 700
    return (
        Xdf.iloc[:n_train].reset_index(drop=True),
        pd.Series(y[:n_train]),
        Xdf.iloc[n_train:].reset_index(drop=True),
        pd.Series(y[n_train:]),
    )


def test_isotonic_fit_does_not_raise(overconfident_setup):
    X_tr, y_tr, X_cal, y_cal = overconfident_setup
    base = DeterministicXGBClassifier(params=XGBParams(n_estimators=200, max_depth=6))
    base.fit(X_tr, y_tr)
    cal = fit_isotonic_calibrator(base, X_cal, y_cal)
    p = cal.predict_proba(X_cal)
    assert p.shape == (len(X_cal), 2)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_calibration_improves_brier_on_calib_set(overconfident_setup):
    X_tr, y_tr, X_cal, y_cal = overconfident_setup
    base = DeterministicXGBClassifier(params=XGBParams(n_estimators=200, max_depth=6))
    base.fit(X_tr, y_tr)
    raw = base.predict_proba(X_cal)[:, 1]
    cal_obj = fit_isotonic_calibrator(base, X_cal, y_cal)
    calibrated = cal_obj.predict_proba(X_cal)[:, 1]
    # Isotonic on its own training set is the empirical-frequency mapping;
    # by construction Brier(cal) <= Brier(raw) on that set.
    assert brier_score_loss(y_cal, calibrated) <= brier_score_loss(y_cal, raw) + 1e-9


def test_calibrator_rejects_single_class():
    X_tr = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"))
    y_tr = pd.Series([0, 1] * 25)
    base = DeterministicXGBClassifier(params=XGBParams(n_estimators=20)).fit(X_tr, y_tr)

    # Calibration set with all zeros -> should be rejected
    X_cal = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
    y_cal = pd.Series([0] * 20)
    with pytest.raises(ValueError, match="only one class"):
        fit_isotonic_calibrator(base, X_cal, y_cal)


def test_calibrated_xgb_wrapper_round_trip(overconfident_setup):
    X_tr, y_tr, X_cal, y_cal = overconfident_setup
    base = DeterministicXGBClassifier(params=XGBParams(n_estimators=80)).fit(X_tr, y_tr)
    cal = fit_isotonic_calibrator(base, X_cal, y_cal)
    cxgb = CalibratedXGB(base, cal)

    raw = cxgb.predict_raw(X_cal)
    calibrated = cxgb.predict_calibrated(X_cal)
    assert raw.shape == calibrated.shape
    # Raw and calibrated may differ; calibrated must remain in [0, 1]
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_column_drift_rejected_through_wrapper(overconfident_setup):
    X_tr, y_tr, X_cal, y_cal = overconfident_setup
    base = DeterministicXGBClassifier(params=XGBParams(n_estimators=20)).fit(X_tr, y_tr)
    cal = fit_isotonic_calibrator(base, X_cal, y_cal)
    cxgb = CalibratedXGB(base, cal)
    bad = X_cal.rename(columns={X_cal.columns[0]: "renamed"})
    with pytest.raises(ValueError, match="Feature column drift"):
        cxgb.predict_proba(bad)
