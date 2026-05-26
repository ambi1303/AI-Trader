"""Probability calibration via isotonic regression on a held-out slice.

We use sklearn's CalibratedClassifierCV(method='isotonic', cv='prefit'):
  - 'prefit' means we pass an *already-fit* base estimator.
  - The isotonic step then learns a non-parametric monotone mapping from raw
    probabilities to actual frequencies on the calibration set.

Why isotonic, not Platt:
  - Platt (sigmoid) assumes the miscalibration is sigmoidal -- often false for
    tree ensembles, especially with class imbalance and scale_pos_weight.
  - Isotonic is non-parametric and tracks the *empirical* reliability curve,
    which matches what we measure with ECE/Brier in metrics.py.

Why a separate calibration set (not nested CV here):
  - With time-ordered data we MUST not shuffle. The cleanest pattern is:
        train  -> fit XGB
        calib  -> fit isotonic on top
        test   -> evaluate the calibrated probabilities
    All three are non-overlapping and forward in time. CalibratedClassifierCV
    with cv='prefit' is exactly the right tool for this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.xgboost_classifier import DeterministicXGBClassifier
from src.utils.logger import get_logger

log = get_logger("models.calib")


def fit_isotonic_calibrator(
    base_model: DeterministicXGBClassifier,
    X_calib: pd.DataFrame,
    y_calib: pd.Series | np.ndarray,
) -> CalibratedClassifierCV:
    """Wrap a *prefit* model in an isotonic calibrator.

    Returns a sklearn CalibratedClassifierCV. Use `.predict_proba` on it to
    get calibrated probabilities. The returned object also exposes the
    underlying base model under `.estimator`.
    """
    if not isinstance(X_calib, pd.DataFrame):
        raise TypeError("X_calib must be a DataFrame")
    y_arr = np.asarray(y_calib).astype(int)
    if len(np.unique(y_arr)) < 2:
        raise ValueError(
            "Calibration set has only one class; isotonic calibration "
            "requires both 0 and 1 to be present"
        )

    # NOTE: sklearn 1.5 keyword is `estimator=`; older versions used `base_estimator=`.
    cal = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv="prefit")
    cal.fit(X_calib, y_arr)
    log.info(
        "Fit isotonic calibrator on {} rows, {:.2%} positive",
        len(X_calib),
        y_arr.mean(),
    )
    return cal


class CalibratedXGB:
    """Convenience wrapper that owns (base_model, calibrator) together.

    Why we don't just use CalibratedClassifierCV directly: it stores a CLONE of
    the base estimator, which means our DeterministicXGBClassifier's
    `_feature_names` check no longer fires on the cloned copy (sklearn clones
    only `__init__` params). We keep the original base_model for predict-time
    column-drift detection, and we use the calibrator only to map raw probs
    to calibrated probs via isotonic regression on top of the SAME base.
    """

    def __init__(
        self,
        base_model: DeterministicXGBClassifier,
        calibrator: CalibratedClassifierCV,
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Run column-drift check via base_model first.
        _ = self.base_model.predict_proba(X)  # raises on mismatch
        return self.calibrator.predict_proba(X)

    def predict_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(X)[:, 1]
