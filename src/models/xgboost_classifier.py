"""Deterministic XGBoost classifier wrapper.

Determinism in xgboost requires:
  - tree_method='hist' (default in 2.x but pinned for clarity)
  - nthread=1                 (multi-threaded hist is non-deterministic)
  - seed=<fixed>
  - shuffle off in any wrapping CV (we don't use it; folds are time-based)

The class follows scikit-learn's fit/predict_proba contract so it composes
with CalibratedClassifierCV. We also expose `train_with_validation` which
does explicit early stopping on a held-out time-ordered val set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils.logger import get_logger

log = get_logger("models.xgb")


# ---------------------------------------------------------------------------
# Default hyperparameters -- conservative; overrideable by Optuna later.
# ---------------------------------------------------------------------------


@dataclass
class XGBParams:
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    seed: int = 42
    nthread: int = 1
    tree_method: str = "hist"
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"

    def to_xgb_kwargs(self, *, scale_pos_weight: float = 1.0) -> dict:
        # XGBoost's sklearn API expects `n_estimators` separately and the rest
        # via the constructor. `random_state` is the canonical name.
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "random_state": self.seed,
            "n_jobs": self.nthread,
            "tree_method": self.tree_method,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": 0,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class DeterministicXGBClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible wrapper around xgb.XGBClassifier.

    The point of this class is *not* to add intelligence; it is to:
      1. enforce single-thread + fixed seed so repeated training is byte-identical
      2. carry around the feature-column order (names) so we can refuse to
         predict on a frame whose columns drifted
      3. keep `scale_pos_weight` derivation in one place
    """

    def __init__(
        self,
        params: XGBParams | None = None,
        *,
        scale_pos_weight: float | None = None,
    ) -> None:
        self.params = params or XGBParams()
        self.scale_pos_weight = scale_pos_weight
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self.classes_: np.ndarray = np.array([0, 1])

    # -- sklearn API -------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "DeterministicXGBClassifier":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame so we can pin column order")
        y_arr = np.asarray(y).astype(int)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D")

        spw = self.scale_pos_weight
        if spw is None:
            n_pos = int((y_arr == 1).sum())
            n_neg = int((y_arr == 0).sum())
            spw = (n_neg / n_pos) if n_pos > 0 else 1.0

        kwargs = self.params.to_xgb_kwargs(scale_pos_weight=spw)
        if early_stopping_rounds is not None and eval_set is not None:
            kwargs["early_stopping_rounds"] = early_stopping_rounds

        self._model = xgb.XGBClassifier(**kwargs)
        self._feature_names = list(X.columns)

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [(es_x.values, np.asarray(es_y).astype(int))
                                      for es_x, es_y in eval_set]
            fit_kwargs["verbose"] = False

        self._model.fit(X.values, y_arr, **fit_kwargs)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fit yet")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if list(X.columns) != self._feature_names:
            raise ValueError(
                "Feature column drift: trained on "
                f"{len(self._feature_names)} cols, got {len(X.columns)}; first "
                f"mismatch index = {_first_mismatch(self._feature_names, list(X.columns))}"
            )
        return self._model.predict_proba(X.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # -- introspection ----------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Model not fit yet")
        return list(self._feature_names)

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not fit yet")
        booster = self._model.get_booster()
        scores = booster.get_score(importance_type="gain")
        # XGBoost names features f0, f1, ... when fit on numpy arrays.
        idx_to_name = {f"f{i}": n for i, n in enumerate(self._feature_names or [])}
        out = {idx_to_name.get(k, k): v for k, v in scores.items()}
        s = pd.Series(out, name="gain").sort_values(ascending=False)
        return s

    def get_raw_bytes(self) -> bytes:
        """Stable byte representation for determinism tests."""
        if self._model is None:
            raise RuntimeError("Model not fit yet")
        return self._model.get_booster().save_raw("ubj")


def _first_mismatch(a: list[str] | None, b: list[str]) -> int:
    if a is None:
        return 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


class DeterministicXGBMulticlass(BaseEstimator, ClassifierMixin):
    """Deterministic multiclass (softprob) XGBoost for the SELL/HOLD/BUY head.

    Same determinism guarantees as :class:`DeterministicXGBClassifier` plus
    a pinned ``num_class``. We expose the sklearn classifier contract so the
    isotonic calibrator (one-vs-rest) can wrap it.
    """

    def __init__(
        self,
        params: XGBParams | None = None,
        *,
        num_class: int = 3,
    ) -> None:
        self.params = params or XGBParams()
        self.num_class = num_class
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self.classes_: np.ndarray = np.arange(num_class)

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> "DeterministicXGBMulticlass":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame so we can pin column order")
        y_arr = np.asarray(y).astype(int)
        kwargs = self.params.to_xgb_kwargs()
        # Override the binary objective with multiclass softprob.
        kwargs.update(
            objective="multi:softprob",
            num_class=self.num_class,
            eval_metric="mlogloss",
        )
        kwargs.pop("scale_pos_weight", None)  # not valid for multiclass
        self._model = xgb.XGBClassifier(**kwargs)
        self._feature_names = list(X.columns)
        self._model.fit(X.values, y_arr)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fit yet")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if list(X.columns) != self._feature_names:
            raise ValueError(
                "Feature column drift: trained on "
                f"{len(self._feature_names or [])} cols, got {len(X.columns)}"
            )
        return self._model.predict_proba(X.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Model not fit yet")
        return list(self._feature_names)
