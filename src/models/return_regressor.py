"""Deterministic XGBoost regressor for the forward-return / price-target head.

This is the "how much" model that complements the binary "will it go up"
classifier and the tri-class verdict. It predicts the expected forward
return over the model's horizon; the predictor turns that into a target
price (``close * (1 + predicted_return)``).

Determinism mirrors :class:`DeterministicXGBClassifier`: single thread,
fixed seed, ``hist`` tree method, pinned feature-column order so a column
drift between training and inference is caught loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from src.utils.logger import get_logger

log = get_logger("models.regressor")


@dataclass
class XGBRegParams:
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
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"

    def to_xgb_kwargs(self) -> dict:
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
            "verbosity": 0,
        }


class DeterministicXGBRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around xgb.XGBRegressor."""

    def __init__(self, params: XGBRegParams | None = None) -> None:
        self.params = params or XGBRegParams()
        self._model: xgb.XGBRegressor | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> "DeterministicXGBRegressor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame so we can pin column order")
        y_arr = np.asarray(y).astype(float)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D")
        self._model = xgb.XGBRegressor(**self.params.to_xgb_kwargs())
        self._feature_names = list(X.columns)
        self._model.fit(X.values, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fit yet")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if list(X.columns) != self._feature_names:
            raise ValueError(
                "Feature column drift: trained on "
                f"{len(self._feature_names or [])} cols, got {len(X.columns)}"
            )
        return self._model.predict(X.values)

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Model not fit yet")
        return list(self._feature_names)
