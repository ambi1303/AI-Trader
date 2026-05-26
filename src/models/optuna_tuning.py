"""Time-budgeted Optuna hyperparameter search.

Search is constrained to:
  - small / shallow trees                     (overfitting control)
  - fixed seed and single-thread              (determinism)
  - validation Brier as the objective         (we calibrate later, so we want
                                                the BASE classifier to be as
                                                well-ranked AND well-scored as
                                                possible BEFORE calibration)
  - per-trial early stopping on the val set   (cheap)

We use TPESampler with a fixed seed so the same DB + same code reproduces
exactly the same trial sequence. Pruning uses MedianPruner to kill bad trials
early.

The driver reports the best trial and returns the chosen XGBParams; the
caller is responsible for retraining + calibrating + registering the final
model.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.models.metrics import evaluate_classifier
from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams
from src.utils.logger import get_logger

log = get_logger("models.optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptunaResult:
    best_params: XGBParams
    best_value: float          # validation Brier
    n_trials_completed: int
    elapsed_seconds: float


def _sample_params(trial: optuna.Trial, seed: int) -> XGBParams:
    return XGBParams(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 7),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        seed=seed,
    )


def search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int = 60,
    timeout_seconds: int | None = 60 * 30,
    seed: int = 42,
) -> OptunaResult:
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_val, pd.DataFrame):
        raise TypeError("X_train and X_val must be DataFrames")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("Train or val slice has only one class")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed, multivariate=True),
        pruner=MedianPruner(n_warmup_steps=5),
    )

    def _objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, seed=seed)
        model = DeterministicXGBClassifier(params=params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
        )
        proba = model.predict_proba(X_val)[:, 1]
        rep = evaluate_classifier(np.asarray(y_val).astype(int), proba)
        if math.isnan(rep.brier):
            return 1.0
        return rep.brier

    t0 = time.time()
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best_trial = study.best_trial
    log.info(
        "Optuna done: best Brier={:.4f} in {} trials ({:.1f}s)",
        best_trial.value, len(study.trials), elapsed,
    )
    best_params = XGBParams(
        n_estimators=best_trial.params["n_estimators"],
        max_depth=best_trial.params["max_depth"],
        learning_rate=best_trial.params["learning_rate"],
        min_child_weight=best_trial.params["min_child_weight"],
        subsample=best_trial.params["subsample"],
        colsample_bytree=best_trial.params["colsample_bytree"],
        reg_lambda=best_trial.params["reg_lambda"],
        reg_alpha=best_trial.params["reg_alpha"],
        gamma=best_trial.params["gamma"],
        seed=seed,
    )
    return OptunaResult(
        best_params=best_params,
        best_value=best_trial.value,
        n_trials_completed=len(study.trials),
        elapsed_seconds=elapsed,
    )
