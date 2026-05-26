"""ML / model layer (Week 3).

Submodules:
- dataset: builds the training matrix from feature_data with leakage-safe targets.
- xgboost_classifier: deterministic XGBoost training with class-weight handling.
- calibration: isotonic calibration of raw probabilities.
- threshold_tuning: pick decision threshold by expected utility (not accuracy).
- walk_forward: expanding-window evaluation with retrain-and-recalibrate.
- registry: versioned model save/load with metadata + git sha + feature hash.
- predict: inference path that writes predictions_log with feature snapshot.
- metrics: Brier, ECE, ROC-AUC, PR-AUC reporting.
- optuna_tuning: time-budgeted hyperparameter search.
"""

FEATURE_SET_VERSION = 1

