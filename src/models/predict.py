"""Inference: run a registered model on the most recent feature row per
symbol and persist predictions to predictions_log.

Why we snapshot the feature vector into predictions_log (not just the
probability):
  - Auditability. When a backtest or live trade looks weird, we need to know
    exactly which inputs the model saw, not approximate them by re-querying
    feature_data (which may have changed due to corrected corporate actions
    or backfills).
  - Reproducibility. predictions_log is the single source of truth for
    "what did the model think on date X".

We do NOT write a signal here -- that decision belongs in the risk/signal
layer (Week 4) which combines calibrated_prob, threshold, position sizing,
and the trading calendar.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.models.registry import ModelMetadata, load_model
from src.utils.db import execute, fetch_all
from src.utils.logger import get_logger

log = get_logger("models.predict")


def _load_feature_row(symbol: str, on_date: date | None) -> pd.DataFrame:
    if on_date is None:
        rows = fetch_all(
            "SELECT * FROM feature_data WHERE symbol = ? "
            "ORDER BY feature_date DESC LIMIT 1",
            (symbol,),
        )
    else:
        rows = fetch_all(
            "SELECT * FROM feature_data WHERE symbol = ? AND feature_date = ?",
            (symbol, on_date.isoformat()),
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["feature_date"] = pd.to_datetime(df["feature_date"]).dt.date
    return df


def _restrict_to_feature_columns(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature row missing required columns "
            f"(model trained with newer feature set?): {missing[:5]}..."
        )
    return df[feature_columns].copy()


def predict_for_symbol(
    run_id: str,
    symbol: str,
    *,
    on_date: date | None = None,
    persist: bool = True,
) -> dict | None:
    """Predict for one symbol on the latest available feature row (or `on_date`).

    Returns dict with raw_prob, calibrated_prob, snapshot — or None if no
    feature row is available.
    """
    cxgb, meta = load_model(run_id)
    row = _load_feature_row(symbol, on_date)
    if row.empty:
        log.warning("No feature row for {} on {}", symbol, on_date)
        return None
    if row["feature_date"].iloc[0] is None:
        return None

    X = _restrict_to_feature_columns(row, meta.feature_columns)

    raw = float(cxgb.predict_raw(X)[0])
    cal = float(cxgb.predict_calibrated(X)[0])

    snapshot = X.iloc[0].to_dict()
    feature_date = row["feature_date"].iloc[0]

    if persist:
        execute(
            """
            INSERT INTO predictions_log
                (run_id, symbol, prediction_date, raw_prob, calibrated_prob,
                 feature_snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                meta.run_id, symbol, str(feature_date), raw, cal,
                json.dumps(snapshot, default=str),
            ),
        )

    return {
        "run_id": meta.run_id,
        "symbol": symbol,
        "prediction_date": str(feature_date),
        "raw_prob": raw,
        "calibrated_prob": cal,
        "threshold": meta.threshold,
        "would_signal": (
            cal >= meta.threshold if meta.threshold is not None else None
        ),
    }


def predict_for_universe(
    run_id: str,
    symbols: list[str],
    *,
    on_date: date | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    out = []
    for s in symbols:
        r = predict_for_symbol(run_id, s, on_date=on_date, persist=persist)
        if r is not None:
            out.append(r)
    return pd.DataFrame(out)


def get_metadata(run_id: str) -> ModelMetadata:
    _, meta = load_model(run_id)
    return meta
