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

from src.models.dataset import add_cross_sectional_features
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


def _resolve_universe_date(on_date: date | None) -> str | None:
    """The date we score a cross-sectional model on.

    Cross-sectional features require a single shared date so the per-day
    ranks are well-defined. We use the latest feature_date that exists in
    feature_data (or the caller's explicit date).
    """
    if on_date is not None:
        return on_date.isoformat()
    row = fetch_all("SELECT MAX(feature_date) AS d FROM feature_data")
    return dict(row[0])["d"] if row else None


def _load_universe_rows(on_date_str: str) -> pd.DataFrame:
    """All symbols' feature rows for a single date -- the universe over which
    cross-sectional ranks are computed (must mirror training)."""
    rows = fetch_all(
        "SELECT * FROM feature_data WHERE feature_date = ?", (on_date_str,)
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["feature_date"] = pd.to_datetime(df["feature_date"]).dt.date
    return df


def predict_for_universe_cross_sectional(
    run_id: str,
    symbols: list[str] | None,
    *,
    on_date: date | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """Score a cross-sectional model.

    Loads the WHOLE universe present on the target date, computes the per-day
    rank features exactly as training did, then scores each requested symbol
    (or all of them if `symbols` is falsy). Persists one predictions_log row
    per symbol.
    """
    cxgb, meta = load_model(run_id)
    on_str = _resolve_universe_date(on_date)
    if on_str is None:
        log.warning("No feature_data available to score.")
        return pd.DataFrame()

    universe = _load_universe_rows(on_str)
    if universe.empty:
        log.warning("No feature rows on {}", on_str)
        return pd.DataFrame()

    # Ranks must be computed over the full universe present that day.
    universe, _ = add_cross_sectional_features(universe)

    wanted = {s.upper() for s in symbols} if symbols else None
    feature_date = universe["feature_date"].iloc[0]
    out = []
    for _, row in universe.iterrows():
        sym = row["symbol"]
        if wanted is not None and sym not in wanted:
            continue
        X = _restrict_to_feature_columns(
            pd.DataFrame([row]), meta.feature_columns
        )
        raw = float(cxgb.predict_raw(X)[0])
        cal = float(cxgb.predict_calibrated(X)[0])
        if persist:
            execute(
                """
                INSERT INTO predictions_log
                    (run_id, symbol, prediction_date, raw_prob, calibrated_prob,
                     feature_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.run_id, sym, str(feature_date), raw, cal,
                    json.dumps(X.iloc[0].to_dict(), default=str),
                ),
            )
        out.append({
            "run_id": meta.run_id,
            "symbol": sym,
            "prediction_date": str(feature_date),
            "raw_prob": raw,
            "calibrated_prob": cal,
            "threshold": meta.threshold,
            "would_signal": (
                cal >= meta.threshold if meta.threshold is not None else None
            ),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("calibrated_prob", ascending=False).reset_index(
            drop=True
        )
    return df


def predict_for_universe(
    run_id: str,
    symbols: list[str],
    *,
    on_date: date | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    # Cross-sectional models need the whole-universe batch path; everything
    # else can score symbol-by-symbol.
    _, meta = load_model(run_id)
    if getattr(meta, "cross_sectional_features", False):
        return predict_for_universe_cross_sectional(
            run_id, symbols, on_date=on_date, persist=persist
        )
    out = []
    for s in symbols:
        r = predict_for_symbol(run_id, s, on_date=on_date, persist=persist)
        if r is not None:
            out.append(r)
    return pd.DataFrame(out)


def get_metadata(run_id: str) -> ModelMetadata:
    _, meta = load_model(run_id)
    return meta
