"""Learned multi-horizon return model (one XGBoost regressor per horizon).

This is the ML sibling of :mod:`src.analysis.forecast` (the transparent
drift+volatility projection). Instead of assuming a Brownian drift, we *learn*
the expected forward log-return at each horizon from the same engineered
features the classifier already uses, plus per-day cross-sectional rank
companions (relative strength / value / quality), which is where multi-week and
multi-month edge actually lives.

Design
------
* **One regressor per horizon.** Horizons 1W..1Y (5/21/63/126/252 trading days)
  each get a :class:`DeterministicXGBRegressor` predicting
  ``ln(close[t+h] / close[t])``. The multi-year "long run" (3Y) is intentionally
  *not* learned -- there is not enough non-overlapping history to fit it
  honestly -- so callers keep using the analytic projection for that horizon.
* **Honest, held-out bands.** Each horizon is trained on the earliest dates and
  evaluated on a later, unseen slice. The std of the held-out residual becomes
  the 1-sigma band half-width and feeds ``prob_up``; the rank IC and RMSE on the
  same slice are recorded so a human can judge per-horizon skill. The *deployed*
  model is then refit on all rows to use maximum data.
* **No look-ahead.** Targets come from the SAME bhavcopy close used to build
  features (via :func:`src.models.dataset._next_day_returns_per_symbol`), and
  the train/test split is strictly time-ordered (no shuffling).
* **Same output shape as the analytic forecaster** so the storage / UI layer is
  identical -- a horizon dict carries expected/low/high price, return %, prob_up
  and a verdict, just tagged ``method="ml"``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.analysis.forecast import _BAND_Z, _TRADING_DAYS_YEAR, _verdict
from src.models.dataset import (
    _META_COLS,
    _NON_FEATURE_COLS,
    _load_feature_data,
    _next_day_returns_per_symbol,
    add_cross_sectional_features,
    time_based_split,
)
from src.models.registry import _new_run_id, feature_hash
from src.models.return_regressor import DeterministicXGBRegressor, XGBRegParams
from src.utils.db import execute, fetch_all, fetch_one
from src.utils.logger import get_logger

log = get_logger("models.horizon")

_N = NormalDist()

MODEL_NAME = "horizon_forecaster"
_BUNDLE_FILE = "bundle.joblib"
_BASE_DIR = "data/models/horizon"

# Horizons we LEARN (label -> trading days). 3Y is left to the analytic model.
ML_HORIZONS: tuple[tuple[str, int], ...] = (
    ("1W", 5),
    ("1M", 21),
    ("3M", 63),
    ("6M", 126),
    ("1Y", 252),
)

# A horizon with fewer than this many (feature, target) rows is skipped --
# fitting a 1-year forward return on a few hundred rows is not credible.
_MIN_ROWS = 400
# Floor on the residual sigma so prob_up can't blow up if a horizon happens to
# fit its held-out slice almost perfectly (over-optimistic band).
_MIN_RESID_STD = 1e-3


@dataclass
class HorizonModel:
    """A single trained horizon: the regressor plus its held-out diagnostics."""

    label: str
    horizon_days: int
    regressor: DeterministicXGBRegressor
    resid_std: float                 # std of held-out log-return residual
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HorizonBundle:
    """All per-horizon models + the shared inference contract."""

    models: dict[str, HorizonModel]
    feature_columns: list[str]
    feature_hash: str
    cross_sectional: bool
    long_run_annual: float
    trained_from: str | None
    trained_to: str | None
    created_at: str
    run_id: str

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": MODEL_NAME,
            "created_at": self.created_at,
            "feature_hash": self.feature_hash,
            "cross_sectional": self.cross_sectional,
            "long_run_annual": self.long_run_annual,
            "trained_from": self.trained_from,
            "trained_to": self.trained_to,
            "n_features": len(self.feature_columns),
            "horizons": {
                lbl: {"horizon_days": m.horizon_days,
                      "resid_std": round(m.resid_std, 6),
                      **{k: round(v, 6) for k, v in m.metrics.items()}}
                for lbl, m in self.models.items()
            },
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _spearman_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation (no scipy dependency) between prediction and outcome."""
    if len(pred) < 3:
        return 0.0
    s = pd.Series(pred).corr(pd.Series(actual), method="spearman")
    return float(s) if s == s else 0.0  # NaN-safe (constant input -> NaN)


def _fit_horizon(
    label: str,
    days: int,
    feat_x: pd.DataFrame,
    feature_cols: list[str],
    *,
    params: XGBRegParams,
) -> HorizonModel | None:
    """Fit one horizon regressor; return None when there isn't enough data."""
    syms = feat_x["symbol"].unique().tolist()
    fwd = _next_day_returns_per_symbol(syms, horizon=days)
    df = feat_x.merge(fwd, on=["symbol", "feature_date"], how="inner")
    df = df.dropna(subset=["fwd_return"])
    # Guard against degenerate returns (corporate-action artefacts) before log.
    df = df[df["fwd_return"] > -0.99].copy()
    if len(df) < _MIN_ROWS:
        log.warning("horizon {} skipped: only {} rows (< {})",
                    label, len(df), _MIN_ROWS)
        return None

    df = df.sort_values(["feature_date", "symbol"]).reset_index(drop=True)
    y = np.log1p(df["fwd_return"].astype(float).to_numpy())
    X = df[feature_cols].copy()
    meta = df[["symbol", "feature_date"]].copy()

    train_idx, val_idx, test_idx = time_based_split(
        meta, train_frac=0.70, val_frac=0.15
    )
    # Held-out = val + test (everything after the training cut), so the band /
    # IC reflect genuinely future, unseen dates.
    holdout_idx = np.concatenate([val_idx, test_idx]) if len(test_idx) else val_idx
    if len(train_idx) < _MIN_ROWS // 2 or len(holdout_idx) < 50:
        log.warning("horizon {} skipped: thin split "
                    "(train={}, holdout={})", label, len(train_idx), len(holdout_idx))
        return None

    fit_reg = DeterministicXGBRegressor(params)
    fit_reg.fit(X.iloc[train_idx], y[train_idx])
    pred_h = fit_reg.predict(X.iloc[holdout_idx])
    actual_h = y[holdout_idx]

    resid = actual_h - pred_h
    resid_std = max(float(np.std(resid)), _MIN_RESID_STD)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ic = _spearman_ic(pred_h, actual_h)
    # Directional hit-rate among non-trivial predictions.
    hit = float(np.mean(np.sign(pred_h) == np.sign(actual_h)))

    # Deploy a model trained on ALL rows (max data); keep the held-out stats.
    final_reg = DeterministicXGBRegressor(params)
    final_reg.fit(X, y)

    log.info("horizon {} ({}d): n={} ic={:.3f} rmse={:.4f} hit={:.3f} "
             "resid_std={:.4f}", label, days, len(df), ic, rmse, hit, resid_std)
    return HorizonModel(
        label=label, horizon_days=days, regressor=final_reg,
        resid_std=resid_std,
        metrics={"n_rows": float(len(df)), "n_holdout": float(len(holdout_idx)),
                 "ic": ic, "rmse": rmse, "mae": mae, "hit_rate": hit},
    )


def train_horizon_models(
    symbols: list[str] | None = None,
    *,
    horizons: tuple[tuple[str, int], ...] = ML_HORIZONS,
    cross_sectional: bool = True,
    long_run_annual: float = 0.11,
    params: XGBRegParams | None = None,
) -> HorizonBundle:
    """Train one regressor per horizon and return an unsaved bundle."""
    params = params or XGBRegParams()
    feat = _load_feature_data(symbols)
    if feat.empty:
        raise RuntimeError("No feature_data rows to train on.")

    base_cols = [
        c for c in feat.columns
        if c not in _META_COLS and c not in _NON_FEATURE_COLS
    ]
    if cross_sectional:
        feat, xs_cols = add_cross_sectional_features(feat)
        feature_cols = base_cols + xs_cols
    else:
        feature_cols = base_cols

    models: dict[str, HorizonModel] = {}
    for label, days in horizons:
        m = _fit_horizon(label, days, feat, feature_cols, params=params)
        if m is not None:
            models[label] = m
    if not models:
        raise RuntimeError("No horizon could be trained (insufficient data).")

    trained_from = str(feat["feature_date"].min())
    trained_to = str(feat["feature_date"].max())
    return HorizonBundle(
        models=models,
        feature_columns=feature_cols,
        feature_hash=feature_hash(feature_cols),
        cross_sectional=cross_sectional,
        long_run_annual=long_run_annual,
        trained_from=trained_from,
        trained_to=trained_to,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_id=_new_run_id(MODEL_NAME),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_bundle(bundle: HorizonBundle, *, base_dir: str | Path = _BASE_DIR) -> Path:
    base_path = Path(base_dir).resolve()
    run_dir = base_path / bundle.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, run_dir / _BUNDLE_FILE, compress=3)
    (run_dir / "metadata.json").write_text(
        json.dumps(bundle.metadata(), indent=2, default=str), encoding="utf-8"
    )
    execute(
        """
        INSERT OR REPLACE INTO model_runs
            (run_id, model_name, git_sha, feature_hash,
             trained_from, trained_to, metrics_json, artifact_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (bundle.run_id, MODEL_NAME, "n/a", bundle.feature_hash,
         bundle.trained_from, bundle.trained_to,
         json.dumps(bundle.metadata()["horizons"], default=str), str(run_dir)),
    )
    log.info("Saved horizon bundle run_id={} -> {}", bundle.run_id, run_dir)
    return run_dir


def load_bundle(run_id: str, *, base_dir: str | Path = _BASE_DIR) -> HorizonBundle:
    run_dir = Path(base_dir).resolve() / run_id
    bundle: HorizonBundle = joblib.load(run_dir / _BUNDLE_FILE)
    expected = feature_hash(bundle.feature_columns)
    if expected != bundle.feature_hash:
        raise ValueError(
            f"Horizon bundle feature-hash mismatch: {bundle.feature_hash} "
            f"vs recomputed {expected}"
        )
    return bundle


def latest_bundle(
    *, base_dir: str | Path = _BASE_DIR, db_path: str | None = None
) -> HorizonBundle | None:
    """Most recently registered horizon bundle, or None if none trained yet."""
    row = fetch_one(
        "SELECT run_id, artifact_path FROM model_runs WHERE model_name = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (MODEL_NAME,), db_path=db_path,
    )
    if not row:
        return None
    try:
        path = row["artifact_path"]
        base = Path(path).parent if path else Path(base_dir)
        return load_bundle(row["run_id"], base_dir=base)
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.warning("Could not load latest horizon bundle {}: {}",
                    row["run_id"], exc)
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _horizon_dict(model: HorizonModel, last_close: float, r_pred: float) -> dict[str, Any]:
    """Build the standard horizon output dict from a predicted log-return."""
    band = _BAND_Z * model.resid_std
    expected = last_close * math.exp(r_pred)
    low = last_close * math.exp(r_pred - band)
    high = last_close * math.exp(r_pred + band)
    exp_ret_pct = (math.exp(r_pred) - 1.0) * 100.0
    annualized_pct = (
        math.exp(r_pred * _TRADING_DAYS_YEAR / model.horizon_days) - 1.0
    ) * 100.0
    prob_up = _N.cdf(r_pred / band) if band > 0 else (1.0 if r_pred > 0 else 0.0)
    label_txt, tone = _verdict(exp_ret_pct, prob_up)
    return {
        "label": model.label,
        "horizon_days": model.horizon_days,
        "expected_price": round(expected, 2),
        "low_price": round(low, 2),
        "high_price": round(high, 2),
        "expected_return_pct": round(exp_ret_pct, 1),
        "annualized_return_pct": round(annualized_pct, 1),
        "band_pct": round((math.exp(band) - 1.0) * 100.0, 1),
        "prob_up_pct": round(prob_up * 100.0, 1),
        "verdict": label_txt,
        "tone": tone,
        "method": "ml",
    }


def _resolve_universe_date(as_of: str, db_path: str | None) -> str | None:
    row = fetch_one(
        "SELECT MAX(feature_date) AS d FROM feature_data WHERE feature_date <= ?",
        (as_of,), db_path=db_path,
    )
    return row["d"] if row and row["d"] else None


def _load_universe(on_date: str, db_path: str | None) -> pd.DataFrame:
    rows = fetch_all(
        "SELECT * FROM feature_data WHERE feature_date = ?", (on_date,),
        db_path=db_path,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["feature_date"] = pd.to_datetime(df["feature_date"]).dt.date
    return df


def predict_universe(
    bundle: HorizonBundle,
    as_of: str,
    *,
    db_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Forecast every symbol present on the as-of (or prior) feature date.

    Returns ``{symbol: forecast_dict}`` where each forecast dict matches the
    analytic forecaster's shape (``available``, ``last_close``, ``horizons``).
    Cross-sectional rank features are computed over the whole universe on that
    date, exactly mirroring training.
    """
    on_date = _resolve_universe_date(as_of, db_path)
    if on_date is None:
        return {}
    universe = _load_universe(on_date, db_path)
    if universe.empty:
        return {}
    if bundle.cross_sectional:
        universe, _ = add_cross_sectional_features(universe)

    missing = [c for c in bundle.feature_columns if c not in universe.columns]
    if missing:
        raise ValueError(
            f"Universe missing model features (feature set drift?): {missing[:5]}"
        )

    X_all = universe[bundle.feature_columns].copy()
    # Predict each horizon once for the whole universe (vectorised), then slice.
    preds: dict[str, np.ndarray] = {}
    for label, model in bundle.models.items():
        preds[label] = model.regressor.predict(X_all)

    out: dict[str, dict[str, Any]] = {}
    closes = universe["close"].to_numpy()
    syms = universe["symbol"].tolist()
    for i, sym in enumerate(syms):
        close = closes[i]
        if close is None or pd.isna(close) or close <= 0:
            continue
        close = float(close)
        horizons = [
            _horizon_dict(model, close, float(preds[label][i]))
            for label, model in bundle.models.items()
        ]
        out[sym] = {
            "available": True,
            "symbol": sym,
            "method": "ml",
            "model_run_id": bundle.run_id,
            "last_close": round(close, 2),
            "as_of_date": str(on_date),
            "horizons": horizons,
        }
    return out


def predict_symbol(
    bundle: HorizonBundle,
    symbol: str,
    as_of: str,
    *,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Single-symbol convenience wrapper around :func:`predict_universe`."""
    allout = predict_universe(bundle, as_of, db_path=db_path)
    return allout.get(symbol.upper().strip(), {"available": False, "symbol": symbol})
