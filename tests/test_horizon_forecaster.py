"""Unit + integration tests for the learned multi-horizon forecaster."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import forecast_store
from src.analysis.forecast_store import forecast_symbol, latest_forecast, store_forecast
from src.db.migrate import apply_schema
from src.models.horizon_forecaster import (
    HorizonModel,
    _horizon_dict,
    latest_bundle,
    save_bundle,
    train_horizon_models,
)
from src.models.return_regressor import DeterministicXGBRegressor
from src.utils import db as db_mod

# Short horizons so a ~90-day synthetic history can train them.
_TEST_HORIZONS = (("1W", 5), ("1M", 21))


# --------------------------------------------------------------------------
# Pure: horizon dict from a predicted log-return
# --------------------------------------------------------------------------


def _dummy_model(resid_std: float, days: int = 21) -> HorizonModel:
    return HorizonModel(label="1M", horizon_days=days,
                        regressor=DeterministicXGBRegressor(),
                        resid_std=resid_std, metrics={})


def test_horizon_dict_band_brackets_expected():
    d = _horizon_dict(_dummy_model(0.15), last_close=100.0, r_pred=0.05)
    assert d["low_price"] < d["expected_price"] < d["high_price"]
    assert d["expected_price"] == pytest.approx(100.0 * math.exp(0.05), abs=0.01)
    assert d["method"] == "ml"


def test_horizon_dict_prob_up_tracks_sign():
    up = _horizon_dict(_dummy_model(0.15), last_close=100.0, r_pred=0.10)
    flat = _horizon_dict(_dummy_model(0.15), last_close=100.0, r_pred=0.0)
    down = _horizon_dict(_dummy_model(0.15), last_close=100.0, r_pred=-0.10)
    assert up["prob_up_pct"] > 50.0 > down["prob_up_pct"]
    assert flat["prob_up_pct"] == pytest.approx(50.0, abs=0.1)


# --------------------------------------------------------------------------
# Integration: train -> save -> predict -> persist on a synthetic DB
# --------------------------------------------------------------------------


@pytest.fixture()
def ml_db(tmp_path, monkeypatch):
    db_file = tmp_path / "ml.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    apply_schema(db_path=db_file)
    forecast_store.clear_bundle_cache()

    rng = np.random.default_rng(7)
    dates = [d.date().isoformat()
             for d in pd.bdate_range("2022-01-03", periods=90)]
    symbols = [f"SYM{i:02d}" for i in range(35)]

    price_rows, feat_rows = [], []
    for s in symbols:
        c = 100.0 * (1.0 + 0.5 * rng.random())
        closes = []
        for _ in dates:
            c *= math.exp(rng.normal(0.0006, 0.02))
            closes.append(c)
        closes = np.array(closes)
        for i, d in enumerate(dates):
            cl = float(closes[i])
            price_rows.append((s, d, cl, cl * 1.01, cl * 0.99, cl, 100000,
                               "bhavcopy"))
            mom20 = float(closes[i] / closes[i - 20] - 1.0) if i >= 20 else 0.0
            mom60 = 0.0
            ret5 = float(closes[i] / closes[i - 5] - 1.0) if i >= 5 else 0.0
            feat_rows.append((s, d, cl, mom20, mom60, ret5, 0.02, 0.012))

    with db_mod.transaction() as conn:
        conn.executemany(
            "INSERT INTO price_data "
            "(symbol, bar_date, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", price_rows,
        )
        conn.executemany(
            "INSERT INTO feature_data "
            "(symbol, feature_date, close, mom_20d, mom_60d, ret_5d, "
            " vol_20d, atr_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", feat_rows,
        )
    return db_file, dates[-1]


def test_train_predict_persist_ml(ml_db, tmp_path):
    _, last_date = ml_db
    models_dir = tmp_path / "models"

    bundle = train_horizon_models(cross_sectional=False, horizons=_TEST_HORIZONS)
    assert set(bundle.models) <= {"1W", "1M"} and bundle.models
    for m in bundle.models.values():
        assert m.resid_std > 0
        assert np.isfinite(m.metrics["ic"])

    save_bundle(bundle, base_dir=models_dir)
    forecast_store.clear_bundle_cache()

    # latest_bundle must round-trip from the registry (via artifact_path).
    reloaded = latest_bundle(base_dir=models_dir)
    assert reloaded is not None
    assert reloaded.run_id == bundle.run_id

    out = forecast_symbol("SYM00", last_date)
    assert out["available"] and out["method"] == "ml"
    methods = {h["label"]: h.get("method") for h in out["horizons"]}
    assert methods.get("1W") == "ml" and methods.get("1M") == "ml"
    # 3Y is never learned -> always the analytic projection.
    assert methods.get("3Y") == "drift"
    for h in out["horizons"]:
        assert h["low_price"] < h["expected_price"] < h["high_price"]

    store_forecast("SYM00", last_date)
    rows = {r["horizon_label"]: r for r in latest_forecast("SYM00")}
    assert rows["1W"]["method"] == "ml"
    assert rows["1W"]["model_run_id"] == bundle.run_id
    assert rows["3Y"]["method"] == "drift"


def test_analytic_fallback_when_no_bundle(ml_db):
    """With no trained bundle, forecasting must fall back to the analytic
    projection (method='drift') with no regression."""
    _, last_date = ml_db
    forecast_store.clear_bundle_cache()
    out = forecast_symbol("SYM01", last_date)
    assert out["available"]
    assert all(h.get("method") == "drift" for h in out["horizons"])
