"""Tests for model registry: save/load round-trip + feature-hash integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import CalibratedXGB, fit_isotonic_calibrator
from src.models.registry import (
    delete_run,
    feature_hash,
    latest_run_id,
    load_model,
    save_model,
)
from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams
from src.utils import db as db_mod


@pytest.fixture()
def temp_db_and_models(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    schema = """
    CREATE TABLE model_runs (
        run_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        git_sha TEXT,
        feature_hash TEXT,
        trained_from TEXT,
        trained_to TEXT,
        metrics_json TEXT,
        artifact_path TEXT,
        created_at TEXT NOT NULL DEFAULT '2024-01-01'
    );
    """
    db_mod.execute_script(schema)
    return tmp_path / "models"


def _trained_pair():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 6)),
                     columns=[f"f{i}" for i in range(6)])
    y = pd.Series((rng.uniform(size=n) < 0.4).astype(int))

    base = DeterministicXGBClassifier(params=XGBParams(n_estimators=40)).fit(X, y)
    cal = fit_isotonic_calibrator(base, X.iloc[200:], y.iloc[200:])
    return CalibratedXGB(base, cal), list(X.columns), X


def test_feature_hash_is_order_sensitive():
    a = ["f1", "f2", "f3"]
    b = ["f3", "f2", "f1"]
    assert feature_hash(a) != feature_hash(b)
    assert feature_hash(a) == feature_hash(["f1", "f2", "f3"])


def test_save_and_load_round_trip(temp_db_and_models):
    base_dir = temp_db_and_models
    cxgb, cols, X = _trained_pair()
    meta = save_model(
        cxgb,
        model_name="unit_test",
        feature_columns=cols,
        trained_from="2024-01-01",
        trained_to="2024-06-30",
        metrics={"brier": 0.21, "ece": 0.05},
        threshold=0.65,
        base_dir=base_dir,
    )
    loaded, loaded_meta = load_model(meta.run_id, base_dir=base_dir)
    assert loaded_meta.run_id == meta.run_id
    assert loaded_meta.feature_hash == meta.feature_hash
    assert loaded_meta.threshold == 0.65
    assert loaded_meta.feature_columns == cols
    # Predictions must match exactly across the round trip.
    p_orig = cxgb.predict_calibrated(X)
    p_load = loaded.predict_calibrated(X)
    np.testing.assert_allclose(p_orig, p_load, atol=1e-12)


def test_latest_run_id_returns_most_recent(temp_db_and_models):
    base_dir = temp_db_and_models
    cxgb, cols, _ = _trained_pair()

    save_model(cxgb, model_name="m1",
               feature_columns=cols, trained_from=None, trained_to=None,
               metrics={}, threshold=None, base_dir=base_dir)
    second = save_model(cxgb, model_name="m1",
                        feature_columns=cols, trained_from=None, trained_to=None,
                        metrics={}, threshold=None, base_dir=base_dir)
    assert latest_run_id("m1") == second.run_id
    assert latest_run_id("does_not_exist") is None


def test_delete_run_cleans_up(temp_db_and_models):
    base_dir = temp_db_and_models
    cxgb, cols, _ = _trained_pair()
    meta = save_model(
        cxgb, model_name="to_delete",
        feature_columns=cols, trained_from=None, trained_to=None,
        metrics={}, threshold=None, base_dir=base_dir,
    )
    delete_run(meta.run_id, base_dir=base_dir)
    with pytest.raises(FileNotFoundError):
        load_model(meta.run_id, base_dir=base_dir)
    assert latest_run_id("to_delete") is None
