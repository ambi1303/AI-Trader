"""Determinism + smoke tests for the XGBoost wrapper.

If these fail it means an upstream change (xgboost version, BLAS lib, or
a config drift in XGBParams) introduced non-determinism. That's a Gate-3
blocker -- a non-deterministic model cannot be re-validated, audited, or
explained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams


@pytest.fixture()
def synth_dataset():
    rng = np.random.default_rng(0)
    n = 600
    n_feat = 12
    X = rng.standard_normal((n, n_feat))
    coefs = rng.standard_normal(n_feat)
    logits = X @ coefs + rng.standard_normal(n) * 0.5
    y = (logits > 0).astype(int)
    cols = [f"f_{i}" for i in range(n_feat)]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


def test_two_fits_produce_identical_bytes(synth_dataset):
    X, y = synth_dataset
    p = XGBParams(n_estimators=80, max_depth=4, seed=42)
    m1 = DeterministicXGBClassifier(params=p).fit(X, y)
    m2 = DeterministicXGBClassifier(params=p).fit(X, y)
    assert m1.get_raw_bytes() == m2.get_raw_bytes()


def test_different_seeds_produce_different_models(synth_dataset):
    X, y = synth_dataset
    m1 = DeterministicXGBClassifier(params=XGBParams(n_estimators=80, seed=1)).fit(X, y)
    m2 = DeterministicXGBClassifier(params=XGBParams(n_estimators=80, seed=2)).fit(X, y)
    assert m1.get_raw_bytes() != m2.get_raw_bytes()


def test_predict_proba_returns_valid_probabilities(synth_dataset):
    X, y = synth_dataset
    m = DeterministicXGBClassifier(params=XGBParams(n_estimators=50)).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_column_drift_is_rejected(synth_dataset):
    X, y = synth_dataset
    m = DeterministicXGBClassifier().fit(X, y)
    bad = X.rename(columns={X.columns[0]: "renamed"})
    with pytest.raises(ValueError, match="Feature column drift"):
        m.predict_proba(bad)


def test_scale_pos_weight_auto_balances():
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame(rng.standard_normal((n, 5)),
                     columns=[f"f{i}" for i in range(5)])
    # 5% positive class: class imbalance scenario
    y = pd.Series((rng.uniform(size=n) < 0.05).astype(int))
    m = DeterministicXGBClassifier(params=XGBParams(n_estimators=30)).fit(X, y)
    # Auto-derived spw = neg/pos. With 5% positive that's ~19.
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    expected = n_neg / n_pos
    assert pytest.approx(m._model.scale_pos_weight, rel=0.001) == expected


def test_feature_importance_uses_real_names(synth_dataset):
    X, y = synth_dataset
    m = DeterministicXGBClassifier().fit(X, y)
    imp = m.feature_importance()
    # importance keys must match real column names, not f0/f1/...
    for k in imp.index:
        assert k in X.columns
