"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.models.metrics import (
    evaluate_classifier,
    expected_calibration_error,
    reliability_curve,
)


def test_perfect_classifier_has_zero_brier_and_zero_ece():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    rep = evaluate_classifier(y, p)
    assert rep.brier == 0.0
    assert rep.ece == 0.0
    assert rep.roc_auc == 1.0


def test_random_classifier_has_brier_around_quarter():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=10000)
    p = np.full(10000, 0.5)
    rep = evaluate_classifier(y, p)
    # Brier of constant 0.5 against 50/50 labels is exactly 0.25
    assert abs(rep.brier - 0.25) < 1e-10


def test_ece_increases_with_miscalibration():
    y = np.zeros(1000, dtype=int)
    y[:300] = 1  # 30% positive
    # Well-calibrated: predict 0.3 everywhere
    p_good = np.full(1000, 0.3)
    # Miscalibrated: predict 0.8 everywhere
    p_bad = np.full(1000, 0.8)
    e_good = expected_calibration_error(y, p_good)
    e_bad = expected_calibration_error(y, p_bad)
    assert e_bad > e_good


def test_single_class_returns_none_aucs():
    y = np.zeros(100, dtype=int)
    p = np.random.default_rng(0).uniform(size=100)
    rep = evaluate_classifier(y, p)
    assert rep.roc_auc is None
    assert rep.pr_auc is None
    assert np.isnan(rep.log_loss)


def test_reliability_curve_partitions_correctly():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85])
    curve = reliability_curve(y, p, n_bins=5)
    assert len(curve) == 5
    total = sum(c for _, _, c in curve)
    assert total == len(y)


def test_empty_input_does_not_crash():
    rep = evaluate_classifier(np.array([]), np.array([]))
    assert rep.n == 0
