"""Unit tests for the pure return-correlation math."""

from __future__ import annotations

from src.portfolio.correlation import max_corr_to, pearson


def _ser(fn, n=50):
    return {f"d{i}": fn(i) for i in range(n)}


def test_identical_series_correlate_perfectly():
    a = _ser(lambda i: 0.01 * (i % 5 - 2))
    assert pearson(a, dict(a)) == 1.0


def test_anti_correlated_series():
    a = _ser(lambda i: 1.0 if i % 2 == 0 else -1.0)
    b = _ser(lambda i: -1.0 if i % 2 == 0 else 1.0)
    assert pearson(a, b) == -1.0


def test_insufficient_overlap_returns_none():
    a = _ser(lambda i: float(i), n=10)
    b = _ser(lambda i: float(i), n=10)
    assert pearson(a, b, min_overlap=40) is None


def test_only_common_dates_are_used():
    a = {"d0": 1.0, "d1": -1.0, "d2": 1.0, "x": 99.0}
    b = {"d0": 1.0, "d1": -1.0, "d2": 1.0, "y": -99.0}
    # Common dates d0..d2 are identical -> corr 1.0; the disjoint keys ignored.
    assert pearson(a, b, min_overlap=3) == 1.0


def test_constant_series_is_undefined():
    a = _ser(lambda i: 5.0)            # zero variance
    b = _ser(lambda i: float(i))
    assert pearson(a, b) is None


def test_max_corr_to_picks_highest_positive_and_ignores_hedges():
    sym = _ser(lambda i: 1.0 if i % 2 == 0 else -1.0)
    peers = {
        "TWIN": dict(sym),                                      # corr +1.0
        "HEDGE": _ser(lambda i: -1.0 if i % 2 == 0 else 1.0),   # corr -1.0
        "NOISE": _ser(lambda i: 1.0 if (i // 2) % 2 == 0 else -1.0),
    }
    c, peer = max_corr_to(sym, peers)
    assert peer == "TWIN"
    assert c == 1.0


def test_max_corr_to_empty_peers():
    sym = _ser(lambda i: float(i))
    assert max_corr_to(sym, {}) == (None, None)
