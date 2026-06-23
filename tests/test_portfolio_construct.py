"""Unit tests for the greedy diversification gate."""

from __future__ import annotations

from src.portfolio.construct import DiversificationGate, PortfolioConfig


def _ser(fn, n=50):
    return {f"d{i}": fn(i) for i in range(n)}


_UP = _ser(lambda i: 1.0 if i % 2 == 0 else -1.0)
_PERIOD4 = _ser(lambda i: 1.0 if (i // 2) % 2 == 0 else -1.0)


def test_disabled_gate_admits_everything():
    g = DiversificationGate(returns={"A": _UP, "B": dict(_UP)}, betas={},
                            cfg=PortfolioConfig(enabled=False))
    g.accept("A")
    assert g.admits("B").ok is True


def test_correlation_cap_blocks_twin():
    g = DiversificationGate(returns={"A": _UP, "B": dict(_UP)}, betas={},
                            cfg=PortfolioConfig(max_correlation=0.80))
    assert g.admits("A").ok is True
    g.accept("A")
    adm = g.admits("B")
    assert adm.ok is False
    assert "correlation" in adm.reason and "A" in adm.reason


def test_uncorrelated_name_admitted():
    g = DiversificationGate(returns={"A": _UP, "C": _PERIOD4}, betas={},
                            cfg=PortfolioConfig(max_correlation=0.80))
    g.accept("A")
    assert g.admits("C").ok is True


def test_negative_correlation_is_not_blocked():
    hedge = _ser(lambda i: -1.0 if i % 2 == 0 else 1.0)
    g = DiversificationGate(returns={"A": _UP, "H": hedge}, betas={},
                            cfg=PortfolioConfig(max_correlation=0.80))
    g.accept("A")
    assert g.admits("H").ok is True               # corr -1.0 is a hedge, allow


def test_gate_diversifies_against_held_book():
    # No accepts yet, but "HELD" is already in the book -> a twin is blocked.
    g = DiversificationGate(returns={"HELD": _UP, "TWIN": dict(_UP)}, betas={},
                            held={"HELD"}, cfg=PortfolioConfig())
    assert g.admits("TWIN").ok is False


def test_single_beta_cap():
    g = DiversificationGate(returns={}, betas={"WILD": 2.5},
                            cfg=PortfolioConfig(max_single_beta=2.0))
    adm = g.admits("WILD")
    assert adm.ok is False
    assert "beta" in adm.reason


def test_average_beta_cap_after_grace():
    cfg = PortfolioConfig(max_single_beta=2.0, max_avg_beta=1.40, beta_grace=2)
    betas = {"X": 1.6, "Y": 1.6, "Z": 1.6}
    g = DiversificationGate(returns={}, betas=betas, cfg=cfg)
    # First two are under the single cap and within grace -> admitted.
    assert g.admits("X").ok is True
    g.accept("X")
    assert g.admits("Y").ok is True
    g.accept("Y")
    # Third would push avg beta to 1.6 > 1.40 -> blocked.
    adm = g.admits("Z")
    assert adm.ok is False
    assert "avg beta" in adm.reason


def test_missing_data_is_permissive():
    # Unknown symbol (no returns, no beta) sails through.
    g = DiversificationGate(returns={"A": _UP}, betas={"A": 1.0},
                            cfg=PortfolioConfig())
    g.accept("A")
    assert g.admits("UNKNOWN").ok is True
