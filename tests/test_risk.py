"""Unit tests for the risk-management engine (pure functions)."""

from __future__ import annotations

from src.analysis import risk


# ---- position sizing ------------------------------------------------------


def test_position_size_standard_case() -> None:
    r = risk.position_size(100_000, 2, 1000, 950, target=1150)
    assert r["valid"] is True
    assert r["shares"] == 40                 # 2000 budget / 50 risk/share
    assert r["risk_amount"] == 2000.0
    assert r["risk_pct_actual"] == 2.0
    assert r["position_value"] == 40_000.0
    assert r["risk_reward"] == 3.0           # (1150-1000)/(1000-950)


def test_position_size_rejects_stop_above_entry() -> None:
    r = risk.position_size(100_000, 2, 1000, 1050)
    assert r["valid"] is False
    assert any("below the entry" in n for n in r["notes"])


def test_position_size_capped_by_capital_on_tight_stop() -> None:
    r = risk.position_size(100_000, 2, 1000, 998)
    assert r["valid"] is True
    assert r["shares"] == 100                # capital cap: 100000/1000
    assert r["risk_pct_actual"] < 2.0        # full budget not deployed
    assert any("tight" in n.lower() for n in r["notes"])


def test_position_size_lot_rounding() -> None:
    r = risk.position_size(100_000, 2, 1000, 950, lot_size=25)
    assert r["shares"] % 25 == 0
    assert r["shares"] == 25                  # 40 floored to nearest 25


def test_position_size_missing_inputs_is_invalid() -> None:
    assert risk.position_size(0, 2, 1000, 950)["valid"] is False


# ---- portfolio health -----------------------------------------------------


def test_portfolio_health_empty() -> None:
    assert risk.portfolio_health([]) == {"has_positions": False}


def test_portfolio_health_flags_single_name_concentration() -> None:
    positions = [{"symbol": "TCS", "sector": "IT", "qty": 100, "price": 4000}]
    h = risk.portfolio_health(positions, {"TCS": 0.02})
    assert h["has_positions"] is True
    assert h["n_positions"] == 1
    assert h["max_position"]["weight_pct"] == 100.0
    # Single name + one sector + too few names -> low diversification.
    assert h["diversification_score"] < 50
    texts = " ".join(w["text"] for w in h["warnings"]).lower()
    assert "concentration" in texts or "diversification" in texts
    assert h["var_95_1d_rupees"] > 0


def test_portfolio_health_balanced_scores_well() -> None:
    positions = [
        {"symbol": "TCS", "sector": "IT", "qty": 10, "price": 4000},      # 40k
        {"symbol": "HDFCBANK", "sector": "BANK", "qty": 20, "price": 1500},  # 30k
        {"symbol": "SUNPHARMA", "sector": "PHARMA", "qty": 30, "price": 1000},  # 30k
        {"symbol": "ITC", "sector": "FMCG", "qty": 80, "price": 400},      # 32k
        {"symbol": "MARUTI", "sector": "AUTO", "qty": 3, "price": 11000},  # 33k
    ]
    vols = {p["symbol"]: 0.015 for p in positions}
    h = risk.portfolio_health(positions, vols)
    assert h["n_positions"] == 5
    assert h["max_position"]["weight_pct"] <= 30
    assert h["top_sector"]["weight_pct"] <= 40
    assert h["diversification_score"] >= 80
    # Sectors sum to ~100%.
    assert abs(sum(s["weight_pct"] for s in h["sectors"]) - 100.0) < 1.0


def test_portfolio_var_scales_with_volatility() -> None:
    pos = [{"symbol": "X", "sector": "Z", "qty": 10, "price": 1000}]  # 10k
    low = risk.portfolio_health(pos, {"X": 0.01})["var_95_1d_rupees"]
    high = risk.portfolio_health(pos, {"X": 0.04})["var_95_1d_rupees"]
    assert high > low
