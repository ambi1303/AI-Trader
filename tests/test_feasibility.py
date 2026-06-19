"""Unit tests for the profit-target feasibility model (pure, no DB/network)."""

from __future__ import annotations

from src.analysis.feasibility import (
    feasible_target_pct,
    target_feasibility,
    touch_prob,
)


def _p(**kw) -> float:
    base = {"target_pct": 10.0, "horizon_days": 30, "daily_vol": 0.02}
    base.update(kw)
    return target_feasibility(**base)["prob_touch"]


def test_unavailable_without_volatility() -> None:
    assert target_feasibility(target_pct=10, horizon_days=30,
                              daily_vol=None)["available"] is False
    assert target_feasibility(target_pct=10, horizon_days=30,
                              daily_vol=0.0)["available"] is False


def test_unavailable_for_nonpositive_target_or_time() -> None:
    assert target_feasibility(target_pct=0, horizon_days=30,
                              daily_vol=0.02)["available"] is False
    assert target_feasibility(target_pct=10, horizon_days=0,
                              daily_vol=0.02)["available"] is False


def test_probability_is_bounded() -> None:
    for t in (3, 10, 25, 40):
        for d in (5, 30, 90):
            p = _p(target_pct=t, horizon_days=d)
            assert 0.0 <= p <= 1.0


def test_more_time_raises_touch_probability() -> None:
    assert _p(horizon_days=10) < _p(horizon_days=30) < _p(horizon_days=90)


def test_bigger_target_lowers_probability() -> None:
    assert _p(target_pct=5) > _p(target_pct=15) > _p(target_pct=30)


def test_higher_volatility_raises_probability() -> None:
    assert _p(daily_vol=0.01) < _p(daily_vol=0.02) < _p(daily_vol=0.04)


def test_positive_trend_helps_vs_negative() -> None:
    up = _p(mom_20d=0.10, mom_60d=0.25)
    flat = _p()
    down = _p(mom_20d=-0.10, mom_60d=-0.25)
    assert up > flat > down


def test_result_shape_and_verdict() -> None:
    r = target_feasibility(target_pct=10, horizon_days=30, daily_vol=0.025,
                           last_close=100.0, mom_20d=0.05, mom_60d=0.12,
                           atr_pct=0.025)
    assert r["available"] is True
    assert r["verdict"] in ("Likely", "Possible", "Unlikely", "Very unlikely")
    assert r["tone"] in ("good", "ok", "warn", "bad")
    assert r["target_price"] == 110.0
    assert isinstance(r["notes"], list) and r["notes"]
    assert 0.0 <= r["prob_touch_pct"] <= 100.0


def test_touch_probability_at_least_terminal() -> None:
    # Reaching the level *any time* must be >= reaching it only at the close.
    r = target_feasibility(target_pct=12, horizon_days=40, daily_vol=0.02,
                           mom_20d=0.03, mom_60d=0.06)
    assert r["prob_touch_pct"] >= r["prob_terminal_pct"] - 1e-6


# ---------------------------------------------------------------------------
# Auto-trader gating helpers
# ---------------------------------------------------------------------------


def test_feasible_target_returns_full_target_when_easy() -> None:
    # High vol + long window -> even +10% is comfortably reachable, so the
    # full conviction target is kept.
    v = feasible_target_pct(min_prob=0.50, max_target_pct=10.0, floor_pct=5.0,
                            horizon_days=90, daily_vol=0.05)
    assert v == 10.0


def test_feasible_target_none_when_floor_unreachable() -> None:
    # Tiny vol, no trend, short window -> can't even touch +5% -> skip (None).
    v = feasible_target_pct(min_prob=0.50, max_target_pct=20.0, floor_pct=5.0,
                            horizon_days=20, daily_vol=0.005)
    assert v is None


def test_feasible_target_trims_to_feasible_level() -> None:
    # Conviction target (+30%) is too rich, but a smaller target clears the
    # bar -> returns a value between the floor and the max whose touch
    # probability is right around the threshold.
    v = feasible_target_pct(min_prob=0.50, max_target_pct=30.0, floor_pct=5.0,
                            horizon_days=60, daily_vol=0.02)
    assert v is not None
    assert 5.0 <= v <= 30.0
    p = touch_prob(target_pct=v, horizon_days=60, daily_vol=0.02)
    assert p >= 0.48                              # ~ min_prob (search tolerance)


def test_touch_prob_zero_when_vol_missing() -> None:
    assert touch_prob(target_pct=5, horizon_days=30, daily_vol=None) == 0.0
