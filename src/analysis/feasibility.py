"""Profit-target feasibility: 'can this stock realistically reach +X% within
N days, given its own conditions?'

This is a transparent, free, no-ML estimate. We model the daily log price as an
arithmetic Brownian motion with drift::

    X_t = mu * t + sigma * W_t

where ``sigma`` is the stock's recent daily volatility and ``mu`` is a
conservatively-damped estimate of its recent momentum (drift per day). The
probability of *touching* an upside target level ``b = ln(1 + target%)`` at any
point within the horizon ``T`` days is the classic first-passage (running-max)
formula for Brownian motion with drift:

    P(max_{0..T} X >= b) = Phi((mu*T - b)/(sigma*sqrt(T)))
                           + exp(2*mu*b/sigma^2) * Phi((-mu*T - b)/(sigma*sqrt(T)))

We report the *touch* probability (because a take-profit triggers the moment
price reaches the level, not only at the close), plus the terminal probability,
the typical 1-sigma move over the horizon, and plain-language conditions. None
of this is a guarantee -- it's a conditions-based reality check, not advice.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

_N = NormalDist()

# Momentum rarely persists fully; damp the recent pace before projecting it
# forward so the estimate is conservative rather than trend-extrapolating.
_MOMENTUM_PERSISTENCE = 0.5
# Cap |drift| at this many daily sigmas so a recent spike can't dominate.
_DRIFT_CAP_SIGMAS = 1.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _verdict(p: float) -> tuple[str, str]:
    """(label, tone) for a touch probability."""
    if p >= 0.65:
        return "Likely", "good"
    if p >= 0.45:
        return "Possible", "ok"
    if p >= 0.25:
        return "Unlikely", "warn"
    return "Very unlikely", "bad"


def target_feasibility(
    *,
    target_pct: float,
    horizon_days: float,
    daily_vol: float | None,
    last_close: float | None = None,
    mom_20d: float | None = None,
    mom_60d: float | None = None,
    atr_pct: float | None = None,
) -> dict[str, Any]:
    """Estimate whether ``+target_pct`` is reachable within ``horizon_days``.

    ``daily_vol`` is the std of 1-day log returns (e.g. feature ``vol_20d``).
    Returns a dict the UI can render directly; ``available`` is False when
    there isn't enough information to compute a sensible estimate.
    """
    if (target_pct is None or target_pct <= 0 or horizon_days is None
            or horizon_days <= 0 or daily_vol is None or daily_vol <= 0):
        return {"available": False}

    sigma = float(daily_vol)
    T = float(horizon_days)
    b = math.log1p(target_pct / 100.0)            # log-return barrier (>0)
    sqrt_t = math.sqrt(T)

    # ---- drift (per-day) from damped recent momentum --------------------
    comps: list[float] = []
    if mom_20d is not None:
        comps.append(mom_20d / 20.0)
    if mom_60d is not None:
        comps.append(mom_60d / 60.0)
    drift = (sum(comps) / len(comps)) * _MOMENTUM_PERSISTENCE if comps else 0.0
    cap = _DRIFT_CAP_SIGMAS * sigma
    drift = max(-cap, min(cap, drift))

    # ---- first-passage (touch) probability ------------------------------
    z1 = (drift * T - b) / (sigma * sqrt_t)
    z2 = (-drift * T - b) / (sigma * sqrt_t)
    # exp() guarded against overflow for large positive drift/barrier.
    expo = 2.0 * drift * b / (sigma * sigma)
    expo = min(expo, 50.0)
    p_touch = _clamp(_N.cdf(z1) + math.exp(expo) * _N.cdf(z2))

    # Terminal (close-at-horizon) probability, for context.
    p_terminal = _clamp(1.0 - _N.cdf((b - drift * T) / (sigma * sqrt_t)))

    label, tone = _verdict(p_touch)

    # ---- context numbers -------------------------------------------------
    typical_move_pct = (math.exp(sigma * sqrt_t) - 1.0) * 100.0   # ~1-sigma band
    drift_move_pct = (math.exp(drift * T) - 1.0) * 100.0
    required_daily_pct = target_pct / T
    daily_move_pct = (atr_pct * 100.0) if atr_pct else (sigma * 100.0)
    # Expected days to touch if there's positive drift (rough: b / drift).
    expected_days = (b / drift) if drift > 1e-9 else None

    notes: list[str] = []
    if drift > 0:
        notes.append(f"Recent trend adds an estimated +{drift_move_pct:.1f}% of "
                     f"drift over {int(T)} days.")
    elif drift < 0:
        notes.append(f"Recent trend is negative (~{drift_move_pct:.1f}% drift "
                     f"over {int(T)} days), working against the target.")
    else:
        notes.append("No clear trend; relying on volatility alone to reach the target.")
    notes.append(f"Typical swing over {int(T)} days is about "
                 f"\u00b1{typical_move_pct:.1f}% (1\u03c3).")
    notes.append(f"Hitting +{target_pct:.0f}% needs ~{required_daily_pct:.2f}%/day; "
                 f"the stock moves ~{daily_move_pct:.1f}%/day on average.")
    if expected_days is not None and expected_days <= T * 3:
        notes.append(f"At the current pace it could take ~{expected_days:.0f} "
                     f"trading days to reach the target.")

    out: dict[str, Any] = {
        "available": True,
        "target_pct": round(float(target_pct), 1),
        "horizon_days": int(T),
        "prob_touch": round(p_touch, 4),
        "prob_touch_pct": round(p_touch * 100.0, 1),
        "prob_terminal_pct": round(p_terminal * 100.0, 1),
        "verdict": label,
        "tone": tone,
        "typical_move_pct": round(typical_move_pct, 1),
        "drift_move_pct": round(drift_move_pct, 1),
        "required_daily_pct": round(required_daily_pct, 2),
        "daily_move_pct": round(daily_move_pct, 2),
        "expected_days": (round(expected_days) if expected_days is not None
                          and expected_days <= T * 5 else None),
        "notes": notes,
    }
    if last_close:
        out["target_price"] = round(last_close * (1.0 + target_pct / 100.0), 2)
        out["last_close"] = round(float(last_close), 2)
    return out


def touch_prob(
    *,
    target_pct: float,
    horizon_days: float,
    daily_vol: float | None,
    mom_20d: float | None = None,
    mom_60d: float | None = None,
    atr_pct: float | None = None,
) -> float:
    """Just the touch probability (0..1); 0.0 when not computable."""
    r = target_feasibility(
        target_pct=target_pct, horizon_days=horizon_days, daily_vol=daily_vol,
        mom_20d=mom_20d, mom_60d=mom_60d, atr_pct=atr_pct,
    )
    return float(r.get("prob_touch", 0.0)) if r.get("available") else 0.0


def feasible_target_pct(
    *,
    min_prob: float,
    max_target_pct: float,
    floor_pct: float,
    horizon_days: float,
    daily_vol: float | None,
    mom_20d: float | None = None,
    mom_60d: float | None = None,
    atr_pct: float | None = None,
    iters: int = 18,
) -> float | None:
    """Largest target (between ``floor_pct`` and ``max_target_pct``) whose
    touch probability over ``horizon_days`` is at least ``min_prob``.

    Returns ``None`` when even the floor target can't clear the bar -- i.e. the
    stock is too sluggish to plausibly reach the minimum profit in the window,
    so the auto-trader should pass. Touch probability is monotonically
    decreasing in the target, so a binary search finds the best level.
    """
    if daily_vol is None or daily_vol <= 0 or horizon_days <= 0:
        return None
    hi = max(floor_pct, float(max_target_pct))
    lo = float(floor_pct)

    def _p(tp: float) -> float:
        return touch_prob(target_pct=tp, horizon_days=horizon_days,
                          daily_vol=daily_vol, mom_20d=mom_20d,
                          mom_60d=mom_60d, atr_pct=atr_pct)

    if _p(hi) >= min_prob:        # full conviction target already feasible
        return round(hi, 1)
    if _p(lo) < min_prob:         # even the floor is out of reach -> skip
        return None
    # Search for the largest feasible target in (lo, hi).
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _p(mid) >= min_prob:
            lo = mid
        else:
            hi = mid
    return round(lo, 1)
