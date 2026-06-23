"""Multi-horizon price-target projection (transparent, no-ML).

For a stock we project an *expected* price and a probability band at a set of
horizons -- 1 week, 1 month, 3/6 months, 1 year, and a multi-year "long run" --
using the same drift + volatility model as :mod:`src.analysis.feasibility`:

    expected log-return over T days  R(T) = mom_contrib(T) + mu_long * T
    expected price                   close * exp(R(T))
    1-sigma band                     close * exp(R(T) +/- z * sigma * sqrt(T))

where ``sigma`` is the stock's recent daily log-return volatility (feature
``vol_20d``) and the drift has two parts:

  * **Momentum** -- a conservatively damped/capped daily drift from recent
    momentum (``mom_20d`` / ``mom_60d``), exactly as in the feasibility model.
    Crucially its *cumulative* contribution **saturates** (recent trend can't be
    extrapolated linearly for years) and is hard-capped, so a 2-month spike
    never implies a 10x over 3 years.
  * **Long-run drift** -- a modest baseline equity log-drift per day that
    dominates the long horizons, so multi-year targets compound at a sane rate
    rather than tracking short-term momentum.

The band widens with ``sqrt(T)`` -- honest about how uncertain a 1-year or
multi-year single-stock target is. This is a conditions-based projection, not a
guarantee or advice.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

_N = NormalDist()

# (label, trading days). ~21 trading days per month, ~252 per year.
HORIZONS: tuple[tuple[str, int], ...] = (
    ("1W", 5),
    ("1M", 21),
    ("3M", 63),
    ("6M", 126),
    ("1Y", 252),
    ("3Y", 756),        # "long run"
)

# Momentum drift handling (mirrors feasibility, plus a saturation + total cap).
_MOMENTUM_PERSISTENCE = 0.5         # recent pace rarely persists fully
_DRIFT_CAP_SIGMAS = 1.0             # cap |daily momentum drift| at 1 daily sigma
_MOM_TAU_DAYS = 60.0                # momentum effect saturates over ~this window
_MOM_CONTRIB_CAP = 0.40             # |cumulative momentum log-return| <= 0.40 (~49%)

# Conservative long-run equity drift (nominal). Indian large-caps have
# historically compounded ~11%/yr; we use that as the baseline that takes over
# at long horizons. Override via ``long_run_annual`` if you have a better prior.
_DEFAULT_LONG_RUN_ANNUAL = 0.11
_TRADING_DAYS_YEAR = 252

# Band width: 1 sigma ~ 68% interval.
_BAND_Z = 1.0


def _daily_momentum_drift(mom_20d: float | None, mom_60d: float | None,
                          sigma: float) -> float:
    """Damped, sigma-capped per-day momentum drift (same recipe as feasibility)."""
    comps: list[float] = []
    if mom_20d is not None:
        comps.append(mom_20d / 20.0)
    if mom_60d is not None:
        comps.append(mom_60d / 60.0)
    drift = (sum(comps) / len(comps)) * _MOMENTUM_PERSISTENCE if comps else 0.0
    cap = _DRIFT_CAP_SIGMAS * sigma
    return max(-cap, min(cap, drift))


def _momentum_contribution(drift_daily: float, days: float) -> float:
    """Cumulative momentum log-return over ``days`` with saturation + a hard cap.

    Linear (~drift*days) for short horizons, saturating to ``drift*tau`` as the
    horizon grows, then clamped to +/- ``_MOM_CONTRIB_CAP`` so recent momentum
    can never dominate a multi-year projection.
    """
    raw = drift_daily * _MOM_TAU_DAYS * (1.0 - math.exp(-days / _MOM_TAU_DAYS))
    return max(-_MOM_CONTRIB_CAP, min(_MOM_CONTRIB_CAP, raw))


def _verdict(expected_return_pct: float, prob_up: float) -> tuple[str, str]:
    """(label, tone) summarising the horizon outlook."""
    if prob_up >= 0.60 and expected_return_pct > 0:
        return "Bullish", "good"
    if prob_up >= 0.50:
        return "Mildly bullish", "ok"
    if prob_up >= 0.40:
        return "Neutral", "warn"
    return "Bearish", "bad"


def project_horizon(
    *,
    last_close: float,
    sigma: float,
    drift_mom_daily: float,
    days: int,
    long_run_daily: float,
    label: str = "",
) -> dict[str, Any]:
    """Expected price + 1-sigma band + up-probability at one horizon."""
    T = float(days)
    sqrt_t = math.sqrt(T)
    mom = _momentum_contribution(drift_mom_daily, T)
    r = mom + long_run_daily * T                       # expected log-return
    band = _BAND_Z * sigma * sqrt_t

    expected = last_close * math.exp(r)
    low = last_close * math.exp(r - band)
    high = last_close * math.exp(r + band)

    exp_ret_pct = (math.exp(r) - 1.0) * 100.0
    annualized_pct = (math.exp(r * _TRADING_DAYS_YEAR / T) - 1.0) * 100.0
    # Terminal probability of finishing above today's price.
    prob_up = _N.cdf(r / band) if band > 0 else (1.0 if r > 0 else 0.0)
    label_txt, tone = _verdict(exp_ret_pct, prob_up)

    return {
        "label": label,
        "horizon_days": int(days),
        "expected_price": round(expected, 2),
        "low_price": round(low, 2),
        "high_price": round(high, 2),
        "expected_return_pct": round(exp_ret_pct, 1),
        "annualized_return_pct": round(annualized_pct, 1),
        "band_pct": round((math.exp(band) - 1.0) * 100.0, 1),
        "prob_up_pct": round(prob_up * 100.0, 1),
        "verdict": label_txt,
        "tone": tone,
    }


def forecast_stock(
    *,
    last_close: float | None,
    daily_vol: float | None,
    mom_20d: float | None = None,
    mom_60d: float | None = None,
    horizons: tuple[tuple[str, int], ...] = HORIZONS,
    long_run_annual: float = _DEFAULT_LONG_RUN_ANNUAL,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Project price targets across ``horizons`` for one stock.

    ``daily_vol`` is the std of 1-day log returns (feature ``vol_20d``).
    Returns ``{"available": False}`` when there isn't enough to project.
    """
    if last_close is None or last_close <= 0 or daily_vol is None or daily_vol <= 0:
        return {"available": False, "symbol": symbol}

    sigma = float(daily_vol)
    drift_mom_daily = _daily_momentum_drift(mom_20d, mom_60d, sigma)
    long_run_daily = math.log1p(long_run_annual) / _TRADING_DAYS_YEAR

    rows = [
        project_horizon(
            last_close=float(last_close), sigma=sigma,
            drift_mom_daily=drift_mom_daily, days=days,
            long_run_daily=long_run_daily, label=label,
        )
        for label, days in horizons
    ]
    return {
        "available": True,
        "symbol": symbol,
        "last_close": round(float(last_close), 2),
        "daily_vol_pct": round(sigma * 100.0, 2),
        "momentum_drift_daily_pct": round(drift_mom_daily * 100.0, 3),
        "horizons": rows,
    }
