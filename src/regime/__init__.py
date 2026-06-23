"""Market-regime engine: classify the market, then route to a strategy.

Phase 1A: a transparent, rule-based regime classifier (NIFTY trend + India VIX
+ market breadth) with hysteresis to prevent daily whipsaw, plus a strategy
router that turns a regime label into a concrete trading plan. An HMM upgrade
can come later only if it beats these rules out-of-sample.
"""

from __future__ import annotations

# Regime labels -- imported widely, so define them in one place.
BULL_TREND = "BULL_TREND"
BEAR_TREND = "BEAR_TREND"
RANGE = "RANGE"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
CRISIS = "CRISIS"

ALL_REGIMES = (BULL_TREND, BEAR_TREND, RANGE, HIGH_VOLATILITY, CRISIS)

__all__ = [
    "BULL_TREND", "BEAR_TREND", "RANGE", "HIGH_VOLATILITY", "CRISIS",
    "ALL_REGIMES",
]
