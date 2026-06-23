"""Strategy router: turn a regime label into a concrete trading plan (pure).

Instead of running one model in all conditions, we pick *what to do* based on
the regime:

    Regime            Plan
    ----------------  --------------------------------------------------------
    BULL_TREND        momentum (default factor blend), full book
    HIGH_VOLATILITY   breakout, conservative book (fewer holdings, higher bar)
    RANGE             mean-reversion (buy oversold dips in healthy names)
    BEAR_TREND        defensive: manage exits only, no new entries
    CRISIS            defensive: manage exits only, no new entries
    None              fail-open to momentum (regime step skipped/unavailable)

This reuses the existing ``StrategyConfig`` + ``generate_strategy_signals``
machinery: the router chooses the engine (scorer), the config, and whether new
entries are allowed -- it does not re-implement signal logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.regime import (
    BEAR_TREND,
    BULL_TREND,
    CRISIS,
    HIGH_VOLATILITY,
    RANGE,
)
from src.signals.strategy import StrategyConfig


@dataclass(frozen=True)
class StrategyPlan:
    engine: str                 # momentum | mean_reversion | breakout | defensive
    config: StrategyConfig
    allow_new_entries: bool
    note: str = ""


def _high_vol_config() -> StrategyConfig:
    """Trade smaller and pickier when volatility is elevated: fewer slots, a
    higher score bar, tighter sector cap. The reconciler still trails/exits
    open names with the shared risk config."""
    return StrategyConfig(
        target_holdings=12,
        max_per_sector=3,
        min_score=65.0,
    )


def select_strategy(regime: str | None) -> StrategyPlan:
    """Map a regime label to a trading plan. ``None`` fails open to momentum so
    a skipped/unavailable regime step never silently halts the book."""
    if regime in (CRISIS, BEAR_TREND):
        return StrategyPlan(
            engine="defensive",
            config=StrategyConfig(),
            allow_new_entries=False,
            note=f"{regime}: defensive -- manage exits only, no new entries",
        )
    if regime == HIGH_VOLATILITY:
        return StrategyPlan(
            engine="breakout",
            config=_high_vol_config(),
            allow_new_entries=True,
            note="HIGH_VOLATILITY: breakout, reduced book + higher score bar",
        )
    if regime == RANGE:
        return StrategyPlan(
            engine="mean_reversion",
            config=StrategyConfig(),
            allow_new_entries=True,
            note="RANGE: mean-reversion (buy oversold dips in healthy names)",
        )
    # BULL_TREND or None -> full momentum book.
    return StrategyPlan(
        engine="momentum",
        config=StrategyConfig(),
        allow_new_entries=True,
        note=("BULL_TREND: momentum, full book" if regime == BULL_TREND
              else "regime unavailable: fail-open to momentum"),
    )
