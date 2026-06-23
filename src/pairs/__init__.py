"""Pairs trading: find cointegrated pairs and signal on the spread.

Pipeline (per the classic statistical-arbitrage recipe):

    Universe -> Cointegrated Pairs -> Spread -> Z-Score -> Trade Signal

* :mod:`src.pairs.cointegration` -- pure stats: OLS hedge ratio, an
  Augmented Dickey-Fuller t-statistic, mean-reversion half-life, and an
  Engle-Granger two-step cointegration screen.
* :mod:`src.pairs.spread`        -- pure: standardise the spread (z-score) and
  map it to a desired action on the spread.
* :mod:`src.pairs.scan`          -- DB IO: pull daily closes from ``price_data``,
  scan sector-grouped candidate pairs, and persist the cointegrated set + their
  current signal to the ``pairs`` table.

Scope note: this is **research / signal generation only**. Executing a pair
needs a short leg, and the paper trader is LONG-only in v1 -- so these signals
are surfaced for analysis, not auto-traded, until short support lands.

The cointegration/spread math is pure (NumPy only, no DB) so it's fully
unit-testable in isolation.
"""

from __future__ import annotations

__all__ = ["cointegration", "spread", "scan"]
