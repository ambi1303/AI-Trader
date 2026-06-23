"""Portfolio construction: turn a ranked list of single-name ideas into a
*diversified book*.

The scorer/router answers "is this a good trade?" one name at a time. That's
not enough: three names can each score 95 and all be the same IT bet, so naively
filling slots top-down quietly concentrates risk. This package adds the
portfolio-level view that professional systems run *between* signal ranking and
position sizing:

* :mod:`src.portfolio.correlation` -- pure return-correlation math.
* :mod:`src.portfolio.construct`   -- a greedy diversification gate that caps
  pairwise correlation and portfolio beta as slots are filled.
* :mod:`src.portfolio.store`       -- DB IO: trailing daily returns + beta from
  ``feature_data``.

Everything in ``correlation`` and ``construct`` is pure (no DB / network) so the
selection logic is unit-testable in isolation.
"""

from __future__ import annotations

from src.portfolio.construct import DiversificationGate, PortfolioConfig

__all__ = ["DiversificationGate", "PortfolioConfig"]
