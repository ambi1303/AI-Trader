"""Greedy diversification gate (pure: no DB, no network).

Given candidates already ranked best-first, we fill slots top-down but *admit*
each name only if it keeps the book diversified:

* **Correlation cap** -- reject a candidate whose trailing-return correlation to
  any already-accepted name (or any name already held) exceeds
  ``max_correlation``. This is what stops "Stock A/B/C all = IT" from becoming
  one giant correlated bet, even when each scores highly on its own.

* **Beta budget** -- reject a name whose own beta is implausibly high
  (``max_single_beta``), or one that would push the equal-weight average beta of
  the *new* entries above ``max_avg_beta`` (only once a couple of names are in,
  so the first pick is never blocked for being a bit racy).

The gate is seeded with the symbols already held, so new entries are diversified
against the existing book, not just against each other. Equal-weight is used as
a transparent proxy for portfolio beta -- final rupee weights come from the
sizer downstream, but equal-weight keeps the rule explainable and stable.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.portfolio.correlation import ReturnSeries, max_corr_to


@dataclass(frozen=True)
class PortfolioConfig:
    """Knobs for the diversification gate."""
    enabled: bool = True
    max_correlation: float = 0.80   # cap pairwise trailing-return correlation
    min_overlap: int = 40           # min shared days before trusting a corr
    max_single_beta: float = 2.0    # reject lone hyper-beta names
    max_avg_beta: float = 1.40      # cap equal-weight avg beta of new entries
    beta_grace: int = 2             # don't enforce avg-beta until N names in


@dataclass
class Admission:
    ok: bool
    reason: str = ""


class DiversificationGate:
    """Stateful greedy gate. Call :meth:`admits` before taking a name, then
    :meth:`accept` once it's actually added to the book.

    ``returns``: symbol -> {date: ret_1d}. ``betas``: symbol -> beta (may be
    partial; missing entries simply skip the beta checks). ``held``: symbols
    already in the book (seed the correlation peer set).
    """

    def __init__(
        self,
        returns: dict[str, ReturnSeries] | None,
        betas: dict[str, float] | None,
        held: set[str] | list[str] | None = None,
        cfg: PortfolioConfig | None = None,
    ) -> None:
        self.returns: dict[str, ReturnSeries] = returns or {}
        self.betas: dict[str, float] = betas or {}
        self.cfg = cfg or PortfolioConfig()
        held = set(held or ())
        # Peers we diversify *against* start as the held book.
        self._accepted: set[str] = set(held)
        self._held: set[str] = set(held)
        self._new_betas: list[float] = []

    def admits(self, symbol: str, sector: str | None = None) -> Admission:
        cfg = self.cfg
        if not cfg.enabled:
            return Admission(True)

        # ---- correlation cap ----
        sym_ret = self.returns.get(symbol)
        if sym_ret:
            peers = {s: self.returns[s] for s in self._accepted
                     if s != symbol and s in self.returns}
            if peers:
                c, peer = max_corr_to(sym_ret, peers, cfg.min_overlap)
                if c is not None and c > cfg.max_correlation:
                    return Admission(
                        False,
                        f"correlation {c:.2f} with {peer} exceeds "
                        f"{cfg.max_correlation:.2f}",
                    )

        # ---- beta budget ----
        b = self.betas.get(symbol)
        if b is not None:
            if b > cfg.max_single_beta:
                return Admission(
                    False, f"beta {b:.2f} exceeds {cfg.max_single_beta:.2f}")
            if len(self._new_betas) >= cfg.beta_grace:
                prospective = (sum(self._new_betas) + b) / (len(self._new_betas) + 1)
                if prospective > cfg.max_avg_beta:
                    return Admission(
                        False,
                        f"avg beta would rise to {prospective:.2f} (cap "
                        f"{cfg.max_avg_beta:.2f})",
                    )

        return Admission(True)

    def accept(self, symbol: str) -> None:
        """Record ``symbol`` as taken so later candidates diversify against it."""
        self._accepted.add(symbol)
        b = self.betas.get(symbol)
        if b is not None and symbol not in self._held:
            self._new_betas.append(b)

    @property
    def accepted_new(self) -> int:
        """How many *new* entries (excluding the seeded held book) accepted."""
        return len(self._accepted) - len(self._held & self._accepted)
