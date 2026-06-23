"""Return-correlation math (pure: no DB, no network).

Correlations are computed on *overlapping* daily-return histories, keyed by
date, so two series with slightly different calendars (holidays, late listings)
align on their common dates rather than by position. We require a minimum
overlap before trusting a correlation -- a handful of shared days produces
noisy, near-meaningless coefficients.

Returns are expressed as a ``{date_str: ret_1d}`` mapping per symbol.
"""

from __future__ import annotations

import math

ReturnSeries = dict[str, float]


def pearson(a: ReturnSeries, b: ReturnSeries, min_overlap: int = 40) -> float | None:
    """Pearson correlation of two return series over their common dates.

    Returns ``None`` when the overlap is below ``min_overlap`` or either series
    is constant over the overlap (zero variance -> correlation undefined).
    """
    common = a.keys() & b.keys()
    if len(common) < min_overlap:
        return None

    xs = [a[d] for d in common]
    ys = [b[d] for d in common]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n

    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None

    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    c = sxy / math.sqrt(sxx * syy)
    # Guard against tiny FP overshoot beyond [-1, 1].
    return max(-1.0, min(1.0, c))


def max_corr_to(
    sym_returns: ReturnSeries,
    others: dict[str, ReturnSeries],
    min_overlap: int = 40,
) -> tuple[float | None, str | None]:
    """Highest *positive* correlation of ``sym_returns`` to any series in
    ``others``, with the peer symbol that produced it.

    We track the signed maximum (not absolute) on purpose: two names moving
    together is the concentration risk we want to cap; a strongly *negative*
    correlation is a hedge and should never block a pick. Returns
    ``(None, None)`` when nothing has sufficient overlap.
    """
    best: float | None = None
    best_sym: str | None = None
    for osym, oret in others.items():
        c = pearson(sym_returns, oret, min_overlap)
        if c is None:
            continue
        if best is None or c > best:
            best = c
            best_sym = osym
    return best, best_sym
