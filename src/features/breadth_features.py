"""Market-breadth features (pure, no IO).

Breadth measures *participation*: a rally where 350/500 names are above their
50-DMA is healthy; one led by 5 names is fragile. These cross-sectional
aggregates are often better market-health signals than any single-stock RSI,
and they feed the regime classifier (``src/regime/classifier.py``).

Every function here is pure: it takes the day's universe rows (one dict per
symbol, already the latest feature row <= as_of) and returns a plain dict. All
IO -- pulling the cross-section out of ``feature_data`` -- lives in
``src/regime/store.py``, mirroring how ``feature_builder`` wraps the pure math
in ``statistical_features`` / ``technical_indicators``.

Expected per-row keys (all optional; missing/None are skipped per metric):
    dist_ema_50_pct   stock's % distance from its 50-EMA   (>0 == above)
    dist_ema_200_pct  stock's % distance from its 200-EMA  (>0 == above)
    dd_from_high_252d  drawdown from the 1y high (<=0; ~0 == at new high)
    ret_1d            today's 1-day return (sign -> advancer/decliner)
"""

from __future__ import annotations

from typing import Any

# A name within this fraction of its 252-day high counts as a "new high".
_NEW_HIGH_TOL = 0.005


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pct_positive(rows: list[dict[str, Any]], key: str) -> tuple[float, int]:
    """(% of rows where row[key] > 0, count of usable rows). 0.0 if none."""
    vals = [v for r in rows if (v := _f(r.get(key))) is not None]
    if not vals:
        return 0.0, 0
    pos = sum(1 for v in vals if v > 0.0)
    return 100.0 * pos / len(vals), len(vals)


def compute_breadth(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one day's universe cross-section into breadth metrics.

    Returns a dict with ``available`` False when there's nothing to measure, so
    the regime classifier can fall back gracefully rather than treat an empty
    universe as a breadth collapse.
    """
    n = len(rows)
    if n == 0:
        return {
            "available": False,
            "universe_count": 0,
            "pct_above_50dma": None,
            "pct_above_200dma": None,
            "adv_decl_ratio": None,
            "new_high_pct": None,
            "breadth_score": None,
        }

    pct_above_50dma, _ = _pct_positive(rows, "dist_ema_50_pct")
    pct_above_200dma, _ = _pct_positive(rows, "dist_ema_200_pct")

    rets = [v for r in rows if (v := _f(r.get("ret_1d"))) is not None]
    advancers = sum(1 for v in rets if v > 0.0)
    decliners = sum(1 for v in rets if v < 0.0)
    adv_decl_ratio = advancers / max(1, decliners)

    dds = [v for r in rows if (v := _f(r.get("dd_from_high_252d"))) is not None]
    new_high_pct = (
        100.0 * sum(1 for v in dds if v >= -_NEW_HIGH_TOL) / len(dds)
        if dds else 0.0
    )

    # Composite 0..100: equal weight on short- and long-term participation.
    # Deliberately simple and explainable; the classifier applies the bands.
    breadth_score = round(0.5 * pct_above_50dma + 0.5 * pct_above_200dma, 1)

    return {
        "available": True,
        "universe_count": n,
        "pct_above_50dma": round(pct_above_50dma, 1),
        "pct_above_200dma": round(pct_above_200dma, 1),
        "adv_decl_ratio": round(adv_decl_ratio, 3),
        "new_high_pct": round(new_high_pct, 1),
        "breadth_score": breadth_score,
    }
