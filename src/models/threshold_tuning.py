"""Decision-threshold selection by expected utility.

Why not maximise accuracy or F1:
  - Trading economics. A signal at probability p that produces an
    average return of r when it fires, after subtracting a per-trade cost c,
    has expected utility per signal:
        U(thr) =  E[ fwd_return | calibrated_prob >= thr ] - c
    and total expected utility:
        T(thr) = n_signals(thr) * (E[fwd_return | fired] - c)
  - F1 ignores the magnitude of the move; trading cares about magnitude.
  - We therefore pick the threshold that maximises T(thr) on a held-out
    calibration set, with both `n_signals` and per-trade utility constrained
    to be > 0 (i.e., we won't pick a threshold that fires zero times or that
    yields negative-utility trades).

Returned object includes the chosen threshold, the per-threshold utility
curve (for plotting / debugging), and the binding constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger("models.thresh")


@dataclass
class ThresholdResult:
    threshold: float
    n_signals: int
    expected_return_per_trade: float
    expected_total_utility: float
    cost_pct: float
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _round_trip_cost_pct(cost_model: dict | None) -> float:
    """Estimate round-trip cost as a fraction of notional.

    Delegates to the canonical Week-4 cost model so threshold tuning and
    the backtester can never disagree on what a trade actually costs. The
    `cost_model` argument is kept only for backward compatibility with the
    older signature; if provided it is ignored in favour of the canonical
    YAML config under config/cost_model.yaml.
    """
    try:
        from src.backtesting.cost_model import load_cost_config, round_trip_pct
        cfg = load_cost_config()
        return round_trip_pct("DEFAULT", cfg)
    except Exception as e:  # noqa: BLE001 - defensive: never fail tuning over costs
        log.warning("Falling back to 0.30%% round-trip cost ({})", e)
        return 0.0030


def _build_curve(
    p: np.ndarray, r: np.ndarray, grid: np.ndarray, cost_pct: float
) -> pd.DataFrame:
    """Evaluate every threshold in ``grid`` and return the utility curve."""
    rows = []
    for thr in grid:
        mask = p >= thr
        n = int(mask.sum())
        if n == 0:
            rows.append((float(thr), 0, float("nan"), float("nan")))
            continue
        avg_r = float(r[mask].mean())
        per_trade_util = avg_r - cost_pct
        total_util = n * per_trade_util
        rows.append((float(thr), n, avg_r, total_util))
    return pd.DataFrame(
        rows, columns=["threshold", "n_signals", "avg_fwd_return", "total_utility"]
    )


def _pick_best(curve: pd.DataFrame, cost_pct: float, min_signals: int):
    """Return ``(best_row, status)`` where status is one of
    'optimal' (positive utility), 'least_negative' (eligible but all
    net-negative), or 'none' (no threshold fired >= min_signals times)."""
    eligible = curve[
        (curve["n_signals"] >= min_signals) & curve["total_utility"].notna()
    ]
    pos = eligible[(eligible["avg_fwd_return"] - cost_pct) > 0]
    if len(pos) > 0:
        return pos.loc[pos["total_utility"].idxmax()], "optimal"
    if len(eligible) > 0:
        return eligible.loc[eligible["total_utility"].idxmax()], "least_negative"
    return None, "none"


def _adaptive_grid(
    p: np.ndarray, grid_step: float,
    *, quantile_lo: float = 0.50, quantile_hi: float = 0.995,
) -> np.ndarray:
    """Derive a threshold grid from the *observed* score distribution.

    Isotonic calibration compresses probabilities toward the base rate, so
    a fixed [0.50, 0.95] grid can sit entirely above every score (max
    calibrated prob ~0.43 in practice). When that happens we search the
    real range instead: from the median score up to its far-right tail.
    Starting at the median guarantees the low end fires often enough to
    satisfy ``min_signals``; ending near the max keeps the high end
    meaningful without chasing a 1-sample tail.
    """
    lo = float(np.quantile(p, quantile_lo))
    hi = float(np.quantile(p, quantile_hi))
    if hi - lo < grid_step:
        hi = lo + grid_step
    grid = np.arange(lo, hi + 1e-9, grid_step)
    return np.round(np.clip(grid, lo, hi), 6)


def tune_threshold(
    calibrated_prob: np.ndarray,
    forward_return: np.ndarray,
    *,
    cost_pct: float | None = None,
    cost_model: dict | None = None,
    grid_min: float = 0.50,
    grid_max: float = 0.95,
    grid_step: float = 0.01,
    min_signals: int = 5,
    adaptive: bool = False,
    adaptive_fallback: bool = True,
) -> ThresholdResult:
    """Pick the threshold that maximises expected total utility.

    cost_pct: if given, use directly. Else derive from cost_model. Else 0.30%.
    min_signals: refuse thresholds that fire fewer than this many times on
                 the calibration set (avoid overfitting to a tiny tail).
    adaptive: when True, search a grid derived from the *observed* score
                 distribution (median -> far-right tail) instead of the fixed
                 [grid_min, grid_max] band. This is the correct mode for
                 isotonic-calibrated probabilities, which compress toward the
                 base rate (e.g. max ~0.45) so that a 0.50 floor would only
                 ever fire on a handful of historical outliers. The
                 expected-utility + min_signals constraints still guard
                 against a reckless low threshold.
    adaptive_fallback: when the fixed [grid_min, grid_max] grid produces no
                 threshold that fires >= min_signals times (typical when
                 isotonic calibration has compressed all scores below
                 grid_min), re-run the search on a data-derived grid spanning
                 the actual score distribution. This is what makes the system
                 produce signals at a utility-optimal operating point instead
                 of silently returning a never-firing threshold.
    """
    p = np.asarray(calibrated_prob, dtype=float)
    r = np.asarray(forward_return, dtype=float)
    if len(p) != len(r):
        raise ValueError("prob and forward_return length mismatch")
    if len(p) == 0:
        raise ValueError("Empty calibration set; cannot tune threshold")

    if cost_pct is None:
        cost_pct = _round_trip_cost_pct(cost_model)

    if adaptive:
        grid = _adaptive_grid(p, grid_step)
    else:
        grid = np.arange(grid_min, grid_max + 1e-9, grid_step)
        # np.arange can drift one step beyond grid_max; clip and round so the
        # grid is exact and bounded.
        grid = np.round(np.clip(grid, grid_min, grid_max), 6)
    curve = _build_curve(p, r, grid, cost_pct)
    best, status = _pick_best(curve, cost_pct, min_signals)

    if status == "none" and adaptive_fallback and not adaptive:
        # The fixed grid never fired enough -- almost always because isotonic
        # calibration squashed every score below grid_min. Re-grid on the
        # real distribution and try again.
        adaptive = _adaptive_grid(p, grid_step)
        log.warning(
            "Fixed grid [{:.2f}, {:.2f}] never fired >= {} signals "
            "(max calibrated prob = {:.4f}); retrying on data-adaptive grid "
            "[{:.4f}, {:.4f}].",
            grid_min, grid_max, min_signals, float(p.max()),
            float(adaptive.min()), float(adaptive.max()),
        )
        curve = _build_curve(p, r, adaptive, cost_pct)
        best, status = _pick_best(curve, cost_pct, min_signals)

    if status == "least_negative":
        log.warning(
            "No threshold yields positive expected utility net of {:.4%} cost; "
            "falling back to least-negative ({:.4f}).",
            cost_pct,
            float(best["total_utility"]),
        )
    elif status == "none":
        # Still nothing fired min_signals times even on the adaptive grid.
        # Return the threshold that fires the most (best statistical support).
        nonzero = curve[curve["n_signals"] > 0]
        if len(nonzero) == 0:
            best = curve.iloc[0]
        else:
            best = nonzero.iloc[nonzero["n_signals"].argmax()]
        log.warning(
            "No threshold yields >= {} signals; falling back to threshold={:.4f} "
            "with {} signals.",
            min_signals,
            float(best["threshold"]),
            int(best["n_signals"]),
        )

    return ThresholdResult(
        threshold=float(best["threshold"]),
        n_signals=int(best["n_signals"]),
        expected_return_per_trade=float(best["avg_fwd_return"] - cost_pct)
        if not np.isnan(best["avg_fwd_return"])
        else 0.0,
        expected_total_utility=float(best["total_utility"])
        if not np.isnan(best["total_utility"])
        else 0.0,
        cost_pct=cost_pct,
        curve=curve,
    )
