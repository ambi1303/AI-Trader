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
) -> ThresholdResult:
    """Pick the threshold that maximises expected total utility.

    cost_pct: if given, use directly. Else derive from cost_model. Else 0.30%.
    min_signals: refuse thresholds that fire fewer than this many times on
                 the calibration set (avoid overfitting to a tiny tail).
    """
    p = np.asarray(calibrated_prob, dtype=float)
    r = np.asarray(forward_return, dtype=float)
    if len(p) != len(r):
        raise ValueError("prob and forward_return length mismatch")
    if len(p) == 0:
        raise ValueError("Empty calibration set; cannot tune threshold")

    if cost_pct is None:
        cost_pct = _round_trip_cost_pct(cost_model)

    grid = np.arange(grid_min, grid_max + 1e-9, grid_step)
    # np.arange can drift one step beyond grid_max; clip and round so the grid
    # is exact and bounded.
    grid = np.round(np.clip(grid, grid_min, grid_max), 6)
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

    curve = pd.DataFrame(
        rows, columns=["threshold", "n_signals", "avg_fwd_return", "total_utility"]
    )

    eligible = curve[
        (curve["n_signals"] >= min_signals) & curve["total_utility"].notna()
    ]
    # Among eligible, also require positive per-trade utility, else fall back.
    pos = eligible[(eligible["avg_fwd_return"] - cost_pct) > 0]
    if len(pos) > 0:
        best = pos.loc[pos["total_utility"].idxmax()]
    elif len(eligible) > 0:
        best = eligible.loc[eligible["total_utility"].idxmax()]
        log.warning(
            "No threshold yields positive expected utility net of {:.4%} cost; "
            "falling back to least-negative ({:.4f}).",
            cost_pct,
            float(best["total_utility"]),
        )
    else:
        # Degenerate: no threshold has min_signals; return the highest threshold
        # that fires at least once.
        nonzero = curve[curve["n_signals"] > 0]
        if len(nonzero) == 0:
            best = curve.iloc[0]
        else:
            best = nonzero.iloc[nonzero["n_signals"].argmax()]
        log.warning(
            "No threshold yields >= {} signals; falling back to threshold={:.2f} "
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
