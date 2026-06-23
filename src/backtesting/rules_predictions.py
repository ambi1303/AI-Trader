"""Replay the rules strategy over history into a backtestable predictions frame.

The production rules path (:func:`src.signals.strategy.generate_strategy_signals`)
scores the *latest* universe cross-section, lets the regime router pick the
engine, and queues BUY signals. To measure whether the regime routing + the
three scorers actually help, we need to replay that same selection logic over
*every historical day* and feed it to the proven bar-by-bar
:func:`src.backtesting.engine.run_backtest` (which owns entry/exit/cost/sizing
and per-regime trade tagging).

This module is the bridge. For each trading day ``D`` it:

  1. reads the ``feature_data`` cross-section as-of ``D`` (technicals **and**
     as-of-joined fundamentals already live there -- no look-ahead),
  2. resolves the regime active on ``D`` (latest ``market_regime`` row <= D),
  3. asks the router for the engine + config (or, in baseline mode, forces
     plain momentum with no routing),
  4. scores the cross-section with that engine and keeps the best
     ``top_n_per_day`` names,

and emits one ``(symbol, feature_date, calibrated_prob)`` row per kept name.
``calibrated_prob`` is the production Kelly mapping ``_score_to_prob(score)`` so
the engine sizes positions exactly as the live book would.

Defensive regimes (BEAR_TREND / CRISIS) emit no new entries -- mirroring
``select_strategy(...).allow_new_entries == False`` -- so the only positions
carried through a bear market are ones opened earlier (the engine still
trails/exits them).

Fidelity caveats (documented, not hidden):
  - The diversification gate and the feasibility target-trim are *selection-time*
    refinements applied live; this v1 harness validates the scorer + router
    layer and leans on the engine's own sector / position caps for
    concentration control. They can be layered in later.
  - Within a single day the engine fills candidates in symbol order until caps
    bind, not strictly best-first; capping emission at the book size keeps that
    effect small.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import pandas as pd

from src.regime.router import select_strategy
from src.regime.store import regime_history
from src.signals.strategy import StrategyConfig, _score_to_prob, score_universe
from src.utils.db import fetch_all
from src.utils.logger import get_logger

log = get_logger("backtest.rules_predictions")

# Columns the three scorers (momentum / mean_reversion / breakout) read. All
# already as-of correct in feature_data, so there is no forward leakage.
_FEATURE_COLS = (
    "close", "dist_ema_200_pct", "dist_ema_50_pct", "rsi_14", "mom_60d",
    "ret_20d", "dd_from_high_252d", "macd_hist", "bb_pct_b", "vol_ratio_20d",
    "roe", "debt_to_equity", "profit_margin", "pe_ttm",
)


@dataclass
class RulesPredictions:
    """Output bundle: the predictions frame plus the regime timeline + a
    per-(symbol, day) audit frame (score / engine / regime)."""
    predictions: pd.DataFrame          # symbol, feature_date, calibrated_prob
    regime_by_date: dict[str, str]     # as_of_date -> regime (for engine tagging)
    audit: pd.DataFrame                # symbol, feature_date, score, engine, regime


def _sector_map(db_path: str | None = None) -> dict[str, str]:
    rows = fetch_all("SELECT symbol, sector FROM stock_sectors", db_path=db_path)
    return {r["symbol"]: r["sector"] for r in rows}


def _load_feature_cross_sections(
    symbols: list[str] | None,
    start: str,
    end: str,
    db_path: str | None = None,
) -> pd.DataFrame:
    cols = ", ".join(("symbol", "feature_date", *_FEATURE_COLS))
    clauses = ["feature_date >= ?", "feature_date <= ?"]
    params: list[object] = [start, end]
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        clauses.append(f"symbol IN ({placeholders})")
        params.extend(symbols)
    where = " AND ".join(clauses)
    rows = fetch_all(
        f"SELECT {cols} FROM feature_data WHERE {where} "  # noqa: S608 - cols are literal
        "ORDER BY feature_date, symbol",
        tuple(params), db_path=db_path,
    )
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _regime_asof_getter(regime_by_date: dict[str, str]):
    """latest regime label on/at-or-before a date string (None if unknown)."""
    keys = sorted(regime_by_date)
    if not keys:
        return lambda _d: None

    def get(d: str) -> str | None:
        idx = bisect.bisect_right(keys, d) - 1
        return regime_by_date[keys[idx]] if idx >= 0 else None

    return get


def build_rules_predictions(
    *,
    start: str,
    end: str,
    universe: list[str] | None = None,
    regime_routing: bool = True,
    base_config: StrategyConfig | None = None,
    top_n_per_day: int | None = None,
    db_path: str | None = None,
) -> RulesPredictions:
    """Replay the rules selection over ``[start, end]``.

    ``regime_routing=True``  -> engine + config chosen per day by the router
                                (defensive days emit nothing).
    ``regime_routing=False`` -> baseline: plain momentum every day (the A/B
                                control for "does routing add value?").

    ``universe`` defaults to the mapped ``stock_sectors`` names (the tradeable
    book). ``top_n_per_day`` defaults to the active config's ``target_holdings``.
    """
    sectors = _sector_map(db_path=db_path)
    if universe is None:
        universe = sorted(sectors)
    feats = _load_feature_cross_sections(universe, start, end, db_path=db_path)
    if feats.empty:
        log.warning("No feature_data in [{}, {}] for {} symbols.",
                    start, end, len(universe or []))
        return RulesPredictions(
            predictions=pd.DataFrame(
                columns=["symbol", "feature_date", "calibrated_prob"]),
            regime_by_date={},
            audit=pd.DataFrame(
                columns=["symbol", "feature_date", "score", "engine", "regime"]),
        )

    regime_by_date = regime_history(start, end, db_path=db_path)
    regime_at = _regime_asof_getter(regime_by_date)

    pred_rows: list[dict] = []
    audit_rows: list[dict] = []
    skipped_defensive = 0

    for fdate, sub in feats.groupby("feature_date", sort=True):
        d = str(fdate)
        regime = regime_at(d)

        if regime_routing:
            plan = select_strategy(regime)
            if not plan.allow_new_entries:
                skipped_defensive += 1
                continue
            engine, cfg = plan.engine, plan.config
        else:
            engine = "momentum"
            cfg = base_config or StrategyConfig()

        rows = sub.to_dict("records")
        for r in rows:
            r["sector"] = sectors.get(r["symbol"], "UNKNOWN")

        cands = score_universe(rows, cfg, engine=engine)
        limit = top_n_per_day if top_n_per_day is not None else cfg.target_holdings
        for cand in cands[:limit]:
            pred_rows.append({
                "symbol": cand.symbol,
                "feature_date": d,
                "calibrated_prob": _score_to_prob(cand.score),
            })
            audit_rows.append({
                "symbol": cand.symbol, "feature_date": d,
                "score": cand.score, "engine": engine, "regime": regime,
            })

    pred_df = pd.DataFrame(
        pred_rows, columns=["symbol", "feature_date", "calibrated_prob"])
    audit_df = pd.DataFrame(
        audit_rows, columns=["symbol", "feature_date", "score", "engine", "regime"])

    log.info(
        "Rules predictions | routing={} | days_scored={} | defensive_skipped={} "
        "| signals={} | symbols={}",
        regime_routing, feats["feature_date"].nunique() - skipped_defensive,
        skipped_defensive, len(pred_df), pred_df["symbol"].nunique()
        if not pred_df.empty else 0,
    )
    return RulesPredictions(
        predictions=pred_df, regime_by_date=regime_by_date, audit=audit_df)
