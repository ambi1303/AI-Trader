"""Per-regime backtest analysis (pure).

Splits a backtest's trades + equity curve by the market regime that was active
at trade entry / on each day, so we can answer the question that justifies the
whole regime engine: *does each strategy actually earn its keep in the regime
it claims to help?* (e.g. mean-reversion should make money in RANGE, not just
ride momentum in BULL_TREND.)

Pure: takes DataFrames, returns a plain nested dict. No IO, no plotting.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

_TRADING_DAYS = 252


def _trade_stats(grp: pd.DataFrame) -> dict[str, Any]:
    net = grp["net_pnl"].astype(float)
    n = int(len(grp))
    wins = int((net > 0).sum())
    gross_win = float(net[net > 0].sum())
    gross_loss = float(-net[net < 0].sum())
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else None
    avg_hold = (round(float(grp["holding_days"].astype(float).mean()), 1)
                if "holding_days" in grp else None)
    return {
        "n_trades": n,
        "win_rate_pct": round(100.0 * wins / n, 1) if n else 0.0,
        "net_pnl": round(float(net.sum()), 2),
        "avg_net_pnl": round(float(net.mean()), 2) if n else 0.0,
        "profit_factor": pf,
        "avg_holding_days": avg_hold,
    }


def _daily_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=0))
    if sd <= 0:
        return 0.0
    return round(float(r.mean()) / sd * math.sqrt(_TRADING_DAYS), 2)


def summarize_by_regime(
    trades: pd.DataFrame | None,
    equity: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{regime: {trade + equity stats}}``.

    Trades are grouped by ``entry_regime``; the equity curve (if it carries a
    ``regime`` column) contributes a per-regime daily-return Sharpe and the
    number of days spent in each regime. Unknown/None tags bucket as
    ``"UNKNOWN"`` so a regime-agnostic backtest still returns a single bucket.
    """
    out: dict[str, dict[str, Any]] = {}

    if (trades is not None and not trades.empty
            and "entry_regime" in trades.columns):
        tagged = trades.copy()
        tagged["entry_regime"] = tagged["entry_regime"].fillna("UNKNOWN")
        for regime, grp in tagged.groupby("entry_regime"):
            out[str(regime)] = _trade_stats(grp)

    if (equity is not None and not equity.empty
            and "regime" in equity.columns):
        eq = equity.copy()
        eq["_ret"] = eq["equity"].astype(float).pct_change()
        eq["regime"] = eq["regime"].fillna("UNKNOWN")
        for regime, grp in eq.groupby("regime"):
            bucket = out.setdefault(str(regime), {})
            bucket["days_in_regime"] = int(len(grp))
            bucket["daily_sharpe"] = _daily_sharpe(grp["_ret"])

    return out
