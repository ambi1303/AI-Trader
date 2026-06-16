"""Performance metrics for an equity curve and a trade ledger.

All return-based metrics accept a *daily* return series. Annualisation uses
TRADING_DAYS = 252 (Indian markets ~250-252 active sessions).

Computed:
  - Total return, CAGR
  - Sharpe (risk-free = 0; daily-stdev * sqrt(252))
  - Sortino (downside-only stdev)
  - Calmar (CAGR / max DD)
  - MAR  (annualised return / max DD)
  - Max drawdown, max DD duration
  - Hit rate, expectancy, profit factor, avg win/loss
  - Turnover (gross traded notional / mean equity)
  - Cost drag (sum of costs / sum of |gross PnL|)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestMetrics:
    n_trades: int
    days: int
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    max_drawdown_days: int
    calmar: float
    mar: float
    hit_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float
    profit_factor: float
    turnover: float
    cost_drag_pct: float
    cost_to_gross_pnl_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Drawdown helpers
# ---------------------------------------------------------------------------


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max - 1.0)


def _max_drawdown_and_duration(equity: pd.Series) -> tuple[float, int]:
    if len(equity) == 0:
        return (0.0, 0)
    dd = _drawdown_series(equity)
    max_dd = float(dd.min())

    # Duration: longest run of bars below previous peak.
    in_dd = dd < 0
    longest = 0
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return (max_dd, int(longest))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    *,
    equity_curve: pd.DataFrame,        # columns: bar_date, equity (and optional cash, daily_pnl)
    trades: pd.DataFrame,              # columns: gross_pnl, net_pnl, cost_rupees, entry_price, qty
    initial_capital: float,
) -> BacktestMetrics:
    if equity_curve.empty:
        return BacktestMetrics(
            n_trades=0, days=0, total_return_pct=0.0, cagr_pct=0.0,
            sharpe=0.0, sortino=0.0, max_drawdown_pct=0.0, max_drawdown_days=0,
            calmar=0.0, mar=0.0, hit_rate_pct=0.0, avg_win_pct=0.0,
            avg_loss_pct=0.0, expectancy_pct=0.0, profit_factor=0.0,
            turnover=0.0, cost_drag_pct=0.0, cost_to_gross_pnl_pct=0.0,
        )

    eq = equity_curve.sort_values("bar_date").reset_index(drop=True)
    equity = eq["equity"].astype(float)
    days = len(equity)

    # Returns. The first pct_change is NaN by definition; we DROP it (rather
    # than fillna(0)) so that "constant return every day" yields zero variance
    # as one would intuitively expect for Sharpe.
    daily_ret = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)

    years = max(days / TRADING_DAYS, 1.0 / TRADING_DAYS)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0

    std = float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else 0.0
    sharpe = (
        float(daily_ret.mean() / std * np.sqrt(TRADING_DAYS))
        if std > 0 else 0.0
    )

    # Sortino uses the canonical *target downside deviation* (target/MAR = 0):
    # the root-mean-square of the negative excursions over ALL periods, i.e.
    # sqrt(mean(min(r, 0)^2)). This is the standard definition (Sortino & Price)
    # and, unlike the sample std of just the negative subset, guarantees the
    # denominator <= total stdev, so |Sortino| >= |Sharpe| for a given-sign mean.
    if len(daily_ret) > 0:
        neg = daily_ret.clip(upper=0.0)
        dstd = float(np.sqrt((neg ** 2).mean()))
    else:
        dstd = 0.0
    sortino = (
        float(daily_ret.mean() / dstd * np.sqrt(TRADING_DAYS))
        if dstd > 0 else 0.0
    )

    max_dd, max_dd_days = _max_drawdown_and_duration(equity)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    annualised = cagr
    mar = float(annualised / abs(max_dd)) if max_dd < 0 else 0.0

    # Trade stats
    if trades.empty:
        hit, avg_w, avg_l, expectancy, pf = 0.0, 0.0, 0.0, 0.0, 0.0
        turnover, cost_drag, cost_to_gross = 0.0, 0.0, 0.0
        n_trades = 0
    else:
        n_trades = len(trades)
        # Per-trade returns (relative to entry notional)
        notional = (trades["entry_price"] * trades["qty"]).astype(float).clip(lower=1e-9)
        ret_pct = trades["net_pnl"].astype(float) / notional
        wins = ret_pct[ret_pct > 0]
        losses = ret_pct[ret_pct < 0]
        hit = float((ret_pct > 0).mean()) if n_trades else 0.0
        avg_w = float(wins.mean()) if len(wins) else 0.0
        avg_l = float(losses.mean()) if len(losses) else 0.0
        expectancy = float(ret_pct.mean()) if n_trades else 0.0
        gross_wins = float(trades["net_pnl"][trades["net_pnl"] > 0].sum())
        gross_losses = float(-trades["net_pnl"][trades["net_pnl"] < 0].sum())
        pf = float(gross_wins / gross_losses) if gross_losses > 0 else float("inf")

        gross_traded = float(notional.sum() * 2)  # buy + sell legs
        mean_equity = float(equity.mean())
        turnover = float(gross_traded / mean_equity) if mean_equity > 0 else 0.0
        cost_total = float(trades["cost_rupees"].sum())
        gross_pnl_abs = float(trades["gross_pnl"].abs().sum())
        cost_to_gross = float(cost_total / gross_pnl_abs) if gross_pnl_abs > 0 else 0.0
        cost_drag = float(cost_total / initial_capital) if initial_capital > 0 else 0.0

    return BacktestMetrics(
        n_trades=int(n_trades),
        days=int(days),
        total_return_pct=total_return * 100.0,
        cagr_pct=cagr * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd * 100.0,
        max_drawdown_days=int(max_dd_days),
        calmar=calmar,
        mar=mar,
        hit_rate_pct=hit * 100.0,
        avg_win_pct=avg_w * 100.0,
        avg_loss_pct=avg_l * 100.0,
        expectancy_pct=expectancy * 100.0,
        profit_factor=pf,
        turnover=turnover,
        cost_drag_pct=cost_drag * 100.0,
        cost_to_gross_pnl_pct=cost_to_gross * 100.0,
    )
