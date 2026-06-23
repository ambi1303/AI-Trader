"""Persistence and pretty-printing for BacktestResult.

Persistence rule: ONE bt_run_id == ONE consistent set of rows across
backtest_runs / backtest_equity / backtest_trades. We use a single
transaction so a partial failure never leaves dangling equity rows
without a parent run.
"""

from __future__ import annotations

import json

from src.backtesting.engine import BacktestResult
from src.utils.db import transaction
from src.utils.logger import get_logger

log = get_logger("backtest.report")


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def persist(result: BacktestResult, *, model_run_id: str | None = None) -> None:
    """Insert run + equity curve + trades atomically."""
    with transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO backtest_runs
                (bt_run_id, model_run_id, name, start_date, end_date,
                 initial_capital, config_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.bt_run_id, model_run_id, result.name,
                result.start_date, result.end_date,
                result.initial_capital,
                json.dumps(result.config, default=str),
                json.dumps(_metrics_with_regime(result), default=str),
            ),
        )
        # Replace any existing rows for this bt_run_id
        conn.execute("DELETE FROM backtest_equity WHERE bt_run_id = ?",
                     (result.bt_run_id,))
        if not result.equity_curve.empty:
            rows = [
                (result.bt_run_id, r.bar_date, r.cash, r.equity,
                 int(r.open_count), r.daily_pnl)
                for r in result.equity_curve.itertuples(index=False)
            ]
            conn.executemany(
                """
                INSERT INTO backtest_equity
                    (bt_run_id, bar_date, cash, equity, open_count, daily_pnl)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        conn.execute("DELETE FROM backtest_trades WHERE bt_run_id = ?",
                     (result.bt_run_id,))
        if not result.trades.empty:
            has_regime = "entry_regime" in result.trades.columns
            tr_rows = [
                (
                    result.bt_run_id, r.symbol, r.side,
                    r.entry_date, r.exit_date,
                    float(r.entry_price), float(r.exit_price), int(r.qty),
                    float(r.gross_pnl), float(r.cost_rupees), float(r.net_pnl),
                    int(r.holding_days), r.exit_reason,
                    None if r.entry_prob is None else float(r.entry_prob),
                    None if r.threshold is None else float(r.threshold),
                    getattr(r, "entry_regime", None) if has_regime else None,
                )
                for r in result.trades.itertuples(index=False)
            ]
            conn.executemany(
                """
                INSERT INTO backtest_trades
                    (bt_run_id, symbol, side, entry_date, exit_date,
                     entry_price, exit_price, qty, gross_pnl, cost_rupees,
                     net_pnl, holding_days, exit_reason, entry_prob, threshold,
                     entry_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tr_rows,
            )

    log.info(
        "Persisted backtest {} | trades={} equity_rows={}",
        result.bt_run_id, len(result.trades), len(result.equity_curve),
    )


def _metrics_with_regime(result: BacktestResult) -> dict:
    """Aggregate metrics dict with the per-regime breakdown folded in, so the
    regime analysis is queryable from backtest_runs.metrics_json without a
    second schema column."""
    md = result.metrics.as_dict()
    if result.by_regime:
        md["by_regime"] = result.by_regime
    return md


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def format_summary(result: BacktestResult) -> str:
    m = result.metrics
    lines = [
        f"=== Backtest: {result.name} ({result.bt_run_id}) ===",
        f"  Window         : {result.start_date} -> {result.end_date}  ({m.days} days)",
        f"  Capital        : initial={result.initial_capital:,.0f} | "
        f"final equity multiplier={1.0 + m.total_return_pct/100:.4f}",
        f"  Trades         : n={m.n_trades} | hit={m.hit_rate_pct:.1f}% | "
        f"avg_win={m.avg_win_pct:.2f}% | avg_loss={m.avg_loss_pct:.2f}%",
        f"  Returns        : total={m.total_return_pct:.2f}% | CAGR={m.cagr_pct:.2f}%",
        f"  Risk           : Sharpe={m.sharpe:.2f} | Sortino={m.sortino:.2f} | "
        f"MaxDD={m.max_drawdown_pct:.2f}% ({m.max_drawdown_days}d) | "
        f"Calmar={m.calmar:.2f}",
        f"  Trade quality  : expectancy={m.expectancy_pct:.3f}% | PF={m.profit_factor:.2f}",
        f"  Cost diagnostics: drag/capital={m.cost_drag_pct:.2f}% | "
        f"cost/|grossPnL|={m.cost_to_gross_pnl_pct:.1f}% | turnover={m.turnover:.2f}x",
    ]
    if result.by_regime:
        lines.append("  By regime:")
        for regime in sorted(result.by_regime):
            s = result.by_regime[regime]
            pf = s.get("profit_factor")
            pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else "n/a"
            lines.append(
                f"    {regime:<16} n={s.get('n_trades', 0):<4} "
                f"hit={s.get('win_rate_pct', 0):.0f}% "
                f"net={s.get('net_pnl', 0):,.0f} PF={pf_str} "
                f"sharpe={s.get('daily_sharpe', 0)} "
                f"days={s.get('days_in_regime', 0)}"
            )
    return "\n".join(lines)
