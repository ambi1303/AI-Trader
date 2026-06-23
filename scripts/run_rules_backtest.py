"""Backtest the RULES strategy (scorers + regime router) over history.

Unlike ``scripts.run_backtest`` (which replays ML walk-forward predictions),
this replays the transparent factor/mean-reversion/breakout scorers and the
regime router via :func:`src.backtesting.rules_predictions.build_rules_predictions`,
then runs the same bar-by-bar engine for entry/exit/cost/sizing and per-regime
trade tagging.

By default it runs an A/B: the regime-ROUTED book vs a momentum-only BASELINE
(routing disabled) over the same window, so you can see whether routing the
strategy by regime actually changes risk-adjusted returns -- with a per-regime
P&L breakdown for the routed run.

Usage:
    python -m scripts.run_rules_backtest --start 2019-04-01 --end 2026-06-19
    python -m scripts.run_rules_backtest --routed-only
"""

from __future__ import annotations

import argparse
import sys

from scripts.run_backtest import _build_atr, _load_prices, _load_sectors
from src.backtesting.cost_model import load_cost_config
from src.backtesting.engine import BacktestResult, EngineConfig, run_backtest
from src.backtesting.report import format_summary, persist
from src.backtesting.risk import RiskConfig
from src.backtesting.rules_predictions import build_rules_predictions
from src.db.migrate import apply_schema
from src.signals.strategy import StrategyConfig
from src.utils.logger import get_logger

log = get_logger("script.run_rules_backtest")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2019-04-01")
    p.add_argument("--end", default="2026-06-19")
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--top-n", type=int, default=None,
                   help="Max signals per day (default = config target_holdings)")
    p.add_argument("--routed-only", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--no-persist", action="store_true")
    # Exit-geometry knobs (the engine, not the scorer, owns exits). Defaults
    # mirror the live book's caps with a roomier stop than the 2-ATR engine
    # default, which the harness showed was churning trades out at ~-2%.
    p.add_argument("--stop-atr", type=float, default=2.5)
    p.add_argument("--target-atr", type=float, default=5.0)
    p.add_argument("--trail-atr", type=float, default=3.5)
    p.add_argument("--no-trail", action="store_true",
                   help="Disable the trailing stop (let targets/time exit)")
    p.add_argument("--max-hold", type=int, default=60)
    p.add_argument("--target-holdings", type=int, default=22)
    p.add_argument("--max-per-sector", type=int, default=6)
    return p.parse_args(argv)


def _risk_from_args(args) -> RiskConfig:
    """RiskConfig assembled from CLI knobs so the exit geometry can be swept
    without code edits (the harness's main tuning surface)."""
    return RiskConfig(
        stop_atr_mult=args.stop_atr,
        take_profit_atr_mult=args.target_atr,
        use_trailing_stop=not args.no_trail,
        trail_atr_mult=args.trail_atr,
        max_holding_days=args.max_hold,
        max_concurrent_positions=args.target_holdings,
        max_per_sector=args.max_per_sector,
        min_profit_pct=5.0,
    )


def _run_one(*, label: str, routed: bool, args) -> BacktestResult | None:
    rp = build_rules_predictions(
        start=args.start, end=args.end,
        regime_routing=routed, base_config=StrategyConfig(),
        top_n_per_day=args.top_n,
    )
    if rp.predictions.empty:
        log.warning("[{}] no rules signals in window -- nothing to backtest.", label)
        return None

    symbols = rp.predictions["symbol"].unique().tolist()
    prices = _load_prices(symbols)
    if prices.empty:
        log.error("[{}] no price_data for {} symbols.", label, len(symbols))
        return None
    atr_df = _build_atr(prices)
    sectors = _load_sectors()

    cfg = EngineConfig(
        initial_capital=args.initial_capital,
        risk=_risk_from_args(args),
        cost=load_cost_config(),
        name=label,
    )
    res = run_backtest(
        predictions=rp.predictions, prices=prices, atr=atr_df, sectors=sectors,
        threshold=0.0, cfg=cfg,
        regime_by_date=rp.regime_by_date if routed else None,
    )
    print()
    print(format_summary(res))
    print()
    if not args.no_persist:
        persist(res)
    return res


def _print_ab(routed: BacktestResult | None, baseline: BacktestResult | None) -> None:
    if not (routed and baseline):
        return
    r, b = routed.metrics, baseline.metrics
    print("=" * 72)
    print("A/B: regime-ROUTED vs momentum-only BASELINE")
    print(f"{'metric':<22}{'routed':>16}{'baseline':>16}{'delta':>16}")
    rows = [
        ("total_return_%", r.total_return_pct, b.total_return_pct),
        ("CAGR_%", r.cagr_pct, b.cagr_pct),
        ("Sharpe", r.sharpe, b.sharpe),
        ("Sortino", r.sortino, b.sortino),
        ("MaxDD_%", r.max_drawdown_pct, b.max_drawdown_pct),
        ("Calmar", r.calmar, b.calmar),
        ("hit_rate_%", r.hit_rate_pct, b.hit_rate_pct),
        ("expectancy_%", r.expectancy_pct, b.expectancy_pct),
        ("profit_factor", r.profit_factor, b.profit_factor),
        ("n_trades", float(r.n_trades), float(b.n_trades)),
    ]
    for name, rv, bv in rows:
        print(f"{name:<22}{rv:>16.2f}{bv:>16.2f}{rv - bv:>+16.2f}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    apply_schema()

    routed = baseline = None
    if not args.baseline_only:
        routed = _run_one(label="rules_routed", routed=True, args=args)
    if not args.routed_only:
        baseline = _run_one(label="rules_baseline", routed=False, args=args)
    _print_ab(routed, baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
