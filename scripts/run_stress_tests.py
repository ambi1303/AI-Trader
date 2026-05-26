"""Run the backtester restricted to each pre-defined stress window.

Mirrors run_backtest.py but slices predictions and prices to one window at
a time and runs the engine N times. The output is a per-window summary
dictionary you can compare against the full-period results.

Usage:
    python -m scripts.run_stress_tests
    python -m scripts.run_stress_tests --regen
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from scripts.run_backtest import (
    _build_atr,
    _load_predictions,
    _load_prices,
    _load_sectors,
    _resolve_threshold,
)
from src.backtesting.cost_model import load_cost_config
from src.backtesting.engine import EngineConfig, run_backtest
from src.backtesting.report import format_summary, persist
from src.backtesting.scenarios import STRESS_WINDOWS
from src.utils.logger import get_logger

log = get_logger("script.stress")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default=None)
    p.add_argument("--regen", action="store_true")
    p.add_argument("--model-name", default="xgb_v1")
    p.add_argument("--run-id", default=None)
    p.add_argument("--symbols", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--persist", action="store_true",
                   help="Save each window's run to the DB")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    preds = _load_predictions(args)
    if preds.empty:
        log.error("No predictions")
        return 1
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols else preds["symbol"].unique().tolist()
    )
    prices = _load_prices(symbols)
    if prices.empty:
        log.error("No prices")
        return 1
    atr_df = _build_atr(prices)
    sectors = _load_sectors()
    threshold, model_run_id = _resolve_threshold(args)
    cost_cfg = load_cost_config()

    summaries = []
    for w in STRESS_WINDOWS:
        log.info("=== Stress window: {} ({} -> {}) ===", w.name, w.start, w.end)
        wp = preds[
            (pd.to_datetime(preds["feature_date"]).dt.date >= w.start)
            & (pd.to_datetime(preds["feature_date"]).dt.date <= w.end)
        ]
        wpr = prices[
            (pd.to_datetime(prices["bar_date"]).dt.date >= w.start)
            & (pd.to_datetime(prices["bar_date"]).dt.date <= w.end)
        ]
        watr = atr_df[
            (pd.to_datetime(atr_df["bar_date"]).dt.date >= w.start)
            & (pd.to_datetime(atr_df["bar_date"]).dt.date <= w.end)
        ]
        if wp.empty or wpr.empty:
            log.warning("Insufficient data in window {}; skipping.", w.name)
            continue

        cfg = EngineConfig(
            initial_capital=args.initial_capital,
            cost=cost_cfg,
            name=f"stress_{w.name}",
        )
        res = run_backtest(
            predictions=wp, prices=wpr, atr=watr, sectors=sectors,
            threshold=threshold, cfg=cfg,
        )
        print()
        print(format_summary(res))
        if args.persist:
            persist(res, model_run_id=model_run_id)
        summaries.append({
            "window": w.name,
            "description": w.description,
            "metrics": res.metrics.as_dict(),
        })

    print("\n=== Stress summary ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
