"""Run a backtest using Week-3 walk-forward predictions.

Pipeline:
  1. Load walk-forward predictions for the chosen model (Parquet from
     scripts.walk_forward_eval --save) OR re-generate them on-the-fly.
  2. Load BhavCopy OHLC + compute ATR(14) per (symbol, date).
  3. Load stock_sectors mapping for sector concentration caps.
  4. Pull production threshold from the registered model.
  5. Run the engine, persist results to backtest_runs / equity / trades.
  6. Print the summary.

Usage:
    python -m scripts.run_backtest --predictions data/reports/wf_predictions_*.parquet
    python -m scripts.run_backtest --regen --symbols TCS,INFY,RELIANCE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.backtesting.cost_model import load_cost_config
from src.backtesting.engine import EngineConfig, run_backtest
from src.backtesting.report import format_summary, persist
from src.backtesting.risk import RiskConfig
from src.backtesting.sizing import SizingConfig
from src.features.technical_indicators import atr as atr_fn
from src.models.dataset import build_training_matrix
from src.models.registry import latest_run_id, load_model
from src.models.walk_forward import (
    WalkForwardConfig,
    aggregate_walk_forward,
    run_walk_forward,
)
from src.utils.db import fetch_all
from src.utils.logger import get_logger

log = get_logger("script.run_backtest")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default=None,
                   help="Parquet with cols (symbol, feature_date, calibrated_prob)")
    p.add_argument("--regen", action="store_true",
                   help="Re-run walk-forward to regenerate predictions in-memory")
    p.add_argument("--model-name", default="xgb_v1")
    p.add_argument("--run-id", default=None)
    p.add_argument("--symbols", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the registry-stored decision threshold")
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--name", default="full")
    return p.parse_args(argv)


def _load_predictions(args) -> pd.DataFrame:
    if args.predictions:
        path = Path(args.predictions)
        log.info("Loading predictions from {}", path)
        df = pd.read_parquet(path)
    elif args.regen:
        log.info("Regenerating predictions via walk-forward")
        symbols = (
            [s.strip().upper() for s in args.symbols.split(",")]
            if args.symbols else None
        )
        m = build_training_matrix(symbols)
        cfg = WalkForwardConfig(
            initial_train_days=120, calib_days=30, test_days=30, step_days=30,
            min_train_rows=60,
        )
        results = run_walk_forward(m, cfg=cfg)
        agg = aggregate_walk_forward(results)
        if "test_predictions" not in agg:
            log.error("Walk-forward produced no completed folds")
            sys.exit(1)
        df = agg["test_predictions"]
    else:
        log.error("Specify either --predictions <parquet> or --regen")
        sys.exit(1)
    df["feature_date"] = pd.to_datetime(df["feature_date"]).dt.date
    return df[["symbol", "feature_date", "calibrated_prob"]].copy()


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(symbols))
    rows = fetch_all(
        f"""
        SELECT symbol, bar_date, open, high, low, close
        FROM   price_data
        WHERE  source = 'bhavcopy' AND symbol IN ({placeholders})
        ORDER BY bar_date, symbol
        """,  # noqa: S608
        tuple(symbols),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    return df


def _build_atr(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["symbol", "bar_date", "atr"])
    parts = []
    for sym, sub in prices.groupby("symbol", sort=False):
        sub = sub.sort_values("bar_date").reset_index(drop=True)
        a = atr_fn(
            high=sub["high"], low=sub["low"], close=sub["close"], period=period
        )
        parts.append(pd.DataFrame({
            "symbol": sym, "bar_date": sub["bar_date"], "atr": a.values,
        }))
    return pd.concat(parts, ignore_index=True)


def _load_sectors() -> dict[str, str]:
    rows = fetch_all("SELECT symbol, sector FROM stock_sectors")
    return {r["symbol"]: r["sector"] for r in rows}


def _resolve_threshold(args) -> tuple[float, str | None]:
    if args.threshold is not None:
        return args.threshold, None
    run_id = args.run_id or latest_run_id(args.model_name)
    if not run_id:
        log.error("No model registered under name '{}'.", args.model_name)
        sys.exit(1)
    _, meta = load_model(run_id)
    if meta.threshold is None:
        log.warning("Model has no stored threshold; defaulting to 0.55")
        return 0.55, run_id
    return float(meta.threshold), run_id


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    preds = _load_predictions(args)
    if preds.empty:
        log.error("No predictions available")
        return 1

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols else preds["symbol"].unique().tolist()
    )
    prices = _load_prices(symbols)
    if prices.empty:
        log.error("No price data for {} symbols", len(symbols))
        return 1

    atr_df = _build_atr(prices)
    sectors = _load_sectors()
    threshold, model_run_id = _resolve_threshold(args)

    log.info(
        "Backtest input: predictions={} prices={} atr={} symbols={} "
        "sectors={} threshold={:.3f}",
        len(preds), len(prices), len(atr_df), len(symbols),
        len(sectors), threshold,
    )

    cost_cfg = load_cost_config()
    cfg = EngineConfig(
        initial_capital=args.initial_capital,
        sizing=SizingConfig(),
        risk=RiskConfig(),
        cost=cost_cfg,
        name=args.name,
    )

    res = run_backtest(
        predictions=preds, prices=prices, atr=atr_df, sectors=sectors,
        threshold=threshold, cfg=cfg,
    )
    print()
    print(format_summary(res))
    print()
    persist(res, model_run_id=model_run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
