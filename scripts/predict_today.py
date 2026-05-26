"""Inference script: load the latest registered model and predict for today.

Reads the most recent feature_data row per symbol and writes results into
predictions_log along with a feature snapshot. Does NOT generate trade
signals; that is the responsibility of the Week-4 risk/signal layer.

Usage:
    python -m scripts.predict_today --model-name xgb_v1
    python -m scripts.predict_today --run-id xgb_v1-20260101T120000Z-abcd1234
    python -m scripts.predict_today --symbols TCS,INFY,RELIANCE
"""

from __future__ import annotations

import argparse
import sys

from src.models.predict import predict_for_universe
from src.models.registry import latest_run_id
from src.utils.db import fetch_all
from src.utils.logger import get_logger

log = get_logger("script.predict")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--model-name", default="xgb_v1")
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols. Default: all in feature_data.")
    p.add_argument("--no-persist", action="store_true",
                   help="Do not write to predictions_log (dry run).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    run_id = args.run_id or latest_run_id(args.model_name)
    if not run_id:
        log.error("No registered model found for name '{}'. Train one first.",
                  args.model_name)
        return 1

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        rows = fetch_all("SELECT DISTINCT symbol FROM feature_data ORDER BY symbol")
        symbols = [r["symbol"] for r in rows]

    log.info("Predicting for {} symbols using run_id={}", len(symbols), run_id)
    df = predict_for_universe(run_id, symbols, persist=not args.no_persist)
    if df.empty:
        log.warning("No predictions produced (no feature rows available).")
        return 0
    df_sorted = df.sort_values("calibrated_prob", ascending=False)
    log.info("Top 10 by calibrated probability:\n{}",
             df_sorted.head(10).to_string(index=False))
    n_signals = int((df["would_signal"].fillna(False)).sum())
    log.success("{}/{} symbols would generate a BUY signal at current threshold.",
                n_signals, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
