"""Fast walk-forward predictions for the Week-4 smoke. Saves a parquet to
data/reports/wf_smoke.parquet that run_backtest.py can consume."""
from __future__ import annotations

import os
import sys
import time

from src.models.dataset import build_training_matrix
from src.models.walk_forward import (
    WalkForwardConfig,
    aggregate_walk_forward,
    run_walk_forward,
)


def main() -> int:
    t0 = time.time()
    m = build_training_matrix(["TCS", "INFY", "RELIANCE"])
    cfg = WalkForwardConfig(
        initial_train_days=730, calib_days=60, test_days=180,
        step_days=180, min_train_rows=200,
    )
    results = run_walk_forward(m, cfg=cfg)
    agg = aggregate_walk_forward(results)
    print(
        f"folds_completed={agg.get('folds_completed', 0)} "
        f"predictions={agg.get('n_test_predictions', 0)} "
        f"elapsed={time.time() - t0:.1f}s"
    )
    if "test_predictions" not in agg:
        print("No predictions; bailing.")
        return 1
    os.makedirs("data/reports", exist_ok=True)
    out = "data/reports/wf_smoke.parquet"
    agg["test_predictions"].to_parquet(out, index=False)
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
