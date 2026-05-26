"""Quick inspection of the model registry, predictions_log, and metrics.

Usage:
    python -m scripts.inspect_models
"""

from __future__ import annotations

import json
import sys

from src.utils.db import fetch_all


def main(argv: list[str] | None = None) -> int:
    runs = fetch_all(
        "SELECT run_id, model_name, feature_hash, trained_from, trained_to, "
        "created_at, metrics_json FROM model_runs ORDER BY created_at DESC, "
        "rowid DESC LIMIT 5"
    )
    print("=== model_runs (latest 5) ===")
    if not runs:
        print("  (none)")
    else:
        for r in runs:
            run_id = r["run_id"]
            name = r["model_name"]
            fh = r["feature_hash"]
            print(f"  {run_id}")
            print(f"    name={name} feat_hash={fh[:12]}")
            print(f"    trained {r['trained_from']} -> {r['trained_to']}")
            print(f"    created_at={r['created_at']}")

    preds = fetch_all(
        "SELECT COUNT(*) as n, COUNT(DISTINCT symbol) as n_sym, "
        "MAX(prediction_date) as last_date FROM predictions_log"
    )
    p = preds[0]
    print()
    print("=== predictions_log ===")
    print(f"  rows={p['n']} symbols={p['n_sym']} last_date={p['last_date']}")

    if runs:
        m = json.loads(runs[0]["metrics_json"])
        print()
        print("=== latest model metrics (test slice) ===")
        for k in ("test_raw", "test_calibrated", "threshold", "split_sizes"):
            print(f"  {k}: {m.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
