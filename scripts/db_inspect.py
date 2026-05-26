"""Print the current DB state in a single glance.

Used by the README and as a quick sanity check before / after running
the daily pipeline. Read-only -- never mutates the database.
"""

from __future__ import annotations

from src.utils.db import fetch_one


CHECKS = [
    ("price_data rows",     "SELECT COUNT(*) AS n FROM price_data"),
    ("latest price bar",    "SELECT MAX(bar_date) AS d FROM price_data"),
    ("price symbols",       "SELECT COUNT(DISTINCT symbol) AS n FROM price_data"),
    ("feature rows",        "SELECT COUNT(*) AS n FROM feature_data"),
    ("latest features",     "SELECT MAX(feature_date) AS d FROM feature_data"),
    ("model_runs",          "SELECT COUNT(*) AS n FROM model_runs"),
    ("latest model run_id", "SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1"),
    ("predictions total",   "SELECT COUNT(*) AS n FROM predictions_log"),
    ("latest prediction",   "SELECT MAX(prediction_date) AS d FROM predictions_log"),
    ("signals total",       "SELECT COUNT(*) AS n FROM signal_outbox"),
    ("signals today",       "SELECT COUNT(*) AS n FROM signal_outbox WHERE signal_date = date('now')"),
    ("open paper trades",   "SELECT COUNT(*) AS n FROM paper_trades WHERE status='open'"),
    ("closed paper trades", "SELECT COUNT(*) AS n FROM paper_trades WHERE status='closed'"),
    ("validation failures", "SELECT COUNT(*) AS n FROM validation_failures"),
]


def main() -> int:
    print(f"{'Check':24}  Result")
    print("-" * 60)
    for label, sql in CHECKS:
        try:
            row = fetch_one(sql)
            value = dict(row) if row else None
        except Exception as exc:  # noqa: BLE001
            value = f"ERROR: {exc}"
        print(f"{label:24}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
