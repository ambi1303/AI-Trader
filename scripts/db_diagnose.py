"""Show what's happening at the price/feature/prediction edge.

Helpful when the dashboard is empty: tells you whether it's a calendar
issue (NSE holiday), a feature-build gap, or a prediction-date mismatch.
"""

from __future__ import annotations

from src.utils.db import fetch_all


def show(title: str, sql: str, params: tuple = ()) -> None:
    print(f"\n=== {title} ===")
    rows = fetch_all(sql, params)
    if not rows:
        print("  (no rows)")
        return
    for r in rows:
        print(" ", dict(r))


def main() -> int:
    show(
        "Latest feature_date per symbol (top 10)",
        "SELECT symbol, MAX(feature_date) AS latest "
        "FROM feature_data GROUP BY symbol "
        "ORDER BY latest DESC, symbol LIMIT 10",
    )
    show(
        "Trading-calendar rows from 2026-05-20 onwards",
        "SELECT cal_date, is_holiday, description "
        "FROM trading_calendar "
        "WHERE cal_date >= '2026-05-20' ORDER BY cal_date",
    )
    show(
        "RELIANCE price bars from 2026-05-20 onwards",
        "SELECT bar_date, close, source FROM price_data "
        "WHERE symbol='RELIANCE' AND bar_date >= '2026-05-20' "
        "ORDER BY bar_date",
    )
    show(
        "Latest predictions (top 5 most recent dates)",
        "SELECT prediction_date, COUNT(*) AS n, "
        "       MAX(calibrated_prob) AS max_prob "
        "FROM predictions_log GROUP BY prediction_date "
        "ORDER BY prediction_date DESC LIMIT 5",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
