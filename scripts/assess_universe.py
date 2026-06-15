"""Assess price-data coverage to size a wider trading universe.

Read-only. Prints, from price_data (bhavcopy source):
  - total distinct symbols
  - how many have >= N bars of history (a few N buckets)
  - how many are already in feature_data
  - a liquidity-ranked shortlist (by median rupee turnover over the last
    ~60 bars) so we can pick a tractable mid/small-cap expansion.
"""

from __future__ import annotations

import sys

from src.utils.db import fetch_all, fetch_one
from src.utils.logger import get_logger

log = get_logger("script.assess")


def _one(query: str, params: tuple = ()):
    """fetch_one -> dict (sqlite3.Row has no .get)."""
    row = fetch_one(query, params)
    return dict(row) if row else {}


def main() -> int:
    total = _one(
        "SELECT COUNT(DISTINCT symbol) AS n FROM price_data WHERE source='bhavcopy'"
    )
    print(f"Distinct bhavcopy symbols: {total.get('n', 0)}")

    for n in (250, 400, 500, 750, 1000):
        row = _one(
            """
            SELECT COUNT(*) AS n FROM (
              SELECT symbol FROM price_data WHERE source='bhavcopy'
              GROUP BY symbol HAVING COUNT(*) >= ?
            )
            """,
            (n,),
        )
        print(f"  symbols with >= {n:>4} bars: {row.get('n', 0)}")

    feat = _one("SELECT COUNT(DISTINCT symbol) AS n FROM feature_data")
    print(f"Symbols already in feature_data: {feat.get('n', 0)}")

    latest = _one(
        "SELECT MAX(bar_date) AS d FROM price_data WHERE source='bhavcopy'"
    )
    cutoff = _one(
        "SELECT MIN(bar_date) AS d FROM ("
        " SELECT DISTINCT bar_date FROM price_data WHERE source='bhavcopy'"
        " ORDER BY bar_date DESC LIMIT 60)"
    )
    last_d = latest.get("d")
    cut_d = cutoff.get("d")
    print(f"Latest bar: {last_d} | 60-bar liquidity window starts: {cut_d}")

    # Liquidity ranking: median(close*volume) over the recent window, for
    # symbols with a decent history. Top of the list = most liquid; the
    # mid/small-cap band sits a few hundred rows down.
    rows = fetch_all(
        """
        SELECT symbol,
               COUNT(*)                         AS bars,
               AVG(close * volume)              AS avg_turnover
        FROM   price_data
        WHERE  source='bhavcopy' AND bar_date >= ?
        GROUP  BY symbol
        HAVING COUNT(*) >= 30
        ORDER  BY avg_turnover DESC
        """,
        (cut_d,),
    )
    print(f"\nLiquidity-ranked symbols (recent window): {len(rows)} qualify")
    print("Rank  Symbol           AvgTurnover(Cr)")
    for i, r in enumerate(rows):
        if i < 15 or (200 <= i < 205) or (490 <= i < 495):
            cr = (r["avg_turnover"] or 0) / 1e7  # rupees -> crore
            print(f"{i+1:>4}  {r['symbol']:<16} {cr:>12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
