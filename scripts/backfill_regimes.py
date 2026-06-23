"""Backfill the ``market_regime`` table over a historical window.

Walks every NIFTY trading day in ``[--start, --end]`` in chronological order and
calls :func:`src.regime.store.store_regime`, which classifies the regime using
only data available on/at-or-before that day (no look-ahead) and persists one
row per day. Because the classifier reads the *previous* day's regime for
hysteresis, the chronological order matters -- do not parallelise.

Typical use (after deep index history is ingested):

    python -m scripts.backfill_regimes --start 2019-01-01 --end 2026-06-19
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from src.db.migrate import apply_schema
from src.regime.store import NIFTY_SYMBOL, store_regime
from src.utils.db import fetch_all
from src.utils.logger import get_logger


def _trading_dates(start: str, end: str, db_path: str | None = None) -> list[str]:
    rows = fetch_all(
        "SELECT bar_date FROM index_data "
        "WHERE index_symbol = ? AND bar_date >= ? AND bar_date <= ? "
        "ORDER BY bar_date",
        (NIFTY_SYMBOL, start, end), db_path=db_path,
    )
    return [r["bar_date"] for r in rows]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill market_regime over a window.")
    p.add_argument("--start", required=True, help="ISO date, e.g. 2019-01-01")
    p.add_argument("--end", required=True, help="ISO date, e.g. 2026-06-19")
    p.add_argument("--db", dest="db_path", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.backfill_regimes")
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    apply_schema(db_path=args.db_path)
    dates = _trading_dates(args.start, args.end, db_path=args.db_path)
    if not dates:
        log.warning("No NIFTY trading dates in [{}, {}].", args.start, args.end)
        return 1

    log.info("Backfilling regimes for {} trading days: {} -> {}",
             len(dates), dates[0], dates[-1])
    counts: Counter[str] = Counter()
    for i, d in enumerate(dates, 1):
        payload = store_regime(d, db_path=args.db_path)
        counts[payload["regime"]] += 1
        if i % 100 == 0:
            log.info("  ... {}/{} ({})", i, len(dates), d)

    log.info("Done. Regime distribution:")
    total = sum(counts.values())
    for regime, n in counts.most_common():
        log.info("  {:<16} {:>5}  ({:>5.1f}%)", regime, n, 100.0 * n / total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
