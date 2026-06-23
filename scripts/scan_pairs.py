"""Scan the universe for cointegrated pairs and persist signals.

Research / signal-generation only: pairs need a short leg, and the paper trader
is LONG-only in v1, so these are surfaced for analysis, not auto-traded.

Examples
--------
Scan as of today with defaults::

    python -m scripts.scan_pairs

Scan a specific date, show only actionable (entry) signals::

    python -m scripts.scan_pairs --as-of 2026-06-17 --signals-only
"""

from __future__ import annotations

import argparse
from datetime import date

from src.db.migrate import apply_schema
from src.pairs.scan import PairScanConfig, scan_pairs
from src.utils.logger import get_logger

log = get_logger("scripts.scan_pairs")


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan for cointegrated pairs.")
    ap.add_argument("--as-of", default=date.today().isoformat(),
                    help="Scan date (ISO YYYY-MM-DD). Default: today.")
    ap.add_argument("--lookback", type=int, default=252,
                    help="Trading-day history for the fit (default 252).")
    ap.add_argument("--min-corr", type=float, default=0.30,
                    help="Return-correlation pre-filter (default 0.30, loose).")
    ap.add_argument("--level", type=float, default=0.05,
                    help="Engle-Granger significance level (default 0.05).")
    ap.add_argument("--signals-only", action="store_true",
                    help="Print only actionable LONG/SHORT_SPREAD pairs.")
    args = ap.parse_args()

    apply_schema()
    cfg = PairScanConfig(
        lookback_days=args.lookback, min_corr=args.min_corr, level=args.level)
    out = scan_pairs(as_of=args.as_of, cfg=cfg)

    print(f"\nPairs scan {out['as_of']}: tested={out['tested']} "
          f"cointegrated={out['cointegrated']} actionable={out['actionable']}")
    pairs = out["pairs"]
    pairs.sort(key=lambda r: abs(r["zscore"]), reverse=True)
    shown = 0
    for r in pairs:
        if args.signals_only and r["signal"] not in ("LONG_SPREAD", "SHORT_SPREAD"):
            continue
        hl = f"{r['half_life']:.0f}d" if r["half_life"] is not None else "n/a"
        print(f"  {r['symbol_y']:>14} ~ {r['symbol_x']:<14} "
              f"[{r['sector']:<12}] beta={r['beta']:+.2f} "
              f"z={r['zscore']:+.2f} half-life={hl:>5} "
              f"adf={r['adf_tstat']:+.2f} -> {r['signal']}")
        shown += 1
    if shown == 0:
        print("  (no pairs to show)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
