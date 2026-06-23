"""Multi-horizon price-target forecasts for one or many stocks.

Projects an expected price + a 1-sigma probability band at 1W / 1M / 3M / 6M /
1Y / 3Y (long run) using the drift + volatility model in
:mod:`src.analysis.forecast`, prints a readable table, and (unless
``--no-persist``) upserts the rows into ``price_forecasts``.

Usage:
    python -m scripts.forecast_targets --symbols TCS,RELIANCE
    python -m scripts.forecast_targets                 # whole mapped universe
    python -m scripts.forecast_targets --symbols INFY --as-of 2026-06-19
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from src.analysis.forecast_store import forecast_symbol, store_forecast
from src.db.migrate import apply_schema
from src.utils.db import fetch_all
from src.utils.logger import get_logger

log = get_logger("script.forecast_targets")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None,
                   help="Comma-separated; default = mapped stock_sectors universe")
    p.add_argument("--as-of", default=None, help="ISO date; default = today")
    p.add_argument("--long-run-annual", type=float, default=None,
                   help="Override the long-run annual drift (e.g. 0.11)")
    p.add_argument("--no-persist", action="store_true")
    return p.parse_args(argv)


def _universe(args) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    rows = fetch_all("SELECT symbol FROM stock_sectors ORDER BY symbol")
    return [r["symbol"] for r in rows]


def _print_forecast(out: dict) -> None:
    sym = out.get("symbol", "?")
    if not out.get("available"):
        print(f"\n{sym}: no data to forecast.")
        return
    print(f"\n=== {sym}  (as-of {out.get('as_of_date')}) ===")
    print(f"  last close = {out['last_close']:,.2f} | daily vol = "
          f"{out['daily_vol_pct']:.2f}% | momentum drift = "
          f"{out['momentum_drift_daily_pct']:+.3f}%/day")
    print(f"  {'horizon':<8}{'expected':>12}{'low (1s)':>12}{'high (1s)':>12}"
          f"{'return':>10}{'ann.':>9}{'P(up)':>8}  outlook")
    for h in out["horizons"]:
        print(
            f"  {h['label']:<8}{h['expected_price']:>12,.2f}"
            f"{h['low_price']:>12,.2f}{h['high_price']:>12,.2f}"
            f"{h['expected_return_pct']:>9.1f}%{h['annualized_return_pct']:>8.1f}%"
            f"{h['prob_up_pct']:>7.0f}%  {h['verdict']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    apply_schema()
    as_of = args.as_of or date.today().isoformat()
    symbols = _universe(args)
    if not symbols:
        log.error("No symbols to forecast.")
        return 1

    n_ok = 0
    for sym in symbols:
        if args.no_persist:
            out = forecast_symbol(sym, as_of, long_run_annual=args.long_run_annual)
        else:
            out = store_forecast(sym, as_of, long_run_annual=args.long_run_annual)
        if len(symbols) <= 30:
            _print_forecast(out)
        if out.get("available"):
            n_ok += 1

    log.info("Forecast complete: {}/{} symbols had data{}.",
             n_ok, len(symbols), "" if args.no_persist else " (persisted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
