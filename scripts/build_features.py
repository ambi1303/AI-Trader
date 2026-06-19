"""Build the feature matrix for one or more symbols."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from tqdm import tqdm

import pandas as pd

from src.features.feature_builder import build_for_symbol, load_market_context
from src.utils.db import fetch_all
from src.utils.logger import get_logger


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, help="Comma-separated; default = all in v_universe_today")
    p.add_argument("--top-liquid", type=int, default=None,
                   help="Build for the top-N most liquid symbols (by recent "
                        "avg rupee turnover) instead of v_universe_today. "
                        "Use this to expand into liquid mid/small-caps.")
    p.add_argument("--min-bars", type=int, default=500,
                   help="With --top-liquid: require at least this many bhavcopy "
                        "bars of history (default 500 ~ 2 years).")
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--limit", type=int)
    p.add_argument("--incremental", action="store_true",
                   help="Only (re)compute the new tail per symbol and skip "
                        "symbols already up to date. Recommended for daily runs.")
    p.add_argument("--lookback-bars", type=int, default=600,
                   help="With --incremental: prior bars to load as warm-up so "
                        "recursive indicators converge (default 600 ~ 2.5y).")
    return p.parse_args(argv)


# ETF / fund / debt instrument name fragments. These are NOT stock picks --
# a liquid/debt fund mechanically drifts up (interest accrual) and so games a
# "will it rise?" target, while index ETFs have no idiosyncratic signal. We
# keep the universe to actual equities.
_NON_EQUITY_TOKENS = (
    "BEES", "ETF", "LIQUID", "LIQUIDADD", "LIQUIDPLUS", "LIQUIDCASE",
    "GOLD", "SILVER", "GILT", "GSEC", "SDL", "GROWTH", "MON100",
    "MAFANG", "HNGSNG", "NIFTY", "SENSEX", "BANKBEES", "PSUBNKBEES",
    "JUNIORBEES", "MOM100", "MOMENTUM50", "MIDCAP", "SMALLCAP",
    "MASPTOP50", "MAHKTECH", "ALPHA", "LOWVOL", "QUAL30", "VAL30",
    "EQUAL50", "CONS", "CPSEETF", "BHARATBOND", "EBBETF", "LICMFGOLD",
    "AXISGOLD", "SETFGOLD", "QGOLD", "GOLDBEES", "SILVERBEES",
    "LICNETFN50", "LICNETFGSC", "LICNFNHGP", "TNIDETF",
)


def _is_equity(symbol: str) -> bool:
    s = symbol.upper()
    return not any(tok in s for tok in _NON_EQUITY_TOKENS)


def _resolve_liquid_universe(top_n: int, min_bars: int) -> list[str]:
    """Top-N *equity* symbols by average rupee turnover over the most recent
    ~60 trading days, restricted to those with >= min_bars of history.

    Liquidity ordering keeps us in names that are actually tradeable (real
    fills, sane spreads) while still reaching well past the hyper-efficient
    large-cap core into the mid/small-cap band where TA edge can survive.
    ETFs/funds/debt instruments are excluded (see _NON_EQUITY_TOKENS).
    """
    cut = fetch_all(
        "SELECT MIN(bar_date) AS d FROM ("
        " SELECT DISTINCT bar_date FROM price_data WHERE source='bhavcopy'"
        " ORDER BY bar_date DESC LIMIT 60)"
    )
    cut_d = dict(cut[0])["d"] if cut else None
    # Over-fetch, then filter non-equities, then trim to top_n.
    rows = fetch_all(
        """
        SELECT symbol
        FROM   price_data
        WHERE  source='bhavcopy'
        GROUP  BY symbol
        HAVING COUNT(*) >= ?
        ORDER  BY AVG(CASE WHEN bar_date >= ? THEN close * volume END) DESC
        LIMIT  ?
        """,
        (int(min_bars), cut_d, int(top_n) * 2),
    )
    eq = [r["symbol"] for r in rows if _is_equity(r["symbol"])]
    return eq[: int(top_n)]


def _resolve_symbols(arg: str | None) -> list[str]:
    if arg:
        return [s.strip().upper() for s in arg.split(",") if s.strip()]
    rows = fetch_all("SELECT symbol FROM v_universe_today ORDER BY symbol")
    return [r["symbol"] for r in rows]


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.build_features")
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    if args.top_liquid:
        symbols = _resolve_liquid_universe(args.top_liquid, args.min_bars)
        log.info("Liquid universe: top {} (>= {} bars) -> {} symbols",
                 args.top_liquid, args.min_bars, len(symbols))
    else:
        symbols = _resolve_symbols(args.symbols)
    if args.limit:
        symbols = symbols[: args.limit]

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    log.info("Feature build: {} symbols (incremental={})",
             len(symbols), bool(args.incremental))

    # Load the market-context indices once and share a sector-index cache so a
    # batch build doesn't re-query the full Nifty/VIX/sector history per symbol.
    nifty_close, vix_close = load_market_context()
    sector_cache: dict[str, pd.Series] = {}

    total = 0
    skipped = 0
    for sym in tqdm(symbols, desc="features"):
        s = build_for_symbol(
            sym, start=start, end=end,
            incremental=args.incremental, lookback_bars=args.lookback_bars,
            nifty_close=nifty_close, vix_close=vix_close,
            sector_close_cache=sector_cache,
        )
        total += s.rows_out
        if s.rows_in == 0 and s.rows_out == 0:
            skipped += 1
    log.info("Done. Total feature rows written: {} (symbols skipped as "
             "up-to-date: {})", total, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
