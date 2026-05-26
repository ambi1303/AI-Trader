"""Sanity-check the feature_data table for ranges and last-day snapshots.

Usage:
    python -m scripts.spot_check_features --symbols RELIANCE,TCS,INFY
"""

from __future__ import annotations

import argparse
import sys

from src.utils.db import fetch_all


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        rows = fetch_all(
            "SELECT DISTINCT symbol FROM feature_data ORDER BY symbol LIMIT 10"
        )
        syms = [r["symbol"] for r in rows]
    placeholders = ",".join("?" * len(syms))

    print("=" * 76)
    print(f"feature_data coverage for: {syms}")
    print("=" * 76)
    rows = fetch_all(
        f"""
        SELECT symbol, COUNT(*) AS n,
               MIN(feature_date) AS first,
               MAX(feature_date) AS last
        FROM   feature_data
        WHERE  symbol IN ({placeholders})
        GROUP BY symbol
        """,
        tuple(syms),
    )
    for r in rows:
        print(f"  {r['symbol']:<10} rows={r['n']:>4}  {r['first']} -> {r['last']}")
    print()

    print("=" * 76)
    print("Range checks (RSI/ADX/Stoch must be [0,100], BB%B usually ~[-0.5,1.5])")
    print("=" * 76)
    rows = fetch_all(
        f"""
        SELECT symbol,
          ROUND(MIN(rsi_14),2)  AS rsi_min,  ROUND(MAX(rsi_14),2)  AS rsi_max,
          ROUND(MIN(adx_14),2)  AS adx_min,  ROUND(MAX(adx_14),2)  AS adx_max,
          ROUND(MIN(stoch_k),2) AS sk_min,   ROUND(MAX(stoch_k),2) AS sk_max,
          ROUND(MIN(bb_pct_b),3) AS bb_min,  ROUND(MAX(bb_pct_b),3) AS bb_max,
          ROUND(MIN(atr_pct)*100,3) AS atrp_min,
          ROUND(MAX(atr_pct)*100,3) AS atrp_max
        FROM   feature_data
        WHERE  symbol IN ({placeholders})
        GROUP BY symbol
        """,
        tuple(syms),
    )
    for r in rows:
        d = dict(r)
        print(f"  {d['symbol']}")
        print(f"    rsi_14   in [{d['rsi_min']:>6}, {d['rsi_max']:>6}]")
        print(f"    adx_14   in [{d['adx_min']:>6}, {d['adx_max']:>6}]")
        print(f"    stoch_k  in [{d['sk_min']:>6}, {d['sk_max']:>6}]")
        print(f"    bb_pct_b in [{d['bb_min']:>6}, {d['bb_max']:>6}]")
        print(f"    atr_pct  in [{d['atrp_min']:>6}%, {d['atrp_max']:>6}%]")
    print()

    print("=" * 76)
    print("Last-day feature snapshot")
    print("=" * 76)
    rows = fetch_all(
        f"""
        WITH last_dates AS (
          SELECT symbol, MAX(feature_date) AS d
          FROM   feature_data
          WHERE  symbol IN ({placeholders})
          GROUP BY symbol
        )
        SELECT f.symbol, f.feature_date,
               ROUND(f.close,2)              AS close,
               ROUND(f.rsi_14,1)             AS rsi,
               ROUND(f.adx_14,1)             AS adx,
               ROUND(f.macd,3)               AS macd,
               ROUND(f.dist_ema_50_pct*100,2)  AS dist_ema50_pct,
               ROUND(f.bb_pct_b,3)           AS bb_pctb,
               ROUND(f.vix_level,2)          AS vix,
               ROUND(f.beta_60d,3)           AS beta_60d,
               ROUND(f.corr_60d,3)           AS corr_60d,
               ROUND(f.sector_rs_20d*100,3)  AS sector_rs_pct
        FROM   feature_data f
        JOIN   last_dates l ON l.symbol = f.symbol AND l.d = f.feature_date
        ORDER BY f.symbol
        """,
        tuple(syms),
    )
    for r in rows:
        d = dict(r)
        print(f"  {d['symbol']} @ {d['feature_date']}")
        print(f"    close={d['close']}  rsi={d['rsi']}  adx={d['adx']}  macd={d['macd']}")
        print(f"    dist_ema50%={d['dist_ema50_pct']}  bb_pctb={d['bb_pctb']}  vix={d['vix']}")
        print(f"    beta_60d={d['beta_60d']}  corr_60d={d['corr_60d']}  sector_rs%={d['sector_rs_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
