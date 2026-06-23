"""Pairs scan: find cointegrated pairs in the universe and persist signals.

DB IO around the pure stats in :mod:`src.pairs.cointegration` /
:mod:`src.pairs.spread`. We only test pairs **within the same sector** (so a
cointegration hit has an economic rationale, not just a spurious fit) and
pre-filter by return correlation to keep the O(n^2) scan cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import combinations
from typing import Any

import numpy as np

from src.pairs import cointegration as coint
from src.pairs import spread as spr
from src.utils.db import fetch_all, transaction
from src.utils.logger import get_logger

log = get_logger("pairs.scan")

# Source preference when the same bar exists from multiple providers.
_SOURCE_PRIORITY = {"bhavcopy": 0, "yfinance": 1, "nsepython": 2}


@dataclass(frozen=True)
class PairScanConfig:
    lookback_days: int = 252        # ~1y of trading history for the fit
    min_obs: int = 60               # min overlapping days to even try a pair
    # Loose return-correlation pre-filter: just a cheap way to drop wholly
    # unrelated names. It is deliberately NOT high -- cointegration is about
    # co-movement in *levels*, and genuinely cointegrated pairs (e.g. a hedge
    # with a small beta) can have only moderate daily-return correlation, so a
    # tight gate here wrongly discards exactly the pairs we want. The
    # within-sector restriction is the real economic filter.
    min_corr: float = 0.30
    level: float = 0.05             # Engle-Granger significance level
    z_window: int = 60              # trailing window for the z-score
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 3.5
    max_half_life: float = 120.0
    min_half_life: float = 1.0
    max_per_sector: int = 40        # guard against pathological sector sizes


def _as_of(as_of: str | None) -> str:
    return as_of or date.today().isoformat()


def _cutoff(as_of: str, lookback_days: int) -> str:
    try:
        anchor = datetime.fromisoformat(as_of).date()
    except ValueError:
        anchor = date.today()
    return (anchor - timedelta(days=int(lookback_days * 1.7) + 20)).isoformat()


def _load_prices(as_of: str, lookback_days: int) -> dict[str, dict[str, float]]:
    """``{symbol: {bar_date: price}}`` preferring adj_close, deduped by source."""
    cutoff = _cutoff(as_of, lookback_days)
    rows = fetch_all(
        """
        SELECT symbol, bar_date, close, adj_close, source
        FROM   price_data
        WHERE  bar_date <= ? AND bar_date >= ?
        ORDER BY symbol, bar_date
        """,
        (as_of, cutoff),
    )
    best_src: dict[tuple[str, str], int] = {}
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        px = r["adj_close"] if r["adj_close"] is not None else r["close"]
        if px is None or px <= 0:
            continue
        key = (r["symbol"], r["bar_date"])
        pr = _SOURCE_PRIORITY.get(r["source"], 99)
        if key in best_src and pr >= best_src[key]:
            continue
        best_src[key] = pr
        out.setdefault(r["symbol"], {})[r["bar_date"]] = float(px)
    return out


def _sectors() -> dict[str, str]:
    rows = fetch_all("SELECT symbol, sector FROM stock_sectors")
    return {r["symbol"]: (r["sector"] or "UNKNOWN") for r in rows}


def _align(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(a.keys() & b.keys())
    ya = np.array([a[d] for d in common], dtype=float)
    xa = np.array([b[d] for d in common], dtype=float)
    return ya, xa


def _ret_corr(ya: np.ndarray, xa: np.ndarray) -> float:
    if ya.size < 3:
        return 0.0
    ry = np.diff(np.log(ya))
    rx = np.diff(np.log(xa))
    if ry.std() == 0 or rx.std() == 0:
        return 0.0
    return float(np.corrcoef(ry, rx)[0, 1])


def scan_pairs(as_of: str | None = None,
               cfg: PairScanConfig | None = None) -> dict[str, Any]:
    """Scan sector-grouped pairs, keep the cointegrated set, persist + return a
    summary. Idempotent per ``as_of`` (replaces that date's rows)."""
    cfg = cfg or PairScanConfig()
    asof = _as_of(as_of)
    prices = _load_prices(asof, cfg.lookback_days)
    sectors = _sectors()

    by_sector: dict[str, list[str]] = {}
    for sym, series in prices.items():
        if len(series) < cfg.min_obs:
            continue
        sec = sectors.get(sym)
        if not sec or sec == "UNKNOWN":
            continue
        by_sector.setdefault(sec, []).append(sym)

    results: list[dict[str, Any]] = []
    n_tested = 0
    for sec, syms in by_sector.items():
        if len(syms) > cfg.max_per_sector:
            log.info("pairs: sector {} has {} names (> {}), skipping",
                     sec, len(syms), cfg.max_per_sector)
            continue
        for a, b in combinations(sorted(syms), 2):       # deterministic (Y=a, X=b)
            ya, xa = _align(prices[a], prices[b])
            if ya.size < cfg.min_obs:
                continue
            if abs(_ret_corr(ya, xa)) < cfg.min_corr:
                continue
            n_tested += 1
            res = coint.engle_granger(
                ya, xa, level=cfg.level,
                max_half_life=cfg.max_half_life, min_half_life=cfg.min_half_life,
            )
            if not res.cointegrated:
                continue
            sig = spr.signal_for(
                ya, xa, beta=res.beta, alpha=res.alpha, window=cfg.z_window,
                entry=cfg.entry, exit=cfg.exit, stop=cfg.stop,
            )
            results.append({
                "symbol_y": a, "symbol_x": b, "sector": sec,
                "beta": res.beta, "alpha": res.alpha, "adf_tstat": res.adf_tstat,
                "half_life": res.half_life, "corr": _ret_corr(ya, xa),
                "spread_mean": sig.mean, "spread_std": sig.std,
                "zscore": sig.zscore, "signal": sig.signal, "n_obs": res.n_obs,
            })

    _persist(asof, results)

    actionable = [r for r in results
                  if r["signal"] in (spr.LONG_SPREAD, spr.SHORT_SPREAD)]
    log.info("pairs scan {}: tested={} cointegrated={} actionable={}",
             asof, n_tested, len(results), len(actionable))
    return {
        "as_of": asof, "tested": n_tested,
        "cointegrated": len(results), "actionable": len(actionable),
        "pairs": results,
    }


def _persist(as_of: str, results: list[dict[str, Any]]) -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM pairs WHERE as_of_date = ?", (as_of,))
        for r in results:
            conn.execute(
                """
                INSERT INTO pairs
                    (as_of_date, symbol_y, symbol_x, sector, beta, alpha,
                     adf_tstat, half_life, corr, spread_mean, spread_std,
                     zscore, signal, n_obs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (as_of, r["symbol_y"], r["symbol_x"], r["sector"], r["beta"],
                 r["alpha"], r["adf_tstat"], r["half_life"], r["corr"],
                 r["spread_mean"], r["spread_std"], r["zscore"], r["signal"],
                 r["n_obs"]),
            )


def latest_pairs(as_of: str | None = None,
                 signal_only: bool = False) -> list[dict[str, Any]]:
    """Read back persisted pairs for ``as_of`` (latest scan if None)."""
    asof = as_of
    if asof is None:
        row = fetch_all("SELECT MAX(as_of_date) AS d FROM pairs")
        asof = row[0]["d"] if row else None
    if not asof:
        return []
    sql = "SELECT * FROM pairs WHERE as_of_date = ?"
    params: tuple[Any, ...] = (asof,)
    if signal_only:
        sql += " AND signal IN ('LONG_SPREAD', 'SHORT_SPREAD')"
    sql += " ORDER BY ABS(zscore) DESC"
    return [dict(r) for r in fetch_all(sql, params)]
