"""DB IO for portfolio construction: trailing daily returns + beta.

Pulls from ``feature_data`` (one row per symbol per day). Kept separate from the
pure math in :mod:`src.portfolio.correlation` / :mod:`src.portfolio.construct`
so those stay unit-testable without a database.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from src.utils.db import fetch_all
from src.utils.logger import get_logger
from src.portfolio.correlation import ReturnSeries

log = get_logger("portfolio.store")


def _as_of_date(as_of: str | None) -> str:
    return as_of or date.today().isoformat()


def _calendar_cutoff(as_of: str, lookback_days: int) -> str:
    """Calendar cutoff that comfortably contains ``lookback_days`` *trading*
    days (markets run ~5/7, minus holidays), with margin so the min-overlap
    check has room to work."""
    try:
        anchor = datetime.fromisoformat(as_of).date()
    except ValueError:
        anchor = date.today()
    span = int(lookback_days * 1.7) + 20
    return (anchor - timedelta(days=span)).isoformat()


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def load_returns(
    symbols: set[str] | list[str],
    as_of: str | None = None,
    lookback_days: int = 90,
    *,
    db_path: str | None = None,  # accepted for symmetry / tests; unused (global conn)
) -> dict[str, ReturnSeries]:
    """``{symbol: {feature_date: ret_1d}}`` over a trailing window up to
    ``as_of``. Symbols with no usable returns are simply absent."""
    syms = [s for s in dict.fromkeys(symbols)]  # de-dupe, keep order
    if not syms:
        return {}

    asof = _as_of_date(as_of)
    cutoff = _calendar_cutoff(asof, lookback_days)

    rows = fetch_all(
        f"""
        SELECT symbol, feature_date, ret_1d
        FROM   feature_data
        WHERE  symbol IN ({_placeholders(len(syms))})
          AND  feature_date <= ?
          AND  feature_date >= ?
          AND  ret_1d IS NOT NULL
        ORDER BY symbol, feature_date
        """,
        (*syms, asof, cutoff),
    )

    out: dict[str, ReturnSeries] = {}
    for r in rows:
        try:
            ret = float(r["ret_1d"])
        except (TypeError, ValueError):
            continue
        out.setdefault(r["symbol"], {})[r["feature_date"]] = ret

    # Trim each series to the most recent ``lookback_days`` observations.
    for sym, series in out.items():
        if len(series) > lookback_days:
            keep = sorted(series)[-lookback_days:]
            out[sym] = {d: series[d] for d in keep}
    return out


def load_betas(
    symbols: set[str] | list[str],
    as_of: str | None = None,
    *,
    db_path: str | None = None,
) -> dict[str, float]:
    """Latest ``beta_60d`` per symbol as of ``as_of`` (missing -> absent)."""
    syms = [s for s in dict.fromkeys(symbols)]
    if not syms:
        return {}

    asof = _as_of_date(as_of)
    rows = fetch_all(
        f"""
        SELECT fd.symbol, fd.beta_60d
        FROM   feature_data fd
        JOIN  (SELECT symbol, MAX(feature_date) AS md
               FROM feature_data
               WHERE feature_date <= ? AND beta_60d IS NOT NULL
                 AND symbol IN ({_placeholders(len(syms))})
               GROUP BY symbol) x
          ON  x.symbol = fd.symbol AND x.md = fd.feature_date
        """,
        (asof, *syms),
    )
    out: dict[str, float] = {}
    for r in rows:
        try:
            out[r["symbol"]] = float(r["beta_60d"])
        except (TypeError, ValueError):
            continue
    return out


def load_inputs(
    symbols: set[str] | list[str],
    as_of: str | None = None,
    lookback_days: int = 90,
) -> dict[str, Any]:
    """Convenience: returns + betas in one call."""
    return {
        "returns": load_returns(symbols, as_of, lookback_days),
        "betas": load_betas(symbols, as_of),
    }
