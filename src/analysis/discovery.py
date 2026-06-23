"""AI Stock Discovery / Screener.

Scans the stored universe (latest ``feature_data`` + ``fundamental_data`` +
``predictions_log`` row per symbol) and ranks names by preset strategies:
value, growth, quality, momentum, breakout, oversold-in-uptrend, and the
model's top-conviction picks.

It reads only the *latest* stored row per symbol (one cheap pass, cached ~10
min) -- no per-request recompute -- so it's instant and works against either the
local SQLite or the cloud Postgres mirror (same ``fetch_all`` backend, same
``?`` placeholders). Pure ranking logic; never writes to the DB.

Field scale conventions (consistent with the rest of the app):
* ratios from ``fundamental_data`` are fractions (roe 0.18 == 18%);
* returns/momentum/drawdown/dist_ema from ``feature_data`` are fractions
  (mom_60d 0.12 == +12%, dd_from_high_252d -0.05 == 5% below the 1y high).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from src.utils.logger import get_logger

log = get_logger("analysis.discovery")

_CACHE_TTL_S = 600.0
_cache: dict[str, Any] = {"at": 0.0, "rows": None}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Strategy catalogue (order = display order)
# ---------------------------------------------------------------------------

STRATEGIES: list[dict[str, str]] = [
    {"key": "top_conviction", "label": "Top conviction",
     "blurb": "Highest model 'up' probability for the latest scored day."},
    {"key": "value", "label": "Undervalued",
     "blurb": "Low P/E and P/B with respectable returns on equity."},
    {"key": "growth", "label": "Growth",
     "blurb": "Strong revenue growth and high ROE."},
    {"key": "quality", "label": "Quality",
     "blurb": "High ROE and margins with a low debt load."},
    {"key": "momentum", "label": "Momentum",
     "blurb": "Trending up, above the 50-EMA, healthy RSI."},
    {"key": "breakout", "label": "Breakout",
     "blurb": "Within ~5% of the 52-week high and still rising."},
    {"key": "oversold", "label": "Oversold dip",
     "blurb": "Oversold (RSI<35) but still above the 200-EMA uptrend."},
]
_STRATEGY_KEYS = {s["key"] for s in STRATEGIES}


# ---------------------------------------------------------------------------
# Data: latest stored row per symbol (cached)
# ---------------------------------------------------------------------------


def _load_rows() -> list[dict[str, Any]]:
    """Merge the latest feature/fundamental/prediction row for every symbol."""
    from src.utils.db import fetch_all

    feats = fetch_all(
        """
        SELECT fd.symbol, fd.close, fd.rsi_14, fd.macd_hist,
               fd.mom_20d, fd.mom_60d, fd.ret_20d, fd.dd_from_high_252d,
               fd.dist_ema_50_pct, fd.dist_ema_200_pct, fd.vol_20d,
               fd.bb_pct_b, fd.vol_ratio_20d, fd.atr_pct, fd.stoch_k
        FROM   feature_data fd
        JOIN  (SELECT symbol, MAX(feature_date) AS md
               FROM feature_data GROUP BY symbol) x
          ON  x.symbol = fd.symbol AND x.md = fd.feature_date
        """
    )
    funds = fetch_all(
        """
        SELECT f.symbol, f.pe_ttm, f.pb, f.roe, f.debt_to_equity,
               f.profit_margin, f.revenue_growth, f.earnings_growth,
               f.dividend_yield, f.market_cap
        FROM   fundamental_data f
        JOIN  (SELECT symbol, MAX(as_of_date) AS md
               FROM fundamental_data WHERE source='yfinance_snapshot'
               GROUP BY symbol) x
          ON  x.symbol = f.symbol AND x.md = f.as_of_date
        WHERE  f.source = 'yfinance_snapshot'
        """
    )
    sectors = fetch_all("SELECT symbol, sector FROM stock_sectors")

    pred_rows: list[Any] = []
    pdate = fetch_all("SELECT MAX(prediction_date) AS d FROM predictions_log")
    latest = pdate[0]["d"] if pdate else None
    if latest:
        pred_rows = fetch_all(
            """
            SELECT symbol, calibrated_prob, verdict, predicted_return, target_price
            FROM   predictions_log
            WHERE  prediction_date = ?
              AND  run_id = (SELECT run_id FROM predictions_log
                             WHERE prediction_date = ?
                             ORDER BY id DESC LIMIT 1)
            """,
            (latest, latest),
        )

    by_sym: dict[str, dict[str, Any]] = {}
    for r in feats:
        by_sym[r["symbol"]] = dict(r)
    fund_map = {r["symbol"]: dict(r) for r in funds}
    sec_map = {r["symbol"]: r["sector"] for r in sectors}
    pred_map = {r["symbol"]: dict(r) for r in pred_rows}

    rows: list[dict[str, Any]] = []
    for sym, row in by_sym.items():
        row.update(fund_map.get(sym, {}))
        row["sector"] = sec_map.get(sym, "UNKNOWN")
        row.update(pred_map.get(sym, {}))
        rows.append(row)
    return rows


def _rows(force: bool = False) -> list[dict[str, Any]]:
    now = time.monotonic()
    with _lock:
        if (not force and _cache["rows"] is not None
                and (now - _cache["at"]) < _CACHE_TTL_S):
            return _cache["rows"]
    try:
        rows = _load_rows()
    except Exception as exc:  # noqa: BLE001
        log.warning("discovery load failed: {}", exc)
        rows = []
    with _lock:
        _cache["rows"] = rows
        _cache["at"] = now
    return rows


# ---------------------------------------------------------------------------
# Strategy scorers: return a 0-100 fit score, or None to exclude the row
# ---------------------------------------------------------------------------


def _g(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _score_top_conviction(r: dict[str, Any]) -> float | None:
    p = _g(r, "calibrated_prob")
    return None if p is None else _clamp(p * 100)


def _score_value(r: dict[str, Any]) -> float | None:
    pe, pb, roe = _g(r, "pe_ttm"), _g(r, "pb"), _g(r, "roe")
    if pe is None or pe <= 0 or pe >= 22:
        return None
    if pb is not None and pb > 4:
        return None
    if roe is not None and roe < 0.10:
        return None
    score = _clamp(100 - pe * 3.0)
    if pb is not None:
        score += _clamp(30 - pb * 6, 0, 30) * 0.3
    if roe is not None:
        score += _clamp(roe * 100, 0, 30) * 0.3
    return _clamp(score)


def _score_growth(r: dict[str, Any]) -> float | None:
    rev, earn, roe = _g(r, "revenue_growth"), _g(r, "earnings_growth"), _g(r, "roe")
    if rev is None or rev < 0.15:
        return None
    if roe is not None and roe < 0.12:
        return None
    if earn is not None and earn < -0.05:
        return None
    score = _clamp(rev * 100 * 2)
    if roe is not None:
        score = 0.6 * score + 0.4 * _clamp(roe * 100 * 3)
    return _clamp(score)


def _score_quality(r: dict[str, Any]) -> float | None:
    roe, de, margin = _g(r, "roe"), _g(r, "debt_to_equity"), _g(r, "profit_margin")
    if roe is None or roe < 0.15:
        return None
    if de is not None and de > 1.0:
        return None
    if margin is not None and margin < 0.08:
        return None
    score = _clamp(roe * 100 * 2.5)
    if margin is not None:
        score = 0.6 * score + 0.4 * _clamp(margin * 100 * 3)
    if de is not None:
        score += _clamp(20 - de * 20, 0, 20) * 0.2
    return _clamp(score)


def _score_momentum(r: dict[str, Any]) -> float | None:
    m60, r20, d50, rsi = (_g(r, "mom_60d"), _g(r, "ret_20d"),
                          _g(r, "dist_ema_50_pct"), _g(r, "rsi_14"))
    if m60 is None or m60 < 0.08:
        return None
    if r20 is not None and r20 < 0:
        return None
    if d50 is not None and d50 < 0:          # below 50-EMA -> not momentum
        return None
    if rsi is not None and (rsi < 50 or rsi > 80):
        return None
    return _clamp(50 + m60 * 100 * 1.5)


def _score_breakout(r: dict[str, Any]) -> float | None:
    dd, d50, m20 = (_g(r, "dd_from_high_252d"), _g(r, "dist_ema_50_pct"),
                    _g(r, "mom_20d"))
    if dd is None or dd < -0.06:             # >6% below 52w high -> not breakout
        return None
    if d50 is not None and d50 < 0:
        return None
    if m20 is not None and m20 < 0:
        return None
    # Closer to the high (dd near 0) scores higher.
    return _clamp(100 + dd * 100 * 8)        # dd=-0.06 -> ~52, dd=0 -> 100


def _score_oversold(r: dict[str, Any]) -> float | None:
    rsi, d200 = _g(r, "rsi_14"), _g(r, "dist_ema_200_pct")
    if rsi is None or rsi >= 35:
        return None
    if d200 is None or d200 < 0:             # must still be above the 200-EMA
        return None
    return _clamp((35 - rsi) * 3 + 40)


_SCORERS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "top_conviction": _score_top_conviction,
    "value": _score_value,
    "growth": _score_growth,
    "quality": _score_quality,
    "momentum": _score_momentum,
    "breakout": _score_breakout,
    "oversold": _score_oversold,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(strategy: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return the top ``limit`` matches for ``strategy``, ranked by fit score.

    Each result carries the key metrics the UI shows plus a ``score`` (0-100,
    strategy fit) and the model's ``calibrated_prob`` where available.
    """
    scorer = _SCORERS.get(strategy)
    if scorer is None:
        return []
    out: list[dict[str, Any]] = []
    for r in _rows():
        s = scorer(r)
        if s is None:
            continue
        out.append({
            "symbol": r["symbol"],
            "sector": r.get("sector", "UNKNOWN"),
            "score": round(s),
            "close": r.get("close"),
            "pe_ttm": r.get("pe_ttm"),
            "roe": r.get("roe"),
            "revenue_growth": r.get("revenue_growth"),
            "debt_to_equity": r.get("debt_to_equity"),
            "mom_60d": r.get("mom_60d"),
            "ret_20d": r.get("ret_20d"),
            "rsi_14": r.get("rsi_14"),
            "dd_from_high_252d": r.get("dd_from_high_252d"),
            "calibrated_prob": r.get("calibrated_prob"),
            "verdict": r.get("verdict"),
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[: int(limit)]


def universe_count() -> int:
    return len(_rows())
