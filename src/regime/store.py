"""IO for the regime engine: read market context, classify, persist.

All database access for the regime engine lives here (the classifier and
breadth math stay pure). One row per day lands in ``market_regime``; the daily
pipeline calls :func:`store_regime` after features are built, and the signal
step reads :func:`latest_regime` to pick a strategy.

Look-ahead safety: every query is bounded by ``as_of`` (``<=``), and the
breadth cross-section uses the single most-recent ``feature_date <= as_of`` so
all names are measured on the same day.
"""

from __future__ import annotations

import json
from datetime import date
from statistics import fmean
from typing import Any

from src.features.breadth_features import compute_breadth
from src.regime.classifier import RegimeInputs, classify_regime
from src.utils.db import execute, fetch_all, fetch_one
from src.utils.logger import get_logger

log = get_logger("regime.store")

NIFTY_SYMBOL = "^NSEI"
VIX_SYMBOL = "^INDIAVIX"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _breadth_cross_section(as_of: str, db_path: str | None = None) -> list[dict[str, Any]]:
    """Latest feature_data cross-section on the most recent date <= as_of."""
    row = fetch_one(
        "SELECT MAX(feature_date) AS d FROM feature_data WHERE feature_date <= ?",
        (as_of,), db_path=db_path,
    )
    d = row["d"] if row else None
    if not d:
        return []
    rows = fetch_all(
        "SELECT dist_ema_50_pct, dist_ema_200_pct, dd_from_high_252d, ret_1d "
        "FROM feature_data WHERE feature_date = ?",
        (d,), db_path=db_path,
    )
    return [dict(r) for r in rows]


def _index_closes(symbol: str, as_of: str, db_path: str | None = None) -> list[float]:
    rows = fetch_all(
        "SELECT close FROM index_data WHERE index_symbol = ? AND bar_date <= ? "
        "ORDER BY bar_date",
        (symbol, as_of), db_path=db_path,
    )
    out: list[float] = []
    for r in rows:
        try:
            if r["close"] is not None:
                out.append(float(r["close"]))
        except (TypeError, ValueError):
            continue
    return out


def _nifty_vix_inputs(as_of: str, db_path: str | None = None) -> dict[str, Any]:
    """NIFTY MA booleans + latest VIX as of ``as_of`` (no look-ahead)."""
    closes = _index_closes(NIFTY_SYMBOL, as_of, db_path=db_path)
    last = closes[-1] if closes else None
    ma50 = fmean(closes[-50:]) if len(closes) >= 50 else None
    ma200 = fmean(closes[-200:]) if len(closes) >= 200 else None

    nifty_above_ma200 = (last > ma200) if (last is not None and ma200 is not None) else None
    nifty_ma50_gt_ma200 = (ma50 > ma200) if (ma50 is not None and ma200 is not None) else None

    vix_closes = _index_closes(VIX_SYMBOL, as_of, db_path=db_path)
    vix = vix_closes[-1] if vix_closes else None

    return {
        "nifty_above_ma200": nifty_above_ma200,
        "nifty_ma50_gt_ma200": nifty_ma50_gt_ma200,
        "vix": vix,
    }


def previous_regime(as_of: str, db_path: str | None = None) -> str | None:
    """Regime label of the most recent day strictly before ``as_of``."""
    row = fetch_one(
        "SELECT regime FROM market_regime WHERE as_of_date < ? "
        "ORDER BY as_of_date DESC LIMIT 1",
        (as_of,), db_path=db_path,
    )
    return row["regime"] if row else None


def regime_history(
    start: str | None = None,
    end: str | None = None,
    db_path: str | None = None,
) -> dict[str, str]:
    """Map of ``as_of_date -> regime`` over an optional window, for feeding
    ``run_backtest(regime_by_date=...)`` so trades get tagged with the regime
    active at entry."""
    clauses, params = [], []
    if start:
        clauses.append("as_of_date >= ?")
        params.append(start)
    if end:
        clauses.append("as_of_date <= ?")
        params.append(end)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = fetch_all(
        f"SELECT as_of_date, regime FROM market_regime{where} ORDER BY as_of_date",
        tuple(params), db_path=db_path,
    )
    return {r["as_of_date"]: r["regime"] for r in rows}


def latest_regime(as_of: str | None = None, db_path: str | None = None) -> str | None:
    """Most recent regime label on/at-or-before ``as_of`` (None if unknown)."""
    as_of = as_of or date.today().isoformat()
    row = fetch_one(
        "SELECT regime FROM market_regime WHERE as_of_date <= ? "
        "ORDER BY as_of_date DESC LIMIT 1",
        (as_of,), db_path=db_path,
    )
    return row["regime"] if row else None


# ---------------------------------------------------------------------------
# Compute + persist
# ---------------------------------------------------------------------------


def compute_regime(as_of: str, db_path: str | None = None) -> dict[str, Any]:
    """Compute (but do not store) the regime payload for ``as_of``."""
    breadth = compute_breadth(_breadth_cross_section(as_of, db_path=db_path))
    ctx = _nifty_vix_inputs(as_of, db_path=db_path)

    inputs = RegimeInputs(
        nifty_ma50_gt_ma200=ctx["nifty_ma50_gt_ma200"],
        nifty_above_ma200=ctx["nifty_above_ma200"],
        vix=ctx["vix"],
        breadth_score=breadth.get("breadth_score"),
        pct_above_200dma=breadth.get("pct_above_200dma"),
    )
    prev = previous_regime(as_of, db_path=db_path)
    result = classify_regime(inputs, prev=prev)

    return {
        "as_of_date": as_of,
        "regime": result.regime,
        "prev_regime": prev,
        "reasons": result.reasons,
        "context": ctx,
        "breadth": breadth,
    }


def _bool_to_int(value: bool | None) -> int | None:
    return None if value is None else (1 if value else 0)


def store_regime(as_of: str, db_path: str | None = None) -> dict[str, Any]:
    """Compute the regime for ``as_of`` and upsert it into ``market_regime``."""
    payload = compute_regime(as_of, db_path=db_path)
    ctx = payload["context"]
    breadth = payload["breadth"]

    execute(
        """
        INSERT INTO market_regime
            (as_of_date, regime, nifty_above_ma200, nifty_ma50_gt_ma200, vix,
             pct_above_50dma, pct_above_200dma, adv_decl_ratio, breadth_score,
             scores_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(as_of_date) DO UPDATE SET
            regime              = excluded.regime,
            nifty_above_ma200   = excluded.nifty_above_ma200,
            nifty_ma50_gt_ma200 = excluded.nifty_ma50_gt_ma200,
            vix                 = excluded.vix,
            pct_above_50dma     = excluded.pct_above_50dma,
            pct_above_200dma    = excluded.pct_above_200dma,
            adv_decl_ratio      = excluded.adv_decl_ratio,
            breadth_score       = excluded.breadth_score,
            scores_json         = excluded.scores_json
        """,
        (
            as_of,
            payload["regime"],
            _bool_to_int(ctx["nifty_above_ma200"]),
            _bool_to_int(ctx["nifty_ma50_gt_ma200"]),
            ctx["vix"],
            breadth.get("pct_above_50dma"),
            breadth.get("pct_above_200dma"),
            breadth.get("adv_decl_ratio"),
            breadth.get("breadth_score"),
            json.dumps(payload, default=str),
        ),
        db_path=db_path,
    )
    log.info("regime {} -> {} ({})", as_of, payload["regime"],
             "; ".join(payload["reasons"]))
    return payload
