"""Read-only DB queries for the web dashboard.

Every function here is:
* parameterised (no string concatenation into SQL),
* read-only (only ``SELECT``s),
* defensive about missing tables / NULL columns so the UI degrades
  gracefully on a fresh DB rather than throwing 500s,
* typed via dataclasses so the templates can dot-access fields (e.g.
  ``{{ row.symbol }}``) and tests can assert on shape.

We deliberately do NOT import anything from ``src.notifications`` --
the dashboard reuses the same DB but its presentation logic is its own
concern.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from src.utils.db import fetch_all, fetch_one
from src.utils.logger import get_logger

log = get_logger("web.queries")


# ---------------------------------------------------------------------------
# Row types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalRow:
    symbol: str
    signal_date: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: int
    confidence: float
    status: str           # pending | executed | skipped | failed
    sector: str | None


@dataclass(frozen=True)
class OpenPositionRow:
    paper_id: int
    symbol: str
    sector: str
    qty: int
    entry_date: str
    entry_price: float
    last_close: float | None
    stop_loss: float | None
    take_profit: float | None
    unrealised_pnl: float
    unrealised_pnl_pct: float
    holding_days: int


@dataclass(frozen=True)
class ClosedTradeRow:
    paper_id: int
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    net_pnl: float
    pnl_pct: float
    exit_reason: str
    holding_days: int


@dataclass(frozen=True)
class CandidateRow:
    """A ranked model prediction for the latest scored date.

    Distinct from a SignalRow: a candidate is *every* symbol the model
    scored, ranked by conviction, whether or not it crossed the action
    threshold. This powers the watchlist so the dashboard is useful for
    analysis even on days when nothing fires.
    """
    symbol: str
    prediction_date: str
    raw_prob: float
    calibrated_prob: float
    sector: str | None
    threshold: float | None
    distance_to_threshold: float | None   # calibrated - threshold (neg = below)
    would_fire: bool


@dataclass(frozen=True)
class FreshnessRow:
    latest_price_date: str | None
    latest_feature_date: str | None
    latest_prediction_date: str | None
    days_since_last_price: int | None
    is_stale: bool


@dataclass(frozen=True)
class ModelRow:
    run_id: str
    model_name: str
    trained_from: str | None
    trained_to: str | None
    threshold: float | None
    created_at: str


@dataclass(frozen=True)
class DashboardSnapshot:
    """Bundle returned to the home page so the template renders in one go."""
    as_of: str
    signals: list[SignalRow] = field(default_factory=list)
    open_positions: list[OpenPositionRow] = field(default_factory=list)
    closed_recent: list[ClosedTradeRow] = field(default_factory=list)
    realised_pnl_30d: float = 0.0
    unrealised_pnl: float = 0.0
    win_rate_30d_pct: float = 0.0
    n_open: int = 0
    freshness: FreshnessRow | None = None
    model: ModelRow | None = None
    universe_size: int = 0
    candidates: list[CandidateRow] = field(default_factory=list)
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        # asdict on a None field is fine; we just want JSON-clean booleans.
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_fetch(query: str, params: tuple = ()) -> list[dict[str, Any]]:
    """fetch_all that swallows missing-table errors so the dashboard can
    show "no data yet" instead of a 500 on a fresh deploy."""
    try:
        return [dict(r) for r in fetch_all(query, params)]
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard query failed (returning empty): {}", exc)
        return []


def _safe_fetch_one(query: str, params: tuple = ()) -> dict[str, Any] | None:
    try:
        row = fetch_one(query, params)
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard query failed (returning None): {}", exc)
        return None


def _holding_days(entry: str, today: str) -> int:
    try:
        return max(0, (date.fromisoformat(today) - date.fromisoformat(entry)).days)
    except (ValueError, TypeError):
        return 0


def _coerce_threshold(raw: Any) -> float | None:
    """metrics_json.threshold may be a number or {"value": ..., ...} dict.

    Same shape-tolerant logic the report builder uses; duplicated here
    so the web layer doesn't need to import the notifications module.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        v = raw.get("value")
        if isinstance(v, (int, float)):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Section loaders
# ---------------------------------------------------------------------------


def get_latest_signal_date() -> str | None:
    """Return the most recent ``signal_date`` present in ``signal_outbox``,
    or ``None`` if the table is empty. We use this so the dashboard can
    show "the latest available run" rather than going dark on weekends,
    holidays, and 4-day-old features."""
    row = _safe_fetch_one(
        "SELECT MAX(signal_date) AS d FROM signal_outbox"
    )
    return (row or {}).get("d")


def get_today_signals(as_of: str | None = None) -> list[SignalRow]:
    """Signals for ``as_of`` (defaults to today). If today has no signals
    we fall back to the latest signal date we have, so the user sees
    *something* even if features/predictions are stale."""
    sd = as_of or date.today().isoformat()
    sql = """
        SELECT s.symbol, s.signal_date, s.side, s.entry_price, s.stop_loss,
               s.take_profit, s.qty, s.confidence, s.status,
               ss.sector AS sector
        FROM   signal_outbox s
        LEFT   JOIN stock_sectors ss ON ss.symbol = s.symbol
        WHERE  s.signal_date = ?
        ORDER  BY s.confidence DESC
    """
    rows = _safe_fetch(sql, (sd,))
    if not rows and as_of is None:
        # Caller didn't pin a date and today is dry -- fall back so the
        # dashboard isn't blank when the latest run is from Friday.
        latest = get_latest_signal_date()
        if latest and latest != sd:
            rows = _safe_fetch(sql, (latest,))
    return [SignalRow(
        symbol=r["symbol"],
        signal_date=r["signal_date"],
        side=r["side"],
        entry_price=float(r["entry_price"] or 0.0),
        stop_loss=float(r["stop_loss"] or 0.0),
        take_profit=float(r["take_profit"] or 0.0),
        qty=int(r["qty"] or 0),
        confidence=float(r["confidence"] or 0.0),
        status=r["status"],
        sector=r["sector"],
    ) for r in rows]


def get_latest_prediction_date() -> str | None:
    """Most recent ``prediction_date`` present in ``predictions_log``."""
    row = _safe_fetch_one(
        "SELECT MAX(prediction_date) AS d FROM predictions_log"
    )
    return (row or {}).get("d")


def get_top_candidates(
    limit: int = 15,
    as_of: str | None = None,
    threshold: float | None = None,
) -> list[CandidateRow]:
    """Ranked model predictions for the latest scored date.

    Always returns the model's most confident names (by calibrated
    probability) regardless of whether they crossed the action threshold,
    so the dashboard can show a watchlist on days when no signal fires.

    We pin to the *most recently inserted* run for that date so re-running
    predict_today (e.g. after a retrain) doesn't mix stale rows from older
    model runs into the ranking.
    """
    pred_date = as_of or get_latest_prediction_date()
    if not pred_date:
        return []
    if threshold is None:
        model = get_latest_model()
        threshold = model.threshold if model else None

    rows = _safe_fetch(
        """
        SELECT p.symbol, p.prediction_date, p.raw_prob, p.calibrated_prob,
               ss.sector AS sector
        FROM   predictions_log p
        LEFT   JOIN stock_sectors ss ON ss.symbol = p.symbol
        WHERE  p.prediction_date = ?
          AND  p.run_id = (
                 SELECT run_id FROM predictions_log
                 WHERE  prediction_date = ?
                 ORDER  BY id DESC LIMIT 1
               )
        ORDER  BY p.calibrated_prob DESC, p.raw_prob DESC
        LIMIT  ?
        """,
        (pred_date, pred_date, int(limit)),
    )
    out: list[CandidateRow] = []
    for r in rows:
        cal = float(r["calibrated_prob"] or 0.0)
        dist = (cal - threshold) if threshold is not None else None
        out.append(CandidateRow(
            symbol=r["symbol"],
            prediction_date=r["prediction_date"],
            raw_prob=float(r["raw_prob"] or 0.0),
            calibrated_prob=cal,
            sector=r["sector"],
            threshold=threshold,
            distance_to_threshold=dist,
            would_fire=(threshold is not None and cal >= threshold),
        ))
    return out


def get_open_positions(as_of: str | None = None) -> list[OpenPositionRow]:
    today = as_of or date.today().isoformat()
    rows = _safe_fetch(
        """
        SELECT pt.id, pt.symbol, pt.sector, pt.qty, pt.entry_date,
               pt.entry_price, pt.stop_loss, pt.take_profit
        FROM   paper_trades pt
        WHERE  pt.status = 'open'
        ORDER  BY pt.entry_date ASC, pt.id ASC
        """
    )
    out: list[OpenPositionRow] = []
    for r in rows:
        sym = r["symbol"]
        last = _safe_fetch_one(
            "SELECT close FROM price_data WHERE symbol = ? AND bar_date <= ? "
            "AND source = 'yfinance' ORDER BY bar_date DESC LIMIT 1",
            (sym, today),
        )
        last_close = float(last["close"]) if last and last.get("close") is not None else None
        qty = int(r["qty"] or 0)
        entry = float(r["entry_price"] or 0.0)
        if last_close is not None and qty > 0 and entry > 0:
            unreal = (last_close - entry) * qty
            unreal_pct = (last_close - entry) / entry * 100.0
        else:
            unreal = 0.0
            unreal_pct = 0.0
        out.append(OpenPositionRow(
            paper_id=int(r["id"]),
            symbol=sym,
            sector=r["sector"] or "UNKNOWN",
            qty=qty,
            entry_date=r["entry_date"] or "",
            entry_price=entry,
            last_close=last_close,
            stop_loss=(None if r["stop_loss"] is None else float(r["stop_loss"])),
            take_profit=(None if r["take_profit"] is None else float(r["take_profit"])),
            unrealised_pnl=unreal,
            unrealised_pnl_pct=unreal_pct,
            holding_days=_holding_days(r["entry_date"] or today, today),
        ))
    return out


def get_recent_closed(window_days: int = 30, limit: int = 50) -> list[ClosedTradeRow]:
    rows = _safe_fetch(
        """
        SELECT id, symbol, entry_date, exit_date, entry_price, exit_price,
               qty, pnl_rupees, pnl_pct, exit_reason
        FROM   paper_trades
        WHERE  status = 'closed' AND exit_date IS NOT NULL
          AND  exit_date >= date('now', ?)
        ORDER  BY exit_date DESC, id DESC
        LIMIT  ?
        """,
        (f"-{int(window_days)} days", int(limit)),
    )
    out: list[ClosedTradeRow] = []
    for r in rows:
        out.append(ClosedTradeRow(
            paper_id=int(r["id"]),
            symbol=r["symbol"],
            entry_date=r["entry_date"] or "",
            exit_date=r["exit_date"] or "",
            entry_price=float(r["entry_price"] or 0.0),
            exit_price=float(r["exit_price"] or 0.0),
            qty=int(r["qty"] or 0),
            net_pnl=float(r["pnl_rupees"] or 0.0),
            pnl_pct=float(r["pnl_pct"] or 0.0),
            exit_reason=r["exit_reason"] or "",
            holding_days=_holding_days(r["entry_date"] or "", r["exit_date"] or ""),
        ))
    return out


def get_freshness(as_of: str | None = None) -> FreshnessRow:
    today = as_of or date.today().isoformat()
    p = _safe_fetch_one(
        "SELECT MAX(bar_date) AS d FROM price_data WHERE source = 'yfinance'"
    )
    f = _safe_fetch_one("SELECT MAX(feature_date) AS d FROM feature_data")
    pr = _safe_fetch_one("SELECT MAX(prediction_date) AS d FROM predictions_log")
    latest_price = (p or {}).get("d")
    days_since: int | None = None
    if latest_price:
        try:
            days_since = (
                date.fromisoformat(today) - date.fromisoformat(latest_price)
            ).days
        except (ValueError, TypeError):
            days_since = None
    return FreshnessRow(
        latest_price_date=latest_price,
        latest_feature_date=(f or {}).get("d"),
        latest_prediction_date=(pr or {}).get("d"),
        days_since_last_price=days_since,
        is_stale=(days_since is not None and days_since > 3),
    )


def get_latest_model() -> ModelRow | None:
    row = _safe_fetch_one(
        """
        SELECT run_id, model_name, trained_from, trained_to,
               metrics_json, created_at
        FROM   model_runs
        ORDER  BY created_at DESC, rowid DESC
        LIMIT  1
        """
    )
    if not row:
        return None
    try:
        metrics = json.loads(row.get("metrics_json") or "{}")
    except json.JSONDecodeError:
        metrics = {}
    return ModelRow(
        run_id=row["run_id"],
        model_name=row["model_name"],
        trained_from=row.get("trained_from"),
        trained_to=row.get("trained_to"),
        threshold=_coerce_threshold(metrics.get("threshold")),
        created_at=row["created_at"],
    )


def get_universe_size() -> int:
    row = _safe_fetch_one("SELECT COUNT(*) AS n FROM v_universe_today")
    return int((row or {}).get("n") or 0)


def build_dashboard_snapshot(as_of: str | None = None) -> DashboardSnapshot:
    """One round-trip helper for the home page; aggregates the sections
    above plus a few rolled-up portfolio numbers.

    The displayed ``as_of`` becomes the *effective* signal date -- if
    today has no signals we surface the latest one we have instead, so
    the page never silently lies about freshness.
    """
    today = as_of or date.today().isoformat()
    signals = get_today_signals(today)
    # If we fell back, use that date as the headline "as of"; otherwise
    # keep today's date so the user sees a fresh run.
    effective_as_of = signals[0].signal_date if signals else today

    open_pos = get_open_positions(today)
    closed = get_recent_closed(window_days=30, limit=50)
    realised = sum(t.net_pnl for t in closed)
    unrealised = sum(p.unrealised_pnl for p in open_pos)
    if closed:
        win_rate = (sum(1 for t in closed if t.net_pnl > 0) / len(closed)) * 100.0
    else:
        win_rate = 0.0
    model = get_latest_model()
    threshold = model.threshold if model else None
    candidates = get_top_candidates(limit=15, threshold=threshold)
    return DashboardSnapshot(
        as_of=effective_as_of,
        signals=signals,
        open_positions=open_pos,
        closed_recent=closed,
        realised_pnl_30d=realised,
        unrealised_pnl=unrealised,
        win_rate_30d_pct=win_rate,
        n_open=len(open_pos),
        freshness=get_freshness(today),
        model=model,
        universe_size=get_universe_size(),
        candidates=candidates,
        threshold=threshold,
    )


# Useful for /healthz JSON without doing the full snapshot.
def get_health() -> dict[str, Any]:
    fr = get_freshness()
    return {
        "ok": True,
        "now_utc": datetime.now(timezone.utc).isoformat(),
        "freshness": asdict(fr),
    }
