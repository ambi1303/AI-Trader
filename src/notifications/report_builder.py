"""Build a structured DailyReport from the SQLite store.

This module is the single source of truth for *what goes in the daily
notification*. The output is a plain dataclass so templates / PDF / dispatcher
can render it without re-querying the DB.

Design choices
--------------
- Read-only: every query goes through ``fetch_all`` / ``fetch_one`` and never
  touches the writer connection.
- Tolerant of empty tables: any missing data degrades to None / empty list
  and is reported as such in the template, never as an exception.
- No raw user input flows into SQL; symbols and dates are parameterised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.db import fetch_all, fetch_one
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("notifications.report_builder")


# ---------------------------------------------------------------------------
# Data classes (immutable record of the snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRow:
    symbol: str
    calibrated_prob: float | None
    raw_prob: float | None
    prediction_date: str
    is_signal: bool


@dataclass(frozen=True)
class ModelSnapshot:
    run_id: str
    model_name: str
    git_sha: str | None
    feature_hash: str | None
    trained_from: str | None
    trained_to: str | None
    metrics: dict[str, Any]
    threshold: float | None
    created_at: str


@dataclass(frozen=True)
class BacktestSnapshot:
    bt_run_id: str
    name: str
    start_date: str | None
    end_date: str | None
    initial_capital: float
    metrics: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TradeRow:
    symbol: str
    side: str
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
class ValidationSummary:
    total: int
    by_severity: dict[str, int]
    latest_check_date: str | None
    window_days: int = 7   # rolling window the totals were measured over


@dataclass(frozen=True)
class OpenPaperPosition:
    paper_id: int
    symbol: str
    sector: str
    qty: int
    entry_date: str
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    last_close: float | None
    unrealised_pnl: float
    unrealised_pnl_pct: float
    holding_days: int


@dataclass(frozen=True)
class PaperTradeRow:
    """Recently CLOSED paper trade (mirror of TradeRow but from paper_trades)."""
    symbol: str
    side: str
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
class PaperPortfolio:
    open_positions: list[OpenPaperPosition]
    closed_recent: list[PaperTradeRow]
    realised_pnl_window: float          # net pnl of trades closed in window
    unrealised_pnl: float                # sum of marks
    open_count: int
    win_rate_pct_window: float           # of closed trades in window
    window_days: int


@dataclass(frozen=True)
class WalkForwardSummary:
    """Aggregate out-of-sample metrics from the most recent walk-forward eval.

    Walk-forward = train on the past, test on the *next* unseen window, roll
    forward. This is a far more honest read on "does the model work?" than a
    single 70/15/15 split, which can get lucky on one regime. We surface it
    when ``scripts.walk_forward_eval --save`` has been run; otherwise None.
    """
    generated_at: str | None
    folds_completed: int
    n_predictions: int
    auc: float | None             # calibrated, out-of-sample
    brier: float | None
    mean_threshold: float | None
    source_file: str | None


@dataclass(frozen=True)
class DataFreshness:
    latest_price_date: str | None
    latest_feature_date: str | None
    latest_prediction_date: str | None
    days_since_last_price: int | None
    is_stale: bool                       # True if > 3 calendar days old


@dataclass(frozen=True)
class DailyReport:
    """Everything one daily notification needs to render."""

    generated_at: str             # ISO timestamp (UTC) for audit
    report_date: str              # the calendar date the report represents
    universe_size: int
    predictions: list[PredictionRow]
    signals: list[PredictionRow]   # subset of predictions where is_signal=True
    latest_model: ModelSnapshot | None
    latest_backtest: BacktestSnapshot | None
    recent_trades: list[TradeRow]
    validation: ValidationSummary
    paper: PaperPortfolio
    freshness: DataFreshness
    walk_forward: WalkForwardSummary | None = None

    # Derived/quick-glance fields used for the WhatsApp short summary.
    top_n: int = field(default=0)
    threshold_used: float | None = field(default=None)

    @property
    def headline(self) -> str:
        """One-line punch-up used as the subject + first WhatsApp line."""
        n_sig = len(self.signals)
        sharpe = "n/a"
        if self.latest_backtest and "sharpe" in (self.latest_backtest.metrics or {}):
            try:
                sharpe = f"{float(self.latest_backtest.metrics['sharpe']):.2f}"
            except (TypeError, ValueError):
                sharpe = "n/a"
        paper = self.paper
        paper_part = (
            f"open={paper.open_count}, "
            f"unreal={paper.unrealised_pnl:+,.0f}"
            if paper else ""
        )
        return (
            f"{self.report_date}: {n_sig} signal(s) | "
            f"{paper_part} | last-Sharpe={sharpe}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_metrics_json(blob: str | None) -> dict[str, Any]:
    """Defensive JSON load. Bad blobs degrade to {} (logged but non-fatal)."""
    if not blob:
        return {}
    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        # Logged so we notice corruption, but don't crash the report.
        log.warning("Could not parse metrics_json: {}", exc)
    return {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_report_date(override: str | None) -> str:
    """If caller supplies a date use it, else 'today' in IST-local terms.

    We pick local date because the user perceives "today's report" in IST,
    not UTC. The DB stores all dates as ISO YYYY-MM-DD already.
    """
    if override:
        return override
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _load_universe_size() -> int:
    row = fetch_one("SELECT COUNT(*) AS n FROM v_universe_today")
    return int(row["n"]) if row else 0


def _load_predictions(report_date: str, top_n: int, threshold: float | None) -> list[PredictionRow]:
    """Top-N predictions for ``report_date`` ordered by calibrated_prob desc.

    If ``threshold`` is provided, ``is_signal = calibrated_prob >= threshold``.
    """
    rows = fetch_all(
        """
        SELECT symbol, raw_prob, calibrated_prob, prediction_date
        FROM   predictions_log
        WHERE  prediction_date = ?
        ORDER BY (calibrated_prob IS NULL), calibrated_prob DESC
        LIMIT  ?
        """,
        (report_date, int(top_n)),
    )
    out: list[PredictionRow] = []
    for r in rows:
        cp = r["calibrated_prob"]
        sig = (
            cp is not None and threshold is not None and float(cp) >= float(threshold)
        )
        out.append(
            PredictionRow(
                symbol=r["symbol"],
                calibrated_prob=None if cp is None else float(cp),
                raw_prob=None if r["raw_prob"] is None else float(r["raw_prob"]),
                prediction_date=r["prediction_date"],
                is_signal=bool(sig),
            )
        )
    return out


def _coerce_threshold(raw: Any) -> float | None:
    """Same shape-tolerant coercion as ``signals.generator._coerce_threshold``.

    Duplicated here (rather than imported) to keep the report builder
    free of any signal-layer import -- it must remain read-only.
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


def _load_latest_model() -> ModelSnapshot | None:
    """Most recent training run. Tie-break on rowid for determinism."""
    row = fetch_one(
        """
        SELECT run_id, model_name, git_sha, feature_hash,
               trained_from, trained_to, metrics_json, created_at
        FROM   model_runs
        ORDER  BY created_at DESC, rowid DESC
        LIMIT  1
        """
    )
    if not row:
        return None
    metrics = _parse_metrics_json(row["metrics_json"])
    threshold = _coerce_threshold(metrics.get("threshold"))
    return ModelSnapshot(
        run_id=row["run_id"],
        model_name=row["model_name"],
        git_sha=row["git_sha"],
        feature_hash=row["feature_hash"],
        trained_from=row["trained_from"],
        trained_to=row["trained_to"],
        metrics=metrics,
        threshold=threshold,
        created_at=row["created_at"],
    )


def _load_latest_backtest() -> BacktestSnapshot | None:
    # Prefer a representative full-period backtest over one-off STRESS scenarios.
    # Scenario runs are persisted with a 'stress_' name prefix and are *meant*
    # to look bad (they replay corrections/crashes), so anchoring the daily
    # headline on them is misleading. `(name LIKE 'stress_%')` sorts non-stress
    # runs (0) ahead of stress runs (1); we fall back to a stress run only if no
    # canonical backtest exists.
    row = fetch_one(
        """
        SELECT bt_run_id, name, start_date, end_date,
               initial_capital, metrics_json, created_at
        FROM   backtest_runs
        ORDER  BY (name LIKE 'stress_%'), created_at DESC, bt_run_id DESC
        LIMIT  1
        """
    )
    if not row:
        return None
    return BacktestSnapshot(
        bt_run_id=row["bt_run_id"],
        name=row["name"] or "(unnamed)",
        start_date=row["start_date"],
        end_date=row["end_date"],
        initial_capital=float(row["initial_capital"]),
        metrics=_parse_metrics_json(row["metrics_json"]),
        created_at=row["created_at"],
    )


def _load_recent_trades(n: int) -> list[TradeRow]:
    """Most recent N trades from the latest backtest run.

    We pull from the latest run only so the recipient gets a coherent set
    of comparable rows (same window/cost model). Falls back to empty.
    """
    bt = fetch_one(
        """
        SELECT bt_run_id FROM backtest_runs
        ORDER BY (name LIKE 'stress_%'), created_at DESC, bt_run_id DESC
        LIMIT 1
        """
    )
    if not bt:
        return []
    rows = fetch_all(
        """
        SELECT symbol, side, entry_date, exit_date, entry_price, exit_price,
               qty, gross_pnl, cost_rupees, net_pnl, holding_days, exit_reason
        FROM   backtest_trades
        WHERE  bt_run_id = ?
        ORDER  BY exit_date DESC, id DESC
        LIMIT  ?
        """,
        (bt["bt_run_id"], int(n)),
    )
    out: list[TradeRow] = []
    for r in rows:
        entry = float(r["entry_price"])
        exit_ = float(r["exit_price"])
        notional = entry * int(r["qty"])
        pnl_pct = 0.0 if notional == 0 else (float(r["net_pnl"]) / notional) * 100.0
        out.append(
            TradeRow(
                symbol=r["symbol"],
                side=r["side"],
                entry_date=r["entry_date"],
                exit_date=r["exit_date"],
                entry_price=entry,
                exit_price=exit_,
                qty=int(r["qty"]),
                net_pnl=float(r["net_pnl"]),
                pnl_pct=pnl_pct,
                exit_reason=r["exit_reason"],
                holding_days=int(r["holding_days"]),
            )
        )
    return out


def _load_validation_summary(*, window_days: int = 7) -> ValidationSummary:
    """Validation events created in the last ``window_days`` only.

    The previous version returned the all-time count, which on long-running
    DBs balloons to tens of thousands and drowns out today's signal noise.
    """
    rows = fetch_all(
        """
        SELECT severity, COUNT(*) AS n, MAX(created_at) AS last_seen
        FROM   validation_failures
        WHERE  created_at >= datetime('now', ?)
        GROUP  BY severity
        """,
        (f"-{int(window_days)} days",),
    )
    by_severity: dict[str, int] = {}
    last_seen: str | None = None
    total = 0
    for r in rows:
        by_severity[r["severity"]] = int(r["n"])
        total += int(r["n"])
        if r["last_seen"] and (last_seen is None or r["last_seen"] > last_seen):
            last_seen = r["last_seen"]
    return ValidationSummary(
        total=total,
        by_severity=by_severity,
        latest_check_date=last_seen,
        window_days=window_days,
    )


# Price sources the report may read, in preference order on a date tie:
# Angel One (broker feed, freshest/most accurate for NSE) > BhavCopy
# (official EOD) > yfinance (third-party gap filler). This MUST match the
# dashboard's ordering in src/web/queries.py so the report and the web UI
# never disagree on "last price".
_PRICE_SOURCES_IN = "('angelone','bhavcopy','yfinance')"
_SRC_RANK_SQL = (
    "CASE source WHEN 'angelone' THEN 0 WHEN 'bhavcopy' THEN 1 ELSE 2 END"
)


def _last_close_for(symbol: str, on_date: str) -> float | None:
    """Latest close <= ``on_date`` for marking open positions.

    Reads across all price sources and prefers Angel One, then BhavCopy,
    then yfinance on the same bar_date (see _SRC_RANK_SQL).
    """
    row = fetch_one(
        f"""
        SELECT close FROM price_data
        WHERE  symbol = ? AND bar_date <= ? AND source IN {_PRICE_SOURCES_IN}
        ORDER  BY bar_date DESC, {_SRC_RANK_SQL}
        LIMIT 1
        """,
        (symbol, on_date),
    )
    return float(row["close"]) if row and row["close"] is not None else None


def _holding_days_str(entry: str, today: str) -> int:
    try:
        return max(
            0, (date.fromisoformat(today) - date.fromisoformat(entry)).days
        )
    except ValueError:
        return 0


def _load_paper_portfolio(report_date: str, *, window_days: int = 30,
                          recent_n: int = 10) -> PaperPortfolio:
    """Open positions (mark-to-market) + recently closed trades."""
    open_rows = fetch_all(
        """
        SELECT id, symbol, sector, qty, entry_date, entry_price,
               stop_loss, take_profit
        FROM   paper_trades
        WHERE  status = 'open'
        ORDER  BY entry_date ASC, id ASC
        """,
    )
    open_positions: list[OpenPaperPosition] = []
    unreal_total = 0.0
    for r in open_rows:
        last_close = _last_close_for(r["symbol"], report_date)
        qty = int(r["qty"]) if r["qty"] is not None else 0
        entry = float(r["entry_price"]) if r["entry_price"] is not None else 0.0
        if last_close is not None and qty > 0 and entry > 0:
            unreal = (last_close - entry) * qty
            unreal_pct = (last_close - entry) / entry * 100.0
        else:
            unreal = 0.0
            unreal_pct = 0.0
        unreal_total += unreal
        open_positions.append(OpenPaperPosition(
            paper_id=int(r["id"]),
            symbol=r["symbol"],
            sector=r["sector"] or "UNKNOWN",
            qty=qty,
            entry_date=r["entry_date"] or "",
            entry_price=entry,
            stop_loss=(None if r["stop_loss"] is None else float(r["stop_loss"])),
            take_profit=(None if r["take_profit"] is None else float(r["take_profit"])),
            last_close=last_close,
            unrealised_pnl=unreal,
            unrealised_pnl_pct=unreal_pct,
            holding_days=_holding_days_str(r["entry_date"] or report_date, report_date),
        ))

    closed_rows = fetch_all(
        """
        SELECT symbol, side, entry_date, exit_date, entry_price, exit_price,
               qty, pnl_rupees, pnl_pct, exit_reason
        FROM   paper_trades
        WHERE  status = 'closed'
          AND  exit_date IS NOT NULL
          AND  exit_date >= date('now', ?)
        ORDER  BY exit_date DESC, id DESC
        LIMIT  ?
        """,
        (f"-{int(window_days)} days", int(recent_n)),
    )
    closed: list[PaperTradeRow] = []
    realised = 0.0
    win_count = 0
    total_count = 0
    for r in closed_rows:
        net = float(r["pnl_rupees"] or 0.0)
        realised += net
        total_count += 1
        if net > 0:
            win_count += 1
        closed.append(PaperTradeRow(
            symbol=r["symbol"],
            side=r["side"],
            entry_date=r["entry_date"] or "",
            exit_date=r["exit_date"] or "",
            entry_price=float(r["entry_price"] or 0.0),
            exit_price=float(r["exit_price"] or 0.0),
            qty=int(r["qty"] or 0),
            net_pnl=net,
            pnl_pct=float(r["pnl_pct"] or 0.0),
            exit_reason=r["exit_reason"] or "",
            holding_days=_holding_days_str(
                r["entry_date"] or "", r["exit_date"] or report_date
            ),
        ))
    win_rate = 0.0 if total_count == 0 else (win_count / total_count) * 100.0
    return PaperPortfolio(
        open_positions=open_positions,
        closed_recent=closed,
        realised_pnl_window=realised,
        unrealised_pnl=unreal_total,
        open_count=len(open_positions),
        win_rate_pct_window=win_rate,
        window_days=window_days,
    )


def _load_data_freshness(report_date: str) -> DataFreshness:
    """Latest dates seen across price/feature/prediction tables."""
    p = fetch_one(
        f"SELECT MAX(bar_date) AS d FROM price_data WHERE source IN {_PRICE_SOURCES_IN}"
    )
    f = fetch_one("SELECT MAX(feature_date) AS d FROM feature_data")
    pr = fetch_one("SELECT MAX(prediction_date) AS d FROM predictions_log")
    latest_price = p["d"] if p else None
    days_since: int | None = None
    is_stale = False
    if latest_price:
        try:
            last = date.fromisoformat(latest_price)
            ref = date.fromisoformat(report_date)
            days_since = (ref - last).days
            # Staleness is measured in TRADING days, not calendar days, so a
            # weekend or NSE holiday gap never reads as "stale". We count the
            # trading sessions strictly after the last bar up to the report
            # date; a gap of <=1 (just today's not-yet-ingested session) is OK.
            from src.data_validation import calendar_check
            trading_gap = sum(
                1 for d in calendar_check.trading_days_between(last, ref)
                if d > last
            )
            is_stale = trading_gap > 1
        except Exception:  # noqa: BLE001 -- never crash the report on a date quirk
            pass
    return DataFreshness(
        latest_price_date=latest_price,
        latest_feature_date=f["d"] if f else None,
        latest_prediction_date=pr["d"] if pr else None,
        days_since_last_price=days_since,
        is_stale=is_stale,
    )


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _load_latest_walk_forward() -> WalkForwardSummary | None:
    """Read the newest data/reports/wf_meta_*.json, if any.

    Walk-forward results are produced by ``scripts.walk_forward_eval --save``
    (a periodic research step, NOT the daily run because it's expensive).
    The report surfaces the latest saved aggregate as the trustworthy
    out-of-sample model-health read; absent that, returns None.
    """
    reports_dir = project_root() / "data" / "reports"
    if not reports_dir.exists():
        return None
    metas = sorted(reports_dir.glob("wf_meta_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not metas:
        return None
    newest = metas[0]
    try:
        agg = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read walk-forward meta {}: {}", newest.name, exc)
        return None
    if int(agg.get("folds_completed", 0) or 0) <= 0:
        return None
    cal = agg.get("calibrated") or {}
    # Filename stamp wf_meta_YYYYMMDDTHHMMSSZ.json -> readable; fall back to mtime.
    stamp = newest.stem.replace("wf_meta_", "")
    return WalkForwardSummary(
        generated_at=stamp or None,
        folds_completed=int(agg.get("folds_completed", 0) or 0),
        n_predictions=int(agg.get("n_test_predictions", 0) or 0),
        auc=_as_float(cal.get("auc") or cal.get("roc_auc")),
        brier=_as_float(cal.get("brier") or cal.get("brier_score")),
        mean_threshold=_as_float(agg.get("mean_threshold")),
        source_file=newest.name,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_daily_report(
    *,
    report_date: str | None = None,
    top_n: int = 10,
    recent_trades_n: int = 10,
    threshold_override: float | None = None,
) -> DailyReport:
    """Assemble a snapshot for the given report date.

    Parameters
    ----------
    report_date: ISO YYYY-MM-DD; defaults to today's local date.
    top_n: number of predictions to include in the top-N table.
    recent_trades_n: number of recent trades to include.
    threshold_override: if provided, overrides the model's stored threshold
        when deciding which predictions count as ``is_signal``.
    """
    rd = _resolve_report_date(report_date)
    log.info("Building daily report for {} (top_n={})", rd, top_n)

    latest_model = _load_latest_model()
    threshold = (
        threshold_override
        if threshold_override is not None
        else (latest_model.threshold if latest_model else None)
    )

    predictions = _load_predictions(rd, top_n=top_n, threshold=threshold)
    signals = [p for p in predictions if p.is_signal]
    latest_backtest = _load_latest_backtest()
    recent_trades = _load_recent_trades(recent_trades_n)
    validation = _load_validation_summary(window_days=7)
    universe_size = _load_universe_size()
    paper = _load_paper_portfolio(rd, window_days=30, recent_n=recent_trades_n)
    freshness = _load_data_freshness(rd)
    walk_forward = _load_latest_walk_forward()

    return DailyReport(
        generated_at=_utc_now_iso(),
        report_date=rd,
        universe_size=universe_size,
        predictions=predictions,
        signals=signals,
        latest_model=latest_model,
        latest_backtest=latest_backtest,
        recent_trades=recent_trades,
        validation=validation,
        paper=paper,
        freshness=freshness,
        walk_forward=walk_forward,
        top_n=top_n,
        threshold_used=threshold,
    )
