"""Generate ``signal_outbox`` rows from the latest predictions.

Pipeline
--------

1. Resolve the model run (caller-supplied or the registry's latest).
2. Pull all predictions for ``signal_date`` whose ``calibrated_prob`` >=
   the model's stored threshold (or a caller override).
3. Sort by probability desc and keep the top-N where N is bounded by the
   configured ``max_concurrent_positions`` minus how many positions are
   already open in ``paper_trades``. This implements end-to-end discipline:
   we don't queue more signals than the portfolio can carry.
4. For each kept candidate:
   * Look up the latest close + ATR(14) to derive entry, stop, and target.
   * Look up sector mapping for sector-cap accounting downstream.
   * Run the canonical position sizer (fractional Kelly + vol target).
   * Skip if sizing returns qty==0 (e.g., min_trade_rupees not met).
5. Write rows into ``signal_outbox`` using ``INSERT OR IGNORE`` so the
   unique index ``(symbol, signal_date)`` makes re-runs idempotent.

Idempotency / safety
--------------------
- Re-running the generator the same day is a no-op (unique index).
- Predictions that have no matching ATR/price row are *skipped*, not
  flagged as errors -- ATR(14) requires 14 bars of warm-up so a freshly
  listed name can be incomplete.
- All numerical inputs are validated (positive prices, non-NaN ATR) and
  any rejection is logged into ``validation_failures`` for the daily
  health-check section of the report.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.backtesting.risk import RiskConfig
from src.backtesting.sizing import SizingConfig, size_position
from src.utils.db import fetch_all, fetch_one, transaction
from src.utils.logger import get_logger

log = get_logger("signals.generator")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalGenConfig:
    """Knobs for one generator run.

    ``equity`` is the portfolio equity used for sizing. We don't read it
    from the DB on every call because the dispatcher / orchestrator owns
    the running cash balance (Week 5 paper-trading ledger).
    """
    equity: float = 1_000_000.0
    risk: RiskConfig = RiskConfig()
    sizing: SizingConfig = SizingConfig()
    threshold_override: float | None = None
    max_signals_today: int | None = None     # None = use risk.max_concurrent
    log_run_id: str = "signal_gen"

    # --- News-sentiment gate (FinBERT / lexical) -------------------------
    # A soft overlay on the model's pick: veto entries into strongly negative
    # news, and modestly scale size with sentiment. Look-ahead safe (only
    # headlines published on/before the signal date are used). All knobs are
    # conservative and the effect is fully audited in the signal payload.
    use_news_sentiment: bool = True
    sentiment_window_days: int = 7
    sentiment_veto_score: float = -0.35      # veto if mean sentiment <= this ...
    sentiment_min_headlines: int = 2         # ... and at least this many headlines
    sentiment_size_weight: float = 0.25      # size multiplier sensitivity
    sentiment_size_floor: float = 0.75       # clamp: never shrink below this ...
    sentiment_size_cap: float = 1.15         # ... nor grow beyond this


@dataclass(frozen=True)
class GeneratedSignal:
    """In-memory representation of one kept signal.

    Returned to the caller (orchestrator / report) so they can summarise
    without re-querying the DB. Mirrors the columns we INSERT.
    """
    symbol: str
    signal_date: str
    side: str                 # always "BUY" in v1 (long-only)
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: int
    confidence: float
    sector: str
    sizing_rationale: str
    raw_kelly_qty: int
    raw_vol_qty: int
    news_sentiment: float | None = None      # mean signed sentiment, or None
    news_label: str | None = None            # 'positive'|'negative'|'neutral'
    size_factor: float = 1.0                 # sentiment size multiplier applied


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_run_id() -> str | None:
    row = fetch_one(
        "SELECT run_id, metrics_json FROM model_runs "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1"
    )
    return row["run_id"] if row else None


def _coerce_threshold(raw: Any) -> float | None:
    """The ``threshold`` field in model metrics_json is either:
      - a bare number (early/test runs), or
      - a dict like ``{"value": 0.55, ...}`` (production threshold tuner output).
    Both shapes are accepted; everything else returns None.
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


def _resolve_threshold(run_id: str, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    row = fetch_one("SELECT metrics_json FROM model_runs WHERE run_id = ?", (run_id,))
    if not row or not row["metrics_json"]:
        return None
    try:
        data = json.loads(row["metrics_json"])
    except json.JSONDecodeError:
        return None
    return _coerce_threshold(data.get("threshold"))


def _open_position_count() -> int:
    row = fetch_one("SELECT COUNT(*) AS n FROM paper_trades WHERE status = 'open'")
    return int(row["n"]) if row else 0


def _open_per_sector() -> dict[str, int]:
    rows = fetch_all(
        "SELECT sector, COUNT(*) AS n FROM paper_trades "
        "WHERE status = 'open' AND sector IS NOT NULL GROUP BY sector"
    )
    return {r["sector"]: int(r["n"]) for r in rows}


def _existing_signals(signal_date: str) -> set[str]:
    rows = fetch_all(
        "SELECT symbol FROM signal_outbox WHERE signal_date = ?",
        (signal_date,),
    )
    return {r["symbol"] for r in rows}


def _pull_candidates(prediction_date: str, run_id: str, threshold: float) -> list[dict[str, Any]]:
    """Predictions that beat the threshold for the given date and run."""
    rows = fetch_all(
        """
        SELECT symbol, calibrated_prob, raw_prob
        FROM   predictions_log
        WHERE  prediction_date = ?
          AND  run_id = ?
          AND  calibrated_prob IS NOT NULL
          AND  calibrated_prob >= ?
        ORDER  BY calibrated_prob DESC
        """,
        (prediction_date, run_id, float(threshold)),
    )
    return [dict(r) for r in rows]


def _latest_price_atr(symbol: str, on_date: str) -> dict[str, float] | None:
    """Latest close + ATR(14) up to and including ``on_date``.

    We pull from ``feature_data`` because that's where ATR lives, and from
    ``price_data`` for the *raw* close (yfinance source) -- using the same
    source as backtests so live and historical numbers match.
    """
    feat = fetch_one(
        "SELECT close, atr_14 FROM feature_data "
        "WHERE symbol = ? AND feature_date <= ? "
        "ORDER BY feature_date DESC LIMIT 1",
        (symbol, on_date),
    )
    if not feat or feat["atr_14"] is None or feat["close"] is None:
        return None
    close = float(feat["close"])
    atr = float(feat["atr_14"])
    if close <= 0 or atr <= 0:
        return None
    return {"close": close, "atr_14": atr}


def _sector_for(symbol: str) -> str:
    row = fetch_one("SELECT sector FROM stock_sectors WHERE symbol = ?", (symbol,))
    return row["sector"] if row and row["sector"] else "UNKNOWN"


def _news_sentiment(symbol: str, on_date: str, window_days: int) -> dict[str, Any] | None:
    """Trailing-window news sentiment for ``symbol`` as of ``on_date``.

    Best-effort: returns ``None`` if scoring is unavailable so the gate fails
    open (never blocks signal generation on a sentiment hiccup).
    """
    try:
        from src.data_ingestion.finbert_scorer import symbol_sentiment
        agg = symbol_sentiment(symbol, as_of=on_date, window_days=window_days)
    except Exception as exc:  # noqa: BLE001 -- sentiment is an optional overlay
        log.warning("sentiment lookup failed for {}: {}", symbol, exc)
        return None
    return agg if agg.get("available") else None


def _sentiment_size_factor(score: float, cfg: SignalGenConfig) -> float:
    """Map a signed sentiment score to a clamped position-size multiplier."""
    raw = 1.0 + score * cfg.sentiment_size_weight
    return max(cfg.sentiment_size_floor, min(cfg.sentiment_size_cap, raw))


def _record_validation(conn: sqlite3.Connection, *, run_id: str, severity: str,
                       message: str, symbol: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO validation_failures (run_id, check_name, symbol, severity, message)
        VALUES (?, 'signal_generator', ?, ?, ?)
        """,
        (run_id, symbol, severity, message),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_signals(
    *,
    signal_date: str | None = None,
    run_id: str | None = None,
    config: SignalGenConfig | None = None,
) -> list[GeneratedSignal]:
    """Build signal rows for ``signal_date`` from the latest predictions.

    Parameters
    ----------
    signal_date : ISO YYYY-MM-DD; the trading day the signal applies to.
        Defaults to today's local date. Predictions are read for this date.
    run_id : the ``model_runs.run_id`` whose predictions to consume. Defaults
        to the most recent run.
    config : sizing/risk knobs. Defaults to ``SignalGenConfig()``.

    Returns
    -------
    list[GeneratedSignal] : one entry per row written. May be empty if no
    candidates passed the threshold or the portfolio is full.
    """
    cfg = config or SignalGenConfig()
    sd = signal_date or date.today().isoformat()
    rid = run_id or _latest_run_id()
    if rid is None:
        log.warning("No model run found; skipping signal generation")
        return []

    threshold = _resolve_threshold(rid, cfg.threshold_override)
    if threshold is None:
        log.warning("No threshold for run {}; skipping signal generation", rid)
        return []

    candidates = _pull_candidates(sd, rid, threshold)
    if not candidates:
        log.info("No candidates for {} (run={}, threshold={:.3f})",
                 sd, rid, threshold)
        return []

    # Portfolio capacity = max concurrent - already-open positions, capped
    # by an optional explicit limit.
    n_open = _open_position_count()
    capacity = max(0, cfg.risk.max_concurrent_positions - n_open)
    if cfg.max_signals_today is not None:
        capacity = min(capacity, int(cfg.max_signals_today))
    if capacity == 0:
        log.info("Portfolio full (open={}, cap={}); no signals queued",
                 n_open, cfg.risk.max_concurrent_positions)
        return []

    sector_open = _open_per_sector()
    already = _existing_signals(sd)
    kept: list[GeneratedSignal] = []

    with transaction() as conn:
        for cand in candidates:
            if len(kept) >= capacity:
                break
            sym = cand["symbol"]
            if sym in already:
                continue                         # idempotent skip

            quote = _latest_price_atr(sym, sd)
            if quote is None:
                _record_validation(
                    conn, run_id=cfg.log_run_id,
                    severity="warning", symbol=sym,
                    message="missing close/atr for signal generation",
                )
                continue

            sector = _sector_for(sym)
            # sector cap: do not exceed risk.max_per_sector. Count signals we
            # have ALREADY queued in this run too, not just open positions.
            already_open_in_sector = sector_open.get(sector, 0)
            already_kept_in_sector = sum(1 for k in kept if k.sector == sector)
            if (already_open_in_sector + already_kept_in_sector
                    >= cfg.risk.max_per_sector):
                _record_validation(
                    conn, run_id=cfg.log_run_id,
                    severity="info", symbol=sym,
                    message=f"sector cap reached ({sector})",
                )
                continue

            # News-sentiment gate (look-ahead safe; only news <= signal date).
            s_score: float | None = None
            s_label: str | None = None
            size_factor = 1.0
            if cfg.use_news_sentiment:
                senti = _news_sentiment(sym, sd, cfg.sentiment_window_days)
                if senti is not None:
                    s_score = float(senti["score"])
                    s_label = senti.get("label")
                    if (int(senti.get("n", 0)) >= cfg.sentiment_min_headlines
                            and s_score <= cfg.sentiment_veto_score):
                        _record_validation(
                            conn, run_id=cfg.log_run_id,
                            severity="info", symbol=sym,
                            message=(f"negative-news veto "
                                     f"(score={s_score:.2f}, n={senti.get('n')})"),
                        )
                        continue
                    size_factor = _sentiment_size_factor(s_score, cfg)

            entry = quote["close"]
            atr = quote["atr_14"]
            stop = cfg.risk.stop_for(entry, atr)
            target = cfg.risk.target_for(entry, atr)

            decision = size_position(
                prob_win=float(cand["calibrated_prob"]),
                entry_price=entry,
                atr=atr,
                stop_atr_mult=cfg.risk.stop_atr_mult,
                equity=cfg.equity,
                cfg=cfg.sizing,
            )
            # Apply the (clamped) sentiment multiplier to the sized quantity.
            final_qty = int(decision.qty * size_factor) if size_factor != 1.0 \
                else int(decision.qty)
            if final_qty <= 0:
                msg = (f"sizer returned qty=0 ({decision.rationale})"
                       if decision.qty <= 0
                       else f"sentiment shrank qty to 0 (factor={size_factor:.2f})")
                _record_validation(
                    conn, run_id=cfg.log_run_id,
                    severity="info", symbol=sym, message=msg,
                )
                continue

            rationale = decision.rationale
            if size_factor != 1.0:
                rationale = f"{rationale}; sentiment x{size_factor:.2f}"

            payload = {
                "raw_prob": cand["raw_prob"],
                "calibrated_prob": cand["calibrated_prob"],
                "atr_14": atr,
                "sector": sector,
                "sizing_rationale": rationale,
                "raw_kelly_qty": decision.raw_kelly_qty,
                "raw_vol_qty": decision.raw_vol_qty,
                "model_run_id": rid,
                "news_sentiment": round(s_score, 4) if s_score is not None else None,
                "news_label": s_label,
                "sentiment_size_factor": round(size_factor, 3),
            }

            # INSERT OR IGNORE leverages the v4 unique index to make re-runs
            # safe even if `already` was stale (race in a multi-writer setup).
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO signal_outbox
                    (symbol, signal_date, side, entry_price, stop_loss, take_profit,
                     qty, confidence, status, payload_json)
                VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (sym, sd, entry, stop, target, final_qty,
                 float(cand["calibrated_prob"]), json.dumps(payload, default=str)),
            )
            if cur.rowcount == 0:
                # Lost the race -- another process generated this signal first.
                continue

            kept.append(GeneratedSignal(
                symbol=sym, signal_date=sd, side="BUY",
                entry_price=entry, stop_loss=stop, take_profit=target,
                qty=final_qty,
                confidence=float(cand["calibrated_prob"]),
                sector=sector,
                sizing_rationale=rationale,
                raw_kelly_qty=decision.raw_kelly_qty,
                raw_vol_qty=decision.raw_vol_qty,
                news_sentiment=s_score,
                news_label=s_label,
                size_factor=size_factor,
            ))

    log.info(
        "Signal generation complete | date={} | candidates={} | kept={} | "
        "capacity={} | open={}",
        sd, len(candidates), len(kept), capacity, n_open,
    )
    return kept


def generate_exit_signals(
    *,
    signal_date: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Queue EXIT signals for open positions the model now flags SELL.

    This is the SELL half of the "complete buy/sell indication". The system
    is long-only (Indian retail delivery can't short), so a SELL verdict on
    a name we *hold* means "get out", not "go short". We write a side='EXIT'
    row into ``signal_outbox`` for each held name whose latest verdict is
    SELL; the paper reconciler closes the position with reason ``model_sell``.

    Idempotent: re-running the same day is a no-op (one EXIT per
    symbol/date).
    """
    sd = signal_date or date.today().isoformat()
    rid = run_id or _latest_run_id()
    if rid is None:
        return []

    open_pos = fetch_all(
        "SELECT symbol, qty FROM paper_trades WHERE status = 'open'"
    )
    if not open_pos:
        return []

    out: list[dict[str, Any]] = []
    with transaction() as conn:
        for p in open_pos:
            sym = p["symbol"]
            v = fetch_one(
                """
                SELECT verdict, prob_sell, target_price, stop_price
                FROM   predictions_log
                WHERE  symbol = ? AND run_id = ? AND prediction_date <= ?
                       AND verdict IS NOT NULL
                ORDER  BY prediction_date DESC LIMIT 1
                """,
                (sym, rid, sd),
            )
            if not v or v["verdict"] != "SELL":
                continue
            exists = conn.execute(
                "SELECT 1 FROM signal_outbox WHERE symbol = ? AND signal_date = ? "
                "AND side = 'EXIT'",
                (sym, sd),
            ).fetchone()
            if exists:
                continue
            payload = {
                "verdict": "SELL",
                "prob_sell": v["prob_sell"],
                "target_price": v["target_price"],
                "stop_price": v["stop_price"],
                "model_run_id": rid,
            }
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO signal_outbox
                    (symbol, signal_date, side, qty, confidence, status,
                     payload_json)
                VALUES (?, ?, 'EXIT', ?, ?, 'pending', ?)
                """,
                (sym, sd, int(p["qty"] or 0),
                 float(v["prob_sell"]) if v["prob_sell"] is not None else None,
                 json.dumps(payload, default=str)),
            )
            if cur.rowcount == 0:
                continue
            out.append({"symbol": sym, "side": "EXIT", "qty": p["qty"]})

    log.info("Exit-signal generation | date={} | exits_queued={}", sd, len(out))
    return out


def list_pending(signal_date: str | None = None) -> list[dict[str, Any]]:
    """Return pending signals for ``signal_date`` (defaults to today)."""
    sd = signal_date or date.today().isoformat()
    rows = fetch_all(
        """
        SELECT id, symbol, signal_date, side, entry_price, stop_loss,
               take_profit, qty, confidence, payload_json, created_at
        FROM   signal_outbox
        WHERE  signal_date = ? AND status = 'pending'
        ORDER  BY confidence DESC
        """,
        (sd,),
    )
    return [dict(r) for r in rows]
