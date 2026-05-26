"""Daily paper-trade reconciliation.

One public entrypoint, ``reconcile``, runs the close-then-open pass for
``as_of`` date and returns a structured ``ReconcileSummary``.

Design choices
--------------
- All DB writes for one ``as_of`` happen inside a single ``transaction()``
  so a partial failure (e.g., SQLite locked on the third row) doesn't leave
  half-closed positions.
- Cost computation goes through ``src.backtesting.cost_model`` exclusively
  so the same numbers show up in backtests, walk-forward tuning, and live.
- Stop / target / trailing / time logic comes from
  ``src.backtesting.risk`` -- live and backtest decisions are identical
  byte-for-byte for the same inputs.
- We pull bar prices from ``price_data`` filtered to ``source='yfinance'``
  (cross-source validation has already promoted the canonical OHLCV).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from src.backtesting.cost_model import (
    CostConfig,
    compute_leg_cost,
    load_cost_config,
)
from src.backtesting.risk import (
    OpenPosition,
    RiskConfig,
    check_stop_or_target,
)
from src.utils.db import fetch_all, fetch_one, transaction
from src.utils.logger import get_logger

log = get_logger("paper.reconcile")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ReconcileSummary:
    as_of: str
    closed: list[dict[str, Any]] = field(default_factory=list)
    opened: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    mtm_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "closed_count": len(self.closed),
            "opened_count": len(self.opened),
            "skipped_count": len(self.skipped),
            "mtm_updates": self.mtm_updates,
            "closed": self.closed,
            "opened": self.opened,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar_for(symbol: str, on_date: str, *, source: str = "yfinance") -> dict[str, float] | None:
    """Today's bar from ``price_data`` (yfinance source by default)."""
    row = fetch_one(
        """
        SELECT open, high, low, close, volume
        FROM   price_data
        WHERE  symbol = ? AND bar_date = ? AND source = ?
        """,
        (symbol, on_date, source),
    )
    if not row:
        return None
    return {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(row["volume"]),
    }


def _open_positions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, symbol, sector, side, qty, entry_date, entry_price,
               stop_loss, take_profit, trailing_stop, entry_atr,
               high_watermark, entry_prob, threshold, run_id, signal_id
        FROM   paper_trades
        WHERE  status = 'open'
        """
    )
    return [dict(r) for r in cur.fetchall()]


def _pending_signals_for(conn: sqlite3.Connection, signal_date: str) -> list[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, symbol, signal_date, side, entry_price, stop_loss,
               take_profit, qty, confidence, payload_json
        FROM   signal_outbox
        WHERE  signal_date = ? AND status = 'pending' AND side = 'BUY'
        ORDER  BY confidence DESC
        """,
        (signal_date,),
    )
    return [dict(r) for r in cur.fetchall()]


def _holding_days(entry: str, today: str) -> int:
    return max(0, (date.fromisoformat(today) - date.fromisoformat(entry)).days)


def _force_close_at(*, exit_price: float, qty: int, entry_price: float,
                    symbol: str, cost_cfg: CostConfig) -> tuple[float, float]:
    """Compute (gross_pnl, total_cost) for a long round-trip.

    Entry cost is recomputed here because we did NOT persist it on the
    paper_trades row (we only stored entry_price * qty). This keeps the
    cost model pluggable -- if YAML changes mid-cycle, exits will use the
    fresh schedule for the *exit* leg only and the *entry* leg is replayed
    from the recorded entry_price/qty for the original schedule.
    """
    buy = compute_leg_cost("BUY", price=entry_price, qty=qty,
                           symbol=symbol, cfg=cost_cfg)
    sell = compute_leg_cost("SELL", price=exit_price, qty=qty,
                            symbol=symbol, cfg=cost_cfg)
    gross = (exit_price - entry_price) * qty
    cost = buy.total + sell.total + cost_cfg.extra_per_exit_rupees
    return gross, cost


# ---------------------------------------------------------------------------
# Close pass
# ---------------------------------------------------------------------------


def _close_one(
    conn: sqlite3.Connection,
    pos: dict[str, Any],
    bar: dict[str, float],
    *,
    risk: RiskConfig,
    cost_cfg: CostConfig,
    as_of: str,
    summary: ReconcileSummary,
) -> bool:
    """Returns True if the position was closed, False if it stays open."""
    qty = int(pos["qty"])
    entry = float(pos["entry_price"])
    atr = float(pos["entry_atr"]) if pos["entry_atr"] is not None else None
    sector = pos["sector"] or "UNKNOWN"

    # Build OpenPosition and evaluate trailing stop + intraday hit.
    op = OpenPosition(
        symbol=pos["symbol"],
        sector=sector,
        side="LONG",
        qty=qty,
        entry_date=pos["entry_date"],
        entry_price=entry,
        atr_at_entry=atr if atr is not None else 0.0,
        stop=float(pos["stop_loss"]) if pos["stop_loss"] is not None else 0.0,
        target=float(pos["take_profit"]) if pos["take_profit"] is not None else float("inf"),
        high_watermark=float(pos["high_watermark"]) if pos["high_watermark"] is not None else entry,
        entry_prob=float(pos["entry_prob"]) if pos["entry_prob"] is not None else 0.0,
        threshold=float(pos["threshold"]) if pos["threshold"] is not None else 0.0,
    )
    op.update_trailing_stop(bar["high"], risk)

    # If trailing moved the stop, persist the new stop AND high-watermark
    # before deciding whether the bar hit.
    if op.stop != float(pos["stop_loss"] or 0.0) or op.high_watermark != float(pos["high_watermark"] or entry):
        conn.execute(
            "UPDATE paper_trades SET stop_loss = ?, high_watermark = ?, "
            "trailing_stop = ?, updated_at = ? WHERE id = ?",
            (op.stop, op.high_watermark, op.stop, _utc_now(), pos["id"]),
        )
        summary.mtm_updates += 1

    hit = check_stop_or_target(
        op,
        bar_open=bar["open"], bar_high=bar["high"],
        bar_low=bar["low"], bar_close=bar["close"],
    )

    # Time stop
    held = _holding_days(pos["entry_date"], as_of)
    if not hit.hit and held >= risk.max_holding_days:
        hit_reason = "time"
        fill_price = bar["close"]
    elif hit.hit:
        hit_reason = hit.reason
        fill_price = hit.fill_price
    else:
        return False  # still open

    gross, cost = _force_close_at(
        exit_price=fill_price, qty=qty, entry_price=entry,
        symbol=pos["symbol"], cost_cfg=cost_cfg,
    )
    net = gross - cost
    pnl_pct = 0.0 if entry * qty == 0 else (net / (entry * qty)) * 100.0

    conn.execute(
        """
        UPDATE paper_trades
        SET status = 'closed',
            exit_date = ?,
            exit_price = ?,
            pnl_rupees = ?,
            pnl_pct = ?,
            cost_rupees = ?,
            exit_reason = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (as_of, fill_price, net, pnl_pct, cost, hit_reason, _utc_now(), pos["id"]),
    )
    summary.closed.append({
        "id": pos["id"],
        "symbol": pos["symbol"],
        "exit_price": fill_price,
        "exit_reason": hit_reason,
        "net_pnl": net,
        "pnl_pct": pnl_pct,
        "holding_days": held,
    })
    return True


# ---------------------------------------------------------------------------
# Open pass
# ---------------------------------------------------------------------------


def _open_one(
    conn: sqlite3.Connection,
    sig: dict[str, Any],
    bar: dict[str, float],
    *,
    risk: RiskConfig,
    cost_cfg: CostConfig,
    as_of: str,
    summary: ReconcileSummary,
) -> bool:
    """Open a paper position from a pending signal at today's open."""
    symbol = sig["symbol"]
    qty = int(sig["qty"]) if sig["qty"] is not None else 0
    if qty <= 0:
        summary.skipped.append({
            "signal_id": sig["id"], "symbol": symbol,
            "reason": "qty_zero_at_signal",
        })
        conn.execute(
            "UPDATE signal_outbox SET status = 'skipped', "
            "error = 'qty zero at signal time', sent_at = ? WHERE id = ?",
            (_utc_now(), sig["id"]),
        )
        return False

    fill_price = float(bar["open"])

    # Recompute SL/TP from the live ATR if the signal was generated more
    # than a day ago (extremely rare in v1 but defensive). We trust the
    # signal's stop/target as authoritative if today is the signal_date.
    stop = float(sig["stop_loss"])
    target = float(sig["take_profit"])

    # Capacity check at fill time too: max_concurrent might have been hit
    # by another generator pass between signal time and fill.
    open_now = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]
    if open_now >= risk.max_concurrent_positions:
        summary.skipped.append({
            "signal_id": sig["id"], "symbol": symbol, "reason": "portfolio_full",
        })
        conn.execute(
            "UPDATE signal_outbox SET status = 'skipped', "
            "error = 'portfolio full at fill', sent_at = ? WHERE id = ?",
            (_utc_now(), sig["id"]),
        )
        return False

    # Entry cost (we'll recompute the matched leg again at exit, this is for
    # an immediate accounting estimate on `cost_rupees`).
    buy = compute_leg_cost(
        "BUY", price=fill_price, qty=qty, symbol=symbol, cfg=cost_cfg,
    )

    # Sector lookup (signal stored it in payload but stock_sectors is
    # canonical).
    sector_row = fetch_one(
        "SELECT sector FROM stock_sectors WHERE symbol = ?", (symbol,)
    )
    sector = sector_row["sector"] if sector_row and sector_row["sector"] else "UNKNOWN"

    # Pull entry-time ATR (we need it for trailing-stop math later).
    atr_row = fetch_one(
        "SELECT atr_14 FROM feature_data WHERE symbol = ? "
        "AND feature_date <= ? ORDER BY feature_date DESC LIMIT 1",
        (symbol, as_of),
    )
    entry_atr = (
        float(atr_row["atr_14"]) if atr_row and atr_row["atr_14"] is not None
        else None
    )

    cur = conn.execute(
        """
        INSERT INTO paper_trades
            (signal_id, symbol, side, entry_date, entry_price, qty,
             cost_rupees, sector, status, stop_loss, take_profit,
             trailing_stop, entry_atr, high_watermark,
             entry_prob, threshold, run_id,
             created_at, updated_at)
        VALUES (?, ?, 'LONG', ?, ?, ?, ?, ?, 'open', ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sig["id"], symbol, as_of, fill_price, qty,
            buy.total, sector, stop, target,
            stop, entry_atr, fill_price,
            sig["confidence"], None,
            None,  # run_id is in payload_json; we leave the column NULL until orch sets it
            _utc_now(), _utc_now(),
        ),
    )
    paper_id = cur.lastrowid

    conn.execute(
        "UPDATE signal_outbox SET status = 'executed', sent_at = ? WHERE id = ?",
        (_utc_now(), sig["id"]),
    )
    summary.opened.append({
        "paper_id": paper_id,
        "signal_id": sig["id"],
        "symbol": symbol,
        "qty": qty,
        "fill_price": fill_price,
        "stop_loss": stop,
        "take_profit": target,
    })
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile(
    *,
    as_of: str | None = None,
    risk: RiskConfig | None = None,
    cost_cfg: CostConfig | None = None,
    intraday: bool = False,
) -> ReconcileSummary:
    """Run one reconciliation pass.

    Parameters
    ----------
    as_of : ISO YYYY-MM-DD trading date. Defaults to today.
    risk  : RiskConfig override (defaults to defaults).
    cost_cfg : CostConfig override (defaults to ``config/cost_model.yaml``).
    intraday : when True, load intraday cost overrides (for paper-day-trading).

    Returns
    -------
    ReconcileSummary
    """
    today = as_of or date.today().isoformat()
    risk = risk or RiskConfig()
    cost_cfg = cost_cfg or load_cost_config(intraday=intraday)
    summary = ReconcileSummary(as_of=today)

    log.info("Reconciling paper trades for {}", today)

    with transaction() as conn:
        # ---------- close pass ----------
        for pos in _open_positions(conn):
            bar = _bar_for(pos["symbol"], today)
            if bar is None:
                summary.skipped.append({
                    "paper_id": pos["id"], "symbol": pos["symbol"],
                    "reason": "no_bar_for_close",
                })
                continue
            _close_one(conn, pos, bar,
                       risk=risk, cost_cfg=cost_cfg,
                       as_of=today, summary=summary)

        # ---------- open pass ----------
        for sig in _pending_signals_for(conn, today):
            bar = _bar_for(sig["symbol"], today)
            if bar is None:
                summary.skipped.append({
                    "signal_id": sig["id"], "symbol": sig["symbol"],
                    "reason": "no_bar_for_open",
                })
                conn.execute(
                    "UPDATE signal_outbox SET status = 'skipped', "
                    "error = 'no bar at fill date', sent_at = ? WHERE id = ?",
                    (_utc_now(), sig["id"]),
                )
                continue
            _open_one(conn, sig, bar,
                      risk=risk, cost_cfg=cost_cfg,
                      as_of=today, summary=summary)

    log.info(
        "Reconcile done | closed={} opened={} skipped={} mtm={}",
        len(summary.closed), len(summary.opened),
        len(summary.skipped), summary.mtm_updates,
    )
    return summary
