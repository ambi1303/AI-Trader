"""Publish the dashboard subset from local SQLite to the Neon Postgres mirror.

Run after the daily pipeline (or standalone):

    python -m src.cloud.publish

It:
  1. ensures the mirror schema exists on Neon (idempotent),
  2. for each mirrored table, TRUNCATEs and reloads the rows the dashboard
     reads -- small reference/signal/trade tables in full, plus a recent
     (~1 trading year) slice of price_data and the narrow feature overlays,
  3. commits in a single transaction so the dashboard never sees a half-loaded
     state.

The connection string is read from the DATABASE_URL env var (never
hardcoded); Neon requires TLS. The local SQLite DB stays the source of truth.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from src.utils.db import connect as sqlite_connect
from src.utils.logger import get_logger

log = get_logger("cloud.publish")

_SCHEMA_SQL = Path(__file__).resolve().parent / "schema_pg.sql"

# Default history window for the chart slice: ~260 trading days ≈ 380 calendar
# days, matching the dashboard's get_ohlc(limit=260).
_DEFAULT_WINDOW_DAYS = 380

_PRICE_SOURCES_IN = "('angelone','bhavcopy','yfinance')"


class _Spec:
    """A mirrored table: destination columns + the SQLite SELECT that feeds it.

    ``windowed`` tables take a single ``cutoff`` (ISO date) bind parameter;
    ``big`` tables are loaded via COPY for speed.
    """

    def __init__(self, table: str, columns: Sequence[str], select_sql: str,
                 *, windowed: bool = False, big: bool = False) -> None:
        self.table = table
        self.columns = list(columns)
        self.select_sql = select_sql
        self.windowed = windowed
        self.big = big


# Full-copy reference / signal / trade tables (all tiny). Column order here
# MUST match the SELECT and the Postgres schema column list.
_SPECS: list[_Spec] = [
    _Spec(
        "nifty_constituents",
        ["symbol", "start_date", "end_date", "index_name"],
        "SELECT symbol, start_date, end_date, index_name FROM nifty_constituents",
    ),
    _Spec(
        "trading_calendar",
        ["cal_date", "is_holiday", "description", "is_special_session",
         "session_open", "session_close"],
        "SELECT cal_date, is_holiday, description, is_special_session, "
        "session_open, session_close FROM trading_calendar",
    ),
    _Spec(
        "stock_sectors",
        ["symbol", "sector", "sector_index"],
        "SELECT symbol, sector, sector_index FROM stock_sectors",
    ),
    _Spec(
        "model_runs",
        ["run_id", "model_name", "trained_from", "trained_to",
         "metrics_json", "created_at"],
        "SELECT run_id, model_name, trained_from, trained_to, metrics_json, "
        "created_at FROM model_runs",
    ),
    _Spec(
        "predictions_log",
        ["id", "run_id", "symbol", "prediction_date", "raw_prob",
         "calibrated_prob", "verdict", "prob_buy", "prob_hold", "prob_sell",
         "predicted_return", "target_price", "stop_price", "created_at"],
        "SELECT id, run_id, symbol, prediction_date, raw_prob, calibrated_prob, "
        "verdict, prob_buy, prob_hold, prob_sell, predicted_return, "
        "target_price, stop_price, created_at FROM predictions_log",
    ),
    _Spec(
        "signal_outbox",
        ["id", "symbol", "signal_date", "side", "entry_price", "stop_loss",
         "take_profit", "qty", "confidence", "status"],
        "SELECT id, symbol, signal_date, side, entry_price, stop_loss, "
        "take_profit, qty, confidence, status FROM signal_outbox",
    ),
    _Spec(
        "paper_trades",
        ["id", "symbol", "sector", "side", "qty", "entry_date", "exit_date",
         "entry_price", "exit_price", "pnl_rupees", "pnl_pct", "exit_reason",
         "status", "stop_loss", "take_profit"],
        "SELECT id, symbol, sector, side, qty, entry_date, exit_date, "
        "entry_price, exit_price, pnl_rupees, pnl_pct, exit_reason, status, "
        "stop_loss, take_profit FROM paper_trades",
    ),
    _Spec(
        "fundamental_data",
        ["symbol", "as_of_date", "source", "pe_ttm", "pb", "roe",
         "debt_to_equity", "profit_margin", "revenue_growth", "earnings_growth",
         "dividend_yield", "market_cap", "eps_ttm", "book_value"],
        "SELECT symbol, as_of_date, source, pe_ttm, pb, roe, debt_to_equity, "
        "profit_margin, revenue_growth, earnings_growth, dividend_yield, "
        "market_cap, eps_ttm, book_value FROM fundamental_data",
    ),
    # Recent slice only -- these two are loaded via COPY with a date cutoff.
    _Spec(
        "price_data",
        ["symbol", "bar_date", "open", "high", "low", "close", "volume",
         "adj_close", "source", "ingested_at"],
        "SELECT symbol, bar_date, open, high, low, close, volume, adj_close, "
        "source, ingested_at FROM price_data "
        f"WHERE source IN {_PRICE_SOURCES_IN} AND bar_date >= ?",
        windowed=True, big=True,
    ),
    _Spec(
        "feature_data",
        ["symbol", "feature_date", "ema_20", "ema_50", "ema_200", "rsi_14",
         "macd", "macd_signal", "macd_hist"],
        "SELECT symbol, feature_date, ema_20, ema_50, ema_200, rsi_14, macd, "
        "macd_signal, macd_hist FROM feature_data WHERE feature_date >= ?",
        windowed=True, big=True,
    ),
]


def _database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your Neon connection string to .env "
            "(or the environment) before publishing."
        )
    return url


def _split_statements(sql: str) -> Iterable[str]:
    # Drop full-line SQL comments first so semicolons inside comment prose
    # ("...mirrored; heavy blobs...") don't get mistaken for statement
    # terminators. Our schema has no inline (trailing) comments.
    code_lines = [
        line for line in sql.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    for chunk in "\n".join(code_lines).split(";"):
        stmt = chunk.strip()
        if stmt:
            yield stmt


def _apply_schema(pg) -> None:
    sql = _SCHEMA_SQL.read_text(encoding="utf-8")
    with pg.cursor() as cur:
        for stmt in _split_statements(sql):
            cur.execute(stmt)


def _compute_cutoff(sl, window_days: int) -> str:
    row = sl.execute("SELECT MAX(bar_date) AS d FROM price_data").fetchone()
    maxd = row["d"] if row else None
    if not maxd:
        return "0000-00-00"
    try:
        return (date.fromisoformat(maxd) - timedelta(days=window_days)).isoformat()
    except (ValueError, TypeError):
        return "0000-00-00"


def _load_spec(sl, pg, spec: _Spec, cutoff: str) -> int:
    params: tuple = (cutoff,) if spec.windowed else ()
    rows = sl.execute(spec.select_sql, params).fetchall()
    cols = ", ".join(spec.columns)
    with pg.cursor() as cur:
        cur.execute(f"TRUNCATE {spec.table}")
        if not rows:
            return 0
        if spec.big:
            with cur.copy(f"COPY {spec.table} ({cols}) FROM STDIN") as cp:
                for r in rows:
                    cp.write_row(tuple(r))
        else:
            placeholders = ", ".join(["%s"] * len(spec.columns))
            cur.executemany(
                f"INSERT INTO {spec.table} ({cols}) VALUES ({placeholders})",
                [tuple(r) for r in rows],
            )
    return len(rows)


def publish(window_days: int = _DEFAULT_WINDOW_DAYS) -> dict[str, int]:
    """Push the dashboard subset to Neon. Returns {table: rows_loaded}."""
    import psycopg

    url = _database_url()
    summary: dict[str, int] = {}

    sl = sqlite_connect(read_only=True)
    try:
        cutoff = _compute_cutoff(sl, window_days)
        log.info("publishing to Neon (price/feature cutoff >= {})", cutoff)
        with psycopg.connect(url, connect_timeout=20) as pg:
            _apply_schema(pg)
            for spec in _SPECS:
                n = _load_spec(sl, pg, spec, cutoff)
                summary[spec.table] = n
                log.info("  {:<20} {:>8,} rows", spec.table, n)
            pg.commit()
    finally:
        sl.close()

    total = sum(summary.values())
    log.info("publish complete: {:,} rows across {} tables", total, len(summary))
    return summary


def main() -> int:
    try:
        summary = publish()
    except Exception as exc:  # noqa: BLE001
        log.error("publish failed: {}", exc)
        print(f"PUBLISH FAILED: {exc}")
        return 1
    print("Published to Neon:")
    for table, n in summary.items():
        print(f"  {table:<20} {n:>8,} rows")
    print(f"  {'TOTAL':<20} {sum(summary.values()):>8,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
