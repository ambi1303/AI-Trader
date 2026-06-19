"""End-to-end feature builder smoke (DB-backed)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.db.migrate import apply_schema
from src.features.feature_builder import (
    _FEATURE_COLUMNS,
    build_for_symbol,
    compute_features_for_symbol,
)
from src.utils.db import fetch_all, transaction


def _seed_price_data(symbol: str, n: int = 280) -> None:
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.012, size=n)
    close = 1000 * np.exp(np.cumsum(rets))
    intraday = rng.uniform(0.005, 0.02, size=n)
    high = close * (1 + intraday / 2)
    low = close * (1 - intraday / 2)
    open_ = np.clip(close + rng.normal(0, 0.5, size=n), low, high)
    volume = rng.integers(50_000, 5_000_000, size=n).astype(int)

    rows = []
    base = date(2024, 1, 2)
    d = base
    i = 0
    while i < n:
        # Skip weekends roughly
        if d.weekday() < 5:
            rows.append(
                (
                    symbol, d.isoformat(),
                    float(open_[i]), float(high[i]), float(low[i]), float(close[i]),
                    int(volume[i]), None, "bhavcopy",
                )
            )
            i += 1
        d = d + timedelta(days=1)

    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO price_data
              (symbol, bar_date, open, high, low, close, volume, adj_close, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_compute_features_pure_returns_expected_columns() -> None:
    apply_schema()
    rng = np.random.default_rng(1)
    n = 250
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    high = close * 1.01
    low = close * 0.99
    open_ = close.copy()
    volume = pd.Series(rng.integers(100_000, 1_000_000, n), index=idx)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    feat = compute_features_for_symbol(df)
    assert list(feat.columns) == list(_FEATURE_COLUMNS)
    assert len(feat) == n


def test_build_for_symbol_persists_rows() -> None:
    apply_schema()
    _seed_price_data("FAKE", n=260)
    summary = build_for_symbol("FAKE")
    assert summary.rows_in == 260
    assert summary.rows_out == 260

    rows = fetch_all(
        "SELECT COUNT(*) AS n FROM feature_data WHERE symbol = ?", ("FAKE",)
    )
    assert rows[0]["n"] == 260


def test_idempotent_rebuild_does_not_duplicate_rows() -> None:
    apply_schema()
    _seed_price_data("FAKE2", n=250)
    build_for_symbol("FAKE2")
    build_for_symbol("FAKE2")
    rows = fetch_all(
        "SELECT COUNT(*) AS n FROM feature_data WHERE symbol = ?", ("FAKE2",)
    )
    assert rows[0]["n"] == 250


def _last_bar_date(symbol: str) -> date:
    rows = fetch_all(
        "SELECT MAX(bar_date) AS d FROM price_data WHERE symbol = ?", (symbol,)
    )
    return date.fromisoformat(rows[0]["d"])


def _append_one_bar(symbol: str) -> None:
    """Append a single new trading-day bar after the current max date."""
    last = _last_bar_date(symbol)
    nxt = last + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt = nxt + timedelta(days=1)
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO price_data
              (symbol, bar_date, open, high, low, close, volume, adj_close, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'bhavcopy')
            """,
            (symbol, nxt.isoformat(), 1010.0, 1020.0, 1000.0, 1015.0, 1_000_000, None),
        )


def test_incremental_first_run_builds_full_history() -> None:
    # With no existing features, incremental falls back to a full build.
    apply_schema()
    _seed_price_data("INC", n=260)
    summary = build_for_symbol("INC", incremental=True)
    assert summary.rows_out == 260


def test_incremental_skips_when_up_to_date() -> None:
    apply_schema()
    _seed_price_data("INC", n=260)
    build_for_symbol("INC", incremental=True)
    # Nothing new in price_data -> the symbol should be skipped entirely.
    summary = build_for_symbol("INC", incremental=True)
    assert summary.rows_in == 0
    assert summary.rows_out == 0


def test_incremental_appends_only_new_tail() -> None:
    apply_schema()
    _seed_price_data("INC", n=260)
    build_for_symbol("INC", incremental=True)
    _append_one_bar("INC")

    summary = build_for_symbol("INC", incremental=True, lookback_bars=120)
    # Only the single new date is persisted...
    assert summary.rows_out == 1
    # ...while the warm-up window is loaded for indicator convergence.
    assert summary.rows_in > 1

    total = fetch_all(
        "SELECT COUNT(*) AS n FROM feature_data WHERE symbol = ?", ("INC",)
    )
    assert total[0]["n"] == 261


def test_incremental_tail_matches_full_build() -> None:
    # The incrementally-built latest row should match a full rebuild within a
    # tiny tolerance (recursive EMAs converge over the warm-up window).
    apply_schema()
    _seed_price_data("INC", n=400)
    build_for_symbol("INC", incremental=True)
    _append_one_bar("INC")
    build_for_symbol("INC", incremental=True, lookback_bars=300)
    inc_row = fetch_all(
        "SELECT * FROM feature_data WHERE symbol = ? ORDER BY feature_date DESC LIMIT 1",
        ("INC",),
    )[0]

    # Now full-rebuild the same data and compare the latest row.
    build_for_symbol("INC", incremental=False)
    full_row = fetch_all(
        "SELECT * FROM feature_data WHERE symbol = ? ORDER BY feature_date DESC LIMIT 1",
        ("INC",),
    )[0]

    # Fast indicators (close + short-span EMAs) converge over the window and
    # should match the full build essentially exactly.
    for col in ("close", "rsi_14", "atr_14"):
        a, b = inc_row[col], full_row[col]
        if a is None or b is None:
            continue
        assert abs(a - b) <= 1e-6 + 0.001 * abs(b), col
    # EMA200 is the slowest to converge; a 300-bar warm-up leaves a tiny
    # residual (well under 1%), which is immaterial to its trend-gate use.
    ema200_a, ema200_b = inc_row["ema_200"], full_row["ema_200"]
    if ema200_a is not None and ema200_b is not None:
        assert abs(ema200_a - ema200_b) <= 0.01 * abs(ema200_b)
