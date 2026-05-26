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
