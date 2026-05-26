"""Tests for the training-matrix builder.

Critical leakage tests:
  1. y[T] depends only on close[T] and close[T+1]; nothing else.
  2. X[T] depends only on price <= T (we trust feature_builder for that
     -- already audited in tests/test_leakage_audit.py -- and we just
     verify that the matrix preserves date ordering and drops the last
     row of each symbol whose target is undefined).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import (
    build_training_matrix,
    boundary_dates,
    time_based_split,
)
from src.utils import db as db_mod


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_mod, "resolve_db_path",
                        lambda *a, **kw: db_file)

    schema = """
    CREATE TABLE feature_data (
        symbol TEXT NOT NULL,
        feature_date TEXT NOT NULL,
        close REAL,
        volume INTEGER,
        ret_1d REAL,
        rsi_14 REAL,
        ema_20 REAL,
        feature_set_version INTEGER NOT NULL DEFAULT 1,
        computed_at TEXT NOT NULL DEFAULT '2024-01-01',
        PRIMARY KEY (symbol, feature_date)
    );
    CREATE TABLE price_data (
        symbol TEXT NOT NULL,
        bar_date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        adj_close REAL,
        source TEXT NOT NULL,
        ingested_at TEXT NOT NULL DEFAULT '2024-01-01',
        PRIMARY KEY (symbol, bar_date, source)
    );
    """
    db_mod.execute_script(schema)
    return db_file


def _seed(symbols: list[str], start: date, n: int, base: float = 100.0):
    rows_feat = []
    rows_price = []
    for sym_idx, sym in enumerate(symbols):
        price = base + sym_idx * 50
        for i in range(n):
            d = start + timedelta(days=i)
            # deterministic walk so target is well-defined
            price *= 1.0 + (0.005 if i % 3 == 0 else -0.002)
            rows_feat.append(
                (sym, d.isoformat(), price, 1000, 0.001, 55.0, price * 0.99)
            )
            rows_price.append(
                (sym, d.isoformat(), price, price * 1.01, price * 0.99,
                 price, 1000, price, "bhavcopy")
            )
    db_mod.executemany(
        "INSERT INTO feature_data(symbol,feature_date,close,volume,ret_1d,rsi_14,ema_20)"
        " VALUES (?,?,?,?,?,?,?)",
        rows_feat,
    )
    db_mod.executemany(
        "INSERT INTO price_data"
        "(symbol,bar_date,open,high,low,close,volume,adj_close,source)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        rows_price,
    )


def test_matrix_drops_last_row_per_symbol(temp_db):
    _seed(["AAA"], date(2023, 1, 2), 30)
    m = build_training_matrix(["AAA"], target_return_threshold=0.0)
    # 30 input rows -> 29 with defined target
    assert len(m.X) == 29
    assert len(m.y) == 29
    assert len(m.forward_return) == 29
    assert len(m.meta) == 29
    # date strictly < 2023-01-31
    assert m.meta["feature_date"].max() == date(2023, 1, 30)


def test_matrix_target_uses_only_next_day_close(temp_db):
    _seed(["AAA"], date(2023, 1, 2), 10)
    m = build_training_matrix(["AAA"], target_return_threshold=0.001)
    closes = m.X["close"].to_numpy() if "close" in m.X.columns else None
    # close should NOT be in features (excluded as raw price ref)
    assert closes is None
    # Manually check fwd_return values.
    raw_close = pd.DataFrame(
        [dict(r) for r in db_mod.fetch_all(
            "SELECT bar_date, close FROM price_data WHERE symbol='AAA' "
            "AND source='bhavcopy' ORDER BY bar_date"
        )]
    )
    expected_fwd = (
        raw_close["close"].shift(-1) / raw_close["close"] - 1.0
    ).iloc[:-1].to_numpy()
    np.testing.assert_allclose(m.forward_return.to_numpy(), expected_fwd, atol=1e-12)


def test_features_excluded_correctly(temp_db):
    _seed(["AAA"], date(2023, 1, 2), 10)
    m = build_training_matrix(["AAA"], target_return_threshold=0.0)
    # close, volume are NOT features. ret_1d, rsi_14, ema_20 ARE.
    assert "close" not in m.feature_columns
    assert "volume" not in m.feature_columns
    assert "feature_date" not in m.feature_columns
    assert "symbol" not in m.feature_columns
    assert "feature_set_version" not in m.feature_columns
    assert {"ret_1d", "rsi_14", "ema_20"}.issubset(set(m.feature_columns))


def test_time_based_split_is_strictly_ordered(temp_db):
    _seed(["AAA", "BBB"], date(2023, 1, 2), 50)
    m = build_training_matrix(["AAA", "BBB"], target_return_threshold=0.0)
    tr, va, te = time_based_split(m.meta, train_frac=0.7, val_frac=0.15)
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    tr_max = m.meta.iloc[tr]["feature_date"].max()
    va_min = m.meta.iloc[va]["feature_date"].min()
    va_max = m.meta.iloc[va]["feature_date"].max()
    te_min = m.meta.iloc[te]["feature_date"].min()
    # Splits are by row position not by date, so within-day rows can land in
    # different splits. Boundary dates must still be non-decreasing across splits.
    assert tr_max <= va_min, f"train_max={tr_max} > val_min={va_min}"
    assert va_max <= te_min, f"val_max={va_max} > test_min={te_min}"


def test_no_overlap_between_splits(temp_db):
    _seed(["AAA"], date(2023, 1, 2), 100)
    m = build_training_matrix(["AAA"], target_return_threshold=0.0)
    tr, va, te = time_based_split(m.meta, train_frac=0.7, val_frac=0.15)
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert s_tr.isdisjoint(s_va)
    assert s_va.isdisjoint(s_te)
    assert s_tr.isdisjoint(s_te)
    assert len(s_tr) + len(s_va) + len(s_te) == len(m.meta)


def test_boundary_dates_helper(temp_db):
    _seed(["AAA"], date(2023, 1, 2), 20)
    m = build_training_matrix(["AAA"], target_return_threshold=0.0)
    tr, _, _ = time_based_split(m.meta)
    lo, hi = boundary_dates(m.meta, tr)
    assert lo <= hi
    assert lo >= date(2023, 1, 2)
    assert hi <= date(2023, 1, 30)


def test_empty_universe_returns_empty_matrix(temp_db):
    m = build_training_matrix(["DOES_NOT_EXIST"])
    assert m.X.empty
    assert m.y.empty
