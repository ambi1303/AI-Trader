"""Integration test for the pairs scan against a temp SQLite DB."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.pairs.scan import PairScanConfig, latest_pairs, scan_pairs
from src.utils import db as db_mod

AS_OF = "2026-06-17"
N = 300


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "pairs.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    db_mod.execute_script(
        """
        CREATE TABLE price_data (
            symbol TEXT NOT NULL, bar_date TEXT NOT NULL,
            close REAL, adj_close REAL, source TEXT NOT NULL,
            PRIMARY KEY (symbol, bar_date, source)
        );
        CREATE TABLE stock_sectors (symbol TEXT PRIMARY KEY, sector TEXT);
        CREATE TABLE pairs (
            as_of_date TEXT NOT NULL, symbol_y TEXT NOT NULL,
            symbol_x TEXT NOT NULL, sector TEXT, beta REAL NOT NULL,
            alpha REAL NOT NULL, adf_tstat REAL NOT NULL, half_life REAL,
            corr REAL, spread_mean REAL, spread_std REAL, zscore REAL,
            signal TEXT, n_obs INTEGER,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            PRIMARY KEY (as_of_date, symbol_y, symbol_x)
        );
        """
    )
    return db_file


def _dates(n):
    end = date.fromisoformat(AS_OF)
    return [(end - timedelta(days=(n - 1 - i))).isoformat() for i in range(n)]


def _ar1(n, phi, sigma, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, sigma)
    return s


def _rw(n, base, sigma, seed):
    rng = np.random.default_rng(seed)
    return base + np.cumsum(rng.normal(0, sigma, n))


def _seed_prices(symbol, prices, sector):
    dates = _dates(len(prices))
    with db_mod.transaction() as conn:
        conn.execute("INSERT INTO stock_sectors (symbol, sector) VALUES (?, ?)",
                     (symbol, sector))
        for d, px in zip(dates, prices):
            conn.execute(
                "INSERT INTO price_data (symbol, bar_date, close, adj_close, source) "
                "VALUES (?, ?, ?, ?, 'yfinance')",
                (symbol, d, float(px), float(px)),
            )


def test_scan_finds_cointegrated_pair_only(temp_db):
    # IT: COINTA is cointegrated with COINTB (Y = 50 + 1.5*X + persistent spread).
    # X has the larger sigma so returns correlate (clearing the pre-filter) and
    # beta is well-identified.
    x = _rw(N, base=500.0, sigma=2.0, seed=2)
    spread = _ar1(N, phi=0.9, sigma=1.0, seed=3)
    cointa = 50.0 + 1.5 * x + spread
    _seed_prices("COINTB", x, "IT")          # hedge leg (alphabetically X)
    _seed_prices("COINTA", cointa, "IT")     # dependent leg (alphabetically Y)

    # BANK: two independent random walks -> not cointegrated (also corr-filtered).
    _seed_prices("RWALK1", _rw(N, 300.0, 0.5, seed=10), "BANK")
    _seed_prices("RWALK2", _rw(N, 300.0, 0.5, seed=11), "BANK")

    out = scan_pairs(as_of=AS_OF, cfg=PairScanConfig())
    assert out["cointegrated"] == 1

    rows = latest_pairs(AS_OF)
    assert len(rows) == 1
    p = rows[0]
    assert p["symbol_y"] == "COINTA" and p["symbol_x"] == "COINTB"
    assert p["sector"] == "IT"
    assert abs(p["beta"] - 1.5) < 0.2      # ballpark; persistent spread biases OLS
    assert p["adf_tstat"] < -3.34
    assert p["signal"] in ("LONG_SPREAD", "SHORT_SPREAD", "EXIT", "HOLD", "FLAT")
    assert p["n_obs"] == N


def test_scan_is_idempotent_per_date(temp_db):
    x = _rw(N, base=500.0, sigma=2.0, seed=2)
    spread = _ar1(N, phi=0.9, sigma=1.0, seed=3)
    _seed_prices("COINTB", x, "IT")
    _seed_prices("COINTA", 50.0 + 1.5 * x + spread, "IT")

    scan_pairs(as_of=AS_OF)
    scan_pairs(as_of=AS_OF)                  # re-run replaces, no duplicate rows
    n = db_mod.fetch_one("SELECT COUNT(*) AS n FROM pairs WHERE as_of_date = ?",
                         (AS_OF,))["n"]
    assert n == 1
