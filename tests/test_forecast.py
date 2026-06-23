"""Unit + integration tests for the multi-horizon price forecaster."""

from __future__ import annotations

import pytest

from src.analysis.forecast import (
    HORIZONS,
    _DEFAULT_LONG_RUN_ANNUAL,
    _MOM_CONTRIB_CAP,
    forecast_stock,
)
from src.analysis.forecast_store import latest_forecast, store_forecast
from src.utils import db as db_mod


# --------------------------------------------------------------------------
# Pure projection
# --------------------------------------------------------------------------


def test_unavailable_without_inputs():
    assert forecast_stock(last_close=None, daily_vol=0.02)["available"] is False
    assert forecast_stock(last_close=100, daily_vol=None)["available"] is False
    assert forecast_stock(last_close=100, daily_vol=0.0)["available"] is False


def test_all_horizons_present_and_ordered():
    out = forecast_stock(last_close=100.0, daily_vol=0.02,
                         mom_20d=0.04, mom_60d=0.10)
    assert out["available"]
    labels = [h["label"] for h in out["horizons"]]
    assert labels == [lbl for lbl, _ in HORIZONS]
    for h in out["horizons"]:
        assert h["low_price"] < h["expected_price"] < h["high_price"]
        assert 0.0 <= h["prob_up_pct"] <= 100.0


def test_band_widens_with_horizon():
    out = forecast_stock(last_close=100.0, daily_vol=0.02)
    bands = [h["band_pct"] for h in out["horizons"]]
    assert bands == sorted(bands)            # strictly non-decreasing in T
    assert bands[-1] > bands[0]


def test_positive_momentum_lifts_short_term_vs_negative():
    up = forecast_stock(last_close=100.0, daily_vol=0.02,
                       mom_20d=0.08, mom_60d=0.18)
    dn = forecast_stock(last_close=100.0, daily_vol=0.02,
                       mom_20d=-0.08, mom_60d=-0.18)
    # 1-month expected price should be higher with positive momentum.
    assert up["horizons"][1]["expected_price"] > dn["horizons"][1]["expected_price"]


def test_long_run_converges_to_baseline_when_no_momentum():
    out = forecast_stock(last_close=100.0, daily_vol=0.02,
                        mom_20d=0.0, mom_60d=0.0)
    expected_ann = _DEFAULT_LONG_RUN_ANNUAL * 100.0
    for h in out["horizons"]:                # drift-only -> ~baseline at every T
        assert h["annualized_return_pct"] == pytest.approx(expected_ann, abs=0.2)


def test_momentum_contribution_is_capped_long_run():
    """A huge recent spike must not blow up the multi-year target: the momentum
    log-contribution is hard-capped, so 3Y return stays bounded."""
    out = forecast_stock(last_close=100.0, daily_vol=0.05,
                        mom_20d=2.0, mom_60d=5.0)             # absurd momentum
    three_y = out["horizons"][-1]
    # Baseline (3y compounding) + capped momentum, expressed as a return cap.
    import math
    cap_ret = (math.exp(_MOM_CONTRIB_CAP
                        + math.log1p(_DEFAULT_LONG_RUN_ANNUAL) * 3) - 1.0) * 100.0
    assert three_y["expected_return_pct"] <= cap_ret + 1e-6


# --------------------------------------------------------------------------
# DB integration
# --------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "fc.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    db_mod.execute_script(
        """
        CREATE TABLE feature_data (
            symbol TEXT NOT NULL, feature_date TEXT NOT NULL,
            close REAL, vol_20d REAL, mom_20d REAL, mom_60d REAL, atr_pct REAL,
            PRIMARY KEY (symbol, feature_date)
        );
        CREATE TABLE price_forecasts (
            symbol TEXT NOT NULL, as_of_date TEXT NOT NULL,
            horizon_label TEXT NOT NULL, horizon_days INTEGER NOT NULL,
            last_close REAL NOT NULL, expected_price REAL NOT NULL,
            low_price REAL NOT NULL, high_price REAL NOT NULL,
            expected_return_pct REAL NOT NULL, annualized_return_pct REAL,
            prob_up_pct REAL, verdict TEXT, method TEXT, model_run_id TEXT,
            created_at TEXT NOT NULL DEFAULT '2026-01-01',
            PRIMARY KEY (symbol, as_of_date, horizon_label)
        );
        """
    )
    db_mod.execute(
        "INSERT INTO feature_data "
        "(symbol, feature_date, close, vol_20d, mom_20d, mom_60d, atr_pct) "
        "VALUES ('TCS', '2026-06-19', 2000, 0.015, 0.03, 0.09, 0.012)",
    )
    return db_file


def test_store_and_read_back(temp_db):
    out = store_forecast("TCS", "2026-06-19")
    assert out["available"]

    rows = latest_forecast("TCS")
    assert len(rows) == len(HORIZONS)
    assert [r["horizon_label"] for r in rows] == [lbl for lbl, _ in HORIZONS]
    for r in rows:
        assert r["low_price"] < r["expected_price"] < r["high_price"]


def test_store_is_idempotent(temp_db):
    store_forecast("TCS", "2026-06-19")
    store_forecast("TCS", "2026-06-19")             # upsert, not duplicate
    n = db_mod.fetch_one(
        "SELECT COUNT(*) AS c FROM price_forecasts WHERE symbol='TCS'")["c"]
    assert n == len(HORIZONS)


def test_missing_symbol_is_unavailable(temp_db):
    out = store_forecast("NOPE", "2026-06-19")
    assert out["available"] is False
