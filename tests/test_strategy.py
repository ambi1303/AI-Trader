"""Tests for the rules-based factor strategy engine."""

from __future__ import annotations

import pytest

from src.signals import strategy as strat
from src.signals.strategy import (
    StrategyConfig,
    score_row,
    score_row_breakout,
    score_row_mean_reversion,
    score_universe,
)
from src.utils import db as db_mod


# ---------------------------------------------------------------------------
# Pure scorer
# ---------------------------------------------------------------------------


def _good_row(**over):
    r = {
        "symbol": "GOODCO", "sector": "IT", "close": 1000.0,
        "dist_ema_200_pct": 0.12, "dist_ema_50_pct": 0.04, "rsi_14": 58.0,
        "mom_60d": 0.18, "ret_20d": 0.05, "dd_from_high_252d": -0.03,
        "macd_hist": 1.2, "roe": 0.22, "debt_to_equity": 0.2,
        "profit_margin": 0.18, "pe_ttm": 18.0,
    }
    r.update(over)
    return r


def test_good_quality_uptrend_scores_high():
    c = score_row(_good_row(), StrategyConfig())
    assert c is not None
    assert c.score >= 60
    assert any("momentum" in s for s in c.reasons)
    assert any("200-EMA" in s for s in c.reasons)


def test_gate_rejects_below_200ema():
    assert score_row(_good_row(dist_ema_200_pct=-0.05), StrategyConfig()) is None


def test_gate_rejects_overbought_and_falling_knife():
    assert score_row(_good_row(rsi_14=90.0), StrategyConfig()) is None
    assert score_row(_good_row(rsi_14=20.0), StrategyConfig()) is None


def test_gate_rejects_penny_and_negative_momentum_and_deep_drawdown():
    assert score_row(_good_row(close=10.0), StrategyConfig()) is None
    assert score_row(_good_row(mom_60d=-0.02), StrategyConfig()) is None
    assert score_row(_good_row(dd_from_high_252d=-0.45), StrategyConfig()) is None


def test_missing_fundamentals_still_scores_on_price_action():
    row = _good_row(roe=None, pe_ttm=None, profit_margin=None,
                    debt_to_equity=None)
    c = score_row(row, StrategyConfig())
    assert c is not None                      # price-only names still tradeable
    assert c.sub["quality"] == 45.0
    assert c.sub["value"] == 45.0


def test_score_universe_sorts_and_filters_by_min_score():
    rows = [
        _good_row(symbol="A", mom_60d=0.30, roe=0.30),
        _good_row(symbol="B", mom_60d=0.10, roe=0.16),
        _good_row(symbol="C", dist_ema_200_pct=-0.10),   # gated out
    ]
    ranked = score_universe(rows, StrategyConfig(min_score=50))
    syms = [c.symbol for c in ranked]
    assert "C" not in syms
    assert syms == sorted(syms, key=lambda s: {"A": 2, "B": 1}[s], reverse=True)
    assert syms[0] == "A"                      # strongest momentum+quality first


# ---------------------------------------------------------------------------
# Mean-reversion scorer (RANGE regime)
# ---------------------------------------------------------------------------


def _mr_row(**over):
    r = {
        "symbol": "DIPCO", "sector": "FMCG", "close": 1000.0,
        "dist_ema_200_pct": 0.05, "rsi_14": 22.0, "bb_pct_b": 0.05,
        "mom_60d": 0.02, "dd_from_high_252d": -0.08, "roe": 0.25,
    }
    r.update(over)
    return r


def test_mean_reversion_scores_oversold_quality_dip():
    c = score_row_mean_reversion(_mr_row(), StrategyConfig())
    assert c is not None
    assert c.score >= 55
    assert any("oversold" in s or "lower band" in s for s in c.reasons)
    assert set(c.sub) == {"reversion", "quality", "trend"}


def test_mean_reversion_requires_an_oversold_trigger():
    # Not oversold on either RSI or %b -> excluded.
    assert score_row_mean_reversion(
        _mr_row(rsi_14=55.0, bb_pct_b=0.6), StrategyConfig()) is None


def test_mean_reversion_rejects_broken_down_and_falling_knives():
    assert score_row_mean_reversion(
        _mr_row(dist_ema_200_pct=-0.10), StrategyConfig()) is None
    assert score_row_mean_reversion(
        _mr_row(mom_60d=-0.20), StrategyConfig()) is None
    assert score_row_mean_reversion(
        _mr_row(dd_from_high_252d=-0.40), StrategyConfig()) is None


# ---------------------------------------------------------------------------
# Breakout scorer (HIGH_VOLATILITY regime)
# ---------------------------------------------------------------------------


def _bo_row(**over):
    r = {
        "symbol": "BRKO", "sector": "IT", "close": 1000.0,
        "dist_ema_200_pct": 0.10, "rsi_14": 62.0, "dd_from_high_252d": -0.01,
        "vol_ratio_20d": 1.8, "mom_60d": 0.15, "macd_hist": 1.0,
    }
    r.update(over)
    return r


def test_breakout_scores_new_high_on_volume_surge():
    c = score_row_breakout(_bo_row(), StrategyConfig())
    assert c is not None
    assert c.score >= 55
    assert any("52w high" in s for s in c.reasons)
    assert any("volume" in s for s in c.reasons)
    assert set(c.sub) == {"volume", "proximity", "momentum"}


def test_breakout_requires_proximity_and_volume():
    assert score_row_breakout(
        _bo_row(dd_from_high_252d=-0.10), StrategyConfig()) is None     # not near high
    assert score_row_breakout(
        _bo_row(vol_ratio_20d=1.0), StrategyConfig()) is None           # no surge
    assert score_row_breakout(
        _bo_row(rsi_14=40.0), StrategyConfig()) is None                 # below momentum floor


# ---------------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------------


def test_score_universe_dispatches_by_engine():
    rows = [_mr_row(symbol="DIP"), _good_row(symbol="MOM", rsi_14=58.0)]
    # The oversold dip (RSI 22) fails the momentum RSI floor but passes
    # mean-reversion; the momentum name is the reverse.
    mr = {c.symbol for c in score_universe(rows, StrategyConfig(),
                                           engine="mean_reversion")}
    mom = {c.symbol for c in score_universe(rows, StrategyConfig(),
                                            engine="momentum")}
    assert "DIP" in mr and "DIP" not in mom
    assert "MOM" in mom


def test_score_universe_breakout_excludes_non_breakouts():
    rows = [_bo_row(symbol="BRK"),
            _bo_row(symbol="MID", dd_from_high_252d=-0.15, vol_ratio_20d=1.0)]
    out = {c.symbol for c in score_universe(rows, StrategyConfig(),
                                            engine="breakout")}
    assert out == {"BRK"}


# ---------------------------------------------------------------------------
# Signal generation against a temp DB
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "strat.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    db_mod.execute_script(
        """
        CREATE TABLE feature_data (
            symbol TEXT NOT NULL, feature_date TEXT NOT NULL,
            close REAL, atr_14 REAL, vol_20d REAL, mom_20d REAL, mom_60d REAL,
            ret_1d REAL, beta_60d REAL,
            PRIMARY KEY (symbol, feature_date)
        );
        CREATE TABLE signal_outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, signal_date TEXT NOT NULL, side TEXT NOT NULL,
            entry_price REAL, stop_loss REAL, take_profit REAL,
            qty INTEGER, confidence REAL,
            status TEXT NOT NULL DEFAULT 'pending', payload_json TEXT,
            created_at TEXT NOT NULL DEFAULT '2026-01-01', sent_at TEXT, error TEXT
        );
        CREATE UNIQUE INDEX ix_outbox_uq ON signal_outbox(symbol, signal_date);
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, sector TEXT, status TEXT NOT NULL DEFAULT 'open',
            qty INTEGER
        );
        CREATE TABLE validation_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            check_name TEXT NOT NULL, symbol TEXT, issue_date TEXT,
            severity TEXT NOT NULL, message TEXT NOT NULL, details_json TEXT,
            created_at TEXT NOT NULL DEFAULT '2026-01-01'
        );
        """
    )
    return db_file


def _seed_feature(symbol, close, atr, date_="2026-06-17",
                  vol_20d=None, mom_20d=None, mom_60d=None):
    # vol_20d=None leaves the feasibility gate dormant (can't assess -> allow),
    # so the legacy tests behave exactly as before.
    db_mod.execute(
        "INSERT INTO feature_data "
        "(symbol, feature_date, close, atr_14, vol_20d, mom_20d, mom_60d) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (symbol, date_, close, atr, vol_20d, mom_20d, mom_60d),
    )


def test_generate_strategy_signals_writes_buy_rows(temp_db, monkeypatch):
    rows = [
        _good_row(symbol="A", sector="IT", close=1000.0, mom_60d=0.30),
        _good_row(symbol="B", sector="BANK", close=1500.0, mom_60d=0.20),
        _good_row(symbol="C", sector="AUTO", close=800.0, mom_60d=0.12),
    ]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    for r in rows:
        _seed_feature(r["symbol"], r["close"], r["close"] * 0.02)

    out = strat.generate_strategy_signals(signal_date="2026-06-17")
    syms = {s.symbol for s in out}
    assert syms == {"A", "B", "C"}

    saved = db_mod.fetch_all(
        "SELECT symbol, side, status, stop_loss, entry_price, take_profit, qty "
        "FROM signal_outbox"
    )
    assert all(r["side"] == "BUY" and r["status"] == "pending" for r in saved)
    for r in saved:
        assert r["stop_loss"] < r["entry_price"] < r["take_profit"]
        assert r["qty"] > 0


def _seed_return_series(symbol, rets, close, atr, beta=None, end="2026-06-17"):
    """Seed a trailing ret_1d series (one row/day) so the correlation engine has
    real history. The final row (``end``) also carries close/atr for pricing."""
    from datetime import date as _date, timedelta
    end_d = _date.fromisoformat(end)
    n = len(rets)
    for i, r in enumerate(rets):
        d = (end_d - timedelta(days=(n - 1 - i))).isoformat()
        db_mod.execute(
            "INSERT INTO feature_data "
            "(symbol, feature_date, close, atr_14, ret_1d, beta_60d) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, d, close, atr, r, beta),
        )


def test_portfolio_gate_blocks_correlated_name(temp_db, monkeypatch):
    # A and B are perfectly correlated (identical return histories) in different
    # sectors; C is uncorrelated. A scores highest, so A is taken and B is
    # dropped by the correlation cap, while C survives.
    rows = [
        _good_row(symbol="A", sector="IT", close=1000.0, mom_60d=0.30),
        _good_row(symbol="B", sector="BANK", close=1000.0, mom_60d=0.20),
        _good_row(symbol="C", sector="AUTO", close=1000.0, mom_60d=0.12),
    ]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)

    n = 45
    rA = [0.02 if i % 2 == 0 else -0.02 for i in range(n)]
    rC = [0.02 if (i // 2) % 2 == 0 else -0.02 for i in range(n)]
    _seed_return_series("A", rA, 1000.0, 20.0)
    _seed_return_series("B", list(rA), 1000.0, 20.0)   # identical -> corr 1.0
    _seed_return_series("C", rC, 1000.0, 20.0)

    out = strat.generate_strategy_signals(
        signal_date="2026-06-17",
        config=StrategyConfig(require_feasible=False),
    )
    syms = {s.symbol for s in out}
    assert "A" in syms
    assert "C" in syms
    assert "B" not in syms                       # blocked by correlation cap


def test_portfolio_gate_disabled_keeps_correlated_name(temp_db, monkeypatch):
    rows = [
        _good_row(symbol="A", sector="IT", close=1000.0, mom_60d=0.30),
        _good_row(symbol="B", sector="BANK", close=1000.0, mom_60d=0.20),
    ]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    n = 45
    rA = [0.02 if i % 2 == 0 else -0.02 for i in range(n)]
    _seed_return_series("A", rA, 1000.0, 20.0)
    _seed_return_series("B", list(rA), 1000.0, 20.0)

    from src.portfolio.construct import PortfolioConfig
    out = strat.generate_strategy_signals(
        signal_date="2026-06-17",
        config=StrategyConfig(require_feasible=False,
                              portfolio=PortfolioConfig(enabled=False)),
    )
    assert {s.symbol for s in out} == {"A", "B"}   # gate off -> both kept


def test_generate_strategy_signals_respects_sector_cap(temp_db, monkeypatch):
    rows = [_good_row(symbol=f"IT{i}", sector="IT", close=1000.0 + i,
                      mom_60d=0.30 - i * 0.01) for i in range(5)]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    for r in rows:
        _seed_feature(r["symbol"], r["close"], r["close"] * 0.02)

    out = strat.generate_strategy_signals(
        signal_date="2026-06-17",
        config=StrategyConfig(max_per_sector=2),
    )
    assert len(out) == 2                       # sector cap, not capacity


def test_generate_strategy_signals_idempotent_and_skips_held(temp_db, monkeypatch):
    rows = [_good_row(symbol="A", sector="IT", close=1000.0),
            _good_row(symbol="B", sector="BANK", close=1500.0)]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    for r in rows:
        _seed_feature(r["symbol"], r["close"], r["close"] * 0.02)

    first = strat.generate_strategy_signals(signal_date="2026-06-17")
    assert len(first) == 2
    second = strat.generate_strategy_signals(signal_date="2026-06-17")
    assert second == []                        # unique index makes re-run a no-op
    assert db_mod.fetch_one(
        "SELECT COUNT(*) AS n FROM signal_outbox")["n"] == 2


def test_feasibility_gate_skips_sluggish_name(temp_db, monkeypatch):
    # SLOW: tiny daily volatility + negative trend -> can't plausibly reach
    # even +5% in the holding window, so it must be skipped.
    # FAST: healthy volatility -> kept.
    rows = [
        _good_row(symbol="SLOW", sector="IT", close=1000.0, mom_60d=0.18),
        _good_row(symbol="FAST", sector="BANK", close=1000.0, mom_60d=0.18),
    ]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    _seed_feature("SLOW", 1000.0, 1.0, vol_20d=0.001,
                  mom_20d=-0.05, mom_60d=-0.10)
    _seed_feature("FAST", 1000.0, 25.0, vol_20d=0.025,
                  mom_20d=0.04, mom_60d=0.12)

    out = strat.generate_strategy_signals(
        signal_date="2026-06-17",
        config=StrategyConfig(min_feasibility_prob=0.50),
    )
    syms = {s.symbol for s in out}
    assert "FAST" in syms
    assert "SLOW" not in syms


def test_require_feasible_false_keeps_sluggish_name(temp_db, monkeypatch):
    rows = [_good_row(symbol="SLOW", sector="IT", close=1000.0, mom_60d=0.18)]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    _seed_feature("SLOW", 1000.0, 1.0, vol_20d=0.001,
                  mom_20d=-0.05, mom_60d=-0.10)

    out = strat.generate_strategy_signals(
        signal_date="2026-06-17",
        config=StrategyConfig(require_feasible=False),
    )
    assert {s.symbol for s in out} == {"SLOW"}    # gate disabled -> still entered


def test_generate_strategy_signals_capacity(temp_db, monkeypatch):
    rows = [_good_row(symbol=f"S{i}", sector=f"SEC{i}", close=1000.0 + i,
                      mom_60d=0.30 - i * 0.01) for i in range(10)]
    monkeypatch.setattr(strat.discovery, "_rows", lambda force=False: rows)
    for r in rows:
        _seed_feature(r["symbol"], r["close"], r["close"] * 0.02)

    out = strat.generate_strategy_signals(
        signal_date="2026-06-17",
        config=StrategyConfig(target_holdings=3, max_per_sector=10),
    )
    assert len(out) == 3                        # capacity cap
