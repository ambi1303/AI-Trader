"""Tests for the performance-metric layer."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.backtesting.metrics import compute_metrics


def _equity_curve(returns: list[float], start_capital: float = 1_000_000.0) -> pd.DataFrame:
    eq = [start_capital]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    eq = eq[1:]
    dates = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(len(eq))]
    return pd.DataFrame({
        "bar_date": dates,
        "cash": eq,
        "equity": eq,
        "open_count": [0] * len(eq),
        "daily_pnl": [0.0] * len(eq),
    })


def test_max_drawdown_on_known_curve():
    # Curve: 1.00 -> 1.20 -> 0.96 (= -20% from peak) -> 1.05.
    rets = [0.20, -0.20, 0.09375]  # gives 1, 1.2, 0.96, 1.05
    eq = _equity_curve(rets, start_capital=1.0)
    m = compute_metrics(equity_curve=eq, trades=pd.DataFrame(),
                        initial_capital=1.0)
    assert abs(m.max_drawdown_pct - (-20.0)) < 1e-9
    assert m.max_drawdown_days >= 1


def test_sharpe_zero_when_no_variance():
    eq = _equity_curve([0.001] * 252, start_capital=1.0)
    m = compute_metrics(equity_curve=eq, trades=pd.DataFrame(),
                        initial_capital=1.0)
    # Constant daily return -> stdev = 0 -> Sharpe defined as 0 by convention.
    assert m.sharpe == 0.0


def test_sharpe_positive_when_mean_return_positive():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.005, size=252).tolist()
    eq = _equity_curve(rets, start_capital=1_000_000.0)
    m = compute_metrics(equity_curve=eq, trades=pd.DataFrame(),
                        initial_capital=1_000_000.0)
    assert m.sharpe > 0


def test_hit_rate_computed_from_trade_ledger():
    trades = pd.DataFrame([
        {"entry_price": 100, "qty": 10, "gross_pnl": 50, "cost_rupees": 5,
         "net_pnl": 45},
        {"entry_price": 100, "qty": 10, "gross_pnl": -30, "cost_rupees": 5,
         "net_pnl": -35},
        {"entry_price": 100, "qty": 10, "gross_pnl": 100, "cost_rupees": 5,
         "net_pnl": 95},
    ])
    eq = _equity_curve([0.001] * 10, start_capital=1.0)
    m = compute_metrics(equity_curve=eq, trades=trades, initial_capital=1.0)
    assert m.n_trades == 3
    assert m.hit_rate_pct == pytest_approx(2 / 3 * 100)
    # avg_win and avg_loss expressed in % of entry notional
    assert m.avg_win_pct > 0
    assert m.avg_loss_pct < 0


def pytest_approx(v, tol=1e-6):
    """Local approx for float equality without importing pytest only for a numeric check."""
    class _A:
        def __eq__(self, other): return abs(other - v) < tol
        def __repr__(self): return f"~{v}"
    return _A()


def test_empty_inputs_return_zeros():
    m = compute_metrics(equity_curve=pd.DataFrame(),
                        trades=pd.DataFrame(),
                        initial_capital=1_000_000.0)
    assert m.n_trades == 0
    assert m.total_return_pct == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown_pct == 0.0
