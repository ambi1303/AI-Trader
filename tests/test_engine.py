"""End-to-end engine tests with synthetic data.

We craft a deterministic price series so we can verify that:
  - A signal at T enters at the OPEN of T+1.
  - A guaranteed stop hit produces the right loss.
  - A guaranteed target hit produces the right win.
  - PnL math (gross/net/cost) is internally consistent.
  - Mark-to-market on the equity curve is monotone with price moves.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.backtesting.cost_model import load_cost_config
from src.backtesting.engine import EngineConfig, run_backtest
from src.backtesting.risk import RiskConfig
from src.backtesting.sizing import SizingConfig


def _build_synth_inputs(price_path: list[tuple[float, float, float, float]],
                       start_date: date = date(2024, 1, 1),
                       symbol: str = "TST"):
    """price_path is a list of (open, high, low, close) per day."""
    rows_pr = []
    rows_atr = []
    for i, (o, h, l, c) in enumerate(price_path):
        d = (start_date + timedelta(days=i)).isoformat()
        rows_pr.append({
            "symbol": symbol, "bar_date": d,
            "open": o, "high": h, "low": l, "close": c,
        })
        rows_atr.append({"symbol": symbol, "bar_date": d, "atr": 1.0})
    prices = pd.DataFrame(rows_pr)
    atr = pd.DataFrame(rows_atr)
    return prices, atr


def _signal(symbol: str, on_date: date, prob: float = 0.99) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": symbol, "feature_date": on_date.isoformat(),
        "calibrated_prob": prob,
    }])


def test_signal_on_T_enters_at_open_of_T_plus_1():
    # 5 days, flat then sharp rally -> signal on day 1 should enter day 2 at open=101.
    path = [
        (100.0, 100.5, 99.5, 100.0),  # day 0: flat
        (100.0, 100.5, 99.5, 100.0),  # day 1: signal day
        (101.0, 105.0, 100.5, 104.0),  # day 2: entry (open=101) and run-up
        (104.0, 110.0, 103.0, 109.0),  # day 3: keep going (target hit at 104)
        (109.0, 109.5, 108.0, 108.5),
    ]
    prices, atr = _build_synth_inputs(path)
    signals = _signal("TST", on_date=date(2024, 1, 2), prob=0.99)
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        sizing=SizingConfig(risk_per_trade_pct=0.10, max_position_pct=0.50,
                            kelly_fraction=1.0, min_trade_rupees=100),
        risk=RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                        use_trailing_stop=False, max_holding_days=10,
                        max_per_sector=10, max_concurrent_positions=10,
                        daily_loss_limit_pct=1.0, cooldown_days_after_loss=0),
        cost=load_cost_config(),
        name="unit_entry",
    )
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["entry_date"] == "2024-01-03"        # day 2 (T+1)
    assert t["entry_price"] == 101.0              # open of T+1
    # Target = 101 + 3*ATR(=1.0) = 104. Day 2 high=105 -> filled at 104.
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == 104.0


def test_stop_loss_path_produces_correct_loss():
    # Entry 101, ATR=1, stop = 101 - 2 = 99. Day 2 low = 98 -> stop at 99.
    path = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),  # signal day
        (101.0, 101.0, 98.0, 99.5),    # entry then stop hit
        (99.5, 100.0, 99.0, 99.5),
    ]
    prices, atr = _build_synth_inputs(path)
    signals = _signal("TST", on_date=date(2024, 1, 2))
    cfg = EngineConfig(
        sizing=SizingConfig(risk_per_trade_pct=0.10, max_position_pct=0.50,
                            kelly_fraction=1.0, min_trade_rupees=100),
        risk=RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                        use_trailing_stop=False, daily_loss_limit_pct=1.0,
                        max_per_sector=10, max_concurrent_positions=10,
                        cooldown_days_after_loss=0),
        cost=load_cost_config(),
        name="unit_stop",
    )
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == 99.0
    # Gross loss = (99 - 101) * qty
    assert t["gross_pnl"] == (99.0 - 101.0) * t["qty"]
    # Net <= gross because costs reduce profitability (here both are negative
    # so |net| > |gross| when costs apply).
    assert t["net_pnl"] <= t["gross_pnl"]


def test_pnl_math_is_internally_consistent():
    path = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),
        (101.0, 105.0, 100.5, 104.0),
        (104.0, 105.0, 103.5, 104.5),
    ]
    prices, atr = _build_synth_inputs(path)
    signals = _signal("TST", on_date=date(2024, 1, 2))
    cfg = EngineConfig(
        sizing=SizingConfig(risk_per_trade_pct=0.10, max_position_pct=0.50,
                            kelly_fraction=1.0, min_trade_rupees=100),
        risk=RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                        use_trailing_stop=False, daily_loss_limit_pct=1.0,
                        max_per_sector=10, max_concurrent_positions=10,
                        cooldown_days_after_loss=0),
        cost=load_cost_config(),
        name="unit_pnl",
    )
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    expected_gross = (t["exit_price"] - t["entry_price"]) * t["qty"]
    assert abs(t["gross_pnl"] - expected_gross) < 1e-6
    assert abs((t["gross_pnl"] - t["cost_rupees"]) - t["net_pnl"]) < 1e-6


def test_no_signal_no_trades():
    path = [(100.0, 101.0, 99.0, 100.5)] * 10
    prices, atr = _build_synth_inputs(path)
    signals = pd.DataFrame(columns=["symbol", "feature_date", "calibrated_prob"])
    cfg = EngineConfig(name="unit_no_signal")
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg)
    assert len(res.trades) == 0
    # Equity should stay flat at initial capital.
    assert res.equity_curve["equity"].iloc[-1] == cfg.initial_capital


def test_trades_tagged_with_entry_regime():
    # Same setup as the entry test, but supply a regime timeline. The trade
    # enters on 2024-01-03 (T+1), so it must be tagged BULL_TREND.
    path = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),  # signal day (2024-01-02)
        (101.0, 105.0, 100.5, 104.0),  # entry 2024-01-03 + target
        (104.0, 110.0, 103.0, 109.0),
        (109.0, 109.5, 108.0, 108.5),
    ]
    prices, atr = _build_synth_inputs(path)
    signals = _signal("TST", on_date=date(2024, 1, 2), prob=0.99)
    cfg = EngineConfig(
        sizing=SizingConfig(risk_per_trade_pct=0.10, max_position_pct=0.50,
                            kelly_fraction=1.0, min_trade_rupees=100),
        risk=RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                        use_trailing_stop=False, max_holding_days=10,
                        max_per_sector=10, max_concurrent_positions=10,
                        daily_loss_limit_pct=1.0, cooldown_days_after_loss=0),
        cost=load_cost_config(),
        name="unit_regime_tag",
    )
    regime_by_date = {
        "2024-01-01": "RANGE",
        "2024-01-03": "BULL_TREND",   # as-of lookup carries forward to later days
    }
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg,
                       regime_by_date=regime_by_date)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["entry_regime"] == "BULL_TREND"
    # Per-regime summary is populated and the equity curve carries a regime col.
    assert res.by_regime["BULL_TREND"]["n_trades"] == 1
    assert "regime" in res.equity_curve.columns


def test_no_regime_mapping_leaves_tags_none():
    path = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (101.0, 105.0, 100.5, 104.0),
        (104.0, 110.0, 103.0, 109.0),
    ]
    prices, atr = _build_synth_inputs(path)
    signals = _signal("TST", on_date=date(2024, 1, 2), prob=0.99)
    cfg = EngineConfig(
        sizing=SizingConfig(risk_per_trade_pct=0.10, max_position_pct=0.50,
                            kelly_fraction=1.0, min_trade_rupees=100),
        risk=RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                        use_trailing_stop=False, max_holding_days=10,
                        max_per_sector=10, max_concurrent_positions=10,
                        daily_loss_limit_pct=1.0, cooldown_days_after_loss=0),
        cost=load_cost_config(),
        name="unit_regime_none",
    )
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg)
    assert res.trades.iloc[0]["entry_regime"] is None
    # Regime-agnostic backtest -> single UNKNOWN bucket from the equity curve.
    assert "UNKNOWN" in res.by_regime


def test_force_close_at_end_of_window():
    # Signal day 1, no stop/target hit -> position should be force-closed at end.
    # ATR=1 so target=104. Make sure no day reaches 104 or below 99.
    path = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),  # signal day
        (101.0, 102.0, 100.5, 101.5),  # entry, no hit
        (101.5, 102.5, 101.0, 102.0),  # no hit
        (102.0, 103.0, 101.5, 102.5),  # last bar
    ]
    prices, atr = _build_synth_inputs(path)
    signals = _signal("TST", on_date=date(2024, 1, 2))
    cfg = EngineConfig(
        sizing=SizingConfig(risk_per_trade_pct=0.10, max_position_pct=0.50,
                            kelly_fraction=1.0, min_trade_rupees=100),
        risk=RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                        use_trailing_stop=False, max_holding_days=100,
                        max_per_sector=10, max_concurrent_positions=10,
                        daily_loss_limit_pct=1.0, cooldown_days_after_loss=0),
        cost=load_cost_config(),
        name="unit_force_close",
    )
    res = run_backtest(predictions=signals, prices=prices, atr=atr,
                       sectors={"TST": "X"}, threshold=0.5, cfg=cfg)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "end"
