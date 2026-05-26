"""Backtesting layer (Week 4).

Submodules:
- cost_model: realistic per-leg Indian equity costs (brokerage, STT, GST, etc.).
- sizing:     fractional Kelly + volatility targeting + risk-of-ruin caps.
- risk:       ATR stops, trailing stops, time stops, sector caps, daily loss limit.
- engine:     bar-by-bar long-only simulator with realistic next-day-open fills
              and intraday stop/target hit logic from H/L.
- metrics:    Sharpe, Sortino, Calmar, max drawdown, hit rate, expectancy,
              profit factor, turnover, MAR.
- scenarios:  pre-defined stress windows (COVID, election, high-VIX).
- report:     compact JSON-serialisable summaries; persistence to DB tables
              backtest_runs / backtest_equity / backtest_trades.
"""
