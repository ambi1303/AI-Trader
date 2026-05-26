"""Feature engineering (Week 2).

Submodules:
- technical_indicators: RSI, MACD, EMA, Bollinger, ATR, ADX, OBV, Stochastic.
- statistical_features: returns, volatility, momentum, drawdowns, gaps, volume.
- regime_features: Nifty trend, India VIX, beta/correlation, sector RS.
- circuit_features: circuit flags, days-since-circuit, low-volume flag.
- leakage_audit: property-based test that catches future-data leaks.
- feature_builder: orchestrator that produces the wide feature_data row.
"""
