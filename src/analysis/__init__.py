"""On-demand stock analysis (technicals, conviction score, buy/sell zones).

This package powers the "Complete Analysis" experience: given a price history
(and optional fundamentals) it produces a readable technical read, a 0-100
conviction score broken down by factor with plain-language reasons, and
volatility-based buy/sell zones. It is pure/presentation logic -- it makes no
trading decisions of record and never writes to the DB.
"""
