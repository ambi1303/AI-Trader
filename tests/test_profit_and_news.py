"""Pure unit tests (no DB / no network) for the >=5% profit-target floor
and the news headline classifier (merger/demerger detection)."""

from __future__ import annotations

from src.analysis import stock_analysis as sa
from src.backtesting.risk import RiskConfig
from src.data_ingestion.news_scraper import classify_headline, symbol_query
from src.signals.strategy import StrategyConfig, conviction_tp_atr_mult


# --- Minimum profit target -------------------------------------------------


def test_riskconfig_target_floored_at_min_profit() -> None:
    # Low ATR: the pure ATR target (+3*0.5 = +1.5%) is below the 5% floor.
    cfg = RiskConfig(stop_atr_mult=2.0, take_profit_atr_mult=3.0,
                     min_profit_pct=5.0)
    target = cfg.target_for(100.0, 0.5)
    assert target == 105.0


def test_riskconfig_target_uses_atr_when_above_floor() -> None:
    cfg = RiskConfig(take_profit_atr_mult=3.0, min_profit_pct=5.0)
    # ATR target = 100 + 3*10 = 130 (+30%) which clears the floor.
    assert cfg.target_for(100.0, 10.0) == 130.0


def test_riskconfig_default_has_no_floor() -> None:
    # Default min_profit_pct=0 keeps the legacy pure-ATR behaviour (backtests).
    cfg = RiskConfig(take_profit_atr_mult=3.0)
    assert cfg.target_for(100.0, 0.5) == 101.5


def test_zones_floor_first_target_at_five_pct() -> None:
    z = sa.buy_sell_zones(100.0, {"atr": 0.5}, min_profit_pct=5.0)
    assert z["available"] is True
    assert z["target1"] >= 105.0 - 1e-9
    assert z["target1_pct"] >= 5.0 - 1e-9
    assert z["target2_pct"] >= 10.0 - 1e-9
    assert z["min_profit_pct"] == 5.0


def test_zones_keep_atr_target_when_above_floor() -> None:
    z = sa.buy_sell_zones(100.0, {"atr": 20.0}, min_profit_pct=5.0)
    # Wide ATR -> R-multiple target far above +5%.
    assert z["target1_pct"] > 5.0


# --- Conviction-scaled take-profit (let winners run) -----------------------


def test_tp_mult_base_at_min_score() -> None:
    cfg = StrategyConfig(min_score=55.0, base_tp_atr_mult=3.0,
                         strong_tp_atr_mult=8.0)
    assert conviction_tp_atr_mult(55.0, cfg) == 3.0


def test_tp_mult_strong_at_top_score() -> None:
    cfg = StrategyConfig(min_score=55.0, base_tp_atr_mult=3.0,
                         strong_tp_atr_mult=8.0)
    assert conviction_tp_atr_mult(100.0, cfg) == 8.0


def test_tp_mult_monotonic_in_score() -> None:
    cfg = StrategyConfig()
    assert (conviction_tp_atr_mult(60.0, cfg)
            < conviction_tp_atr_mult(75.0, cfg)
            < conviction_tp_atr_mult(95.0, cfg))


def test_high_conviction_targets_higher_than_floor() -> None:
    cfg = StrategyConfig()
    risk = RiskConfig(take_profit_atr_mult=3.0, min_profit_pct=5.0)
    entry, atr = 100.0, 3.0  # 3% ATR
    strong = risk.target_for(entry, atr, conviction_tp_atr_mult(100.0, cfg))
    weak = risk.target_for(entry, atr, conviction_tp_atr_mult(cfg.min_score, cfg))
    assert strong > weak >= 105.0          # weak still respects the +5% floor
    assert (strong / entry - 1.0) * 100.0 > 5.0   # strong runs well past +5%


# --- News classification ---------------------------------------------------


def test_classify_merger() -> None:
    assert classify_headline("ABC Ltd to acquire XYZ in landmark merger") == "M&A"


def test_classify_demerger() -> None:
    assert classify_headline("Board approves demerger of consumer unit") == "M&A"


def test_classify_acquisition() -> None:
    assert classify_headline("Company acquires rival for $1bn") == "M&A"


def test_classify_corporate_action() -> None:
    assert classify_headline("Firm announces 1:1 bonus issue") == "Corporate action"
    assert classify_headline("Stock split record date set") == "Corporate action"


def test_classify_plain_news_is_unclassified() -> None:
    assert classify_headline("Quarterly profit rises 12% on strong demand") is None


def test_symbol_query_includes_symbol_and_context() -> None:
    q = symbol_query("RELIANCE")
    assert "RELIANCE" in q
    assert "NSE" in q
