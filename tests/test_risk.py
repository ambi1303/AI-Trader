"""Tests for the risk overlay logic."""

from __future__ import annotations

from src.backtesting.risk import (
    OpenPosition,
    RiskConfig,
    can_open_new_position,
    check_stop_or_target,
)


def _pos(stop=95.0, target=110.0, entry=100.0) -> OpenPosition:
    return OpenPosition(
        symbol="X", sector="IT", side="LONG", qty=10,
        entry_date="2024-01-01", entry_price=entry, atr_at_entry=2.5,
        stop=stop, target=target, high_watermark=entry,
        entry_prob=0.6, threshold=0.55,
    )


# ---------------------------------------------------------------------------
# Stop / target hit logic
# ---------------------------------------------------------------------------


def test_gap_down_below_stop_fills_at_open():
    p = _pos(stop=95.0)
    hr = check_stop_or_target(p, bar_open=90.0, bar_high=92.0,
                              bar_low=89.0, bar_close=91.0)
    assert hr.hit and hr.reason == "stop"
    assert hr.fill_price == 90.0


def test_gap_up_through_target_fills_at_open():
    p = _pos(target=110.0)
    hr = check_stop_or_target(p, bar_open=115.0, bar_high=116.0,
                              bar_low=114.0, bar_close=115.5)
    assert hr.hit and hr.reason == "target"
    assert hr.fill_price == 115.0


def test_only_stop_in_range_exits_at_stop():
    p = _pos(stop=95.0, target=120.0)
    hr = check_stop_or_target(p, bar_open=100.0, bar_high=102.0,
                              bar_low=94.0, bar_close=98.0)
    assert hr.hit and hr.reason == "stop"
    assert hr.fill_price == 95.0


def test_only_target_in_range_exits_at_target():
    p = _pos(stop=80.0, target=110.0)
    hr = check_stop_or_target(p, bar_open=100.0, bar_high=112.0,
                              bar_low=99.0, bar_close=108.0)
    assert hr.hit and hr.reason == "target"
    assert hr.fill_price == 110.0


def test_both_in_range_assumes_stop_first_pessimistic():
    p = _pos(stop=95.0, target=110.0)
    hr = check_stop_or_target(p, bar_open=100.0, bar_high=112.0,
                              bar_low=94.0, bar_close=108.0)
    assert hr.hit and hr.reason == "stop"
    assert hr.fill_price == 95.0


def test_no_hit_returns_close():
    p = _pos(stop=80.0, target=120.0)
    hr = check_stop_or_target(p, bar_open=100.0, bar_high=105.0,
                              bar_low=98.0, bar_close=103.0)
    assert not hr.hit
    assert hr.fill_price == 103.0


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------


def test_trailing_stop_only_ratchets_up():
    cfg = RiskConfig(use_trailing_stop=True, trail_atr_mult=2.0)
    p = _pos(stop=95.0)
    p.update_trailing_stop(today_high=110.0, cfg=cfg)  # raise stop
    new_stop_after_high = p.stop
    assert new_stop_after_high > 95.0

    p.update_trailing_stop(today_high=100.0, cfg=cfg)  # should NOT lower
    assert p.stop == new_stop_after_high


def test_trailing_stop_disabled_does_nothing():
    cfg = RiskConfig(use_trailing_stop=False)
    p = _pos(stop=95.0)
    p.update_trailing_stop(today_high=200.0, cfg=cfg)
    assert p.stop == 95.0


# ---------------------------------------------------------------------------
# Portfolio-level guards
# ---------------------------------------------------------------------------


def test_max_concurrent_positions_blocks_new_entry():
    cfg = RiskConfig(max_concurrent_positions=2)
    open_ = [_pos(), _pos()]
    allowed, reason = can_open_new_position(
        cfg=cfg, open_positions=open_, sector="IT", today_pnl_pct=0.0,
    )
    assert not allowed and reason == "max_concurrent_positions"


def test_max_per_sector_blocks_concentration():
    cfg = RiskConfig(max_concurrent_positions=10, max_per_sector=2)
    open_ = [_pos(), _pos()]  # two IT positions
    allowed, reason = can_open_new_position(
        cfg=cfg, open_positions=open_, sector="IT", today_pnl_pct=0.0,
    )
    assert not allowed and reason == "max_per_sector"


def test_daily_loss_limit_halts_entries():
    cfg = RiskConfig(daily_loss_limit_pct=0.03)
    allowed, reason = can_open_new_position(
        cfg=cfg, open_positions=[], sector="IT", today_pnl_pct=-0.04,
    )
    assert not allowed and reason == "daily_loss_limit"


def test_all_clear_returns_ok():
    cfg = RiskConfig()
    allowed, reason = can_open_new_position(
        cfg=cfg, open_positions=[], sector="IT", today_pnl_pct=0.0,
    )
    assert allowed and reason == "ok"
