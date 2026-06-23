"""Unit tests for the positions-page roll-up (`summarize_positions`)."""

from __future__ import annotations

import pytest

from src.web.queries import (
    ClosedTradeRow,
    OpenPositionRow,
    summarize_positions,
)


def _open(symbol, entry, qty, last):
    unreal = (last - entry) * qty if last is not None else 0.0
    pct = (last - entry) / entry * 100.0 if (last is not None and entry) else 0.0
    return OpenPositionRow(
        paper_id=1, symbol=symbol, sector="X", qty=qty, entry_date="2026-06-01",
        entry_price=entry, last_close=last, stop_loss=None, take_profit=None,
        unrealised_pnl=unreal, unrealised_pnl_pct=pct, holding_days=10,
    )


def _closed(symbol, net):
    return ClosedTradeRow(
        paper_id=2, symbol=symbol, entry_date="2026-05-01", exit_date="2026-05-10",
        entry_price=100.0, exit_price=100.0 + net, qty=1, net_pnl=net,
        pnl_pct=net, exit_reason="target", holding_days=9,
    )


def test_summary_invested_value_and_pnl():
    opens = [_open("A", 100.0, 10, 120.0), _open("B", 50.0, 10, 45.0)]
    summary = summarize_positions(opens, [])
    assert summary.n_open == 2
    assert summary.invested == pytest.approx(1500.0)     # 1000 + 500
    assert summary.market_value == pytest.approx(1650.0)  # 1200 + 450
    assert summary.unrealised_pnl == pytest.approx(150.0)
    assert summary.unrealised_pnl_pct == pytest.approx(10.0)


def test_summary_unknown_price_is_pnl_neutral():
    # A position with no last close is valued at entry -> 0 unrealised P&L.
    opens = [_open("A", 100.0, 10, None)]
    summary = summarize_positions(opens, [])
    assert summary.invested == pytest.approx(1000.0)
    assert summary.market_value == pytest.approx(1000.0)
    assert summary.unrealised_pnl == pytest.approx(0.0)
    assert summary.unrealised_pnl_pct == pytest.approx(0.0)


def test_summary_realised_and_win_rate():
    closed = [_closed("A", 500.0), _closed("B", -200.0), _closed("C", 0.0)]
    summary = summarize_positions([], closed)
    assert summary.realised_pnl_30d == pytest.approx(300.0)
    assert summary.winners_30d == 1
    assert summary.losers_30d == 1            # break-even (0) counts as neither
    assert summary.closed_30d == 3
    assert summary.win_rate_30d_pct == pytest.approx(100.0 / 3.0)


def test_summary_empty_is_zeroed():
    summary = summarize_positions([], [])
    assert summary.n_open == 0
    assert summary.invested == 0.0
    assert summary.unrealised_pnl_pct == 0.0
    assert summary.win_rate_30d_pct == 0.0
