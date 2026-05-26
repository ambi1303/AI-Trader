"""Pydantic contract guarantees."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.contracts import Bar, BarSource


def _bar(**kwargs) -> Bar:
    base = dict(
        symbol="RELIANCE",
        bar_date=date(2024, 1, 2),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("95"),
        close=Decimal("105"),
        volume=1000,
        adj_close=Decimal("105"),
        source=BarSource.YFINANCE,
    )
    base.update(kwargs)
    return Bar(**base)


def test_bar_basic_ok() -> None:
    b = _bar()
    assert b.symbol == "RELIANCE"


def test_bar_rejects_negative_price() -> None:
    with pytest.raises(ValidationError):
        _bar(close=Decimal("-1"))


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        _bar(volume=-5)


def test_bar_rejects_bad_symbol() -> None:
    with pytest.raises(ValidationError):
        _bar(symbol="bad symbol with space")


def test_bar_is_immutable() -> None:
    b = _bar()
    with pytest.raises(ValidationError):
        b.volume = 2  # type: ignore[misc]
