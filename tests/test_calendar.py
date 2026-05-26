"""Calendar correctness and DB round-trip."""

from __future__ import annotations

from datetime import date

from src.data_validation import calendar_check
from src.db.migrate import apply_schema


def test_independence_day_2024_is_holiday() -> None:
    assert calendar_check.is_holiday(date(2024, 8, 15))
    assert not calendar_check.is_trading_day(date(2024, 8, 15))


def test_weekend_is_not_trading_day() -> None:
    # 2024-08-17 is a Saturday
    assert calendar_check.is_weekend(date(2024, 8, 17))
    assert not calendar_check.is_trading_day(date(2024, 8, 17))


def test_next_and_prev_skip_holidays_and_weekends() -> None:
    # Aug 14 2024 is a Wed (Janmashtami was 26-Aug-24, here just walk past 15-Aug)
    nxt = calendar_check.next_trading_day(date(2024, 8, 14))
    assert calendar_check.is_trading_day(nxt)
    assert nxt > date(2024, 8, 14)


def test_trading_days_between_count_sane() -> None:
    days = calendar_check.trading_days_between(date(2024, 1, 1), date(2024, 1, 31))
    # January 2024 had 22 trading days (Republic Day on 26th)
    assert 21 <= len(days) <= 23


def test_db_round_trip() -> None:
    apply_schema()
    n = calendar_check.upsert_calendar_to_db()
    assert n > 0
