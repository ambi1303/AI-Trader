"""Tests for ``scripts.run_scheduler``.

We only unit-test the pure logic (next-run computation, weekday skip,
signal-flag handling). We do NOT exercise the subprocess invocation in
tests -- ``scripts.run_daily`` is already covered by its own test file
and shelling out from pytest is slow and brittle on Windows.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts.run_scheduler import _next_run


IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# _next_run -- the core scheduling primitive
# ---------------------------------------------------------------------------


def test_next_run_today_if_target_in_future() -> None:
    # Wednesday 10:00 IST -> next 18:00 should be the same day.
    now = datetime(2026, 5, 27, 10, 0, tzinfo=IST)
    result = _next_run(now, hour=18, minute=0, weekdays_only=True)
    assert result == datetime(2026, 5, 27, 18, 0, tzinfo=IST)


def test_next_run_tomorrow_if_target_already_passed() -> None:
    # Wednesday 19:00 IST -> next 18:00 is Thursday.
    now = datetime(2026, 5, 27, 19, 0, tzinfo=IST)
    result = _next_run(now, hour=18, minute=0, weekdays_only=True)
    assert result == datetime(2026, 5, 28, 18, 0, tzinfo=IST)


def test_next_run_skips_saturday_and_sunday() -> None:
    # Friday 19:00 -> next 18:00 must be Monday, not Saturday.
    now = datetime(2026, 5, 29, 19, 0, tzinfo=IST)   # Fri
    result = _next_run(now, hour=18, minute=0, weekdays_only=True)
    assert result == datetime(2026, 6, 1, 18, 0, tzinfo=IST)  # Mon
    assert result.weekday() == 0


def test_next_run_does_not_skip_weekend_when_disabled() -> None:
    # Friday 19:00 with weekdays_only=False -> Saturday is fine.
    now = datetime(2026, 5, 29, 19, 0, tzinfo=IST)   # Fri
    result = _next_run(now, hour=18, minute=0, weekdays_only=False)
    assert result == datetime(2026, 5, 30, 18, 0, tzinfo=IST)  # Sat
    assert result.weekday() == 5


def test_next_run_landing_exactly_on_target_rolls_forward() -> None:
    # If "now" is exactly the boundary, we want the NEXT occurrence,
    # not "now". (Otherwise we'd run twice on the same minute.)
    now = datetime(2026, 5, 27, 18, 0, tzinfo=IST)
    result = _next_run(now, hour=18, minute=0, weekdays_only=True)
    assert result == datetime(2026, 5, 28, 18, 0, tzinfo=IST)


def test_next_run_handles_saturday_now() -> None:
    # Saturday morning -> next 18:00 must be Monday.
    now = datetime(2026, 5, 30, 9, 0, tzinfo=IST)   # Sat
    result = _next_run(now, hour=18, minute=0, weekdays_only=True)
    assert result == datetime(2026, 6, 1, 18, 0, tzinfo=IST)


def test_next_run_handles_sunday_now_after_target() -> None:
    # Sunday 19:00 -> Monday 18:00.
    now = datetime(2026, 5, 31, 19, 0, tzinfo=IST)  # Sun
    result = _next_run(now, hour=18, minute=0, weekdays_only=True)
    assert result == datetime(2026, 6, 1, 18, 0, tzinfo=IST)


def test_next_run_rejects_naive_datetime() -> None:
    # Naive datetimes are a foot-gun: the system would silently pick
    # the local TZ on the host. Refuse them outright.
    naive = datetime(2026, 5, 27, 10, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        _next_run(naive, hour=18, minute=0, weekdays_only=True)


def test_next_run_respects_minute_field() -> None:
    now = datetime(2026, 5, 27, 18, 15, tzinfo=IST)
    result = _next_run(now, hour=18, minute=30, weekdays_only=True)
    assert result == datetime(2026, 5, 27, 18, 30, tzinfo=IST)


# ---------------------------------------------------------------------------
# Argument parsing -- protects against future regressions in defaults
# ---------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no env overrides leak into the test.
    for key in ("SCHED_HOUR", "SCHED_MINUTE", "TZ", "SCHED_WEEKDAYS_ONLY",
                "SCHED_PIPELINE_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("sys.argv", ["run_scheduler"])
    from scripts.run_scheduler import _parse_args
    args = _parse_args()
    assert args.hour == 18
    assert args.minute == 0
    assert args.tz == "Asia/Kolkata"
    assert args.weekdays_only is True
    assert args.timeout_seconds == 15 * 60
    assert args.run_now is False


def test_parse_args_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHED_HOUR", "9")
    monkeypatch.setenv("SCHED_MINUTE", "30")
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("SCHED_WEEKDAYS_ONLY", "false")
    monkeypatch.setenv("SCHED_PIPELINE_TIMEOUT", "300")
    monkeypatch.setattr("sys.argv", ["run_scheduler"])
    from scripts.run_scheduler import _parse_args
    args = _parse_args()
    assert args.hour == 9
    assert args.minute == 30
    assert args.tz == "UTC"
    assert args.weekdays_only is False
    assert args.timeout_seconds == 300


def test_parse_args_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything after '--' must be forwarded verbatim to run_daily."""
    monkeypatch.setattr(
        "sys.argv",
        ["run_scheduler", "--hour", "18", "--",
         "--skip-notify", "--threshold", "0.30"],
    )
    from scripts.run_scheduler import _parse_args
    args = _parse_args()
    # argparse.REMAINDER keeps the '--' itself; main() strips it.
    assert args.pipeline_args == ["--", "--skip-notify", "--threshold", "0.30"]


def test_parse_args_no_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_scheduler", "--hour", "9"])
    from scripts.run_scheduler import _parse_args
    args = _parse_args()
    assert args.pipeline_args == []
