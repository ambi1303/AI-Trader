"""NSE trading calendar.

Loads holidays from config/nse_holidays.yaml at module import.
Provides:
- is_trading_day(d) -> bool
- next_trading_day(d) / prev_trading_day(d)
- trading_days_between(start, end)
- validate_dates(dates) -> list[ValidationIssue]   (used by ingestion gate)

Treats Sat/Sun as non-trading. Special sessions (Muhurat) are still trading
days but with custom hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.contracts import Severity, ValidationIssue
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("validation.calendar")

_CONFIG_PATH = project_root() / "config" / "nse_holidays.yaml"


@dataclass(frozen=True)
class _CalendarData:
    holidays: dict[date, str]
    special_sessions: dict[date, dict]


def _load() -> _CalendarData:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    holidays: dict[date, str] = {}
    for _year, items in (raw.get("holidays") or {}).items():
        for item in items:
            d = date.fromisoformat(item["date"])
            holidays[d] = item.get("description", "")
    specials: dict[date, dict] = {}
    for _year, items in (raw.get("special_sessions") or {}).items():
        for item in items:
            d = date.fromisoformat(item["date"])
            specials[d] = item
    return _CalendarData(holidays=holidays, special_sessions=specials)


_DATA = _load()


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6


def is_holiday(d: date) -> bool:
    return d in _DATA.holidays


def is_trading_day(d: date) -> bool:
    return not is_weekend(d) and not is_holiday(d)


def is_special_session(d: date) -> bool:
    return d in _DATA.special_sessions


def next_trading_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def prev_trading_day(d: date) -> date:
    prv = d - timedelta(days=1)
    while not is_trading_day(prv):
        prv -= timedelta(days=1)
    return prv


def trading_days_between(start: date, end: date) -> list[date]:
    """Inclusive on both ends, ordered ascending."""
    out: list[date] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def reload_calendar() -> None:
    """Test hook: reload after editing the YAML."""
    global _DATA
    _DATA = _load()


def calendar_summary() -> dict:
    return {
        "holidays_loaded": len(_DATA.holidays),
        "special_sessions_loaded": len(_DATA.special_sessions),
    }


def validate_bar_dates(
    run_id: str, dates: list[date]
) -> list[ValidationIssue]:
    """Flag any bar whose date is a non-trading day."""
    issues: list[ValidationIssue] = []
    for d in dates:
        if is_weekend(d):
            issues.append(
                ValidationIssue(
                    run_id=run_id,
                    check_name="calendar.weekend_bar",
                    issue_date=d,
                    severity=Severity.ERROR,
                    message=f"Bar on weekend {d.isoformat()}",
                )
            )
        elif is_holiday(d):
            issues.append(
                ValidationIssue(
                    run_id=run_id,
                    check_name="calendar.holiday_bar",
                    issue_date=d,
                    severity=Severity.WARNING,
                    message=(
                        f"Bar on declared holiday {d.isoformat()}: "
                        f"{_DATA.holidays.get(d, '')}"
                    ),
                )
            )
    return issues


def upsert_calendar_to_db() -> int:
    """Optional: write the calendar into the trading_calendar table for SQL access."""
    from src.utils.db import transaction

    rows = []
    for d, desc in _DATA.holidays.items():
        rows.append((d.isoformat(), 1, desc, 0, None, None))
    for d, item in _DATA.special_sessions.items():
        rows.append(
            (
                d.isoformat(),
                0,
                item.get("description"),
                1,
                item.get("open"),
                item.get("close"),
            )
        )
    if not rows:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO trading_calendar
                (cal_date, is_holiday, description, is_special_session, session_open, session_close)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cal_date) DO UPDATE SET
                is_holiday         = excluded.is_holiday,
                description        = excluded.description,
                is_special_session = excluded.is_special_session,
                session_open       = excluded.session_open,
                session_close      = excluded.session_close
            """,
            rows,
        )
    log.info("Upserted {} calendar entries", len(rows))
    return len(rows)
