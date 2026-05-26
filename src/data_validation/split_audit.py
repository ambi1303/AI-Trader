"""Verify yfinance auto-adjusted closes around known corporate actions.

For each known split/bonus event in corporate_actions, take the closes
immediately before and after the ex-date for both yfinance and bhavcopy.
The yf series should be smooth across the event (ratio close to 1.0)
because yfinance is auto-adjusted; the bhavcopy series should show the
expected jump (ratio matching the action ratio).

We flag deviations beyond a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from src.contracts import Severity, ValidationIssue
from src.data_validation.calendar_check import prev_trading_day
from src.utils.db import fetch_all
from src.utils.logger import get_logger

log = get_logger("validation.split_audit")

# Tolerances
_YF_RATIO_TOL = 0.05      # adjusted yf series should not move more than 5% across event
_BC_RATIO_TOL = 0.10      # bhavcopy ratio should match action ratio within 10%


@dataclass
class SplitAuditReport:
    run_id: str
    events_checked: int
    events_failed: int
    issues: list[ValidationIssue]


def _close_on_or_before(symbol: str, source: str, d: date) -> tuple[date, float] | None:
    row = fetch_all(
        """
        SELECT bar_date, close FROM price_data
        WHERE  symbol = ? AND source = ? AND bar_date <= ?
        ORDER BY bar_date DESC LIMIT 1
        """,
        (symbol, source, d.isoformat()),
    )
    if not row:
        return None
    return datetime.fromisoformat(row[0]["bar_date"]).date(), float(row[0]["close"])


def _close_on_or_after(symbol: str, source: str, d: date) -> tuple[date, float] | None:
    row = fetch_all(
        """
        SELECT bar_date, close FROM price_data
        WHERE  symbol = ? AND source = ? AND bar_date >= ?
        ORDER BY bar_date ASC LIMIT 1
        """,
        (symbol, source, d.isoformat()),
    )
    if not row:
        return None
    return datetime.fromisoformat(row[0]["bar_date"]).date(), float(row[0]["close"])


def _expected_bc_ratio(action_type: str, ratio_from: int | None, ratio_to: int | None) -> float | None:
    """Pre/post raw close ratio (post / pre) implied by the corporate action.

    For a 1:1 bonus (ratio_from=1, ratio_to=1) the post raw close ~= pre / 2.
    For a 1:2 split (FV halved, equivalent to 1:2 share count), post = pre / 2.
    """
    if action_type in ("bonus", "split", "demerger"):
        if ratio_from is None or ratio_to is None or ratio_from == 0:
            return None
        if action_type == "bonus":
            # x:y bonus means y new shares for every x existing, so multiplier = (x+y)/x
            multiplier = (ratio_from + ratio_to) / ratio_from
        else:  # split
            # ratio_from:ratio_to means total shares scale by ratio_from/ratio_to
            # if FV goes from 10 -> 2, ratio_from=5, ratio_to=1 => multiplier=5
            multiplier = ratio_from / ratio_to
        if multiplier <= 0:
            return None
        return 1.0 / multiplier
    return None


def audit_known_events(run_id: str) -> SplitAuditReport:
    rows = fetch_all(
        "SELECT symbol, ex_date, action_type, ratio_from, ratio_to FROM corporate_actions"
    )
    issues: list[ValidationIssue] = []
    checked = 0
    failed = 0

    for r in rows:
        symbol = r["symbol"]
        ex_date = datetime.fromisoformat(r["ex_date"]).date()
        before_target = prev_trading_day(ex_date)

        yf_pre = _close_on_or_before(symbol, "yfinance", before_target)
        yf_post = _close_on_or_after(symbol, "yfinance", ex_date)
        bc_pre = _close_on_or_before(symbol, "bhavcopy", before_target)
        bc_post = _close_on_or_after(symbol, "bhavcopy", ex_date)

        if not (yf_pre and yf_post):
            log.debug("Skipping {} {} - missing yf data", symbol, ex_date)
            continue
        checked += 1

        # 1) Adjusted yfinance should be smooth across the event.
        yf_ratio = yf_post[1] / yf_pre[1] if yf_pre[1] else 0
        if abs(yf_ratio - 1.0) > _YF_RATIO_TOL:
            failed += 1
            issues.append(
                ValidationIssue(
                    run_id=run_id,
                    check_name="split_audit.yf_unadjusted_jump",
                    symbol=symbol,
                    issue_date=ex_date,
                    severity=Severity.ERROR,
                    message=(
                        f"yfinance close jumped {yf_ratio:.3f}x across "
                        f"{symbol} {r['action_type']} on {ex_date} "
                        f"(pre={yf_pre[1]} post={yf_post[1]})"
                    ),
                    details={"yf_ratio": yf_ratio},
                )
            )

        # 2) BhavCopy should jump by the action's expected ratio.
        if bc_pre and bc_post:
            expected = _expected_bc_ratio(
                r["action_type"], r["ratio_from"], r["ratio_to"]
            )
            if expected is not None:
                bc_ratio = bc_post[1] / bc_pre[1] if bc_pre[1] else 0
                if abs(bc_ratio - expected) / expected > _BC_RATIO_TOL:
                    issues.append(
                        ValidationIssue(
                            run_id=run_id,
                            check_name="split_audit.bc_ratio_off",
                            symbol=symbol,
                            issue_date=ex_date,
                            severity=Severity.WARNING,
                            message=(
                                f"BhavCopy ratio {bc_ratio:.3f} differs from "
                                f"expected {expected:.3f} for {symbol} "
                                f"{r['action_type']} on {ex_date}"
                            ),
                            details={
                                "bc_ratio": bc_ratio,
                                "expected_ratio": expected,
                            },
                        )
                    )

    log.info("split_audit checked={} failed={}", checked, failed)
    return SplitAuditReport(
        run_id=run_id, events_checked=checked, events_failed=failed, issues=issues
    )
