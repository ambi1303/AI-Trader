"""Send (or preview) the daily AI trader report.

Examples
--------
Preview without touching email/WhatsApp:

    python -m scripts.send_daily_report --dry-run

Send for today's date with stored model threshold:

    python -m scripts.send_daily_report

Send for a historical date with an explicit threshold override:

    python -m scripts.send_daily_report --date 2025-10-04 --threshold 0.55

Setup (one-time)
----------------
1. Copy ``.env.example`` -> ``.env``.
2. Fill in:
   - SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO  (Gmail App Password works)
   - CALLMEBOT_PHONE, CALLMEBOT_APIKEY              (free WhatsApp; see env file)
3. ``python -m scripts.send_daily_report --dry-run`` to verify rendering.
4. ``python -m scripts.send_daily_report`` once you're happy.

Exit code: 0 on success or dry-run; 1 if every enabled channel failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.notifications.dispatcher import (
    DispatchResult,
    any_channel_sent,
    send_daily,
)
from src.utils.logger import get_logger

log = get_logger("scripts.send_daily_report")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and dispatch the daily AI trader report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date",
        default=None,
        help="ISO YYYY-MM-DD; defaults to today's local date.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the model's stored threshold for the signal calculation.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build & render only; write artefacts to data/reports/notifications/.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to notifications.yaml (defaults to config/notifications.yaml).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the JSON dispatch summary to stdout.",
    )
    return p.parse_args(argv)


def _print_summary(result: DispatchResult) -> None:
    print(json.dumps(result.to_dict(), indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    log.info(
        "Starting daily report | date={} dry_run={} threshold_override={}",
        args.date or "today", args.dry_run, args.threshold,
    )
    result = send_daily(
        report_date=args.date,
        threshold_override=args.threshold,
        dry_run=args.dry_run,
        config_path=args.config,
    )
    if args.print_summary:
        _print_summary(result)
    if args.dry_run:
        log.info("Dry-run artefacts ready: {}", result.artefacts)
        return 0
    if any_channel_sent(result):
        return 0
    log.error("All enabled channels failed or were unconfigured.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
