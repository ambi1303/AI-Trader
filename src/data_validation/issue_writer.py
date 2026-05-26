"""Persist ValidationIssue rows to validation_failures."""

from __future__ import annotations

import json

from src.contracts import ValidationIssue
from src.utils.db import transaction
from src.utils.logger import get_logger

log = get_logger("validation.issue_writer")


def write_issues(issues: list[ValidationIssue]) -> int:
    if not issues:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO validation_failures
                (run_id, check_name, symbol, issue_date, severity, message, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    i.run_id,
                    i.check_name,
                    i.symbol,
                    i.issue_date.isoformat() if i.issue_date else None,
                    i.severity.value,
                    i.message,
                    json.dumps(i.details) if i.details else None,
                )
                for i in issues
            ],
        )
    log.info("Persisted {} validation issues", len(issues))
    return len(issues)
