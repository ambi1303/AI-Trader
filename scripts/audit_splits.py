"""Verify yfinance split/bonus adjustment correctness on known events."""

from __future__ import annotations

import sys
import uuid

from src.data_validation.issue_writer import write_issues
from src.data_validation.split_audit import audit_known_events
from src.utils.logger import get_logger


def main(argv: list[str] | None = None) -> int:
    log = get_logger("scripts.audit_splits")
    run_id = f"split-audit-{uuid.uuid4().hex[:12]}"
    rep = audit_known_events(run_id)
    write_issues(rep.issues)
    log.info(
        "run_id={} checked={} failed={} issues={}",
        run_id,
        rep.events_checked,
        rep.events_failed,
        len(rep.issues),
    )
    if rep.events_failed > 0:
        log.error("Split audit found {} failures", rep.events_failed)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
