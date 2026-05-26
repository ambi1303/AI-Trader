"""End-to-end daily pipeline.

Order of operations (each step is fault-isolated -- a failure logs to
``validation_failures`` and continues with the next step):

    1. Apply schema migrations           [src.db.migrate.apply_schema]
    2. Ingest latest yfinance bars       [scripts.ingest_yfinance]   *optional*
    3. Build features for today          [scripts.build_features]    *optional*
    4. Predict for the universe          [src.models.predict]
    5. Generate signal_outbox rows       [src.signals.generator]
    6. Reconcile paper trades            [src.paper.reconcile]
    7. Send the daily notification       [src.notifications.dispatcher]

Each step writes a ``validation_failures`` row on error and the final report
shows the count under the Health-check section. The script's exit code is
0 if at least the *report* (step 7) succeeded; otherwise 1.

Examples
--------
Full daily run with notifications:

    python -m scripts.run_daily

Skip live data ingestion (useful for backfills / weekend rehearsals):

    python -m scripts.run_daily --skip-ingest --skip-features

Dry-run (build everything but do not actually email/WhatsApp):

    python -m scripts.run_daily --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable

from src.db.migrate import apply_schema
from src.notifications.dispatcher import (
    DispatchResult,
    any_channel_sent,
    send_daily,
)
from src.paper.reconcile import reconcile
from src.signals.generator import SignalGenConfig, generate_signals
from src.utils.db import execute, fetch_all
from src.utils.logger import get_logger

log = get_logger("scripts.run_daily")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DailyRunResult:
    run_started_at: str
    as_of: str
    steps: list[StepResult] = field(default_factory=list)
    dispatch: DispatchResult | None = None

    @property
    def ok(self) -> bool:
        return self.dispatch is not None and any_channel_sent(self.dispatch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_started_at": self.run_started_at,
            "as_of": self.as_of,
            "ok": self.ok,
            "steps": [asdict(s) for s in self.steps],
            "dispatch": (
                self.dispatch.to_dict() if self.dispatch else None
            ),
        }


# ---------------------------------------------------------------------------
# Fault-isolated step runner
# ---------------------------------------------------------------------------


def _record_failure(*, run_id: str, step: str, message: str) -> None:
    """Persist a step failure into validation_failures (best-effort).

    If the DB itself is broken we just log and move on -- otherwise the
    daily script would crash mid-incident which is exactly when you most
    need the notifier to still fire.
    """
    try:
        execute(
            """
            INSERT INTO validation_failures
                (run_id, check_name, severity, message)
            VALUES (?, ?, 'error', ?)
            """,
            (run_id, f"daily_step:{step}", message[:500]),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not persist failure for step {}: {}", step, exc)


def _run_step(
    name: str,
    fn: Callable[[], dict[str, Any]],
    *,
    run_id: str,
    result: DailyRunResult,
) -> StepResult:
    log.info(">> step start: {}", name)
    try:
        summary = fn() or {}
        sr = StepResult(name=name, ok=True, summary=summary)
        log.info("<< step done : {} ({})", name, summary)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=3)
        log.error("step {} failed: {}", name, exc)
        log.debug("traceback:\n{}", tb)
        _record_failure(run_id=run_id, step=name, message=f"{exc}")
        sr = StepResult(name=name, ok=False, error=str(exc))
    result.steps.append(sr)
    return sr


# ---------------------------------------------------------------------------
# Step implementations (thin wrappers around existing modules)
# ---------------------------------------------------------------------------


def _step_migrate() -> dict[str, Any]:
    return {"schema_version": apply_schema()}


def _step_ingest(symbols: list[str] | None) -> dict[str, Any]:
    """Pull latest yfinance bars. Best-effort -- network failures are not
    fatal; the orchestrator continues to predict / report on whatever
    history is already in the DB."""
    from scripts.ingest_yfinance import main as ingest_main
    argv: list[str] = ["--years", "1"]   # short window covers any gap
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    rc = ingest_main(argv)
    return {"return_code": rc}


def _step_ingest_angelone(symbols: list[str] | None) -> dict[str, Any]:
    """Pull EOD bars from Angel One SmartAPI (read-only).

    Skipped silently if credentials are missing -- Angel One is OPTIONAL;
    the rest of the pipeline runs fine on yfinance + BhavCopy alone.
    """
    from scripts.ingest_angelone import main as angelone_main
    argv: list[str] = ["--days", "5"]
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    rc = angelone_main(argv)
    return {"return_code": rc, "skipped": rc == 2}     # rc=2 == no creds


def _step_features(symbols: list[str] | None) -> dict[str, Any]:
    from scripts.build_features import main as features_main
    argv: list[str] = []
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    rc = features_main(argv)
    return {"return_code": rc}


def _step_predict(symbols: list[str] | None) -> dict[str, Any]:
    from scripts.predict_today import main as predict_main
    argv: list[str] = []
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    rc = predict_main(argv)
    return {"return_code": rc}


def _step_generate(as_of: str | None,
                   threshold_override: float | None) -> dict[str, Any]:
    # threshold_override is the same --threshold flag the notify step uses;
    # forwarding it here means a single CLI flag now dictates *both* which
    # signals are generated AND which probabilities the report explains.
    cfg = SignalGenConfig(threshold_override=threshold_override)
    out = generate_signals(signal_date=as_of, config=cfg)
    return {
        "kept": len(out),
        "symbols": [s.symbol for s in out],
    }


def _step_reconcile(as_of: str | None) -> dict[str, Any]:
    summary = reconcile(as_of=as_of)
    return summary.to_dict()


def _step_notify(as_of: str | None, dry_run: bool, threshold: float | None,
                 result: DailyRunResult) -> dict[str, Any]:
    dispatch = send_daily(
        report_date=as_of,
        threshold_override=threshold,
        dry_run=dry_run,
    )
    result.dispatch = dispatch
    return {
        "channels": {k: v.get("status") for k, v in dispatch.channels.items()},
        "errors": list(dispatch.errors.keys()),
    }


# ---------------------------------------------------------------------------
# Top-level main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full daily AI-trader pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--as-of", default=None,
                   help="ISO YYYY-MM-DD; defaults to today's local date.")
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols; default = whole universe.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override model threshold for signal calculation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Render the report but do not send email/WhatsApp.")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip yfinance ingestion (useful when offline).")
    p.add_argument("--use-angelone", action="store_true",
                   help="Also pull EOD bars from Angel One SmartAPI "
                        "(requires ANGEL_* env vars; data-only / no orders).")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip feature rebuild (use latest persisted features).")
    p.add_argument("--skip-predict", action="store_true",
                   help="Skip model inference (use latest predictions_log rows).")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip signal generation (no new entries).")
    p.add_argument("--skip-reconcile", action="store_true",
                   help="Skip paper-trade reconciliation.")
    p.add_argument("--skip-notify", action="store_true",
                   help="Skip the dispatch step.")
    p.add_argument("--print-summary", action="store_true",
                   help="Print the JSON run summary at the end.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    as_of = args.as_of or date.today().isoformat()
    syms = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else None
    )
    run_id = f"daily_{as_of}"

    result = DailyRunResult(
        run_started_at=date.today().isoformat(),
        as_of=as_of,
    )

    # Step 1: migrate (must always run; failure is fatal because the rest
    # of the pipeline assumes the v4 schema).
    if not _run_step("migrate", _step_migrate, run_id=run_id, result=result).ok:
        log.error("Schema migration failed; aborting daily run.")
        return 1

    # Steps 2..6 are fault-isolated.
    if not args.skip_ingest:
        _run_step("ingest_yfinance", lambda: _step_ingest(syms),
                  run_id=run_id, result=result)
    if args.use_angelone and not args.skip_ingest:
        _run_step("ingest_angelone", lambda: _step_ingest_angelone(syms),
                  run_id=run_id, result=result)
    if not args.skip_features:
        _run_step("build_features", lambda: _step_features(syms),
                  run_id=run_id, result=result)
    if not args.skip_predict:
        _run_step("predict_today", lambda: _step_predict(syms),
                  run_id=run_id, result=result)
    if not args.skip_generate:
        _run_step("generate_signals",
                  lambda: _step_generate(as_of, args.threshold),
                  run_id=run_id, result=result)
    if not args.skip_reconcile:
        _run_step("reconcile_paper", lambda: _step_reconcile(as_of),
                  run_id=run_id, result=result)

    # Step 7: notify (always; even if everything else broke we want the
    # report so the human knows what happened).
    if not args.skip_notify:
        _run_step(
            "send_daily_report",
            lambda: _step_notify(as_of, args.dry_run, args.threshold, result),
            run_id=run_id, result=result,
        )

    if args.print_summary:
        print(json.dumps(result.to_dict(), indent=2, default=str))

    if not args.skip_notify and not args.dry_run:
        return 0 if result.ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# A small read-only helper exposed for tests and ad-hoc CLI use.
# ---------------------------------------------------------------------------


def list_recent_failures(limit: int = 20) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT run_id, check_name, severity, message, created_at "
        "FROM validation_failures "
        "WHERE check_name LIKE 'daily_step:%' "
        "ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    return [dict(r) for r in rows]
