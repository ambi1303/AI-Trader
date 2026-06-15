"""Run the daily pipeline on a fixed local-time schedule.

Why we hand-rolled this instead of using ``cron`` or ``APScheduler``:

* ``cron`` inside a container is fragile (the env it inherits is
  almost-empty, signals don't propagate cleanly, and Python tracebacks
  go to a parallel log we then have to forward). Worse, slim Debian
  images have a stripped cron that silently drops jobs longer than
  RLIMIT_NPROC permits.
* ``APScheduler`` is a 5K-LOC dependency for what is genuinely a
  ``datetime + sleep`` problem. Less code = less attack surface.

What this script does:

1. Compute the next ``RUN_HOUR:RUN_MINUTE`` boundary in the ``TZ`` zone
   (defaults to Asia/Kolkata, set by the Dockerfile).
2. Sleep until then in 60-second chunks so SIGTERM is honoured fast.
3. Invoke ``python -m scripts.run_daily`` as a subprocess with the same
   environment we have. We run it as a subprocess (not an in-process
   import) so a hard crash in the pipeline -- segfault from a buggy
   wheel, OOM kill, etc. -- doesn't take the scheduler down with it.
4. Log start, end, exit code, and elapsed time. Loop forever.

Failure semantics:

* If ``run_daily`` exits non-zero, we DO NOT crash the scheduler. The
  daily orchestrator already records each step's failure to the
  ``validation_failures`` table; the dashboard surfaces those. Crashing
  the scheduler container would just put us in a Docker restart loop
  that masks the real problem.
* If the wall clock jumps backward (NTP correction, host suspended)
  we recompute the next boundary on every iteration, so we never run
  twice on the same day.
* On SIGTERM (`docker stop`) we abandon the in-flight sleep but let any
  in-flight subprocess finish. ``docker stop`` has a 10s grace period
  by default; for a 2-5 min pipeline run you should
  ``docker stop --time 600`` if you actually want a clean exit.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger

log = get_logger("scripts.run_scheduler")


# ---------------------------------------------------------------------------
# Signal handling: flip a flag so the main loop exits between sleep ticks.
# We do NOT raise in the handler -- raising from a signal handler in the
# middle of a `subprocess.run()` would orphan the child.
# ---------------------------------------------------------------------------

_should_stop = False


def _request_stop(signum: int, _frame) -> None:
    global _should_stop
    log.info("scheduler: received signal {}, stopping after current iteration",
             signum)
    _should_stop = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


# ---------------------------------------------------------------------------
# Schedule maths
# ---------------------------------------------------------------------------


def _next_run(now: datetime, *, hour: int, minute: int,
              weekdays_only: bool) -> datetime:
    """Smallest datetime > now that lands on the configured local
    HH:MM, optionally skipping Saturday and Sunday.

    All arithmetic is timezone-aware. ``now`` MUST already carry a
    tzinfo or this raises -- silent UTC inference would be a foot-gun.
    """
    if now.tzinfo is None:
        raise ValueError("scheduler requires a timezone-aware 'now'")

    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    # weekday(): Mon=0 ... Sun=6
    while weekdays_only and candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate


def _sleep_until(target: datetime, *, tick_seconds: float = 30.0) -> None:
    """Sleep in small chunks so we react to SIGTERM within a tick.

    ``time.sleep(big_number)`` on Linux is interruptible by signals, but
    the implementation here is portable and explicit -- on Windows the
    behaviour differs and we don't want to rely on it.
    """
    while not _should_stop:
        now = datetime.now(target.tzinfo)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, tick_seconds))


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------


def _run_pipeline(extra_args: list[str], *, timeout_seconds: int) -> int:
    """Spawn ``python -m scripts.run_daily`` and stream its output.

    Returns the subprocess's exit code. We do NOT raise on non-zero
    because the daily orchestrator's job is to be fault-tolerant; if
    it exited 1 we still want to keep the scheduler alive for tomorrow.
    """
    cmd = [sys.executable, "-m", "scripts.run_daily", *extra_args]
    log.info("scheduler: launching pipeline | argv={}", cmd)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            timeout=timeout_seconds,
            # Inherit stdout/stderr so logs flow into Docker's log driver.
            stdout=None,
            stderr=None,
            # Inherit env (TZ, DB_PATH, WEB_*, etc.) -- nothing to override.
        )
    except subprocess.TimeoutExpired:
        log.error("scheduler: pipeline exceeded timeout of {}s; killed",
                  timeout_seconds)
        return -1
    elapsed = time.monotonic() - started
    log.info("scheduler: pipeline exit_code={} elapsed={:.1f}s",
             proc.returncode, elapsed)
    return int(proc.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the AI Trader daily pipeline on a fixed schedule."
    )
    # Defaults match the Windows scheduler script we shipped earlier:
    # 18:00 IST on weekdays.
    p.add_argument("--hour", type=int,
                   default=int(os.getenv("SCHED_HOUR", "18")),
                   help="Local hour (0-23) at which to run. Default: 18.")
    p.add_argument("--minute", type=int,
                   default=int(os.getenv("SCHED_MINUTE", "0")),
                   help="Local minute (0-59) at which to run. Default: 0.")
    p.add_argument("--tz", default=os.getenv("TZ", "Asia/Kolkata"),
                   help="IANA timezone for the schedule. Default: "
                        "Asia/Kolkata (matches Indian market close + ~2h).")
    p.add_argument("--weekdays-only", action="store_true",
                   default=os.getenv("SCHED_WEEKDAYS_ONLY", "true").lower()
                           in {"1", "true", "yes"},
                   help="Skip Saturday and Sunday. Default: true.")
    p.add_argument("--timeout-seconds", type=int,
                   default=int(os.getenv("SCHED_PIPELINE_TIMEOUT",
                                         str(15 * 60))),
                   help="Hard kill the pipeline if it runs this long. "
                        "Default: 900s (15 min) -- well above the typical "
                        "2-5 min real-world run.")
    p.add_argument("--run-now", action="store_true",
                   help="Run the pipeline immediately on startup (in "
                        "addition to the schedule). Useful for first "
                        "deploys to populate an empty DB without "
                        "waiting until 18:00.")
    # Anything after `--` is forwarded verbatim to scripts.run_daily.
    # Example:
    #   python -m scripts.run_scheduler --hour 18 -- --skip-notify --threshold 0.30
    # We use a positional with REMAINDER (rather than action='append')
    # because argparse otherwise tries to interpret flag-shaped values
    # like `--skip-notify` as parser options of run_scheduler itself.
    p.add_argument("pipeline_args", nargs=argparse.REMAINDER,
                   help="After '--', forwarded as-is to scripts.run_daily.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _install_signal_handlers()

    try:
        zone = ZoneInfo(args.tz)
    except Exception as exc:  # noqa: BLE001
        log.error("scheduler: bad TZ '{}': {}", args.tz, exc)
        return 2

    log.info("scheduler: starting | tz={} run_at={:02d}:{:02d} weekdays_only={} "
             "pipeline_timeout={}s run_now={}",
             args.tz, args.hour, args.minute, args.weekdays_only,
             args.timeout_seconds, args.run_now)

    # argparse.REMAINDER keeps the leading "--" separator if present;
    # strip it so we don't pass it on to run_daily (which would treat
    # it as the end of *its* own option parsing -- harmless but ugly).
    pipeline_args = list(args.pipeline_args or [])
    if pipeline_args and pipeline_args[0] == "--":
        pipeline_args = pipeline_args[1:]

    if args.run_now:
        _run_pipeline(pipeline_args, timeout_seconds=args.timeout_seconds)

    while not _should_stop:
        now = datetime.now(zone)
        target = _next_run(now, hour=args.hour, minute=args.minute,
                           weekdays_only=args.weekdays_only)
        log.info("scheduler: next run at {} ({} from now)",
                 target.isoformat(),
                 (target - now))
        _sleep_until(target)
        if _should_stop:
            break
        _run_pipeline(pipeline_args, timeout_seconds=args.timeout_seconds)

    log.info("scheduler: clean shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
