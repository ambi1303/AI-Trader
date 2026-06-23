"""End-to-end daily pipeline.

Order of operations (each step is fault-isolated -- a failure logs to
``validation_failures`` and continues with the next step):

    1. Apply schema migrations           [src.db.migrate.apply_schema]
    2. Ingest latest yfinance bars       [scripts.ingest_yfinance]    *optional*
    3. Refresh fundamentals snapshot     [scripts.ingest_fundamentals] *optional*
    4. Build features for today          [scripts.build_features]     *optional*
    5. Predict for the universe          [src.models.predict]
    6. Generate signal_outbox rows       [src.signals.generator]
    7. Reconcile paper trades            [src.paper.reconcile]
    8. Send the daily notification       [src.notifications.dispatcher]

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
from src.backtesting.risk import RiskConfig
from src.paper.reconcile import reconcile
from src.signals.generator import SignalGenConfig, generate_signals
from src.signals.strategy import (
    StrategyConfig,
    generate_strategy_signals,
    strategy_risk_config,
)
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


def _step_ingest_bhavcopy() -> dict[str, Any]:
    """Ingest the whole-market official NSE EOD (BhavCopy) for the last few
    trading days. Free, no auth, and runs anywhere (local + GitHub Actions),
    so the broad universe the factor strategy trades stays fresh every day.
    Best-effort -- a download miss for one day is non-fatal.
    """
    from datetime import timedelta

    from scripts.ingest_bhavcopy import main as bhav_main
    end = date.today()
    start = end - timedelta(days=7)        # short window covers weekend/holiday gaps
    rc = bhav_main(["--start", start.isoformat(), "--end", end.isoformat()])
    return {"return_code": rc, "window": f"{start.isoformat()}..{end.isoformat()}"}


def _step_fundamentals(symbols: list[str] | None) -> dict[str, Any]:
    """Refresh the current fundamentals snapshot (P/E, ROE, D/E, ...).

    Snapshot-only by default: quarterly history reconstruction is only
    needed at retrain time, so the daily run just refreshes the latest
    point so the as-of join in feature_builder stays current. Best-effort
    -- yfinance network/404 failures are non-fatal.
    """
    from scripts.ingest_fundamentals import main as fundamentals_main
    argv: list[str] = ["--no-reconstruct"]
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    rc = fundamentals_main(argv)
    return {"return_code": rc, "skipped": rc == 2}     # rc=2 == no symbols


def _step_news(symbols: list[str] | None, *, cap: int = 60) -> dict[str, Any]:
    """Best-effort daily news refresh (Google News RSS, free/no-auth).

    Targets the names the user actually looks at: currently-held paper
    positions plus the daily universe (or an explicit --symbols list),
    capped so we don't hammer the feed. Failures for individual symbols are
    swallowed inside ``fetch_for_symbols``; this whole step is non-critical.
    """
    from src.data_ingestion.news_scraper import fetch_for_symbols, symbol_query
    from src.utils.db import fetch_all

    targets: list[str] = []
    if symbols:
        targets = list(symbols)
    else:
        try:
            held = fetch_all(
                "SELECT DISTINCT symbol FROM paper_trades WHERE status = 'open'"
            )
            uni = fetch_all("SELECT symbol FROM v_universe_today ORDER BY symbol")
            seen: set[str] = set()
            for r in [*held, *uni]:
                s = (r["symbol"] or "").upper()
                if s and s not in seen:
                    seen.add(s)
                    targets.append(s)
        except Exception as exc:  # noqa: BLE001
            log.warning("news target resolution failed: {}", exc)
            targets = []

    targets = targets[:cap]
    if not targets:
        return {"return_code": 2, "skipped": True, "symbols": 0}
    n = fetch_for_symbols((s, symbol_query(s)) for s in targets)

    # Score the freshly-ingested (and any other unscored) headlines with FinBERT
    # / lexical fallback. Best-effort: never let sentiment scoring fail the run.
    scored = 0
    backend = "skipped"
    try:
        from src.data_ingestion.finbert_scorer import (
            active_backend,
            score_unscored_headlines,
        )
        scored = score_unscored_headlines()
        backend = active_backend()
    except Exception as exc:  # noqa: BLE001 -- sentiment is non-critical
        log.warning("news sentiment scoring skipped: {}", exc)

    return {"return_code": 0, "symbols": len(targets), "rows": n,
            "scored": scored, "sentiment_backend": backend}


def _step_features(symbols: list[str] | None,
                   broad: bool = False,
                   broad_n: int = 500,
                   incremental: bool = True) -> dict[str, Any]:
    """Build features. With ``broad`` (the rules-strategy default) and no
    explicit symbols, build for the top-N most-liquid equities so the factor
    strategy has a fresh, diversified universe to rank -- not just the ~47
    Nifty names in ``v_universe_today``.

    Incremental by default: symbols already up to date are skipped and only
    the new tail is recomputed, so a daily run is a cheap append rather than a
    full-history rebuild (which took ~20+ min for the broad universe)."""
    from scripts.build_features import main as features_main
    argv: list[str] = []
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    elif broad:
        argv += ["--top-liquid", str(broad_n)]
    if incremental:
        argv += ["--incremental"]
    rc = features_main(argv)
    return {"return_code": rc, "broad": broad and not symbols,
            "incremental": incremental}


def _step_predict(symbols: list[str] | None) -> dict[str, Any]:
    from scripts.predict_today import main as predict_main
    argv: list[str] = []
    if symbols:
        argv += ["--symbols", ",".join(symbols)]
    rc = predict_main(argv)
    return {"return_code": rc}


def _step_regime(as_of: str) -> dict[str, Any]:
    """Classify today's market regime (NIFTY trend + VIX + breadth) and persist
    it. Runs after features so breadth reads a fresh cross-section, and before
    signal generation so the router can pick a strategy off the latest row."""
    from src.regime.store import store_regime

    payload = store_regime(as_of)
    breadth = payload.get("breadth", {})
    return {
        "return_code": 0,
        "regime": payload["regime"],
        "prev_regime": payload.get("prev_regime"),
        "breadth_score": breadth.get("breadth_score"),
        "vix": payload.get("context", {}).get("vix"),
        "reasons": payload.get("reasons", []),
    }


def _step_forecast(as_of: str, symbols: list[str] | None) -> dict[str, Any]:
    """Project + persist multi-horizon price targets (1W..3Y) for the universe.

    Runs after features so each forecast reads the freshest close / volatility /
    momentum. When no explicit ``--symbols`` are given we forecast the mapped
    ``stock_sectors`` book (the tradeable universe), keeping the step bounded.
    """
    from src.analysis.forecast_store import store_universe
    from src.utils.db import fetch_all

    syms = symbols or [r["symbol"] for r in
                       fetch_all("SELECT symbol FROM stock_sectors ORDER BY symbol")]
    n = store_universe(syms, as_of)
    return {"return_code": 0, "forecasted": n, "universe": len(syms)}


def _step_generate(as_of: str | None,
                   threshold_override: float | None,
                   engine: str,
                   risk: RiskConfig | None) -> dict[str, Any]:
    """Queue BUY signals using the selected engine.

    ``rules`` (default): the transparent factor strategy (momentum + quality +
    value + trend) which actually trades a diversified book daily. The regime
    router decides which strategy config to use and whether new entries are
    allowed at all (defensive in BEAR_TREND / CRISIS).
    ``ml``: the legacy model-probability path (kept for comparison / research).
    """
    if engine == "rules":
        from src.regime.router import select_strategy
        from src.regime.store import latest_regime

        regime = latest_regime(as_of)
        plan = select_strategy(regime)
        if not plan.allow_new_entries:
            log.info("regime {} -> defensive; no new entries ({})",
                     regime, plan.note)
            return {
                "engine": "rules",
                "regime": regime,
                "strategy": plan.engine,
                "allow_new_entries": False,
                "note": plan.note,
                "kept": 0,
                "symbols": [],
            }
        plan_risk = strategy_risk_config(plan.config)
        out = generate_strategy_signals(
            signal_date=as_of, config=plan.config, risk=plan_risk,
            engine=plan.engine,
        )
        return {
            "engine": "rules",
            "regime": regime,
            "strategy": plan.engine,
            "allow_new_entries": True,
            "note": plan.note,
            "kept": len(out),
            "symbols": [s.symbol for s in out],
        }
    # threshold_override is the same --threshold flag the notify step uses;
    # forwarding it here means a single CLI flag now dictates *both* which
    # signals are generated AND which probabilities the report explains.
    cfg = SignalGenConfig(threshold_override=threshold_override)
    out = generate_signals(signal_date=as_of, config=cfg)
    return {
        "engine": "ml",
        "kept": len(out),
        "symbols": [s.symbol for s in out],
    }


def _step_reconcile(as_of: str | None,
                    risk: RiskConfig | None) -> dict[str, Any]:
    summary = reconcile(as_of=as_of, risk=risk)
    return summary.to_dict()


def _step_publish_cloud() -> dict[str, Any]:
    """Mirror the dashboard subset to Neon Postgres so the cloud dashboard
    reflects today's run. Self-skips when DATABASE_URL is not configured
    (i.e. when the cloud mirror isn't in use). Best-effort -- a failure here
    never affects the local run or the report."""
    import os
    # Safety: never publish from inside the test suite (pytest sets this),
    # so a test run can't truncate/overwrite the real cloud mirror.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return {"skipped": True, "reason": "pytest"}
    if not (os.getenv("DATABASE_URL") or "").strip():
        return {"skipped": True, "reason": "no DATABASE_URL"}
    from src.cloud.publish import publish
    summary = publish()
    return {"tables": len(summary), "rows": sum(summary.values())}


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
    p.add_argument("--signal-engine", choices=("rules", "ml"), default="rules",
                   help="Which engine queues BUY signals. 'rules' (default) = "
                        "transparent factor strategy that trades a diversified "
                        "book daily; 'ml' = legacy model-probability path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Render the report but do not send email/WhatsApp.")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip yfinance ingestion (useful when offline).")
    p.add_argument("--use-angelone", action="store_true",
                   help="(Deprecated/no-op) Angel One now runs by default; "
                        "kept for backward compatibility.")
    p.add_argument("--skip-angelone", action="store_true",
                   help="Skip Angel One SmartAPI EOD ingestion. By default it "
                        "runs first (preferred price source) and self-skips "
                        "if ANGEL_* env vars are missing; data-only / no orders.")
    p.add_argument("--skip-bhavcopy", action="store_true",
                   help="Skip whole-market BhavCopy EOD ingestion. By default it "
                        "runs so the broad factor-strategy universe stays fresh "
                        "daily (free, no auth, cloud-safe).")
    p.add_argument("--skip-fundamentals", action="store_true",
                   help="Skip the fundamentals snapshot refresh.")
    p.add_argument("--skip-news", action="store_true",
                   help="Skip the daily news/headlines refresh (Google News "
                        "RSS for held names + the daily universe).")
    p.add_argument("--full-features", action="store_true",
                   help="Force a full-history feature rebuild instead of the "
                        "default incremental (new-tail-only) build.")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip feature rebuild (use latest persisted features).")
    p.add_argument("--skip-predict", action="store_true",
                   help="Skip model inference (use latest predictions_log rows).")
    p.add_argument("--skip-regime", action="store_true",
                   help="Skip market-regime classification. By default (rules "
                        "engine) it runs after features and routes strategy "
                        "choice / defensive mode off the latest regime.")
    p.add_argument("--skip-forecast", action="store_true",
                   help="Skip multi-horizon price-target forecasts "
                        "(1W/1M/3M/6M/1Y/3Y) written to price_forecasts.")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip signal generation (no new entries).")
    p.add_argument("--skip-reconcile", action="store_true",
                   help="Skip paper-trade reconciliation.")
    p.add_argument("--skip-notify", action="store_true",
                   help="Skip the dispatch step.")
    p.add_argument("--skip-publish", action="store_true",
                   help="Skip publishing the dashboard subset to the Neon "
                        "cloud mirror. By default it runs last and self-skips "
                        "if DATABASE_URL is not set.")
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

    # The rules engine runs a larger, longer-held diversified book, so signal
    # generation AND paper reconciliation must share the same capacity / sector
    # / holding-period limits. The ML path keeps the conservative defaults.
    shared_risk = (
        strategy_risk_config(StrategyConfig())
        if args.signal_engine == "rules" else None
    )

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
    # Angel One first: preferred (freshest, official NSE) price source that the
    # dashboard reads ahead of yfinance. Self-skips if ANGEL_* creds are absent.
    if not args.skip_angelone:
        _run_step("ingest_angelone", lambda: _step_ingest_angelone(syms),
                  run_id=run_id, result=result)
    # yfinance second: gap-filler / fallback for symbols Angel One can't map.
    if not args.skip_ingest:
        _run_step("ingest_yfinance", lambda: _step_ingest(syms),
                  run_id=run_id, result=result)
    # BhavCopy: whole-market official EOD so the broad factor universe is fresh
    # (only needed for the rules engine; skip it for the light ML-only path).
    if not args.skip_bhavcopy and args.signal_engine == "rules":
        _run_step("ingest_bhavcopy", _step_ingest_bhavcopy,
                  run_id=run_id, result=result)
    if not args.skip_fundamentals:
        _run_step("ingest_fundamentals", lambda: _step_fundamentals(syms),
                  run_id=run_id, result=result)
    if not args.skip_features:
        _broad = args.signal_engine == "rules"
        _incremental = not args.full_features
        _run_step("build_features",
                  lambda: _step_features(syms, broad=_broad,
                                         incremental=_incremental),
                  run_id=run_id, result=result)
    if not args.skip_predict:
        _run_step("predict_today", lambda: _step_predict(syms),
                  run_id=run_id, result=result)
    # Regime classification (rules engine only): runs after features so breadth
    # reads a fresh cross-section, and before generate so the router can pick a
    # strategy / enter defensive mode off the latest regime.
    if not args.skip_regime and args.signal_engine == "rules":
        _run_step("classify_regime", lambda: _step_regime(as_of),
                  run_id=run_id, result=result)
    # Multi-horizon price-target forecasts (after features, independent of the
    # signal engine; fault-isolated so a hiccup never blocks signals).
    if not args.skip_forecast:
        _run_step("forecast_targets", lambda: _step_forecast(as_of, syms),
                  run_id=run_id, result=result)
    if not args.skip_generate:
        _run_step("generate_signals",
                  lambda: _step_generate(as_of, args.threshold,
                                         args.signal_engine, shared_risk),
                  run_id=run_id, result=result)
    if not args.skip_reconcile:
        _run_step("reconcile_paper",
                  lambda: _step_reconcile(as_of, shared_risk),
                  run_id=run_id, result=result)
    # News refresh (after reconcile so it covers today's freshly-held names).
    if not args.skip_news:
        _run_step("refresh_news", lambda: _step_news(syms),
                  run_id=run_id, result=result)

    # Step 7: notify (always; even if everything else broke we want the
    # report so the human knows what happened).
    if not args.skip_notify:
        _run_step(
            "send_daily_report",
            lambda: _step_notify(as_of, args.dry_run, args.threshold, result),
            run_id=run_id, result=result,
        )

    # Step 8: publish to the cloud mirror (last; never blocks the report).
    if not args.skip_publish:
        _run_step("publish_cloud", _step_publish_cloud,
                  run_id=run_id, result=result)

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
