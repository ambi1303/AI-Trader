"""Tests for the Week-5 daily orchestrator (``scripts.run_daily``).

We focus on the *fault-isolation* contract -- the orchestrator MUST:
- never abort the pipeline when a non-critical step throws,
- record every failure into ``validation_failures``,
- still attempt to send the daily report so the human is informed.

We don't exercise the network-dependent ingest/feature/predict steps
here because those have their own dedicated tests; those steps are
turned off via the existing ``--skip-*`` CLI flags.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts import run_daily
from src.utils import db as db_mod

# Steps that hit the network (Angel One, BhavCopy, Google News, cloud publish)
# or do heavy local work. The orchestrator tests only exercise the
# fault-isolation + dispatch contract, so we keep every run fully hermetic --
# otherwise, on a machine with live broker creds, the real pipeline executes.
_OFFLINE_FLAGS = [
    "--skip-ingest", "--skip-angelone", "--skip-bhavcopy",
    "--skip-fundamentals", "--skip-features", "--skip-predict",
    "--skip-regime", "--skip-forecast", "--skip-news", "--skip-publish",
]


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "daily.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    # Ensure run_daily.apply_schema migrates the *test* DB and not the user's.
    schema = """
    CREATE TABLE schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    """
    db_mod.execute_script(schema)
    return db_file


def _redirect_dispatcher_root(monkeypatch, tmp_path):
    """Make sure the dispatcher writes artefacts under tmp, not the repo."""
    monkeypatch.setattr(
        "src.notifications.dispatcher.project_root", lambda: tmp_path
    )
    cfg_path = tmp_path / "notifications.yaml"
    cfg_path.write_text(
        "enabled_channels: {email: false, whatsapp: false}\n"
        "dry_run_output_dir: data/reports/notifications\n",
        encoding="utf-8",
    )
    return cfg_path


def test_orchestrator_skips_run_steps_and_still_dispatches(temp_db, tmp_path,
                                                          monkeypatch):
    cfg_path = _redirect_dispatcher_root(monkeypatch, tmp_path)
    # Force the dispatcher to use our test config.
    monkeypatch.setattr(
        "src.notifications.dispatcher.load_config",
        lambda *a, **kw: __import__(
            "src.notifications.dispatcher", fromlist=["load_config"]
        ).load_config(cfg_path),
    )
    rc = run_daily.main([
        "--dry-run", *_OFFLINE_FLAGS,
        "--skip-generate", "--skip-reconcile",
    ])
    assert rc == 0


def test_orchestrator_records_failures_in_validation_table(
    temp_db, tmp_path, monkeypatch
):
    cfg_path = _redirect_dispatcher_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.notifications.dispatcher.load_config",
        lambda *a, **kw: __import__(
            "src.notifications.dispatcher", fromlist=["load_config"]
        ).load_config(cfg_path),
    )
    # Force generate_signals to blow up so we can verify fault isolation.
    def _boom(*_a, **_kw):  # noqa: ANN001
        raise RuntimeError("synthetic failure for test")
    monkeypatch.setattr("scripts.run_daily.generate_signals", _boom)

    rc = run_daily.main([
        "--dry-run", "--signal-engine", "ml", *_OFFLINE_FLAGS,
        "--skip-reconcile",
    ])
    # Dry run never asks "did anything send"; we just check the script
    # didn't blow up and the failure was recorded.
    assert rc == 0
    rows = db_mod.fetch_all(
        "SELECT check_name, severity, message FROM validation_failures "
        "WHERE check_name = 'daily_step:generate_signals'"
    )
    assert any("synthetic failure" in r["message"] for r in rows)


def test_orchestrator_skip_notify_returns_zero(temp_db, tmp_path, monkeypatch):
    """When --skip-notify is set, exit code is 0 even with no dispatch."""
    rc = run_daily.main([
        *_OFFLINE_FLAGS,
        "--skip-generate", "--skip-reconcile", "--skip-notify",
    ])
    assert rc == 0


def test_list_recent_failures_returns_only_daily_step_rows(temp_db):
    # Apply the full schema so validation_failures exists.
    from src.db.migrate import apply_schema
    apply_schema()
    db_mod.execute(
        "INSERT INTO validation_failures (run_id, check_name, severity, message) "
        "VALUES ('x', 'daily_step:foo', 'error', 'boom')"
    )
    db_mod.execute(
        "INSERT INTO validation_failures (run_id, check_name, severity, message) "
        "VALUES ('x', 'cross_source', 'warning', 'unrelated')"
    )
    out = run_daily.list_recent_failures(limit=10)
    assert len(out) == 1
    assert out[0]["check_name"] == "daily_step:foo"


def test_summary_serialises_to_valid_json(temp_db, tmp_path, monkeypatch,
                                          capsys):
    cfg_path = _redirect_dispatcher_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.notifications.dispatcher.load_config",
        lambda *a, **kw: __import__(
            "src.notifications.dispatcher", fromlist=["load_config"]
        ).load_config(cfg_path),
    )
    rc = run_daily.main([
        "--dry-run", "--print-summary", *_OFFLINE_FLAGS,
        "--skip-generate", "--skip-reconcile",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["as_of"]
    assert any(s["name"] == "send_daily_report" for s in payload["steps"])
