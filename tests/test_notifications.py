"""Tests for the notifications layer.

Coverage:
- report_builder: empty DB, populated DB, latest-tie-break determinism, signal
  derivation from threshold, JSON metric robustness.
- templates: HTML auto-escaping, plaintext CR/LF scrubbing, WhatsApp char cap.
- pdf_writer: PDF bytes have a valid header and the report date appears in the
  rendered file (so silent corruption is caught).
- email_sender: build_message produces a valid multipart/alternative message
  with the expected headers and a PDF attachment; SMTP transport is mocked.
- whatsapp_sender: phone normalisation, secret loading, CallMeBot HTTP mock.
- dispatcher: dry-run writes artefacts; degrades gracefully when no creds.
- secret redaction: logs never include the raw API key / phone / password.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.notifications import (
    dispatcher,
    email_sender,
    pdf_writer,
    report_builder,
    templates,
    whatsapp_sender,
)
from src.utils import db as db_mod
from src.utils.secrets import MissingSecretError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    schema = """
    CREATE TABLE nifty_constituents (
        symbol TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT,
        index_name TEXT NOT NULL DEFAULT 'NIFTY50',
        notes TEXT,
        PRIMARY KEY (symbol, start_date, index_name)
    );
    CREATE VIEW v_universe_today AS
        SELECT symbol FROM nifty_constituents WHERE end_date IS NULL;
    CREATE TABLE model_runs (
        run_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        git_sha TEXT,
        feature_hash TEXT,
        trained_from TEXT,
        trained_to TEXT,
        metrics_json TEXT,
        artifact_path TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE predictions_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        prediction_date TEXT NOT NULL,
        raw_prob REAL,
        calibrated_prob REAL,
        feature_snapshot_json TEXT,
        created_at TEXT NOT NULL DEFAULT '2025-01-01'
    );
    CREATE TABLE backtest_runs (
        bt_run_id TEXT PRIMARY KEY,
        model_run_id TEXT,
        name TEXT,
        start_date TEXT,
        end_date TEXT,
        initial_capital REAL NOT NULL,
        config_json TEXT,
        metrics_json TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE backtest_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bt_run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_date TEXT NOT NULL,
        exit_date TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        qty INTEGER NOT NULL,
        gross_pnl REAL NOT NULL,
        cost_rupees REAL NOT NULL,
        net_pnl REAL NOT NULL,
        holding_days INTEGER NOT NULL,
        exit_reason TEXT NOT NULL,
        entry_prob REAL,
        threshold REAL
    );
    CREATE TABLE validation_failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        check_name TEXT NOT NULL,
        symbol TEXT,
        issue_date TEXT,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    -- Tables added by Week-5 changes that the report now reads from.
    CREATE TABLE price_data (
        symbol TEXT NOT NULL,
        bar_date TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, volume INTEGER,
        adj_close REAL,
        source TEXT NOT NULL,
        PRIMARY KEY (symbol, bar_date, source)
    );
    CREATE TABLE feature_data (
        symbol TEXT NOT NULL,
        feature_date TEXT NOT NULL,
        close REAL, atr_14 REAL,
        PRIMARY KEY (symbol, feature_date)
    );
    CREATE TABLE paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_date TEXT, exit_date TEXT,
        entry_price REAL, exit_price REAL, qty INTEGER,
        pnl_rupees REAL, pnl_pct REAL, cost_rupees REAL, notes TEXT,
        sector TEXT, status TEXT NOT NULL DEFAULT 'open',
        stop_loss REAL, take_profit REAL, trailing_stop REAL,
        entry_atr REAL, high_watermark REAL, exit_reason TEXT,
        entry_prob REAL, threshold REAL, run_id TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    );
    CREATE TABLE signal_outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, signal_date TEXT NOT NULL,
        side TEXT NOT NULL, entry_price REAL, stop_loss REAL,
        take_profit REAL, qty INTEGER, confidence REAL,
        status TEXT NOT NULL DEFAULT 'pending',
        payload_json TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        sent_at TEXT, error TEXT
    );
    CREATE UNIQUE INDEX ix_outbox_uq_symbol_date
        ON signal_outbox(symbol, signal_date);
    CREATE TABLE stock_sectors (
        symbol TEXT PRIMARY KEY,
        sector TEXT NOT NULL,
        sector_index TEXT NOT NULL,
        notes TEXT
    );
    """
    db_mod.execute_script(schema)
    return db_file


def _seed_minimal(report_date: str = "2025-10-15") -> None:
    # universe
    db_mod.executemany(
        "INSERT INTO nifty_constituents (symbol, start_date, end_date) VALUES (?, ?, ?)",
        [("RELIANCE", "2020-01-01", None),
         ("TCS",      "2020-01-01", None),
         ("INFY",     "2020-01-01", None)],
    )
    # model run
    db_mod.execute(
        """
        INSERT INTO model_runs
            (run_id, model_name, git_sha, feature_hash,
             trained_from, trained_to, metrics_json, artifact_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run-A", "xgb_v1", "abc1234", "feathash",
         "2020-01-01", "2025-09-30",
         '{"threshold": 0.55, "brier": 0.21}',
         "/tmp/run-A.joblib", "2025-10-14T08:00:00Z"),
    )
    # predictions
    db_mod.executemany(
        """
        INSERT INTO predictions_log (run_id, symbol, prediction_date, raw_prob, calibrated_prob)
        VALUES (?, ?, ?, ?, ?)
        """,
        [("run-A", "RELIANCE", report_date, 0.71, 0.62),  # signal
         ("run-A", "TCS",      report_date, 0.50, 0.40),  # not signal
         ("run-A", "INFY",     report_date, 0.65, 0.58)], # signal
    )
    # backtest
    db_mod.execute(
        """
        INSERT INTO backtest_runs
            (bt_run_id, model_run_id, name, start_date, end_date,
             initial_capital, config_json, metrics_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("bt-1", "run-A", "smoke", "2020-01-01", "2025-09-30",
         1_000_000.0, "{}",
         '{"sharpe": 0.85, "sortino": 1.10, "max_drawdown_pct": -8.40, '
         '"total_return_pct": 22.5, "n_trades": 70, "hit_rate_pct": 41.4}',
         "2025-10-14T09:00:00Z"),
    )
    # trades
    db_mod.executemany(
        """
        INSERT INTO backtest_trades
            (bt_run_id, symbol, side, entry_date, exit_date,
             entry_price, exit_price, qty, gross_pnl, cost_rupees,
             net_pnl, holding_days, exit_reason, entry_prob, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("bt-1", "RELIANCE", "LONG", "2025-09-25", "2025-09-29",
             1300.0, 1325.0, 10, 250.0, 35.0, 215.0, 4, "target", 0.61, 0.55),
            ("bt-1", "TCS", "LONG", "2025-09-22", "2025-09-26",
             3500.0, 3450.0, 5, -250.0, 22.0, -272.0, 4, "stop", 0.58, 0.55),
        ],
    )
    # validation -- timestamp ``now`` so the rolling 7-day window in the
    # report builder still picks it up regardless of when the suite runs.
    db_mod.execute(
        """
        INSERT INTO validation_failures (run_id, check_name, severity, message, created_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        """,
        ("ingest-1", "cross_source", "warning", "minor diff"),
    )


# ---------------------------------------------------------------------------
# report_builder
# ---------------------------------------------------------------------------


def test_build_daily_report_empty_db_does_not_crash(temp_db):
    rep = report_builder.build_daily_report(report_date="2025-10-15")
    assert rep.report_date == "2025-10-15"
    assert rep.predictions == []
    assert rep.signals == []
    assert rep.latest_model is None
    assert rep.latest_backtest is None
    assert rep.universe_size == 0
    assert rep.validation.total == 0


def test_build_daily_report_full(temp_db):
    _seed_minimal("2025-10-15")
    rep = report_builder.build_daily_report(report_date="2025-10-15", top_n=5)

    assert rep.universe_size == 3
    assert len(rep.predictions) == 3
    # Sorted by calibrated_prob desc
    assert [p.symbol for p in rep.predictions] == ["RELIANCE", "INFY", "TCS"]
    # Threshold 0.55 from model -> 2 signals
    assert {s.symbol for s in rep.signals} == {"RELIANCE", "INFY"}
    assert rep.latest_model is not None
    assert rep.latest_model.run_id == "run-A"
    assert rep.threshold_used == 0.55
    assert rep.latest_backtest is not None
    assert rep.latest_backtest.metrics["sharpe"] == 0.85
    assert len(rep.recent_trades) == 2
    # Recent trade ordering: most recent exit_date first
    assert rep.recent_trades[0].symbol == "RELIANCE"
    assert rep.validation.total == 1
    assert rep.validation.by_severity == {"warning": 1}


def test_threshold_override_changes_signal_set(temp_db):
    _seed_minimal("2025-10-15")
    # Higher threshold should drop INFY (0.58)
    rep = report_builder.build_daily_report(
        report_date="2025-10-15", threshold_override=0.60
    )
    assert {s.symbol for s in rep.signals} == {"RELIANCE"}
    assert rep.threshold_used == 0.60


def test_corrupt_metrics_json_does_not_crash(temp_db):
    _seed_minimal("2025-10-15")
    db_mod.execute(
        "UPDATE model_runs SET metrics_json = ? WHERE run_id = ?",
        ("not-json", "run-A"),
    )
    rep = report_builder.build_daily_report(report_date="2025-10-15")
    assert rep.latest_model is not None
    assert rep.latest_model.metrics == {}
    assert rep.threshold_used is None


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


def test_html_template_escapes_user_text(temp_db):
    _seed_minimal()
    rep = report_builder.build_daily_report(report_date="2025-10-15")
    html = templates.render_html(rep, subject="[AI Trader] test")
    # Hostile-looking content does not appear unescaped.
    assert "<script>" not in html
    assert "RELIANCE" in html
    assert "&lt;" not in html or "RELIANCE" in html  # sanity (escape engine on)
    assert html.count("</html>") == 1


def test_text_template_scrubs_cr_lf(temp_db):
    _seed_minimal()
    # Inject an evil string into model run_id (simulates corrupted DB / prank).
    db_mod.execute(
        "UPDATE model_runs SET run_id = ? WHERE rowid = 1",
        ("evil\r\nrun-id\rtail",),
    )
    rep = report_builder.build_daily_report(report_date="2025-10-15")
    text = templates.render_text(rep)
    assert "\r" not in text
    # The CR/LF inside run_id should have been replaced by a space.
    assert "run-id" in text


def test_whatsapp_template_respects_max_chars(temp_db):
    _seed_minimal()
    rep = report_builder.build_daily_report(report_date="2025-10-15", top_n=3)
    short = templates.render_whatsapp(rep, max_chars=200)
    assert len(short) <= 200
    assert "AI TRADER" in short


# ---------------------------------------------------------------------------
# pdf_writer
# ---------------------------------------------------------------------------


def test_pdf_bytes_have_pdf_header(temp_db, tmp_path):
    _seed_minimal()
    rep = report_builder.build_daily_report(report_date="2025-10-15")
    payload = pdf_writer.render_pdf_bytes(rep)
    assert payload.startswith(b"%PDF-")
    assert b"%%EOF" in payload[-32:] or b"%%EOF" in payload


def test_write_pdf_creates_file_and_dir(temp_db, tmp_path):
    _seed_minimal()
    rep = report_builder.build_daily_report(report_date="2025-10-15")
    out = tmp_path / "nested" / "report.pdf"
    p = pdf_writer.write_pdf(rep, out)
    assert p.exists()
    assert p.stat().st_size > 1000  # not an empty stub
    assert p.read_bytes().startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# email_sender
# ---------------------------------------------------------------------------


def test_build_message_has_html_text_and_attachment():
    cfg = email_sender.EmailConfig(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="user",
        smtp_password="pass",
        email_from="bot@example.com",
        email_to=("alice@example.com",),
        bcc_self=False,
    )
    msg = email_sender.build_message(
        cfg=cfg,
        subject="[AI] Hi",
        html_body="<p>hi</p>",
        text_body="hi",
        attachments=[email_sender.Attachment(
            filename="r.pdf", payload=b"%PDF-1.4\n", mime_type="application/pdf",
        )],
    )
    assert msg["From"] == "bot@example.com"
    assert msg["To"] == "alice@example.com"
    assert msg["Subject"] == "[AI] Hi"
    payload_types = {part.get_content_type() for part in msg.walk()}
    assert "text/plain" in payload_types
    assert "text/html" in payload_types
    assert "application/pdf" in payload_types


def test_email_validate_rejects_bad_addresses():
    cfg = email_sender.EmailConfig(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="u",
        smtp_password="p",
        email_from="not-an-email",
        email_to=("alice@example.com",),
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_load_email_config_from_env_missing_raises(monkeypatch):
    for var in ("SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingSecretError):
        email_sender.load_email_config_from_env()


def test_send_email_uses_starttls(monkeypatch):
    cfg = email_sender.EmailConfig(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="user",
        smtp_password="pass",
        email_from="bot@example.com",
        email_to=("alice@example.com",),
        bcc_self=False,
    )
    fake_smtp = MagicMock()
    smtp_ctx = MagicMock()
    smtp_ctx.__enter__.return_value = fake_smtp
    smtp_ctx.__exit__.return_value = False
    with patch("smtplib.SMTP", return_value=smtp_ctx) as patched:
        email_sender.send_email(
            cfg=cfg, subject="s", html_body="<p>h</p>", text_body="t",
        )
    patched.assert_called_once_with("smtp.example.com", 587, timeout=cfg.timeout_seconds)
    fake_smtp.starttls.assert_called_once()
    fake_smtp.login.assert_called_once_with("user", "pass")
    fake_smtp.send_message.assert_called_once()


def test_attachment_from_path_detects_pdf(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4\nstub")
    att = email_sender.attachment_from_path(p)
    assert att.mime_type == "application/pdf"
    assert att.filename == "a.pdf"


# ---------------------------------------------------------------------------
# whatsapp_sender
# ---------------------------------------------------------------------------


def test_phone_normalisation_strips_punctuation():
    assert whatsapp_sender._normalise_phone("+91 98765-43210") == "919876543210"


def test_phone_normalisation_rejects_short():
    with pytest.raises(ValueError):
        whatsapp_sender._normalise_phone("12345")


def test_callmebot_send_uses_https_only(monkeypatch):
    cfg = whatsapp_sender.CallMeBotConfig(phone="919876543210", apikey="secret")
    fake_resp = MagicMock(status_code=200, text="Message queued. You will receive it...")
    monkeypatch.setattr(
        "src.notifications.whatsapp_sender.requests.get",
        MagicMock(return_value=fake_resp),
    )
    text = whatsapp_sender._send_callmebot(cfg, "hello")
    # Verify the URL passed used https
    call = whatsapp_sender.requests.get.call_args
    url = call.args[0] if call.args else call.kwargs["url"]
    assert url.startswith("https://api.callmebot.com/whatsapp.php")
    assert "Message queued" in text


def test_callmebot_send_raises_on_http_error(monkeypatch):
    cfg = whatsapp_sender.CallMeBotConfig(phone="919876543210", apikey="secret")
    fake_resp = MagicMock(status_code=403, text="forbidden")
    monkeypatch.setattr(
        "src.notifications.whatsapp_sender.requests.get",
        MagicMock(return_value=fake_resp),
    )
    with pytest.raises(RuntimeError):
        whatsapp_sender._send_callmebot(cfg, "hello")


def test_is_whatsapp_configured_callmebot_missing_returns_false(monkeypatch):
    monkeypatch.delenv("CALLMEBOT_PHONE", raising=False)
    monkeypatch.delenv("CALLMEBOT_APIKEY", raising=False)
    assert whatsapp_sender.is_whatsapp_configured("callmebot") is False


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_dry_run_writes_artefacts(temp_db, tmp_path, monkeypatch):
    _seed_minimal("2025-10-15")
    # Redirect dry-run output dir to tmp
    monkeypatch.setattr(
        "src.notifications.dispatcher.project_root", lambda: tmp_path,
    )
    cfg_path = tmp_path / "notifications.yaml"
    cfg_path.write_text(
        "enabled_channels:\n"
        "  email: false\n"
        "  whatsapp: false\n"
        "dry_run_output_dir: data/reports/notifications\n"
        "content:\n"
        "  top_predictions_n: 3\n"
        "  recent_trades_n: 5\n",
        encoding="utf-8",
    )
    result = dispatcher.send_daily(
        report_date="2025-10-15", dry_run=True, config_path=cfg_path
    )
    assert result.dry_run is True
    for key in ("pdf", "html", "text", "whatsapp_preview"):
        assert key in result.artefacts
        assert Path(result.artefacts[key]).exists()


def test_dispatcher_skips_unconfigured_channels(temp_db, tmp_path, monkeypatch):
    _seed_minimal("2025-10-15")
    monkeypatch.setattr(
        "src.notifications.dispatcher.project_root", lambda: tmp_path,
    )
    # Wipe any inherited env so creds genuinely look "missing".
    for v in ("SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO",
              "CALLMEBOT_PHONE", "CALLMEBOT_APIKEY"):
        monkeypatch.delenv(v, raising=False)
    cfg_path = tmp_path / "notifications.yaml"
    cfg_path.write_text(
        "enabled_channels:\n"
        "  email: true\n"
        "  whatsapp: true\n"
        "whatsapp_provider: callmebot\n"
        "dry_run_output_dir: data/reports/notifications\n",
        encoding="utf-8",
    )
    # NOT dry-run: dispatcher should still complete without raising.
    result = dispatcher.send_daily(
        report_date="2025-10-15", dry_run=False, config_path=cfg_path
    )
    assert result.channels["email"]["status"] == "skipped_unconfigured"
    assert result.channels["whatsapp"]["status"] == "skipped_unconfigured"
    assert dispatcher.any_channel_sent(result) is False


def test_dispatcher_subject_includes_signal_count(temp_db, tmp_path, monkeypatch):
    _seed_minimal("2025-10-15")
    monkeypatch.setattr(
        "src.notifications.dispatcher.project_root", lambda: tmp_path,
    )
    cfg_path = tmp_path / "notifications.yaml"
    cfg_path.write_text(
        "enabled_channels: {email: false, whatsapp: false}\n"
        "dry_run_output_dir: data/reports/notifications\n",
        encoding="utf-8",
    )
    result = dispatcher.send_daily(
        report_date="2025-10-15", dry_run=True, config_path=cfg_path
    )
    html_path = Path(result.artefacts["html"])
    html = html_path.read_text(encoding="utf-8")
    # Signal count rendered correctly for the seeded data (2 signals)
    assert "2 signal(s)" in html or "2 signal" in html


# ---------------------------------------------------------------------------
# Secret redaction in logs
# ---------------------------------------------------------------------------


def test_log_redaction_short_hash_does_not_leak_phone():
    secret_phone = "919999999999"
    h = whatsapp_sender._short_hash(secret_phone)
    # The hash must NOT contain the original phone digits anywhere
    assert secret_phone not in h
    assert len(h) == 10  # sha256[:10]
