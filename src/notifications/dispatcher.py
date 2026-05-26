"""Top-level orchestrator for the daily notification.

What it does (in order):

1. Build the structured report from SQLite.
2. Render HTML, plaintext, WhatsApp text, and a one-page PDF.
3. In ``--dry-run`` mode, write all artefacts to ``data/reports/notifications/``
   and stop -- no creds required.
4. Otherwise, send via the channels enabled in ``config/notifications.yaml``,
   degrading gracefully (a misconfigured channel never blocks the others).

The dispatcher is the *only* place that decides:
- which channels are on,
- the email subject,
- the WhatsApp body length,
- what gets attached to the email.

Higher-level callers (CLI, scheduler) just invoke ``send_daily()``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.notifications import (
    email_sender,
    pdf_writer,
    report_builder,
    templates,
    whatsapp_sender,
)
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("notifications.dispatcher")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass
class ContentToggles:
    show_top_predictions: bool = True
    top_predictions_n: int = 10
    show_signals: bool = True
    show_latest_model: bool = True
    show_latest_backtest: bool = True
    show_validation_summary: bool = True
    show_recent_trades: bool = True
    recent_trades_n: int = 10


@dataclass
class NotificationConfig:
    enabled_email: bool = True
    enabled_whatsapp: bool = True
    whatsapp_provider: str = "callmebot"
    subject_prefix: str = "[AI Trader]"
    attach_pdf: bool = True
    attach_csv: bool = False
    bcc_self: bool = True
    whatsapp_max_chars: int = 700
    content: ContentToggles = field(default_factory=ContentToggles)
    dry_run_output_dir: str = "data/reports/notifications"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("notifications.yaml not found at {} -- using defaults",
                    path.as_posix())
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping at the top level"
        )
    return data


def load_config(path: str | Path | None = None) -> NotificationConfig:
    cfg_path = Path(path) if path else (project_root() / "config" / "notifications.yaml")
    raw = _load_yaml(cfg_path)
    enabled = raw.get("enabled_channels", {}) or {}
    email_cfg = raw.get("email", {}) or {}
    wa_cfg = raw.get("whatsapp", {}) or {}
    content = raw.get("content", {}) or {}
    return NotificationConfig(
        enabled_email=bool(enabled.get("email", True)),
        enabled_whatsapp=bool(enabled.get("whatsapp", True)),
        whatsapp_provider=str(raw.get("whatsapp_provider", "callmebot")),
        subject_prefix=str(email_cfg.get("subject_prefix", "[AI Trader]")),
        attach_pdf=bool(email_cfg.get("attach_pdf", True)),
        attach_csv=bool(email_cfg.get("attach_csv", False)),
        bcc_self=bool(email_cfg.get("bcc_self", True)),
        whatsapp_max_chars=int(wa_cfg.get("max_chars", 700)),
        content=ContentToggles(
            show_top_predictions=bool(content.get("show_top_predictions", True)),
            top_predictions_n=int(content.get("top_predictions_n", 10)),
            show_signals=bool(content.get("show_signals", True)),
            show_latest_model=bool(content.get("show_latest_model", True)),
            show_latest_backtest=bool(content.get("show_latest_backtest", True)),
            show_validation_summary=bool(content.get("show_validation_summary", True)),
            show_recent_trades=bool(content.get("show_recent_trades", True)),
            recent_trades_n=int(content.get("recent_trades_n", 10)),
        ),
        dry_run_output_dir=str(raw.get("dry_run_output_dir", "data/reports/notifications")),
    )


# ---------------------------------------------------------------------------
# Outcome + main entrypoint
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    """Auditable record of what happened. Safe to log/serialise."""
    report_date: str
    dry_run: bool
    artefacts: dict[str, str] = field(default_factory=dict)  # name -> path
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_subject(prefix: str, report: report_builder.DailyReport) -> str:
    n_sig = len(report.signals)
    return f"{prefix} {report.report_date} - {n_sig} signal(s)"


def _write_artefact(out_dir: Path, name: str, content: str | bytes) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


def send_daily(
    *,
    report_date: str | None = None,
    threshold_override: float | None = None,
    dry_run: bool = False,
    config_path: str | Path | None = None,
) -> DispatchResult:
    """Build, render, and dispatch the daily report.

    Parameters
    ----------
    report_date : ISO YYYY-MM-DD, defaults to today.
    threshold_override : if provided, used to decide which predictions are
        considered signals (else we use the model's stored threshold).
    dry_run : when True, skips network calls and writes all artefacts to
        ``config.dry_run_output_dir`` for inspection.
    config_path : override path to ``notifications.yaml`` (mainly for tests).
    """
    cfg = load_config(config_path)
    report = report_builder.build_daily_report(
        report_date=report_date,
        top_n=cfg.content.top_predictions_n,
        recent_trades_n=cfg.content.recent_trades_n,
        threshold_override=threshold_override,
    )
    subject = _resolve_subject(cfg.subject_prefix, report)
    html_body = templates.render_html(report, subject=subject)
    text_body = templates.render_text(report)
    wa_body = templates.render_whatsapp(report, max_chars=cfg.whatsapp_max_chars)

    result = DispatchResult(report_date=report.report_date, dry_run=dry_run)

    # Always render the PDF; it's a few KB and the dispatcher returns the
    # path so the caller can repurpose it (Slack, Telegram, manual).
    out_dir_root = project_root() / cfg.dry_run_output_dir
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir_root / report.report_date
    pdf_path = out_dir / f"daily_{stamp}.pdf"
    pdf_writer.write_pdf(report, pdf_path)
    result.artefacts["pdf"] = pdf_path.as_posix()

    # Always persist the HTML / text alongside in dry-run; in non-dry-run
    # we still persist them so we have an audit trail of *exactly* what
    # was sent on a given day (regulatory hygiene).
    html_path = _write_artefact(out_dir, f"daily_{stamp}.html", html_body)
    text_path = _write_artefact(out_dir, f"daily_{stamp}.txt", text_body)
    wa_path = _write_artefact(out_dir, f"daily_{stamp}_whatsapp.txt", wa_body)
    result.artefacts["html"] = html_path.as_posix()
    result.artefacts["text"] = text_path.as_posix()
    result.artefacts["whatsapp_preview"] = wa_path.as_posix()

    if dry_run:
        log.info("Dry run complete: artefacts under {}", out_dir.as_posix())
        return result

    # ------------------------------------------------------------------
    # Email channel
    # ------------------------------------------------------------------
    if cfg.enabled_email:
        if email_sender.is_email_configured():
            try:
                attachments: list[email_sender.Attachment] = []
                if cfg.attach_pdf:
                    attachments.append(
                        email_sender.attachment_from_path(pdf_path, mime_type="application/pdf")
                    )
                msg_id = email_sender.send_email(
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    attachments=attachments,
                )
                result.channels["email"] = {"status": "sent", "message_id": msg_id}
            except Exception as exc:  # noqa: BLE001  -- we want to capture any send failure
                log.error("Email send failed: {}", exc)
                result.errors["email"] = str(exc)
                result.channels["email"] = {"status": "failed"}
        else:
            result.channels["email"] = {"status": "skipped_unconfigured"}
    else:
        result.channels["email"] = {"status": "disabled"}

    # ------------------------------------------------------------------
    # WhatsApp channel
    # ------------------------------------------------------------------
    if cfg.enabled_whatsapp:
        if whatsapp_sender.is_whatsapp_configured(cfg.whatsapp_provider):
            try:
                wa_resp = whatsapp_sender.send_whatsapp(
                    body=wa_body, provider=cfg.whatsapp_provider
                )
                result.channels["whatsapp"] = {"status": "sent", **wa_resp}
            except Exception as exc:  # noqa: BLE001
                log.error("WhatsApp send failed: {}", exc)
                result.errors["whatsapp"] = str(exc)
                result.channels["whatsapp"] = {"status": "failed"}
        else:
            result.channels["whatsapp"] = {"status": "skipped_unconfigured"}
    else:
        result.channels["whatsapp"] = {"status": "disabled"}

    log.info(
        "Daily notification dispatch complete | channels={} | errors={}",
        {k: v.get("status") for k, v in result.channels.items()},
        list(result.errors.keys()),
    )
    return result


# ---------------------------------------------------------------------------
# Helper for scripts: did anything succeed?
# ---------------------------------------------------------------------------


def any_channel_sent(result: DispatchResult) -> bool:
    return any(v.get("status") == "sent" for v in result.channels.values())
