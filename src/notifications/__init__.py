"""Notification & reporting layer.

Submodules
----------
report_builder
    Pulls today's predictions, latest model snapshot, last backtest summary,
    and recent paper trades from SQLite into a single ``DailyReport`` dataclass.
templates
    Jinja2 templates for HTML (rich email body), plaintext (email fallback /
    text MIME part) and a tight WhatsApp text variant (~700 chars).
pdf_writer
    One-page PDF summary using ReportLab. Used as an email attachment so the
    recipient gets a *document* they can save / forward.
email_sender
    SMTP (STARTTLS) sender. Supports HTML body, plaintext alternate, optional
    attachments. Reads creds from .env, never logs them.
whatsapp_sender
    Free CallMeBot provider (HTTPS GET) and optional Twilio provider. Phone
    numbers and API keys come from .env only.
dispatcher
    Top-level orchestrator: builds report, renders templates, writes a PDF,
    and dispatches over the enabled channels with graceful degradation
    (a missing channel never blocks the others). Supports ``--dry-run`` mode
    for previewing without sending.

Security & privacy notes
------------------------
- All credentials are pulled via ``src.utils.secrets.get_secret``.
- Logs never include creds, phone numbers, or recipient emails verbatim;
  values are redacted to a short hash so we keep observability without
  leaking PII.
- The HTTP layer enforces TLS (HTTPS-only for CallMeBot, STARTTLS for SMTP).
- Output to disk in dry-run mode is restricted to ``data/reports/``.
"""
