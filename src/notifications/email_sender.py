"""SMTP sender for the daily report.

Why SMTP (not a SaaS API like SendGrid):
- Zero-cost path uses Gmail + an App Password.
- We control TLS, headers, and retries explicitly.
- No third-party SDK to audit / pin / update.

Security
--------
- ``SMTP.starttls()`` is mandatory; we never fall back to plaintext SMTP.
- All credentials are pulled via ``src.utils.secrets.get_secret`` which raises
  ``MissingSecretError`` if they're missing -- the dispatcher catches this and
  degrades to "email channel disabled" rather than crashing the whole job.
- Recipient addresses and the SMTP user are *not* logged verbatim; we log the
  count and a SHA-256 short-hash so we can correlate runs without leaking PII.
- HTML/text bodies are constructed with ``email.message.EmailMessage`` which
  does correct quoting / encoding (no manual header building).
"""

from __future__ import annotations

import hashlib
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Iterable, Sequence

from src.utils.logger import get_logger
from src.utils.secrets import MissingSecretError, get_secret

log = get_logger("notifications.email_sender")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_from: str
    email_to: tuple[str, ...]
    bcc_self: bool = True
    timeout_seconds: int = 30

    def validate(self) -> None:
        if not self.email_to:
            raise ValueError("EmailConfig.email_to must contain at least one address")
        if "@" not in self.email_from:
            raise ValueError("EmailConfig.email_from is not a valid address")
        for addr in self.email_to:
            if "@" not in addr:
                raise ValueError(f"Recipient is not a valid address: {addr!r}")
        if self.smtp_port not in (25, 465, 587, 2525):
            log.warning("Unusual SMTP port {}; expected 587/465", self.smtp_port)


def _short_hash(value: str) -> str:
    """Stable, non-reversible identifier for logs (no PII leakage)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _split_csv(env_value: str | None) -> tuple[str, ...]:
    if not env_value:
        return ()
    return tuple(addr.strip() for addr in env_value.split(",") if addr.strip())


def load_email_config_from_env() -> EmailConfig:
    """Read SMTP settings from .env. Raises MissingSecretError on gaps.

    SMTP_HOST / SMTP_PORT default to Gmail values to keep the .env minimal.
    Required: SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port_raw = os.getenv("SMTP_PORT", "587").strip()
    try:
        smtp_port = int(smtp_port_raw)
    except ValueError as exc:
        raise ValueError(f"SMTP_PORT must be an integer, got {smtp_port_raw!r}") from exc

    cfg = EmailConfig(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=get_secret("SMTP_USER", required=True) or "",
        smtp_password=get_secret("SMTP_PASSWORD", required=True) or "",
        email_from=get_secret("EMAIL_FROM", required=True) or "",
        email_to=_split_csv(get_secret("EMAIL_TO", required=True)),
        bcc_self=os.getenv("EMAIL_BCC_SELF", "1") not in {"0", "false", "False"},
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Building the MIME message
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attachment:
    filename: str
    payload: bytes
    mime_type: str = "application/octet-stream"


def build_message(
    *,
    cfg: EmailConfig,
    subject: str,
    html_body: str,
    text_body: str,
    attachments: Sequence[Attachment] = (),
) -> EmailMessage:
    """Compose a multipart/alternative email with optional attachments."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.email_from
    msg["To"] = ", ".join(cfg.email_to)
    if cfg.bcc_self:
        msg["Bcc"] = cfg.email_from
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="ai-trader.local")

    msg.set_content(text_body)            # plaintext part (default for clients with HTML off)
    msg.add_alternative(html_body, subtype="html")  # rich part for HTML-capable clients

    for att in attachments:
        maintype, _, subtype = att.mime_type.partition("/")
        msg.add_attachment(
            att.payload,
            maintype=maintype or "application",
            subtype=subtype or "octet-stream",
            filename=att.filename,
        )
    return msg


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


def send_email(
    *,
    cfg: EmailConfig | None = None,
    subject: str,
    html_body: str,
    text_body: str,
    attachments: Sequence[Attachment] = (),
) -> str:
    """Send the daily report. Returns the message-id used.

    Raises:
      MissingSecretError: when ``cfg`` is None and the required env vars
        are not set.
      smtplib.SMTPException: any underlying SMTP error.
    """
    if cfg is None:
        cfg = load_email_config_from_env()

    msg = build_message(
        cfg=cfg,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        attachments=attachments,
    )
    msg_id = msg["Message-ID"]

    log.info(
        "SMTP send via {}:{} | recipients={} | from_hash={} | subject={!r}",
        cfg.smtp_host, cfg.smtp_port,
        len(cfg.email_to),
        _short_hash(cfg.email_from),
        subject,
    )

    context = ssl.create_default_context()
    if cfg.smtp_port == 465:
        with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port,
                              context=context, timeout=cfg.timeout_seconds) as srv:
            srv.login(cfg.smtp_user, cfg.smtp_password)
            srv.send_message(msg)
    else:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_seconds) as srv:
            srv.ehlo()
            # MANDATORY: never fall back to plaintext.
            srv.starttls(context=context)
            srv.ehlo()
            srv.login(cfg.smtp_user, cfg.smtp_password)
            srv.send_message(msg)

    log.info("Email sent | message_id={}", msg_id)
    return msg_id


# ---------------------------------------------------------------------------
# Convenience: build an Attachment from a Path
# ---------------------------------------------------------------------------


def attachment_from_path(path: str | Path, *, mime_type: str | None = None) -> Attachment:
    p = Path(path)
    payload = p.read_bytes()
    mime = mime_type
    if mime is None:
        if p.suffix.lower() == ".pdf":
            mime = "application/pdf"
        elif p.suffix.lower() == ".csv":
            mime = "text/csv"
        elif p.suffix.lower() == ".html":
            mime = "text/html"
        else:
            mime = "application/octet-stream"
    return Attachment(filename=p.name, payload=payload, mime_type=mime)


def is_email_configured() -> bool:
    """Cheap probe used by the dispatcher to skip with a warning when unset."""
    try:
        load_email_config_from_env()
        return True
    except (MissingSecretError, ValueError) as exc:
        log.warning("Email channel not configured: {}", exc)
        return False
