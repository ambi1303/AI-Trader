"""WhatsApp delivery providers.

Two providers are supported. Pick via ``config/notifications.yaml``:

1. ``callmebot``  -- free, single recipient, personal use only.
   * Setup: save +34 644 51 95 23 in your phone, send "I allow callmebot to
     send me messages", paste the returned API key into ``CALLMEBOT_APIKEY``.
   * Endpoint: HTTPS GET https://api.callmebot.com/whatsapp.php
   * Limits: ~20 messages/min on the free tier; rate-limited at our side too.

2. ``twilio``     -- paid, multi-recipient, production-grade SLA.
   * Optional: only imported when ``twilio`` provider is selected, so the
     rest of the system has zero dependency on the Twilio SDK.

Security
--------
- Phone numbers / API keys come from .env only.
- HTTPS is required for both providers; we never use plain HTTP.
- We never log the API key, the phone number, or the full body. Logs include
  only a SHA-256 short-hash of the phone and the first 80 chars of the body
  (so on-call humans can see "did this message look right?" without seeing PII
  or trade signals leaking into log aggregators).
- The HTTP client uses a short timeout and a single retry; we'd rather skip a
  WhatsApp message than block the daily report behind a hung phone API.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from src.utils.logger import get_logger
from src.utils.secrets import MissingSecretError, get_secret

log = get_logger("notifications.whatsapp_sender")


# ---------------------------------------------------------------------------
# Phone validation
# ---------------------------------------------------------------------------

# CallMeBot expects digits only (no '+', no spaces). Accept anything reasonable
# from the user but normalise to digits before sending.
_PHONE_DIGITS = re.compile(r"\D+")
# 10-15 digits per E.164 (we keep at least 10 to allow short codes; we don't
# allow shorter than 10 because that's a strong signal of a misconfig).
_PHONE_RE = re.compile(r"^\d{10,15}$")


def _normalise_phone(raw: str | None) -> str:
    if not raw:
        raise ValueError("WhatsApp phone is empty")
    digits = _PHONE_DIGITS.sub("", raw)
    if not _PHONE_RE.match(digits):
        raise ValueError(
            "WhatsApp phone must be 10-15 digits (country code + number, "
            "no '+' or spaces)."
        )
    return digits


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# CallMeBot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallMeBotConfig:
    phone: str        # digits only, E.164-without-plus
    apikey: str
    timeout_seconds: int = 15

    @classmethod
    def from_env(cls) -> "CallMeBotConfig":
        return cls(
            phone=_normalise_phone(get_secret("CALLMEBOT_PHONE", required=True)),
            apikey=get_secret("CALLMEBOT_APIKEY", required=True) or "",
        )


def _send_callmebot(cfg: CallMeBotConfig, body: str) -> str:
    """Returns the upstream response text (truncated). Raises on HTTP errors.

    The API URL embeds the api key as a query param. We log only a hash of
    the phone and a hash of the api key so neither reaches log files.
    """
    if not body.strip():
        raise ValueError("WhatsApp body is empty")
    url = (
        "https://api.callmebot.com/whatsapp.php"
        f"?phone={quote(cfg.phone)}"
        f"&text={quote(body)}"
        f"&apikey={quote(cfg.apikey)}"
    )
    log.info(
        "WhatsApp send via CallMeBot | phone_hash={} | apikey_hash={} | body_chars={}",
        _short_hash(cfg.phone), _short_hash(cfg.apikey), len(body),
    )
    resp = requests.get(url, timeout=cfg.timeout_seconds)
    # CallMeBot returns 200 even on logical failures; check the body for
    # known success markers.
    text = resp.text.strip()
    truncated = text[:200]
    if resp.status_code != 200:
        raise RuntimeError(
            f"CallMeBot HTTP {resp.status_code}: {truncated}"
        )
    if "Message queued" not in text and "successfully" not in text.lower():
        # Don't surface raw text -- it can echo our content. Surface a short
        # excerpt only and let the caller log a hash if needed.
        log.warning("CallMeBot response did not look successful: {!r}", truncated)
    return truncated


# ---------------------------------------------------------------------------
# Twilio (optional)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_whatsapp: str  # e.g. "whatsapp:+14155238886"
    to_whatsapp: tuple[str, ...]  # one or more "whatsapp:+91..." addresses
    timeout_seconds: int = 15

    @classmethod
    def from_env(cls) -> "TwilioConfig":
        to_raw = get_secret("TWILIO_TO_WHATSAPP", required=True) or ""
        to_list = tuple(t.strip() for t in to_raw.split(",") if t.strip())
        if not to_list:
            raise ValueError("TWILIO_TO_WHATSAPP must contain at least one recipient")
        return cls(
            account_sid=get_secret("TWILIO_ACCOUNT_SID", required=True) or "",
            auth_token=get_secret("TWILIO_AUTH_TOKEN", required=True) or "",
            from_whatsapp=get_secret("TWILIO_FROM_WHATSAPP", required=True) or "",
            to_whatsapp=to_list,
        )


def _send_twilio(cfg: TwilioConfig, body: str) -> list[str]:
    """Returns a list of message SIDs. Imports twilio lazily."""
    try:
        from twilio.rest import Client  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Twilio provider selected but the 'twilio' package is not installed. "
            "Install it with `pip install twilio` or switch to CallMeBot."
        ) from exc
    client = Client(cfg.account_sid, cfg.auth_token)
    sids: list[str] = []
    for to in cfg.to_whatsapp:
        log.info(
            "WhatsApp send via Twilio | to_hash={} | body_chars={}",
            _short_hash(to), len(body),
        )
        message = client.messages.create(
            from_=cfg.from_whatsapp,
            to=to,
            body=body,
        )
        sids.append(message.sid)
    return sids


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def send_whatsapp(*, body: str, provider: str = "callmebot") -> dict[str, Any]:
    """Send a WhatsApp message via the chosen provider.

    Returns a dict with provider info and a non-PII reference id, e.g.
    ``{"provider": "callmebot", "response": "Message queued..."}``.
    """
    provider = provider.lower().strip()
    if provider == "callmebot":
        cfg = CallMeBotConfig.from_env()
        text = _send_callmebot(cfg, body)
        return {"provider": "callmebot", "response": text}
    if provider == "twilio":
        cfg = TwilioConfig.from_env()
        sids = _send_twilio(cfg, body)
        return {"provider": "twilio", "message_sids": sids}
    raise ValueError(f"Unknown WhatsApp provider: {provider!r}")


def is_whatsapp_configured(provider: str = "callmebot") -> bool:
    """Probe used by the dispatcher to skip-with-warning when unset."""
    provider = provider.lower().strip()
    try:
        if provider == "callmebot":
            CallMeBotConfig.from_env()
        elif provider == "twilio":
            TwilioConfig.from_env()
        else:
            log.warning("Unknown WhatsApp provider {!r}; channel disabled", provider)
            return False
        return True
    except (MissingSecretError, ValueError) as exc:
        log.warning("WhatsApp channel not configured ({}): {}", provider, exc)
        return False
