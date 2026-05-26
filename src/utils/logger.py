"""Structured logging via loguru.

- JSON sink in production (LOG_FORMAT=json); human sink otherwise.
- Rotating file logs in LOG_DIR.
- A REDACTION filter strips known-sensitive keys from any record extra.
- Sanitises CR/LF in messages to prevent log injection.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.secrets import get_settings, project_root

_REDACT_KEYS = {
    "telegram_bot_token",
    # Legacy v2-prefixed names (kept so older configs still get redacted).
    "angel_one_api_key",
    "angel_one_pin",
    "angel_one_totp_seed",
    # Current Angel One SmartAPI variable names (.env.example).
    "angel_api_key",
    "angel_api_secret",
    "angel_client_code",
    "angel_mpin",
    "angel_totp_secret",
    # Angel One session response fields. The SmartAPI login returns these
    # JWT-shaped tokens which grant order-placement rights for the day --
    # they MUST never appear in any log file.
    "jwttoken",
    "jwt_token",
    "refreshtoken",
    "refresh_token",
    "feedtoken",
    "feed_token",
    # Web dashboard auth.
    "web_password",
    "web_session_secret",
    "session_cookie",
    "cookie",
    # Generic bucket.
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
}

# Strip CR / LF / NUL from log messages. Defence against log injection
# when rendering attacker-controlled fields (symbols, headlines, etc.).
_BAD_CHARS = re.compile(r"[\r\n\x00]")


def _redact(record: dict[str, Any]) -> dict[str, Any]:
    extra = record.get("extra") or {}
    for k in list(extra.keys()):
        if k.lower() in _REDACT_KEYS:
            extra[k] = "***REDACTED***"
    msg = record.get("message", "")
    if isinstance(msg, str) and _BAD_CHARS.search(msg):
        record["message"] = _BAD_CHARS.sub(" ", msg)
    return record


def _patcher(record: dict[str, Any]) -> None:
    _redact(record)


_INITIALISED = False


def setup_logging() -> None:
    """Idempotent. Safe to call multiple times."""
    global _INITIALISED
    if _INITIALISED:
        return

    settings = get_settings()
    log_dir = Path(settings.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_root() / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.configure(patcher=_patcher)

    log_format_env = os.getenv("LOG_FORMAT", "human").lower()

    if log_format_env == "json":
        logger.add(
            sys.stderr,
            level=settings.log_level,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
                "<level>{level: <8}</level> "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
                "- <level>{message}</level>"
            ),
            backtrace=False,
            diagnose=False,
        )

    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        level=settings.log_level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=True,
    )

    _INITIALISED = True


def get_logger(name: str | None = None):
    setup_logging()
    return logger.bind(component=name) if name else logger
