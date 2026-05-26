"""Single-user auth for the dashboard.

Design
------
* Username + password come from ``.env`` (``WEB_USERNAME`` /
  ``WEB_PASSWORD``). Plaintext-in-.env is acceptable here because
  the file is gitignored AND OS-restricted (chmod 600 / Windows ACL).
* Comparison uses ``hmac.compare_digest`` to defeat timing attacks.
* On successful login we mint an itsdangerous-signed cookie carrying
  ``{"u": username, "exp": <unix-ts>}``. The cookie is HttpOnly,
  Secure (when behind TLS), SameSite=Lax. We do NOT store anything
  user-supplied in the cookie value.
* On every protected request a FastAPI dependency verifies the
  signature, the expiry, and that the username still matches the
  current ``WEB_USERNAME``. Any mismatch -> 401.
* Rotating ``WEB_SESSION_SECRET`` invalidates every existing session
  immediately (signature breaks). Use this as a kill-switch.

What this module DOES NOT do
----------------------------
* Multi-user / role separation (Phase A is single-user).
* Password reset flows (regenerate ``WEB_PASSWORD`` in .env and rotate
  the session secret).
* OAuth (overkill for one user).
"""

from __future__ import annotations

import hmac
import json
import time
from dataclasses import dataclass

from fastapi import Cookie, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from src.utils.logger import get_logger
from src.utils.secrets import get_secret

log = get_logger("web.auth")

_COOKIE_NAME = "ait_session"
_COOKIE_MAX_AGE = 60 * 60 * 12   # 12 hours of inactivity == re-login

# Forbidden usernames -- typo-squatting common values is a low-effort
# defence against credential stuffing scans.
_DISALLOWED_USERNAMES = {"", "admin", "root", "administrator", "user"}


# ---------------------------------------------------------------------------
# Configuration loading (read once at startup)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthConfig:
    username: str
    password: str
    session_secret: str

    @classmethod
    def from_env(cls) -> "AuthConfig":
        username = (get_secret("WEB_USERNAME", required=False) or "").strip()
        password = (get_secret("WEB_PASSWORD", required=False) or "").strip()
        session_secret = (get_secret("WEB_SESSION_SECRET", required=False) or "").strip()

        if not username or not password or not session_secret:
            raise RuntimeError(
                "Web dashboard auth is not configured. Set WEB_USERNAME, "
                "WEB_PASSWORD, and WEB_SESSION_SECRET in .env. See "
                ".env.example for instructions."
            )
        # Soft warning rather than hard fail on weak choices, so the user
        # can still spin it up locally; the README repeats the warning.
        if username.lower() in _DISALLOWED_USERNAMES:
            log.warning(
                "WEB_USERNAME is a common value -- pick something less "
                "guessable to slow down credential-stuffing."
            )
        if len(password) < 12:
            log.warning(
                "WEB_PASSWORD is shorter than 12 chars; recommend >=20."
            )
        if len(session_secret) < 32:
            raise RuntimeError(
                "WEB_SESSION_SECRET must be >=32 chars. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return cls(username=username, password=password,
                   session_secret=session_secret)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="ait_session_v1")


def issue_token(cfg: AuthConfig, *, username: str) -> str:
    payload = json.dumps({"u": username})
    return _serializer(cfg.session_secret).dumps(payload)


def verify_token(cfg: AuthConfig, token: str) -> str | None:
    """Return the username if valid; else None.

    ``URLSafeTimedSerializer`` enforces ``max_age``; we double-check
    the username still matches the env value so rotating WEB_USERNAME
    immediately invalidates any pre-existing tokens.
    """
    try:
        raw = _serializer(cfg.session_secret).loads(token, max_age=_COOKIE_MAX_AGE)
    except SignatureExpired:
        return None
    except BadSignature:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    user = data.get("u") if isinstance(data, dict) else None
    if not isinstance(user, str):
        return None
    if not hmac.compare_digest(user, cfg.username):
        return None
    return user


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def check_credentials(cfg: AuthConfig, *, username: str, password: str) -> bool:
    """Constant-time username + password match.

    Both comparisons happen even if the username is wrong, so an attacker
    cannot use response timing to enumerate valid usernames.
    """
    user_ok = hmac.compare_digest(username.encode("utf-8"),
                                  cfg.username.encode("utf-8"))
    pwd_ok = hmac.compare_digest(password.encode("utf-8"),
                                 cfg.password.encode("utf-8"))
    # `&` (bitwise AND) on bools to avoid short-circuit timing leak.
    return bool(user_ok & pwd_ok)


# ---------------------------------------------------------------------------
# Per-request guard
# ---------------------------------------------------------------------------


def require_user(cfg: AuthConfig):
    """Build a FastAPI dependency that enforces a valid session cookie.

    Wrapping the dependency in a closure lets us inject the AuthConfig
    that the app loaded at startup, while still having FastAPI handle
    cookie parsing and 401 raising in a routes-friendly way.
    """

    def _dep(
        request: Request,
        ait_session: str | None = Cookie(default=None, alias=_COOKIE_NAME),
    ) -> str:
        if ait_session is None:
            _challenge(request, "missing session cookie")
        user = verify_token(cfg, ait_session)
        if user is None:
            _challenge(request, "bad or expired session")
        return user

    return _dep


def _challenge(request: Request, reason: str) -> None:
    # Log the path but NOT the cookie (which is in the redaction set).
    log.info("auth challenge | path={} reason={}", request.url.path, reason)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="authentication required",
        headers={"WWW-Authenticate": 'FormBased realm="ai-trader"'},
    )


# ---------------------------------------------------------------------------
# Cookie helpers (shared between /login and /logout routes)
# ---------------------------------------------------------------------------


def cookie_kwargs(*, secure: bool) -> dict:
    return {
        "key": _COOKIE_NAME,
        "max_age": _COOKIE_MAX_AGE,
        "httponly": True,
        "secure": secure,
        "samesite": "lax",
        "path": "/",
    }


COOKIE_NAME = _COOKIE_NAME   # exported so the app can use the same constant
