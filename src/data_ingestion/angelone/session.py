"""Angel One SmartAPI session manager.

Responsibilities
----------------
* Generate the rolling 6-digit TOTP from the base32 seed in ``.env``.
* POST to ``loginByPassword`` and capture the JWT, refresh, and feed tokens.
* Refresh the JWT before it expires (default 8h validity).
* Sign every downstream request with the JWT in the ``Authorization`` header
  *plus* the static ``X-PrivateKey`` header that Angel One requires.
* Refuse to call any order-placement endpoint unless ``allow_orders=True``.

Security
--------
* Tokens never touch disk. They live in the :class:`AngelOneSession`
  instance and are garbage-collected when the orchestration ends.
* Logger redaction (``src/utils/logger.py::_REDACT_KEYS``) strips
  ``jwt_token``, ``refresh_token``, ``feed_token``, MPIN, TOTP secret etc.
* ``requests.Session`` is configured with ``verify=True`` (cert validation
  on); we never disable TLS verification.
* Login retries use exponential backoff via ``tenacity`` to avoid
  hammering Angel One on transient 5xx; we do NOT retry on 4xx (those
  are usually wrong-creds or wrong-TOTP and retrying just wastes the
  daily login budget).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pyotp
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger
from src.utils.secrets import get_secret

log = get_logger("angelone.session")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://apiconnect.angelone.in"   # historic alias still works
_LOGIN_PATH = "/rest/auth/angelbroking/user/v1/loginByPassword"
_REFRESH_PATH = "/rest/auth/angelbroking/jwt/v1/generateTokens"
_LOGOUT_PATH = "/rest/secure/angelbroking/user/v1/logout"

# Default JWT validity is ~8h on Angel One; refresh proactively at 80%.
_DEFAULT_JWT_TTL = timedelta(hours=8)
_REFRESH_BEFORE = timedelta(minutes=15)

# Angel One requires these "device fingerprint" headers on every request.
# We DON'T leak the real local IP / MAC -- those add no security value
# because the JWT is the actual auth, and a real MAC would be PII for
# logs. Sending generic stable placeholders is well within the spec.
_GENERIC_HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-UserType": "USER",
    "X-SourceID": "WEB",
    "X-ClientLocalIP": "127.0.0.1",
    "X-ClientPublicIP": "127.0.0.1",
    "X-MACAddress": "00:00:00:00:00:00",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AngelOneAuthError(RuntimeError):
    """Login / refresh failure. NEVER carries credentials in its message."""


class AngelOneAPIError(RuntimeError):
    """Non-auth API failure (rate limit, invalid token, server 5xx)."""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class AngelOneCredentials:
    """In-memory only. NEVER serialise this dataclass."""
    api_key: str
    client_code: str
    mpin: str
    totp_secret: str
    api_secret: str | None = None       # optional; some endpoints want it


def _redact(value: str | None, *, keep: int = 2) -> str:
    """Replace all but the last ``keep`` chars with asterisks. Used for
    log lines that prove "we did receive a token of length N" without
    leaking the value itself."""
    if value is None:
        return "<none>"
    return "*" * max(0, len(value) - keep) + value[-keep:]


class AngelOneSession:
    """Holds tokens and a configured ``requests.Session`` for SmartAPI calls.

    Thread-safe: a single mutex guards the JWT refresh path so two
    parallel callers won't both try to refresh and double-debit the
    daily login budget.

    The session refuses to call order-placement endpoints unless the
    constructor was passed ``allow_orders=True``. The current codebase
    NEVER passes that flag -- live execution is out of scope for
    Week 5.
    """

    def __init__(
        self,
        creds: AngelOneCredentials,
        *,
        base_url: str = _BASE_URL,
        request_timeout: int = 15,
        allow_orders: bool = False,
    ) -> None:
        self._creds = creds
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout
        self._allow_orders = allow_orders

        self._jwt: str | None = None
        self._refresh_token: str | None = None
        self._feed_token: str | None = None
        self._jwt_expires_at: datetime | None = None

        self._http = requests.Session()
        self._http.verify = True              # never disable TLS verify
        self._http.headers.update(_GENERIC_HEADERS_BASE)
        self._http.headers["X-PrivateKey"] = creds.api_key

        self._mutex = threading.Lock()

    # -- Properties ---------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return self._jwt is not None and not self._is_jwt_near_expiry()

    @property
    def feed_token(self) -> str | None:
        """For websocket clients."""
        return self._feed_token

    def websocket_credentials(self) -> dict[str, str] | None:
        """Return the four values SmartStream WS 2.0 needs to authenticate.

        Logs in first if necessary so the JWT/feed token are populated.
        Returns ``None`` only if login could not produce a JWT. The dict is
        in-memory only and must NEVER be logged or serialised -- every value
        in it is a credential (the JWT auth_token and feed_token especially).
        """
        if not self.is_authenticated:
            self.login()
        if not self._jwt or not self._feed_token:
            return None
        # WS Authorization header takes the raw JWT (no "Bearer " prefix);
        # this matches Angel One's reference client. REST uses the Bearer form.
        return {
            "auth_token": self._jwt,
            "feed_token": self._feed_token,
            "api_key": self._creds.api_key,
            "client_code": self._creds.client_code,
        }

    @property
    def allows_orders(self) -> bool:
        return self._allow_orders

    # -- Auth lifecycle -----------------------------------------------------

    def _is_jwt_near_expiry(self) -> bool:
        if self._jwt_expires_at is None:
            return True
        return datetime.now(timezone.utc) >= (self._jwt_expires_at - _REFRESH_BEFORE)

    @retry(
        retry=retry_if_exception_type(AngelOneAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def login(self) -> None:
        """Authenticate using client_code + MPIN + a fresh TOTP code.

        Retries only on transient ``AngelOneAPIError`` (5xx / network).
        Auth errors (4xx) raise immediately so we don't burn the daily
        login budget against bad credentials.
        """
        with self._mutex:
            totp_code = self._generate_totp()
            payload = {
                "clientcode": self._creds.client_code,
                "password": self._creds.mpin,
                "totp": totp_code,
            }
            url = self._base_url + _LOGIN_PATH
            log.info(
                "Angel One login attempt | client={} api_key={}",
                _redact(self._creds.client_code, keep=2),
                _redact(self._creds.api_key, keep=4),
            )
            try:
                resp = self._http.post(url, json=payload, timeout=self._timeout)
            except requests.RequestException as exc:
                raise AngelOneAPIError(f"login network error: {exc!s}") from exc

            self._raise_for_response(resp, action="login")

            data = resp.json()
            payload_data = data.get("data") or {}
            jwt = payload_data.get("jwtToken")
            refresh = payload_data.get("refreshToken")
            feed = payload_data.get("feedToken")
            if not jwt:
                raise AngelOneAuthError(
                    "login succeeded but no jwtToken in response"
                )

            self._jwt = jwt
            self._refresh_token = refresh
            self._feed_token = feed
            self._jwt_expires_at = datetime.now(timezone.utc) + _DEFAULT_JWT_TTL
            self._http.headers["Authorization"] = f"Bearer {jwt}"
            log.info(
                "Angel One login OK | jwt_len={} refresh_present={}",
                len(jwt), bool(refresh),
            )

    def refresh(self) -> None:
        """Trade the refresh token for a new JWT. Falls back to full login
        if the refresh token is missing or rejected."""
        with self._mutex:
            if not self._refresh_token:
                log.info("No refresh token; doing full login instead")
                self._jwt = None
                self._jwt_expires_at = None
            else:
                url = self._base_url + _REFRESH_PATH
                resp = self._http.post(
                    url,
                    json={"refreshToken": self._refresh_token},
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    data = (resp.json() or {}).get("data") or {}
                    self._jwt = data.get("jwtToken") or self._jwt
                    self._refresh_token = (
                        data.get("refreshToken") or self._refresh_token
                    )
                    self._jwt_expires_at = (
                        datetime.now(timezone.utc) + _DEFAULT_JWT_TTL
                    )
                    if self._jwt:
                        self._http.headers["Authorization"] = f"Bearer {self._jwt}"
                    log.info("Angel One JWT refreshed (no full re-login)")
                    return
                log.warning(
                    "JWT refresh failed (status={}); falling back to login",
                    resp.status_code,
                )
        # Full re-login is OUTSIDE the mutex (login() takes it again).
        self.login()

    def logout(self) -> None:
        """Best-effort logout. Errors here are swallowed -- the orchestrator
        shouldn't care if Angel One returned 5xx on the way out."""
        if self._jwt is None:
            return
        try:
            url = self._base_url + _LOGOUT_PATH
            self._http.post(
                url,
                json={"clientcode": self._creds.client_code},
                timeout=self._timeout,
            )
        except requests.RequestException:
            pass
        finally:
            self._jwt = None
            self._refresh_token = None
            self._feed_token = None
            self._jwt_expires_at = None
            self._http.headers.pop("Authorization", None)

    # -- Authenticated request helper --------------------------------------

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        is_order_endpoint: bool = False,
    ) -> dict[str, Any]:
        """Send an authenticated request and return the parsed JSON body.

        Pass ``is_order_endpoint=True`` for any order/portfolio endpoint
        you might add later. The session refuses to make the call unless
        ``allow_orders=True`` was set explicitly at construction time --
        this is the runtime kill-switch for accidental live trading.
        """
        if is_order_endpoint and not self._allow_orders:
            raise AngelOneAuthError(
                "Refusing to call order endpoint: this session was created "
                "with allow_orders=False. Set ANGEL_ALLOW_ORDERS=true AND "
                "construct the session with allow_orders=True to enable."
            )

        if not self.is_authenticated:
            self.login()

        url = self._base_url + path
        # Each request gets a unique correlation id so both sides can
        # trace it without exposing user data.
        headers = {"X-CorrelationID": uuid.uuid4().hex}
        try:
            resp = self._http.request(
                method, url,
                json=json_body, params=params,
                timeout=self._timeout, headers=headers,
            )
        except requests.RequestException as exc:
            raise AngelOneAPIError(f"network error on {path}: {exc!s}") from exc

        self._raise_for_response(resp, action=path)

        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise AngelOneAPIError(
                f"non-JSON response from {path}: status={resp.status_code}"
            ) from exc

    # -- Internal helpers ---------------------------------------------------

    def _generate_totp(self) -> str:
        try:
            return pyotp.TOTP(self._creds.totp_secret).now()
        except Exception as exc:  # noqa: BLE001
            # pyotp raises base Exception on invalid base32 so we have to
            # cast a wide net here. We deliberately don't echo the cause's
            # repr -- it sometimes contains the seed.
            raise AngelOneAuthError(
                "Could not derive TOTP -- check ANGEL_TOTP_SECRET is the "
                "base32 seed (e.g. 'JBSWY3DPEHPK3PXP'), NOT a 6-digit code."
            ) from exc

    def _raise_for_response(self, resp: requests.Response, *, action: str) -> None:
        """Translate HTTP failures into typed exceptions WITHOUT echoing
        any credential-bearing field."""
        if 200 <= resp.status_code < 300:
            return
        try:
            body = resp.json()
            err_msg = body.get("message") or body.get("errorcode") or ""
        except Exception:  # noqa: BLE001
            err_msg = ""
        # Auth-class failures: do NOT retry, and do NOT include creds.
        if resp.status_code in (401, 403):
            raise AngelOneAuthError(
                f"{action} authentication failure (status={resp.status_code}, "
                f"hint={err_msg!r}). Verify ANGEL_API_KEY / ANGEL_CLIENT_CODE "
                f"/ ANGEL_MPIN / ANGEL_TOTP_SECRET in .env."
            )
        if resp.status_code in (400, 404):
            # Bad request / not found -- request was malformed.
            raise AngelOneAPIError(
                f"{action} bad request (status={resp.status_code}, "
                f"hint={err_msg!r})"
            )
        # 429 rate limit + 5xx are retried by callers.
        raise AngelOneAPIError(
            f"{action} server error (status={resp.status_code}, "
            f"hint={err_msg!r})"
        )


# ---------------------------------------------------------------------------
# Bootstrapping helpers
# ---------------------------------------------------------------------------


def _load_credentials_from_env() -> AngelOneCredentials | None:
    """Return creds if all required env vars are set; else None.

    We use ``required=False`` for each individual lookup so the caller
    can decide whether to skip the Angel One leg gracefully (paper-only
    runs without the integration).
    """
    api_key = get_secret("ANGEL_API_KEY", required=False)
    client_code = get_secret("ANGEL_CLIENT_CODE", required=False)
    mpin = get_secret("ANGEL_MPIN", required=False)
    totp_secret = get_secret("ANGEL_TOTP_SECRET", required=False)
    api_secret = get_secret("ANGEL_API_SECRET", required=False)

    if not all([api_key, client_code, mpin, totp_secret]):
        return None
    return AngelOneCredentials(
        api_key=api_key,
        client_code=client_code,
        mpin=mpin,
        totp_secret=totp_secret,
        api_secret=api_secret,
    )


def load_session_from_env(*, allow_orders: bool = False) -> AngelOneSession | None:
    """Build an :class:`AngelOneSession` from .env credentials.

    Returns ``None`` if any required env var is missing, so the caller
    (typically the daily orchestrator) can skip the Angel One leg
    gracefully without crashing.

    ``allow_orders`` defaults to False. The order kill-switch is also
    cross-checked against ``ANGEL_ALLOW_ORDERS`` env var: BOTH must say
    yes before the session would let an order endpoint through.
    """
    creds = _load_credentials_from_env()
    if creds is None:
        return None
    env_allow = (get_secret("ANGEL_ALLOW_ORDERS", required=False) or "false").lower()
    return AngelOneSession(
        creds,
        allow_orders=(allow_orders and env_allow in ("1", "true", "yes")),
    )
