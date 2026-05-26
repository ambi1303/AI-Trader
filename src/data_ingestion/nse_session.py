"""HTTP session helper for nseindia.com / nsearchives.

NSE blocks plain requests; you need a browser-like User-Agent and cookies.
We "warm up" the session by hitting the home page first, then make the real
request. Retries with exponential backoff on common transient errors.

Security:
- We never follow redirects to a different host (would defeat allow-listing).
- We never accept non-HTTP(S) schemes.
- All host names are validated against an allowlist before each request.
"""

from __future__ import annotations

from urllib.parse import urlparse

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger
from src.utils.secrets import get_settings

log = get_logger("ingest.nse_session")

_ALLOWED_HOSTS = {
    "www.nseindia.com",
    "nsearchives.nseindia.com",
    "archives.nseindia.com",
    "www1.nseindia.com",
}

_WARMUP_URL = "https://www.nseindia.com/"


class NseHttpError(RuntimeError):
    pass


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise NseHttpError(f"Unsupported URL scheme: {parsed.scheme}")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise NseHttpError(
            f"Host not on allow-list: {parsed.hostname}. "
            f"Allowed: {sorted(_ALLOWED_HOSTS)}"
        )


def make_session() -> requests.Session:
    settings = get_settings()
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": settings.http_user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.nseindia.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
        }
    )
    return s


def warm_up(session: requests.Session) -> None:
    """Visit nseindia.com to populate cookies before fetching archives."""
    settings = get_settings()
    try:
        r = session.get(
            _WARMUP_URL,
            timeout=settings.http_timeout_seconds,
            allow_redirects=False,
        )
        log.debug("NSE warmup status={}", r.status_code)
    except requests.RequestException as e:
        log.warning("NSE warmup failed (will still try archive fetch): {}", str(e))


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, NseHttpError)),
)
def get_bytes(session: requests.Session, url: str) -> bytes:
    _validate_url(url)
    settings = get_settings()
    r = session.get(
        url,
        timeout=settings.http_timeout_seconds,
        allow_redirects=False,  # do not follow redirects to other hosts
    )
    if r.status_code in (403, 503):
        # Re-warm cookies and retry by raising a retryable error
        warm_up(session)
        raise NseHttpError(f"NSE blocked request: HTTP {r.status_code}")
    if r.status_code != 200:
        raise NseHttpError(f"Unexpected status {r.status_code} for URL")
    return r.content
