"""Google News RSS scraper for per-symbol headlines.

This is the *minimum-viable* news pipeline for Week 2:
- Build a Google News RSS URL per symbol query (no API key required).
- Parse the RSS XML safely using defusedxml (XXE-safe).
- Persist to news_headlines with (source, url) UNIQUE so re-runs are idempotent.

What's intentionally NOT here yet (deferred per the Week-2 plan):
- FinBERT scoring. We do not import transformers/torch in this module.
  Reason: per the plan, FinBERT must be measured against 200-300 manually
  labeled Indian headlines first, and only included as a feature if
  macro-F1 >= 0.55. We'll add a `finbert_scorer.py` once that gate is met.
- GDELT. The free GDELT GKG export is large and noisy; defer until we
  have a working FinBERT path that can leverage it.

URL allow-list:
- We only fetch from news.google.com (RSS endpoint). Any redirect is
  not followed. This bounds SSRF risk.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlparse

import requests
from defusedxml import ElementTree as DET  # XXE-safe parser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.db import transaction
from src.utils.logger import get_logger
from src.utils.secrets import get_settings

log = get_logger("ingest.news")

_ALLOWED_HOSTS = {"news.google.com"}
_RSS_BASE = "https://news.google.com/rss/search"


class NewsHttpError(RuntimeError):
    pass


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise NewsHttpError(f"Unsupported URL scheme: {parsed.scheme}")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise NewsHttpError(f"Host not on allow-list: {parsed.hostname}")


def _build_query_url(query: str, *, hl: str = "en-IN", gl: str = "IN", ceid: str = "IN:en") -> str:
    qs = urllib.parse.urlencode({"q": query, "hl": hl, "gl": gl, "ceid": ceid})
    return f"{_RSS_BASE}?{qs}"


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, NewsHttpError)),
)
def _fetch_rss(url: str) -> bytes:
    _validate_url(url)
    settings = get_settings()
    r = requests.get(
        url,
        timeout=settings.http_timeout_seconds,
        allow_redirects=False,
        headers={"User-Agent": settings.http_user_agent},
    )
    if r.status_code != 200:
        raise NewsHttpError(f"Status {r.status_code}")
    return r.content


def _parse_rss(xml_bytes: bytes) -> list[dict]:
    """Return a list of {published_at, title, url, source}."""
    # defusedxml disables external entities and DTDs. Safe against XXE.
    root = DET.fromstring(xml_bytes)
    out: list[dict] = []
    for item in root.iterfind(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_el = item.find("source")
        publisher = (src_el.text.strip() if src_el is not None and src_el.text else "google_news")
        try:
            published_at = (
                datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
                .replace(tzinfo=timezone.utc)
                .isoformat()
            )
        except ValueError:
            published_at = datetime.now(tz=timezone.utc).isoformat()
        if title and link:
            out.append(
                {
                    "title": title,
                    "url": link,
                    "published_at": published_at,
                    "source": f"google_news:{publisher}",
                }
            )
    return out


def fetch_headlines_for_query(query: str) -> list[dict]:
    url = _build_query_url(query)
    log.debug("Fetching news for query={}", query)
    xml = _fetch_rss(url)
    return _parse_rss(xml)


def upsert_headlines(symbol: str | None, items: list[dict]) -> int:
    if not items:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO news_headlines (symbol, published_at, source, title, url)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, url) DO NOTHING
            """,
            [
                (
                    symbol,
                    it["published_at"],
                    it["source"][:200],
                    it["title"][:500],
                    it["url"][:1000],
                )
                for it in items
            ],
        )
    log.info("Upserted news rows: {} (symbol={})", len(items), symbol)
    return len(items)


def fetch_for_symbols(symbol_query_map: Iterable[tuple[str, str]]) -> int:
    """`symbol_query_map` yields (symbol, query_string)."""
    total = 0
    for sym, q in symbol_query_map:
        try:
            items = fetch_headlines_for_query(q)
            total += upsert_headlines(sym, items)
        except Exception as e:  # noqa: BLE001
            log.warning("News fetch failed for {} ({}): {}", sym, q, str(e))
    return total
