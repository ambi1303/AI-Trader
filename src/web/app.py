"""FastAPI dashboard application.

Routes
------
* ``GET  /healthz``         -- unauthenticated; just confirms the process
                               is up and the DB is reachable.
* ``GET  /login``           -- HTML login form.
* ``POST /login``           -- credential check, sets signed cookie.
* ``POST /logout``          -- clears the cookie.
* ``GET  /``                -- mobile-first dashboard (requires session).
* ``GET  /positions``       -- open + closed paper trades.
* ``GET  /history``         -- last N days of signals + trades.
* ``GET  /report.pdf``      -- streams today's PDF if it exists.
* ``GET  /api/snapshot``    -- JSON of the dashboard data (for future
                               native clients).
* ``GET  /api/health``      -- JSON freshness summary.
* ``GET  /manifest.json``   -- PWA manifest so phones can "Add to Home
                               Screen".

Security middleware (always on)
-------------------------------
* Strict ``Content-Security-Policy`` permitting only the Tailwind CDN.
* ``X-Content-Type-Options: nosniff``, ``X-Frame-Options: DENY``,
  ``Referrer-Policy: no-referrer``, and (when ``WEB_FORCE_HSTS=true``)
  HSTS.
* In-memory token bucket rate limiter scoped per IP.
* All cookies are HttpOnly + SameSite=Lax. ``Secure`` is auto-detected
  from the request scheme so localhost dev still works.

Body-size limits and request timeouts are enforced upstream by Uvicorn
and Cloudflare; we don't try to second-guess them here.
"""

from __future__ import annotations

import re
import time
from collections import deque
from pathlib import Path
from typing import Iterable

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, status
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.utils.logger import get_logger
from src.web import auth as web_auth
from src.web import live
from src.web import queries as q

log = get_logger("web.app")

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"
_REPORTS_DIR = Path("data/reports/notifications")  # produced by the daily pipeline

# NSE symbols are uppercase alphanumerics plus '&' and '-' (M&M, BAJAJ-AUTO,
# ARE&M). We allow-list strictly so a path/param can't smuggle anything weird
# even though every downstream query is parameterised.
_SYMBOL_RE = re.compile(r"^[A-Z0-9&\-]{1,20}$")


def _valid_symbol(symbol: str) -> str | None:
    s = (symbol or "").strip().upper()
    return s if _SYMBOL_RE.match(s) else None


# ---------------------------------------------------------------------------
# Rate limiter -- tiny in-memory token bucket, per remote IP.
# Good enough for a single-user dashboard; Cloudflare adds DDoS protection
# in front of this in Phase C.
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, *, max_per_minute: int = 120) -> None:
        self._max = int(max_per_minute)
        self._hits: dict[str, deque[float]] = {}

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        window = self._hits.setdefault(ip, deque(maxlen=self._max + 1))
        # Drop entries older than 60s.
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self._max:
            return False
        window.append(now)
        return True


def _client_ip(request: Request) -> str:
    # When behind Cloudflare Tunnel the original IP is in CF-Connecting-IP.
    # We DO NOT trust X-Forwarded-For from arbitrary clients.
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    cfg = web_auth.AuthConfig.from_env()
    require_user = web_auth.require_user(cfg)
    rate_limiter = _RateLimiter()

    app = FastAPI(
        title="AI Trader Dashboard",
        version="phase-A",
        docs_url=None,        # turn off auto-generated /docs in prod
        redoc_url=None,
        openapi_url=None,
    )

    if _TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    else:                                 # pragma: no cover -- defensive
        raise RuntimeError(f"templates dir missing: {_TEMPLATES_DIR}")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)),
                  name="static")

    # ---- Browser-friendly 401 ---------------------------------------------
    # Browsers visiting / unauthenticated should land on the login form,
    # NOT see a JSON error. Keep the 401 JSON for /api/* and explicit
    # JSON Accept headers so machine clients still see structured errors.

    def _wants_html(request: Request) -> bool:
        if request.url.path.startswith("/api/"):
            return False
        accept = request.headers.get("accept", "")
        return "text/html" in accept.lower()

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request,
                                      exc: StarletteHTTPException):
        if exc.status_code == 401 and _wants_html(request):
            # 303 forces a GET; safe whether the original was GET or POST.
            return RedirectResponse(url="/login",
                                    status_code=status.HTTP_303_SEE_OTHER)
        # Otherwise emit the default JSON shape; preserve any custom headers
        # the route added (e.g. WWW-Authenticate).
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None) or {},
        )

    # ---- Security headers + rate limit ------------------------------------

    @app.middleware("http")
    async def _harden(request: Request, call_next):
        ip = _client_ip(request)
        # Skip rate-limit for /healthz so liveness probes always succeed.
        if request.url.path != "/healthz" and not rate_limiter.allow(ip):
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        response: Response = await call_next(request)
        # Tailwind CDN is the only third-party origin we permit. Inline
        # styles are needed by Tailwind's runtime; inline scripts are
        # NOT allowed -- our templates ship none.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "img-src 'self' data:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'none'; "
            "form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), camera=(), microphone=(), payment=()"
        )
        # Only emit HSTS when the request actually arrived over TLS;
        # over plain HTTP it's a no-op that may confuse browsers.
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        # Sensitive endpoints must never be cached.
        if request.url.path != "/static" and not request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ---- Public endpoints --------------------------------------------------

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        try:
            payload = q.get_health()
            return JSONResponse(payload)
        except Exception as exc:  # noqa: BLE001
            log.error("healthz failure: {}", exc)
            return JSONResponse({"ok": False}, status_code=503)

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_form(request: Request, error: str | None = None):
        return templates.TemplateResponse(
            request, "login.html", {"error": error},
        )

    @app.post("/login", include_in_schema=False)
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        if not web_auth.check_credentials(cfg, username=username,
                                          password=password):
            log.info("failed login | ip={}", _client_ip(request))
            # Identical response shape so timing/length doesn't reveal
            # which field was wrong.
            return RedirectResponse(
                url="/login?error=Invalid+credentials",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        token = web_auth.issue_token(cfg, username=cfg.username)
        resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        resp.set_cookie(
            value=token,
            **web_auth.cookie_kwargs(secure=request.url.scheme == "https"),
        )
        log.info("login ok | ip={}", _client_ip(request))
        return resp

    @app.post("/logout", include_in_schema=False)
    async def logout(request: Request):
        resp = RedirectResponse(url="/login",
                                status_code=status.HTTP_303_SEE_OTHER)
        resp.delete_cookie(web_auth.COOKIE_NAME, path="/")
        return resp

    @app.get("/manifest.json", include_in_schema=False)
    async def manifest() -> JSONResponse:
        return JSONResponse({
            "name": "AI Trader",
            "short_name": "AITrader",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#0f172a",
            "theme_color": "#0f172a",
            "icons": [],
        })

    # ---- Authenticated endpoints ------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def home(request: Request,
                   user: str = Depends(require_user)):
        snap = q.build_dashboard_snapshot()
        return templates.TemplateResponse(
            request, "dashboard.html", {"snap": snap, "user": user},
        )

    @app.get("/positions", response_class=HTMLResponse,
             include_in_schema=False)
    async def positions(request: Request,
                        user: str = Depends(require_user)):
        return templates.TemplateResponse(
            request, "positions.html",
            {
                "open_positions": q.get_open_positions(),
                "closed": q.get_recent_closed(window_days=30, limit=200),
                "user": user,
            },
        )

    @app.get("/history", response_class=HTMLResponse,
             include_in_schema=False)
    async def history(request: Request,
                      user: str = Depends(require_user)):
        return templates.TemplateResponse(
            request, "history.html",
            {
                "closed": q.get_recent_closed(window_days=120, limit=500),
                "user": user,
            },
        )

    @app.get("/stock/{symbol}", response_class=HTMLResponse,
             include_in_schema=False)
    async def stock_detail(request: Request, symbol: str,
                           user: str = Depends(require_user)):
        sym = _valid_symbol(symbol)
        if sym is None:
            raise HTTPException(status_code=404, detail="unknown symbol")
        detail = q.get_stock_detail(sym)
        if detail.get("last_close") is None and detail.get("prediction") is None:
            raise HTTPException(status_code=404, detail="no data for symbol")
        return templates.TemplateResponse(
            request, "stock.html", {"d": detail, "user": user, "symbol": sym},
        )

    @app.get("/api/ohlc/{symbol}", include_in_schema=False)
    async def api_ohlc(symbol: str,
                       user: str = Depends(require_user)) -> JSONResponse:
        sym = _valid_symbol(symbol)
        if sym is None:
            raise HTTPException(status_code=404, detail="unknown symbol")
        return JSONResponse(q.get_ohlc(sym))

    @app.get("/api/ltp/{symbol}", include_in_schema=False)
    async def api_ltp(symbol: str,
                      user: str = Depends(require_user)) -> JSONResponse:
        sym = _valid_symbol(symbol)
        if sym is None:
            raise HTTPException(status_code=404, detail="unknown symbol")
        # The upstream call is blocking I/O -> run off the event loop.
        payload = await run_in_threadpool(live.get_live_quote, sym)
        return JSONResponse(payload)

    @app.get("/api/ltp", include_in_schema=False)
    async def api_ltp_batch(symbols: str = "",
                            user: str = Depends(require_user)) -> JSONResponse:
        # Comma-separated; validate + de-dup + cap so a crafted query can't
        # fan out into an unbounded number of upstream calls.
        seen: list[str] = []
        for raw in (symbols or "").split(","):
            s = _valid_symbol(raw)
            if s and s not in seen:
                seen.append(s)
            if len(seen) >= 30:
                break
        if not seen:
            return JSONResponse({"quotes": {}, "market_open": live.is_market_open()})
        quotes = await run_in_threadpool(live.get_live_quotes, seen)
        return JSONResponse({"quotes": quotes, "market_open": live.is_market_open()})

    @app.get("/report.pdf", include_in_schema=False)
    async def report_pdf(user: str = Depends(require_user)):
        path = _latest_pdf(_REPORTS_DIR)
        if path is None:
            raise HTTPException(status_code=404, detail="no report yet")
        return FileResponse(
            path=str(path),
            media_type="application/pdf",
            filename=path.name,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/snapshot", include_in_schema=False)
    async def api_snapshot(user: str = Depends(require_user)) -> JSONResponse:
        return JSONResponse(q.build_dashboard_snapshot().to_dict())

    @app.get("/api/health", include_in_schema=False)
    async def api_health(user: str = Depends(require_user)) -> JSONResponse:
        return JSONResponse(q.get_health())

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_pdf(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    pdfs: Iterable[Path] = directory.rglob("*.pdf")
    candidates = sorted(pdfs, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None
