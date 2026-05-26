"""Run the dashboard.

Usage::

    python -m scripts.serve_web                # 127.0.0.1:8000 (default)
    python -m scripts.serve_web --host 0.0.0.0 # listen on LAN
    python -m scripts.serve_web --reload       # dev hot-reload

The default bind is loopback only -- to expose to your phone on the
same Wi-Fi, pass ``--host 0.0.0.0`` and find your LAN IP with
``ipconfig`` / ``ifconfig``. NEVER bind 0.0.0.0 on a machine that has
a public IP without a TLS-terminating reverse proxy in front.
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from src.utils.logger import get_logger

log = get_logger("scripts.serve_web")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve the AI Trader read-only dashboard."
    )
    p.add_argument("--host", default=os.getenv("WEB_BIND_HOST", "127.0.0.1"),
                   help="Bind address. Default: 127.0.0.1.")
    p.add_argument("--port", type=int,
                   default=int(os.getenv("WEB_BIND_PORT", "8000")),
                   help="Bind port. Default: 8000.")
    p.add_argument("--reload", action="store_true",
                   help="Enable auto-reload (dev only).")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes. Keep at 1 unless behind a "
                        "load balancer; SQLite + WAL doesn't love many "
                        "writers, and we have at most one user.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.host not in ("127.0.0.1", "localhost") and not _is_explicit_dev():
        log.warning(
            "binding {} -- only do this behind a TLS-terminating proxy "
            "(Cloudflare Tunnel / Caddy). For laptop-only use stay on "
            "127.0.0.1.",
            args.host,
        )
    log.info("starting dashboard | http://{}:{}", args.host, args.port)
    uvicorn.run(
        "src.web.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        factory=True,
        proxy_headers=True,                # honour X-Forwarded-Proto via proxy
        forwarded_allow_ips=os.getenv("WEB_FORWARDED_IPS", "127.0.0.1"),
        access_log=False,                  # we log per request from middleware
    )
    return 0


def _is_explicit_dev() -> bool:
    return os.getenv("WEB_BIND_ALLOW_LAN", "").lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    sys.exit(main())
