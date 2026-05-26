"""Read-only web dashboard.

This package serves a phone-friendly HTML view (Jinja2 + Tailwind via CDN)
plus a tiny JSON API on top of the existing SQLite DB. It is intentionally
write-free: nothing the browser does mutates ``trading.db``. The package
is also intentionally self-contained -- the daily pipeline does NOT
import anything from here, so a bug in the dashboard can never corrupt
the cron run.

Phase A ships this as a local Uvicorn process. Phase B/C will containerise
it and put it behind a Cloudflare Tunnel on Oracle Cloud.
"""
