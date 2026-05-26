# AI Trading System (Indian Equities)

Research and alerting system for Indian equity (NSE) markets using free data
sources, a calibrated ML signal stack, walk-forward backtesting with realistic
Indian costs, and Telegram alerts.

This is **Week 1: Foundation, Data Quality, Compatibility**. v1 is alerts-only.
Auto-execution is v2 and requires SEBI / broker registration work first.

---

## Operational Warnings (read first)

- **OneDrive corrupts SQLite.** Your DB lives at `data/trading.db`. Right-click
  the `data/` folder in Explorer and choose **"Always keep on this device" off**,
  or move the project off OneDrive entirely (e.g. `C:\dev\ai_trading_system`).
  WAL files (`-shm`, `-wal`) are particularly sensitive.
- **Cisco corporate machine.** Anything that places real orders later (v2) must
  be cleared with your employer's policy and SEBI's retail-algo framework.
- **Paper-trade for 60 days minimum** before any real money. No exceptions.

---

## Setup

```powershell
# from inside ai_trading_system/
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env       # then fill in values you need (Telegram is Week 6)
```

If you see install errors on Windows ARM or on Oracle Cloud Ampere ARM:

```bash
python scripts/arm_compat_check.py
```

That script imports every Week-1 dependency and prints a clear pass/fail list.

---

## Running the Week-1 Pipeline

The full smoke pipeline (DB init -> seed -> small ingest -> validators):

```powershell
python -m scripts.run_week1_pipeline --smoke
```

Individual steps:

```powershell
python -m scripts.init_db                                      # create / migrate DB
python -m scripts.seed_data                                    # constituency + corp actions + calendar
python -m scripts.ingest_yfinance --symbols RELIANCE,TCS --start 2024-12-01 --end 2024-12-31
python -m scripts.ingest_bhavcopy --start 2024-12-01 --end 2024-12-31 --symbols RELIANCE,TCS
python -m scripts.validate_data                                # cross-source check
python -m scripts.audit_splits                                 # split / bonus adjustment audit
```

---

## Running the Tests

```powershell
pytest -q
```

All tests use a per-test isolated SQLite DB under `tmp_path`, so they never
touch your real `data/trading.db`.

---

## Project Layout

```
ai_trading_system/
  config/              non-secret config: universe, holidays, cost model
  data/seed/           seed CSVs (constituency, known corp actions)
  data/raw/            cached downloads (yfinance parquet, bhavcopy zips)
  data/trading.db      SQLite (WAL mode, foreign keys on)
  src/
    contracts/         Pydantic types shared everywhere (Bar, etc.)
    utils/             db.py, logger.py, secrets.py
    db/                schema.sql, migrate.py
    data_ingestion/    yfinance_loader, bhavcopy_loader, ...
    data_validation/   cross_source_check, split_audit, calendar_check, ...
    features/  models/  backtesting/  risk/  execution/  monitoring/
                        (placeholders, implemented in later weeks)
  scripts/             entry points: init_db, seed_data, ingest_*, validate_*
  tests/               pytest suite
  logs/                rotated, redacted JSON logs
```

---

## Schema (Week 1)

Created idempotently by `src/db/schema.sql` (apply via `scripts/init_db.py`):

- `nifty_constituents` (point-in-time index membership)
- `trading_calendar` (holidays + special sessions)
- `price_data` (OHLCV per source; CHECK-constrained)
- `corporate_actions`
- `circuit_flags`
- `news_headlines` (Week 2)
- `model_runs`, `predictions_log` (Week 3)
- `signal_outbox`, `paper_trades` (Weeks 5 / 6)
- `validation_failures` (audit trail for every validator run)
- view: `v_universe_today`

---

## Gate 1 (must pass to proceed to Week 2)

1. 100% of historical Nifty 50 constituents (5 years, including delisted)
   loaded into `nifty_constituents`.
2. `validate_data` reports cross-source match rate ≥ 99.5%.
3. `audit_splits` reports zero `events_failed` for all known events that fall
   inside your ingested date range.
4. `arm_compat_check.py` is green on the Oracle ARM VM you intend to deploy to.
5. All `pytest` tests pass on your dev machine.

The `historical_constituents.csv` provided is a **starter set** — extend it
backwards from the NSE quarterly index reviews before claiming Gate 1.

---

## Secrets

- All secrets live in `.env` (gitignored). Never commit them.
- `src/utils/secrets.py` is the only loader. It refuses to log values.
- `src/utils/logger.py` redacts known-sensitive keys (`telegram_bot_token`,
  `angel_one_api_key`, etc.) from any log record.

---

## Daily report: WhatsApp + Email

The notifications layer (`src/notifications/`) renders a one-page report
(HTML + plaintext + PDF + WhatsApp text) from whatever's currently in the DB
and ships it on two channels.

### Quick start

```powershell
# 1. preview (no creds needed)
python -m scripts.send_daily_report --dry-run --print-summary
#    artefacts written to data/reports/notifications/<date>/...

# 2. send for real once .env is filled in
python -m scripts.send_daily_report

# 3. historical replay
python -m scripts.send_daily_report --date 2025-10-04
```

### Email setup (Gmail, free)

1. Enable 2-Step Verification on your Google account.
2. Google Account → Security → App passwords → generate one for "Mail / Other".
3. In `.env`:

   ```env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=you@gmail.com
   SMTP_PASSWORD=<the 16-char app password>
   EMAIL_FROM=you@gmail.com
   EMAIL_TO=you@gmail.com,partner@example.com
   ```

The HTML body renders in Gmail / Outlook / phone clients. A one-page PDF
summary is attached automatically (toggle in `config/notifications.yaml`).

### WhatsApp setup (CallMeBot, free)

1. Save **+34 644 51 95 23** in your phone as `CallMeBot`.
2. From WhatsApp, send to that contact: `I allow callmebot to send me messages`.
3. You will receive an API key in reply.
4. In `.env`:

   ```env
   CALLMEBOT_PHONE=919XXXXXXXXX     # country code + number, no '+' or spaces
   CALLMEBOT_APIKEY=<the key from step 3>
   ```

CallMeBot is single-recipient and rate-limited; it's perfect for a personal
trader. For multi-recipient delivery use Twilio (`whatsapp_provider: twilio`
in `config/notifications.yaml`, then fill the `TWILIO_*` env vars).

### What the recipient sees

- **Email** — coloured HTML body with: today's signals, top-N predictions,
  model snapshot, latest backtest summary (Sharpe / DD / TotRet / hit-rate),
  recent trades, health-check, plus a one-page PDF attachment.
- **WhatsApp** — tight ~700-char text: signal count, list of BUY symbols
  with calibrated probability, last backtest Sharpe / MaxDD / trade count.

Every send writes copies of the rendered HTML / text / WhatsApp body / PDF
to `data/reports/notifications/<date>/` so you have an audit trail of
exactly what was dispatched on each day.

### Disabling a channel

Edit `config/notifications.yaml`:

```yaml
enabled_channels:
  email: true
  whatsapp: false
```

A missing channel never blocks the others — if WhatsApp creds are absent the
email still goes out, and vice versa.

---

## Week 5: signals → paper trades → daily orchestrator → scheduler

Week 5 turns predictions into actionable signals, books them as paper
trades, marks them to market every day, and ships the whole story over
WhatsApp + email — fully unattended on a Windows box.

### One-shot daily run

```powershell
# Full pipeline: ingest -> features -> predict -> signals -> paper -> notify
python -m scripts.run_daily

# Same, but render-only (no email/WhatsApp send)
python -m scripts.run_daily --dry-run --print-summary

# Skip the network-heavy steps when you only want to refresh the report
python -m scripts.run_daily --skip-ingest --skip-features --skip-predict
```

`run_daily` is **fault-isolated**: any single step failure is logged into
`validation_failures` (severity `error`, `check_name = daily_step:<step>`)
and the pipeline keeps going so the *report* still reaches your inbox.

### Signal flow at a glance

1. `predict_today` writes calibrated probabilities into `predictions_log`.
2. `src.signals.generator` filters to `calibrated_prob >= threshold`,
   sizes each candidate via fractional-Kelly + ATR vol-target, and queues
   pending rows in `signal_outbox` (idempotent on `(symbol, signal_date)`).
3. `src.paper.reconcile` — at the start of the next session — fills
   pending signals at the open and closes any open paper trade that hit
   its stop / target / trailing-stop / time-stop. Costs use the same
   Zerodha-style schedule as the backtester.
4. `src.notifications.dispatcher` rebuilds the daily report (now
   including open paper positions, recent paper-trade P&L, win-rate, and
   a data-freshness check) and dispatches.

### Schedule it on Windows

A PowerShell helper registers a Scheduled Task (no admin needed; runs as
the current user when logged in):

```powershell
# from the repo root
.\scripts\setup_windows_scheduler.ps1                       # daily 18:00 IST
.\scripts\setup_windows_scheduler.ps1 -At 18:30 -DryRun     # render-only
.\scripts\setup_windows_scheduler.ps1 -TaskName MyTrader    # custom name
```

It creates/updates a task named `AITrader-Daily` triggered Mon–Fri at
18:00 local time (after NSE close + yfinance EOD). Useful follow-ups:

```powershell
Get-ScheduledTask     -TaskName 'AITrader-Daily' | Format-List
Start-ScheduledTask   -TaskName 'AITrader-Daily'        # run now
Get-ScheduledTaskInfo -TaskName 'AITrader-Daily'        # last run + result
Unregister-ScheduledTask -TaskName 'AITrader-Daily' -Confirm:$false
```

For an always-on Linux / Oracle Cloud VM use cron instead:

```cron
30 12 * * 1-5  cd /opt/ai_trading_system && /opt/ai_trading_system/.venv/bin/python -m scripts.run_daily >> logs/cron.log 2>&1
```
(12:30 UTC = 18:00 IST.)

### Inspecting state

```powershell
# Pending signals for today (no fills yet)
python -c "from src.signals.generator import list_pending; import json; print(json.dumps(list_pending(), indent=2, default=str))"

# Open paper positions + recent fills
python -c "from src.utils.db import fetch_all; import json; rows=[dict(r) for r in fetch_all('SELECT id, symbol, status, entry_date, exit_date, qty, entry_price, exit_price, pnl_rupees, exit_reason FROM paper_trades ORDER BY id DESC LIMIT 10')]; print(json.dumps(rows, indent=2, default=str))"

# Recent step-level failures from the daily orchestrator
python -c "from scripts.run_daily import list_recent_failures; import json; print(json.dumps(list_recent_failures(), indent=2, default=str))"
```

### Safety / paper-only

`run_daily` writes to `signal_outbox` and `paper_trades` only; it never
places real orders, never holds broker credentials, and never charges
brokerage. The `cost_rupees` column in `paper_trades` tracks what the
trade *would* have cost so backtest and paper P&L use the identical
cost schedule.

---

## Web dashboard (Phase A: laptop, Phase C: cloud)

A read-only FastAPI app exposes the SQLite results as a phone-friendly
web view. The dashboard never writes to the DB; it shares the same
file with the daily pipeline (SQLite WAL means readers don't block the
writer and vice versa).

### One-time setup

1. Copy `.env.example` to `.env` and fill in the **three** web fields:

   ```bash
   # username -- pick something other than 'admin' / 'root'
   WEB_USERNAME=trader1

   # password -- generate with:
   #   python -c "import secrets; print(secrets.token_urlsafe(24))"
   WEB_PASSWORD=PASTE_GENERATED_VALUE_HERE

   # session-cookie signing key -- generate with:
   #   python -c "import secrets; print(secrets.token_urlsafe(48))"
   WEB_SESSION_SECRET=PASTE_64_PLUS_CHAR_VALUE_HERE
   ```

2. Make sure `.env` is **not** world-readable. On Windows, right-click
   → Properties → Security → restrict to your user. On POSIX:
   `chmod 600 .env`.

### Run on your laptop

```bash
# default: 127.0.0.1:8000 (laptop only)
python -m scripts.serve_web

# open http://localhost:8000 in any browser, log in, you should see
# today's signals + open paper positions + 30-day P&L.
```

### View it from your phone (same Wi-Fi)

```bash
# Bind on your LAN. Your phone, laptop, and tablet must be on the
# same Wi-Fi. NEVER do this on a network you don't control.
python -m scripts.serve_web --host 0.0.0.0
```

Then on your phone, open `http://<your-laptop-IP>:8000`. Find the IP
with `ipconfig` (Windows) — look for "IPv4 Address" under your Wi-Fi
adapter, e.g. `http://192.168.1.42:8000`.

> Why `--host 0.0.0.0` is gated to LAN-only: any public-Internet
> exposure must go through a TLS-terminating proxy (Cloudflare Tunnel
> in Phase C). Bare HTTP over the open Internet would leak your
> session cookie. The middleware emits HSTS + a strict CSP whenever
> the request scheme is `https`, so once Cloudflare is in front the
> browser will refuse to talk over HTTP.

### What's on each page

| Route            | Shows                                                  |
| ---------------- | ------------------------------------------------------ |
| `/`              | Today's signals, open paper positions, 30d P&L cards   |
| `/positions`     | Full open + closed paper trades                        |
| `/history`       | Last 120d of closed trades + summary stats             |
| `/report.pdf`    | Streams the most recent daily PDF report               |
| `/healthz`       | JSON freshness check (no auth, for cron/uptime probes) |
| `/api/snapshot`  | JSON view of `/` for native clients (auth required)    |

### Security defaults

- HTTP Basic-style login over a session cookie (HttpOnly, SameSite=Lax,
  Secure when behind TLS), signed with `itsdangerous`.
- Constant-time password compare (`hmac.compare_digest`) — same code
  path runs whether the username is right or wrong, so timing attacks
  can't enumerate usernames.
- Strict `Content-Security-Policy` (only the Tailwind CDN is allowed),
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`.
- In-memory rate limit (120 req/min per IP) so the public Internet
  can't trivially DoS the box. Cloudflare adds another layer in Phase C.
- All write paths are absent: the FastAPI app does not import any of
  the ingest / model / signal modules, so a bug in the dashboard
  cannot corrupt your trading data.

### Phase B / C / D (next)

| Phase | What                                                                  |
| ----- | --------------------------------------------------------------------- |
| B     | Dockerfile + docker-compose so it spins up with one command           |
| C     | Provision Oracle Cloud Free ARM VM, deploy via Cloudflare Tunnel      |
| D     | Plumb in Angel One SmartAPI for live data (the package is built)      |

---

## Troubleshooting

- **NSE 403 / 503**: NSE blocks scraping. The session helper warms cookies and
  retries; if persistent, increase `HTTP_MAX_RETRIES` and run from a
  residential / cloud IP rather than corporate proxy.
- **`database is locked`**: another writer is open. Close the dashboard
  (Week 7) or any open `sqlite3` shell. WAL mode keeps readers OK; only one
  writer at a time.
- **`yfinance returned empty`**: check the symbol suffix (`.NS`), and that the
  stock was listed in the requested range. Some delisted names are missing.
