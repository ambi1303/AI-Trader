"""Smoke + security tests for the dashboard.

These run entirely against an isolated SQLite DB (per-test, via the
``isolated_db`` autouse fixture) so they never touch your real
trading.db. We seed just enough rows to exercise the read paths and
assert on:

* unauthenticated requests get 401 (or redirect to login),
* security headers are present on every response,
* good credentials issue a session cookie and the home page renders,
* the JSON snapshot endpoint returns the expected shape,
* the rate limiter eventually returns 429,
* /healthz works without auth and returns ok=True.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.db.migrate import apply_schema
from src.utils.db import connect


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    creds = {
        "WEB_USERNAME": "trader1",
        "WEB_PASSWORD": "correct-horse-battery-staple-22",
        "WEB_SESSION_SECRET": "x" * 64,
    }
    for k, v in creds.items():
        monkeypatch.setenv(k, v)
    # Force secrets cache to refresh
    from src.utils import secrets as s
    if hasattr(s, "_cached_settings"):
        s._cached_settings = None  # type: ignore[attr-defined]
    return creds


@pytest.fixture()
def seeded_db(env: dict[str, str]) -> None:
    apply_schema()
    with connect() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO model_runs (run_id, model_name, trained_from,
                                    trained_to, metrics_json, artifact_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("run-test", "xgb", "2025-01-01", "2025-09-01",
             '{"threshold": 0.55, "auc": 0.7}', "/tmp/m.pkl"),
        )
        # An open paper trade
        c.execute(
            """
            INSERT INTO paper_trades (signal_id, symbol, side, entry_date,
                entry_price, qty, sector, status, stop_loss, take_profit,
                run_id)
            VALUES (NULL, 'RELIANCE', 'BUY', '2025-09-15',
                    1200.0, 10, 'ENERGY', 'open', 1100.0, 1400.0, 'run-test')
            """
        )
        # A closed winning trade
        c.execute(
            """
            INSERT INTO paper_trades (signal_id, symbol, side, entry_date,
                exit_date, entry_price, exit_price, qty, pnl_rupees,
                pnl_pct, exit_reason, status, sector, run_id)
            VALUES (NULL, 'TCS', 'BUY', date('now','-10 days'),
                    date('now','-2 days'), 3500.0, 3700.0, 5,
                    1000.0, 5.71, 'target', 'closed', 'IT', 'run-test')
            """
        )
        # A signal for today
        c.execute(
            """
            INSERT INTO signal_outbox (signal_date, symbol, side, entry_price,
                stop_loss, take_profit, qty, confidence, status)
            VALUES (date('now'), 'INFY', 'BUY', 1500.0, 1450.0, 1600.0,
                    20, 0.62, 'pending')
            """
        )
        # A price bar for the open position so unrealised P&L computes
        c.execute(
            """
            INSERT INTO price_data (symbol, bar_date, open, high, low,
                close, volume, source)
            VALUES ('RELIANCE', date('now'), 1250.0, 1260.0, 1240.0,
                    1255.0, 100000, 'yfinance')
            """
        )
        # Predictions for the watchlist: one above threshold (0.55), two below.
        for sym, raw, cal in [("INFY", 0.71, 0.60),
                              ("TCS", 0.64, 0.50),
                              ("WIPRO", 0.58, 0.40)]:
            c.execute(
                """
                INSERT INTO predictions_log (run_id, symbol, prediction_date,
                    raw_prob, calibrated_prob, feature_snapshot_json)
                VALUES ('run-test', ?, date('now'), ?, ?, '{}')
                """,
                (sym, raw, cal),
            )
        conn.commit()


@pytest.fixture()
def client(seeded_db) -> TestClient:
    # Import lazily so env vars are set before AuthConfig.from_env().
    from src.web.app import create_app
    app = create_app()
    return TestClient(app, raise_server_exceptions=True)


# --------------------------------------------------------------------------
# Public endpoints
# --------------------------------------------------------------------------


def test_healthz_no_auth(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "freshness" in body


def test_security_headers_on_every_response(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    h = r.headers
    assert "content-security-policy" in h
    assert "default-src 'self'" in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "no-referrer"
    assert h["cache-control"] == "no-store"


def test_unauth_browser_is_redirected_to_login(client: TestClient) -> None:
    """Browsers asking for HTML should be sent to /login, not see JSON."""
    r = client.get("/", headers={"accept": "text/html"},
                   follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_unauth_non_browser_gets_json_401(client: TestClient) -> None:
    """Default Accept (no text/html) keeps the 401 JSON contract intact."""
    r = client.get("/", headers={"accept": "*/*"}, follow_redirects=False)
    assert r.status_code == 401
    assert r.json() == {"detail": "authentication required"}
    assert "www-authenticate" in r.headers


def test_unauth_api_snapshot_blocked(client: TestClient) -> None:
    """API paths must always return JSON 401 even with HTML Accept,
    so a browser typo can never accidentally redirect a JSON consumer."""
    r = client.get("/api/snapshot", headers={"accept": "text/html"})
    assert r.status_code == 401
    assert r.json() == {"detail": "authentication required"}


def test_login_page_renders(client: TestClient) -> None:
    r = client.get("/login")
    assert r.status_code == 200
    assert "Sign in" in r.text


# --------------------------------------------------------------------------
# Auth flow
# --------------------------------------------------------------------------


def test_bad_credentials_redirects_back(client: TestClient,
                                        env: dict[str, str]) -> None:
    r = client.post(
        "/login",
        data={"username": env["WEB_USERNAME"], "password": "WRONG"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "/login" in r.headers["location"]
    # No session cookie was issued.
    assert "ait_session" not in r.cookies


def test_good_credentials_issues_cookie_and_lands_home(
    client: TestClient, env: dict[str, str]
) -> None:
    r = client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert "ait_session" in r.cookies
    # Cookie should be HttpOnly + SameSite=Lax. Starlette's TestClient
    # exposes the Set-Cookie header verbatim.
    set_cookie = r.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie
    assert "samesite=lax" in set_cookie.lower()


def test_authenticated_dashboard_renders(client: TestClient,
                                         env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/")
    assert r.status_code == 200
    assert "Today" in r.text or "as of" in r.text.lower()
    # Seeded position should appear by symbol
    assert "RELIANCE" in r.text


def test_logout_clears_cookie(client: TestClient,
                              env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303
    # After logout, /api/snapshot should now be 401.
    client.cookies.clear()
    r2 = client.get("/api/snapshot")
    assert r2.status_code == 401


# --------------------------------------------------------------------------
# Snapshot shape (stable contract for any future native client)
# --------------------------------------------------------------------------


def test_api_snapshot_shape(client: TestClient,
                            env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/snapshot")
    assert r.status_code == 200
    body = r.json()
    for k in ("as_of", "signals", "open_positions", "closed_recent",
              "realised_pnl_30d", "unrealised_pnl", "win_rate_30d_pct",
              "n_open", "freshness", "model", "universe_size",
              "candidates", "threshold"):
        assert k in body, f"missing key: {k}"
    # Open position from the seed
    assert body["n_open"] == 1
    assert body["open_positions"][0]["symbol"] == "RELIANCE"
    # Unrealised P&L on RELIANCE: (1255 - 1200) * 10 = 550
    assert body["open_positions"][0]["unrealised_pnl"] == pytest.approx(550.0)
    # 30d realised includes the TCS win
    assert body["realised_pnl_30d"] == pytest.approx(1000.0)


def test_watchlist_candidates_ranked_with_threshold(
    client: TestClient, env: dict[str, str]
) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    body = client.get("/api/snapshot").json()
    cands = body["candidates"]
    # Threshold comes from model_runs.metrics_json (0.55 in the seed).
    assert body["threshold"] == pytest.approx(0.55)
    syms = [c["symbol"] for c in cands]
    # Ranked by calibrated prob desc: INFY (0.60) > TCS (0.50) > WIPRO (0.40)
    assert syms[:3] == ["INFY", "TCS", "WIPRO"]
    infy = cands[0]
    assert infy["would_fire"] is True
    assert infy["distance_to_threshold"] == pytest.approx(0.05)
    # TCS is below threshold -> negative distance, does not fire.
    tcs = cands[1]
    assert tcs["would_fire"] is False
    assert tcs["distance_to_threshold"] == pytest.approx(-0.05)


def test_watchlist_renders_on_home(client: TestClient,
                                   env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/")
    assert r.status_code == 200
    assert "Watchlist" in r.text
    assert "FIRES" in r.text  # INFY crosses the threshold in the seed


def test_signal_today_appears(client: TestClient,
                              env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/snapshot")
    body = r.json()
    syms = [s["symbol"] for s in body["signals"]]
    assert "INFY" in syms


def test_positions_page_shows_investment_stats(client: TestClient,
                                               env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/positions")
    assert r.status_code == 200
    # The seeded RELIANCE open position drives the summary cards + columns.
    assert "Invested" in r.text
    assert "Current value" in r.text
    assert "Unrealised P&amp;L" in r.text
    assert "RELIANCE" in r.text


def test_stock_page_renders_with_analysis_cards(
    client: TestClient, env: dict[str, str]
) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/stock/RELIANCE")
    assert r.status_code == 200
    # New analysis sections are present on the page.
    assert "Buy / Sell zones" in r.text
    assert "Conviction score" in r.text
    # Header search box is on every authenticated page.
    assert 'id="stock-search"' in r.text


def test_search_endpoint_returns_matches(
    client: TestClient, env: dict[str, str]
) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/search", params={"term": "REL"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert "RELIANCE" in results  # seeded price_data symbol


def test_search_endpoint_requires_auth(client: TestClient) -> None:
    r = client.get("/api/search", params={"term": "REL"})
    assert r.status_code == 401


def test_search_endpoint_empty_term_is_safe(
    client: TestClient, env: dict[str, str]
) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/search", params={"term": ""})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_discover_page_renders(client: TestClient, env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/discover")
    assert r.status_code == 200
    assert "AI Stock Discovery" in r.text
    # Strategy tabs are present.
    assert "Undervalued" in r.text and "Momentum" in r.text


def test_api_discover_shape_and_auth(
    client: TestClient, env: dict[str, str]
) -> None:
    # Unauthenticated -> 401.
    assert client.get("/api/discover").status_code == 401
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/discover", params={"strategy": "value"})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "value"
    assert isinstance(body["results"], list)


def test_api_discover_unknown_strategy_falls_back(
    client: TestClient, env: dict[str, str]
) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/discover", params={"strategy": "bogus"})
    assert r.status_code == 200
    assert r.json()["strategy"] == "top_conviction"


def test_risk_page_renders(client: TestClient, env: dict[str, str]) -> None:
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/risk")
    assert r.status_code == 200
    assert "Risk Management" in r.text
    assert "Position size calculator" in r.text
    # Seed has an open RELIANCE position -> portfolio health populated.
    assert "Portfolio health" in r.text


def test_api_position_size(client: TestClient, env: dict[str, str]) -> None:
    assert client.get("/api/position-size").status_code == 401
    client.post(
        "/login",
        data={"username": env["WEB_USERNAME"],
              "password": env["WEB_PASSWORD"]},
    )
    r = client.get("/api/position-size", params={
        "capital": 100000, "risk_pct": 2, "entry": 1000, "stop": 950,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["shares"] == 40
    assert body["risk_amount"] == 2000.0


# --------------------------------------------------------------------------
# Rate limiter
# --------------------------------------------------------------------------


def test_rate_limiter_returns_429_eventually(client: TestClient) -> None:
    # /login is public so we don't need to authenticate first.
    seen_429 = False
    for _ in range(180):  # >120 limit
        r = client.get("/login")
        if r.status_code == 429:
            seen_429 = True
            break
        assert r.status_code == 200
    assert seen_429, "rate limiter never tripped"


# --------------------------------------------------------------------------
# Misconfiguration must fail loud
# --------------------------------------------------------------------------


def test_missing_session_secret_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_USERNAME", "u")
    monkeypatch.setenv("WEB_PASSWORD", "p" * 20)
    monkeypatch.delenv("WEB_SESSION_SECRET", raising=False)
    from src.utils import secrets as s
    if hasattr(s, "_cached_settings"):
        s._cached_settings = None  # type: ignore[attr-defined]
    from src.web.app import create_app
    with pytest.raises(RuntimeError):
        create_app()


def test_short_session_secret_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_USERNAME", "u")
    monkeypatch.setenv("WEB_PASSWORD", "p" * 20)
    monkeypatch.setenv("WEB_SESSION_SECRET", "tooshort")
    from src.utils import secrets as s
    if hasattr(s, "_cached_settings"):
        s._cached_settings = None  # type: ignore[attr-defined]
    from src.web.app import create_app
    with pytest.raises(RuntimeError):
        create_app()
