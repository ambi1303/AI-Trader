"""Fundamental data ingestion via yfinance (free).

Two complementary sources land in ``fundamental_data``:

1. **Live snapshot** (``source='yfinance_snapshot'``) -- the current
   ``Ticker.info`` valuation/quality/growth ratios, stamped with today's
   date. This is what the dashboard shows for "now".

2. **Reconstructed history** (``source='yfinance_reconstructed'``) -- we
   walk the quarterly income statement + balance sheet and, at each
   quarter-end report date, compute trailing-twelve-month (TTM) EPS, book
   value per share, margins, growth and leverage. Combined with the close
   on that date (from ``price_data``) we derive a *point-in-time* P/E and
   P/B. This gives the model a few years of fundamentals that were
   genuinely knowable at each historical date (no look-ahead), which is
   the honest way to train on valuation factors.

Free-data caveat: yfinance only exposes ~4-5 recent quarters of
statements, so reconstructed fundamental history is shallow (~1-2 years).
Technical/regime features keep their full depth; fundamentals are
forward-filled between report dates by the feature builder.

Security / safety:
* Read-only HTTP via yfinance; no secrets, no order placement.
* Every numeric field passes through ``_num`` which coerces NaN/inf and
  bad types to ``None`` so we never persist garbage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import date

import pandas as pd
import yfinance as yf

from src.utils.db import fetch_all, transaction
from src.utils.logger import get_logger

log = get_logger("ingest.fundamentals")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FundamentalSnapshot:
    """One row destined for ``fundamental_data``."""
    symbol: str
    as_of_date: str                      # ISO YYYY-MM-DD
    source: str                          # yfinance_snapshot | yfinance_reconstructed
    pe_ttm: float | None = None
    pb: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    dividend_yield: float | None = None
    market_cap: float | None = None
    eps_ttm: float | None = None
    book_value: float | None = None


@dataclass(frozen=True)
class FundamentalIngestSummary:
    requested: int
    snapshots: int          # live snapshot rows written
    reconstructed: int      # historical reconstructed rows written
    upserted: int           # total rows upserted
    failed_symbols: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_yf_symbol(symbol: str) -> str:
    return f"{symbol}.NS"


def _num(x) -> float | None:
    """Coerce to a finite float or None (handles NaN/inf/str/bad types)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _norm_ratio_pct(x) -> float | None:
    """Normalise debtToEquity, which yfinance reports as a percent.

    debtToEquity is commonly reported as e.g. ``45.3`` (meaning a 0.45 ratio).
    We treat anything > 1.5 as a percent and divide by 100 so it is a clean
    ratio/fraction downstream. (Real D/E ratios below 1.5 are left as-is.)
    """
    v = _num(x)
    if v is None:
        return None
    return v / 100.0 if v > 1.5 else v


def _div_yield_to_fraction(x) -> float | None:
    """Convert yfinance's dividendYield to a fraction (consistent with roe/margin).

    yfinance reports dividendYield as a PERCENT number, unlike roe/profitMargins
    which come as fractions. Verified against yfinance 0.2.59:
        HDFCBANK -> 1.66 (1.66%), RELIANCE -> 0.45 (0.45%), ITC -> 5.49 (5.49%).
    So we always divide by 100 to store a fraction; the UI then multiplies every
    ratio by 100 uniformly. The old >1.5 heuristic mis-handled sub-1.5% yields
    (e.g. RELIANCE 0.45 became 45%).
    """
    v = _num(x)
    if v is None:
        return None
    return v / 100.0


def _build_session():
    """A curl_cffi Chrome-impersonating session (Yahoo fingerprints clients)."""
    try:
        from curl_cffi import requests as crequests

        return crequests.Session(impersonate="chrome")
    except Exception as e:  # noqa: BLE001
        log.warning("curl_cffi unavailable ({}); using default yfinance", str(e))
        return None


def _load_close_series(symbol: str) -> pd.Series:
    """Daily close for ``symbol`` (bhavcopy preferred, yfinance fallback)."""
    rows = fetch_all(
        """
        SELECT bar_date, close, source FROM price_data
        WHERE  symbol = ? AND source IN ('bhavcopy', 'yfinance')
        ORDER BY bar_date
        """,
        (symbol,),
    )
    if not rows:
        return pd.Series(dtype="float64")
    df = pd.DataFrame([dict(r) for r in rows])
    # Prefer bhavcopy where both exist for a date.
    df["pref"] = (df["source"] == "bhavcopy").astype(int)
    df = (
        df.sort_values(["bar_date", "pref"])
        .drop_duplicates("bar_date", keep="last")
    )
    s = pd.Series(
        df["close"].astype(float).to_numpy(),
        index=pd.to_datetime(df["bar_date"]),
    ).sort_index()
    return s


def _close_asof(close: pd.Series, when: date) -> float | None:
    """Most recent close at or before ``when`` (no look-ahead)."""
    if close.empty:
        return None
    ts = pd.Timestamp(when)
    sub = close.loc[:ts]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def _stmt_row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    """First matching row (by label) from a yfinance statement DataFrame.

    Statements are indexed by line-item name with columns = period-end
    dates (most recent first). We try several aliases since yfinance label
    spelling varies by ticker.
    """
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


# ---------------------------------------------------------------------------
# Live snapshot
# ---------------------------------------------------------------------------


def fetch_snapshot(symbol: str, info: dict) -> FundamentalSnapshot:
    """Build today's snapshot from a yfinance ``info`` dict."""
    return FundamentalSnapshot(
        symbol=symbol,
        as_of_date=date.today().isoformat(),
        source="yfinance_snapshot",
        pe_ttm=_num(info.get("trailingPE")),
        pb=_num(info.get("priceToBook")),
        roe=_num(info.get("returnOnEquity")),
        debt_to_equity=_norm_ratio_pct(info.get("debtToEquity")),
        profit_margin=_num(info.get("profitMargins")),
        revenue_growth=_num(info.get("revenueGrowth")),
        earnings_growth=_num(
            info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
        ),
        dividend_yield=_div_yield_to_fraction(info.get("dividendYield")),
        market_cap=_num(info.get("marketCap")),
        eps_ttm=_num(info.get("trailingEps")),
        book_value=_num(info.get("bookValue")),
    )


# ---------------------------------------------------------------------------
# Reconstructed point-in-time history
# ---------------------------------------------------------------------------


def reconstruct_history(
    symbol: str,
    *,
    income: pd.DataFrame | None,
    balance: pd.DataFrame | None,
    shares: float | None,
    close: pd.Series,
    anchor_eps: float | None = None,
    anchor_book_value: float | None = None,
) -> list[FundamentalSnapshot]:
    """Reconstruct TTM fundamentals at each quarter-end report date.

    For each quarter-end ``q`` with at least 4 trailing quarters available
    we compute TTM net income / revenue (sum of 4 quarters), TTM EPS and
    book value per share, then derive P/E and P/B against ``close`` as-of
    ``q``. Growth is the TTM-vs-prior-year-TTM change where 8 quarters
    exist.

    Scale anchoring: yfinance frequently returns Indian statement values on
    a different scale than ``sharesOutstanding`` (off by ~100x for some
    tickers), which corrupts raw EPS / book-value-per-share and therefore
    P/E and P/B. The *ratios* (ROE, margin, D/E, growth) are scale-free and
    stay correct. To make P/E and P/B trustworthy and cross-stock
    comparable we anchor: scale all reconstructed EPS so the latest matches
    the broker-reported ``anchor_eps`` (``trailingEps`` from .info), and
    likewise for book value vs ``anchor_book_value``. Historical points then
    carry the statement trajectory on a correct absolute scale.
    """
    net_income = _stmt_row(income, "Net Income", "Net Income Common Stockholders")
    revenue = _stmt_row(income, "Total Revenue", "Operating Revenue")
    equity = _stmt_row(
        balance, "Stockholders Equity", "Common Stock Equity",
        "Total Stockholder Equity",
    )
    total_debt = _stmt_row(balance, "Total Debt", "Net Debt")

    if net_income is None or len(net_income) < 4:
        return []

    # Quarter-end dates sorted oldest -> newest.
    q_dates = sorted(pd.to_datetime(net_income.index))
    out: list[FundamentalSnapshot] = []

    def _series_at(s: pd.Series | None, ts) -> float | None:
        if s is None:
            return None
        try:
            return _num(s.get(ts))
        except Exception:  # noqa: BLE001
            return None

    ni = {pd.to_datetime(k): _num(v) for k, v in net_income.items()}
    rev = {pd.to_datetime(k): _num(v) for k, v in revenue.items()} if revenue is not None else {}

    # Pass 1: compute raw (un-anchored) TTM figures per quarter.
    raw: list[dict] = []
    for i, q in enumerate(q_dates):
        if i < 3:
            continue  # need 4 trailing quarters for a TTM figure
        window = q_dates[i - 3: i + 1]
        ni_vals = [ni.get(d) for d in window]
        if any(v is None for v in ni_vals):
            continue
        ttm_ni = float(sum(ni_vals))

        rev_vals = [rev.get(d) for d in window] if rev else []
        ttm_rev = float(sum(rev_vals)) if rev_vals and all(v is not None for v in rev_vals) else None

        eq_q = _series_at(equity, q)
        debt_q = _series_at(total_debt, q)

        eps_raw = (ttm_ni / shares) if (shares and shares > 0) else None
        bvps_raw = (eq_q / shares) if (eq_q and shares and shares > 0) else None

        roe = (ttm_ni / eq_q) if (eq_q and eq_q != 0) else None
        d2e = (debt_q / eq_q) if (debt_q is not None and eq_q and eq_q != 0) else None
        margin = (ttm_ni / ttm_rev) if (ttm_rev and ttm_rev != 0) else None

        rev_growth = None
        earn_growth = None
        if i >= 7:
            prior = q_dates[i - 7: i - 3]
            prior_ni = [ni.get(d) for d in prior]
            if all(v is not None for v in prior_ni):
                base = float(sum(prior_ni))
                if base != 0:
                    earn_growth = ttm_ni / base - 1.0
            if rev:
                prior_rev = [rev.get(d) for d in prior]
                if all(v is not None for v in prior_rev) and ttm_rev is not None:
                    base_r = float(sum(prior_rev))
                    if base_r != 0:
                        rev_growth = ttm_rev / base_r - 1.0

        raw.append({
            "q": q, "eps_raw": eps_raw, "bvps_raw": bvps_raw,
            "roe": roe, "d2e": d2e, "margin": margin,
            "rev_growth": rev_growth, "earn_growth": earn_growth,
        })

    if not raw:
        return []

    # Derive per-stock scale factors so the latest reconstructed EPS / BVPS
    # match the broker snapshot (fixes yfinance unit mismatches and keeps
    # P/E, P/B cross-stock comparable).
    def _scale(anchor, key) -> float | None:
        if anchor is None or anchor <= 0:
            return None
        latest = next((r[key] for r in reversed(raw) if r[key] and r[key] > 0), None)
        if latest is None or latest <= 0:
            return None
        return anchor / latest

    eps_scale = _scale(anchor_eps, "eps_raw")
    bvps_scale = _scale(anchor_book_value, "bvps_raw")

    # Pass 2: apply anchoring + price to emit final snapshots.
    for r in raw:
        q = r["q"]
        px = _close_asof(close, q.date())
        eps_ttm = (r["eps_raw"] * eps_scale) if (r["eps_raw"] and eps_scale) else None
        bvps = (r["bvps_raw"] * bvps_scale) if (r["bvps_raw"] and bvps_scale) else None
        pe = (px / eps_ttm) if (px and eps_ttm and eps_ttm > 0) else None
        pb = (px / bvps) if (px and bvps and bvps > 0) else None
        # sharesOutstanding is on the correct scale (it was the statement
        # values that were mis-scaled), so market cap is simply price*shares.
        mkt_cap = (px * shares) if (px and shares and shares > 0) else None

        out.append(FundamentalSnapshot(
            symbol=symbol,
            as_of_date=q.date().isoformat(),
            source="yfinance_reconstructed",
            pe_ttm=pe,
            pb=pb,
            roe=r["roe"],
            debt_to_equity=r["d2e"],
            profit_margin=r["margin"],
            revenue_growth=r["rev_growth"],
            earnings_growth=r["earn_growth"],
            dividend_yield=None,             # not reconstructable from statements
            market_cap=mkt_cap,
            eps_ttm=eps_ttm,
            book_value=bvps,
        ))
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_PERSIST_COLS = [
    "pe_ttm", "pb", "roe", "debt_to_equity", "profit_margin",
    "revenue_growth", "earnings_growth", "dividend_yield",
    "market_cap", "eps_ttm", "book_value",
]


def _persist(snapshots: list[FundamentalSnapshot]) -> int:
    if not snapshots:
        return 0
    insert_cols = ["symbol", "as_of_date", *_PERSIST_COLS, "source"]
    placeholders = ",".join(["?"] * len(insert_cols))
    update_set = ", ".join(
        f"{c}=excluded.{c}" for c in [*_PERSIST_COLS, "source"]
    )
    sql = (
        f"INSERT INTO fundamental_data ({', '.join(insert_cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(symbol, as_of_date) DO UPDATE SET {update_set}"
    )
    rows = [
        (
            s.symbol, s.as_of_date,
            *[getattr(s, c) for c in _PERSIST_COLS],
            s.source,
        )
        for s in snapshots
    ]
    with transaction() as conn:
        conn.executemany(sql, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ingest_fundamentals(
    symbols: list[str],
    *,
    persist: bool = True,
    reconstruct: bool = True,
) -> FundamentalIngestSummary:
    """Fetch + (optionally) persist fundamentals for ``symbols``.

    Returns a tally for the orchestrator / smoke tests.
    """
    session = _build_session()
    requested = 0
    n_snap = 0
    n_recon = 0
    n_upsert = 0
    failed: list[str] = []

    for sym in symbols:
        sym = sym.upper().strip()
        requested += 1
        batch: list[FundamentalSnapshot] = []
        try:
            tk = yf.Ticker(_to_yf_symbol(sym), session=session)
            info = {}
            try:
                info = tk.info or {}
            except Exception as exc:  # noqa: BLE001
                log.warning("info fetch failed for {}: {}", sym, exc)

            if info:
                batch.append(fetch_snapshot(sym, info))
                n_snap += 1

            if reconstruct:
                shares = _num(info.get("sharesOutstanding")) if info else None
                anchor_eps = _num(info.get("trailingEps")) if info else None
                anchor_bv = _num(info.get("bookValue")) if info else None
                close = _load_close_series(sym)
                try:
                    hist = reconstruct_history(
                        sym,
                        income=tk.quarterly_income_stmt,
                        balance=tk.quarterly_balance_sheet,
                        shares=shares,
                        close=close,
                        anchor_eps=anchor_eps,
                        anchor_book_value=anchor_bv,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("history reconstruct failed for {}: {}", sym, exc)
                    hist = []
                batch.extend(hist)
                n_recon += len(hist)

            if not batch:
                failed.append(sym)
                continue

            if persist:
                n_upsert += _persist(batch)
        except Exception as exc:  # noqa: BLE001
            log.error("Fundamental ingest failed for {}: {}", sym, exc)
            failed.append(sym)

    log.info(
        "Fundamentals | requested={} snapshots={} reconstructed={} "
        "upserted={} failed={}",
        requested, n_snap, n_recon, n_upsert, len(failed),
    )
    return FundamentalIngestSummary(
        requested=requested,
        snapshots=n_snap,
        reconstructed=n_recon,
        upserted=n_upsert,
        failed_symbols=failed,
    )


# Keep dataclass field list authoritative for tests / introspection.
SNAPSHOT_FIELDS = tuple(f.name for f in fields(FundamentalSnapshot))
