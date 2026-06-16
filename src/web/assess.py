"""Human-readable interpretation helpers for the web UI.

This module turns raw model/fundamental numbers into plain-language labels,
health ratings, and warnings so a non-quant customer can read a stock page at
a glance. It is presentation-only -- it makes NO trading decisions and never
writes to the DB. All thresholds are deliberately conservative and documented
inline; tweak them here in one place rather than scattering magic numbers
through the templates.

Three concerns live here:
  1. Fundamental health  -> ("Healthy" / "Average" / "Stretched", tone)
  2. Fundamentals staleness -> how old the snapshot is + a tone
  3. Cost-band gate      -> is the predicted move bigger than round-trip cost?
  4. Plain-language verdict summary sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any

# Tone vocabulary shared with the templates -> Tailwind colour classes.
#   "good"  -> emerald   "ok" -> slate/amber   "warn" -> amber   "bad" -> rose
Tone = str  # one of: good | ok | warn | bad | neutral


@dataclass(frozen=True)
class Rating:
    """A single interpreted metric: the formatted value plus a verdict."""

    label: str          # e.g. "Healthy", "Stretched", "Below average"
    tone: Tone          # good | ok | warn | bad | neutral
    note: str = ""      # optional one-line explanation for a tooltip


# ---------------------------------------------------------------------------
# 1. Fundamental health
# ---------------------------------------------------------------------------
# Bands are intentionally broad and India-large-cap oriented. They are a
# readability aid, NOT a valuation model. "neutral" is returned when a value
# is missing so the UI can grey it out instead of implying a judgement.


def rate_pe(pe: float | None) -> Rating:
    if pe is None:
        return Rating("—", "neutral")
    if pe <= 0:
        return Rating("Loss-making", "warn", "Negative earnings (P/E undefined)")
    if pe < 15:
        return Rating("Cheap", "good", "Low earnings multiple")
    if pe < 30:
        return Rating("Fair", "ok", "Reasonable earnings multiple")
    if pe < 60:
        return Rating("Rich", "warn", "High earnings multiple — priced for growth")
    return Rating("Very rich", "bad", "Extreme multiple — fragile to disappointment")


def rate_roe(roe: float | None) -> Rating:
    # roe is a fraction (0.18 = 18%).
    if roe is None:
        return Rating("—", "neutral")
    pct = roe * 100
    if pct >= 20:
        return Rating("Excellent", "good", "Strong return on equity")
    if pct >= 12:
        return Rating("Good", "good")
    if pct >= 6:
        return Rating("Average", "ok")
    return Rating("Weak", "warn", "Low profitability on equity")


def rate_debt_to_equity(de: float | None) -> Rating:
    if de is None:
        return Rating("—", "neutral")
    if de < 0:
        return Rating("—", "neutral")
    if de <= 0.5:
        return Rating("Low debt", "good", "Conservative balance sheet")
    if de <= 1.0:
        return Rating("Moderate", "ok")
    if de <= 2.0:
        return Rating("Leveraged", "warn", "Meaningful debt load")
    return Rating("High debt", "bad", "Heavily leveraged — sensitive to rates")


def rate_margin(margin: float | None) -> Rating:
    if margin is None:
        return Rating("—", "neutral")
    pct = margin * 100
    if pct >= 20:
        return Rating("High", "good")
    if pct >= 8:
        return Rating("Healthy", "good")
    if pct >= 0:
        return Rating("Thin", "ok")
    return Rating("Negative", "bad", "Loss-making at the net level")


def rate_growth(g: float | None) -> Rating:
    if g is None:
        return Rating("—", "neutral")
    pct = g * 100
    if pct >= 15:
        return Rating("Strong", "good")
    if pct >= 5:
        return Rating("Steady", "ok")
    if pct >= 0:
        return Rating("Flat", "ok")
    return Rating("Shrinking", "warn", "Declining vs a year ago")


# ---------------------------------------------------------------------------
# 2. Fundamentals staleness
# ---------------------------------------------------------------------------


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def staleness_days(as_of_date: str | None, *, today: str | None = None) -> int | None:
    """Calendar days between the fundamental snapshot and `today`."""
    aod = _parse_iso(as_of_date)
    if aod is None:
        return None
    ref = _parse_iso(today) or date.today()
    return max(0, (ref - aod).days)


def rate_staleness(days: int | None) -> Rating:
    """yfinance fundamentals are quarterly. Up to ~100 days is normal (one
    reporting cycle); beyond ~200 days the company has likely reported since
    and our snapshot is genuinely stale."""
    if days is None:
        return Rating("Unknown age", "neutral")
    if days <= 100:
        return Rating(f"{days}d old", "good", "Within the current quarter")
    if days <= 200:
        return Rating(f"{days}d old", "ok", "Around one reporting cycle old")
    if days <= 400:
        return Rating(f"{days}d old", "warn", "Likely a fresher report exists")
    return Rating(f"{days}d old", "bad", "Stale — treat fundamentals with caution")


# ---------------------------------------------------------------------------
# 3. Cost-band gate
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def round_trip_cost_pct() -> float:
    """Representative round-trip trading cost as a percentage (e.g. 0.29).

    Uses the same cost model as the backtester / threshold tuner so the UI
    can't disagree with the engine about what a trade costs. Cached because
    it only depends on static config.
    """
    try:
        from src.backtesting.cost_model import load_cost_config, round_trip_pct
        cfg = load_cost_config()
        return round_trip_pct("__REPRESENTATIVE__", cfg) * 100.0
    except Exception:  # noqa: BLE001 -- UI must never crash on a config issue
        # Fallback to the documented delivery estimate (~0.3%).
        return 0.30


def is_marginal_move(upside_pct: float | None, *, buffer: float = 1.5) -> bool:
    """True when the predicted move is too small to clear trading costs.

    `buffer` requires the expected move to beat cost by a margin (1.5x) before
    we call it tradeable, since the prediction itself is uncertain.
    """
    if upside_pct is None:
        return False
    return abs(upside_pct) < round_trip_cost_pct() * buffer


# ---------------------------------------------------------------------------
# 4. Plain-language verdict summary
# ---------------------------------------------------------------------------


def _fmt_money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"\u20b9{v:,.2f}"


def verdict_summary(detail: dict[str, Any]) -> dict[str, Any]:
    """Build a one/two-sentence, customer-readable summary of the call.

    Returns a dict the template renders directly:
        headline : short sentence
        detail   : longer sentence with target/stop/horizon
        tone     : good | ok | warn | bad | neutral  (drives the banner colour)
        marginal : bool (predicted move < cost band)
    """
    pred = detail.get("prediction") or {}
    symbol = detail.get("symbol", "This stock")
    verdict = (pred.get("verdict") or "").upper()
    upside = detail.get("upside_pct")
    target = pred.get("target_price")
    stop = pred.get("stop_price")
    pred_ret = pred.get("predicted_return")

    if not verdict:
        return {
            "headline": f"No current model call for {symbol}.",
            "detail": "Run the daily prediction to populate a verdict, "
                      "target and stop for this stock.",
            "tone": "neutral",
            "marginal": False,
        }

    # Confidence of the winning class.
    probs = {
        "BUY": pred.get("prob_buy") or 0.0,
        "HOLD": pred.get("prob_hold") or 0.0,
        "SELL": pred.get("prob_sell") or 0.0,
    }
    conf = probs.get(verdict, 0.0) * 100

    tone = {"BUY": "good", "HOLD": "ok", "SELL": "bad"}.get(verdict, "neutral")
    verb = {
        "BUY": "rates this a BUY",
        "HOLD": "suggests HOLD / wait",
        "SELL": "flags SELL (exit / avoid)",
    }.get(verdict, "has no clear call")

    headline = f"The model {verb} with {conf:.0f}% confidence."

    parts: list[str] = []
    if pred_ret is not None:
        parts.append(f"Expected move ~{pred_ret * 100:+.1f}% over about a month")
    if target is not None:
        upside_txt = f" ({upside:+.1f}%)" if upside is not None else ""
        parts.append(f"to a target of {_fmt_money(target)}{upside_txt}")
    if stop is not None:
        parts.append(f"with a suggested stop near {_fmt_money(stop)}")
    detail_sentence = (", ".join(parts) + ".") if parts else ""

    marginal = is_marginal_move(upside)
    return {
        "headline": headline,
        "detail": detail_sentence,
        "tone": tone,
        "marginal": marginal,
    }


def fundamental_ratings(f: dict[str, Any] | None) -> dict[str, Rating]:
    """Bundle every fundamental rating for the template in one call."""
    f = f or {}
    return {
        "pe": rate_pe(f.get("pe_ttm")),
        "roe": rate_roe(f.get("roe")),
        "debt_to_equity": rate_debt_to_equity(f.get("debt_to_equity")),
        "profit_margin": rate_margin(f.get("profit_margin")),
        "revenue_growth": rate_growth(f.get("revenue_growth")),
        "earnings_growth": rate_growth(f.get("earnings_growth")),
    }
