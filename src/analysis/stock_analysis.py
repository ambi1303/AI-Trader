"""Pure stock-analysis engine: technicals, conviction score, buy/sell zones.

Everything here is a pure function of its inputs (a price-history DataFrame and
an optional fundamentals dict). No DB, no network, no global state -- which makes
it trivial to unit-test and safe to call inside a web request.

Inputs
------
``df``: a DataFrame indexed/ordered by date with float columns
    ``open, high, low, close, volume`` (oldest row first).
``fundamentals``: optional dict with the same keys we store in
    ``fundamental_data`` (pe_ttm, pb, roe, debt_to_equity, profit_margin,
    revenue_growth, earnings_growth, ...). All ratios are fractions
    (0.18 == 18%), consistent with the rest of the app.

Outputs (all JSON-serialisable, template-friendly)
------
* ``compute_technicals`` -> readable technical read (trend, RSI, MACD, EMAs,
  ADX strength, ATR, support/resistance, volume trend, 52-week position).
* ``conviction_score`` -> {"overall": 0-100, "factors": [...], "reasons": [...]}
  -- a transparent weighted blend across Fundamentals / Valuation / Technicals /
  Momentum / Risk, renormalised over whatever factors are available.
* ``buy_sell_zones`` -> volatility (ATR) based entry/target/stop bands + R:R.
* ``rule_based_verdict`` -> BUY/HOLD/SELL + confidence derived from conviction
  (used for stocks outside the trained model's universe; clearly labelled).
* ``analyze`` -> orchestrates the above into one bundle.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.features import technical_indicators as ti

Tone = str  # good | ok | warn | bad | neutral


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _last(series: pd.Series | None) -> float | None:
    """Last finite value of a series, or None."""
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return None if (math.isnan(v) or math.isinf(v)) else v


def _pct(a: float | None, b: float | None) -> float | None:
    """(a/b - 1) * 100, guarded."""
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 1. Technical read
# ---------------------------------------------------------------------------


def compute_technicals(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a readable technical snapshot from an OHLCV DataFrame.

    Returns a dict of raw numbers plus labelled interpretations. Missing/short
    data degrades gracefully to ``None`` fields rather than raising.
    """
    out: dict[str, Any] = {"available": False}
    if df is None or df.empty or "close" not in df:
        return out

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    volume = pd.to_numeric(df.get("volume", pd.Series(index=df.index, dtype=float)),
                           errors="coerce")
    if close.dropna().shape[0] < 20:
        return out  # too little history to say anything honest

    last_close = _last(close)
    rsi = _last(ti.rsi(close, 14))
    macd_df = ti.macd(close)
    macd_v = _last(macd_df["macd"])
    macd_sig = _last(macd_df["macd_signal"])
    macd_hist = _last(macd_df["macd_hist"])

    ema20 = _last(ti.ema(close, 20))
    ema50 = _last(ti.ema(close, 50))
    ema200 = _last(ti.ema(close, 200))

    adx_df = ti.adx(high, low, close, 14)
    adx = _last(adx_df["adx_14"])

    atr = _last(ti.atr(high, low, close, 14))
    atr_pct = _pct(last_close + atr, last_close) if (atr and last_close) else None

    # Support / resistance: recent swing levels over ~3 months, plus 52w extremes.
    lookback = min(len(close), 63)
    support = float(low.tail(lookback).min()) if lookback else None
    resistance = float(high.tail(lookback).max()) if lookback else None
    hi_52w = float(high.tail(252).max()) if len(high) else None
    lo_52w = float(low.tail(252).min()) if len(low) else None

    # Volume trend vs its 20-day average.
    vol_last = _last(volume)
    vol_avg20 = _last(volume.rolling(20, min_periods=5).mean())
    vol_ratio = (vol_last / vol_avg20) if (vol_last and vol_avg20) else None

    # Returns / momentum.
    ret_20d = _pct(last_close, _last(close.shift(20)))
    ret_60d = _pct(last_close, _last(close.shift(60)))

    # ---- Interpretations ----
    trend_label, trend_tone = _trend_read(last_close, ema50, ema200, adx)
    rsi_label, rsi_tone = _rsi_read(rsi)
    macd_label, macd_tone = _macd_read(macd_hist)
    strength_label = _adx_read(adx)
    vol_label = _volume_read(vol_ratio)

    out.update({
        "available": True,
        "last_close": last_close,
        "rsi": rsi, "rsi_label": rsi_label, "rsi_tone": rsi_tone,
        "macd": macd_v, "macd_signal": macd_sig, "macd_hist": macd_hist,
        "macd_label": macd_label, "macd_tone": macd_tone,
        "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "adx": adx, "trend_strength": strength_label,
        "trend_label": trend_label, "trend_tone": trend_tone,
        "atr": atr, "atr_pct": atr_pct,
        "support": support, "resistance": resistance,
        "hi_52w": hi_52w, "lo_52w": lo_52w,
        "pct_from_52w_high": _pct(last_close, hi_52w),
        "pct_from_52w_low": _pct(last_close, lo_52w),
        "vol_ratio": vol_ratio, "vol_label": vol_label,
        "ret_20d": ret_20d, "ret_60d": ret_60d,
    })
    return out


def _trend_read(close, ema50, ema200, adx) -> tuple[str, Tone]:
    if close is None or ema50 is None:
        return "Unknown", "neutral"
    strong = (adx or 0) >= 25
    if ema200 is not None and close > ema50 > ema200:
        return ("Strong uptrend" if strong else "Uptrend"), "good"
    if close > ema50:
        return "Recovering / above 50-EMA", "ok"
    if ema200 is not None and close < ema50 < ema200:
        return ("Strong downtrend" if strong else "Downtrend"), "bad"
    return "Weak / below 50-EMA", "warn"


def _rsi_read(rsi) -> tuple[str, Tone]:
    if rsi is None:
        return "—", "neutral"
    if rsi >= 70:
        return "Overbought", "warn"
    if rsi >= 55:
        return "Bullish", "good"
    if rsi >= 45:
        return "Neutral", "ok"
    if rsi >= 30:
        return "Weak", "warn"
    return "Oversold", "ok"  # oversold can be an opportunity, not strictly bad


def _macd_read(hist) -> tuple[str, Tone]:
    if hist is None:
        return "—", "neutral"
    if hist > 0:
        return "Bullish (above signal)", "good"
    return "Bearish (below signal)", "warn"


def _adx_read(adx) -> str:
    if adx is None:
        return "—"
    if adx >= 40:
        return "Very strong"
    if adx >= 25:
        return "Strong"
    if adx >= 20:
        return "Developing"
    return "Weak / ranging"


def _volume_read(ratio) -> str:
    if ratio is None:
        return "—"
    if ratio >= 2.0:
        return "Surging (2x+ avg)"
    if ratio >= 1.3:
        return "Above average"
    if ratio >= 0.7:
        return "Average"
    return "Quiet (below avg)"


# ---------------------------------------------------------------------------
# 2. Conviction score (0-100) with factor breakdown + reasons
# ---------------------------------------------------------------------------

# Factor weights (sum need not be 100; we renormalise over available factors).
_WEIGHTS = {
    "Fundamentals": 25.0,
    "Valuation": 20.0,
    "Technicals": 25.0,
    "Momentum": 15.0,
    "Risk": 15.0,
}


def conviction_score(
    fundamentals: dict[str, Any] | None,
    technicals: dict[str, Any] | None,
) -> dict[str, Any]:
    """Blend factor sub-scores into a 0-100 conviction with explanations."""
    f = fundamentals or {}
    t = technicals or {}

    factors: list[dict[str, Any]] = []
    reasons: list[str] = []

    def add(name: str, score: float | None, reason: str | None) -> None:
        if score is None:
            return
        factors.append({"name": name, "score": round(_clamp(score)),
                        "weight": _WEIGHTS[name]})
        if reason:
            reasons.append(reason)

    add("Fundamentals", *_score_fundamentals(f))
    add("Valuation", *_score_valuation(f))
    add("Technicals", *_score_technicals(t))
    add("Momentum", *_score_momentum(t))
    add("Risk", *_score_risk(t))

    if not factors:
        return {"overall": None, "factors": [], "reasons": [],
                "label": "Insufficient data", "tone": "neutral"}

    wsum = sum(fac["weight"] for fac in factors)
    overall = sum(fac["score"] * fac["weight"] for fac in factors) / wsum
    overall = round(_clamp(overall))
    label, tone = _conviction_label(overall)
    return {"overall": overall, "factors": factors, "reasons": reasons,
            "label": label, "tone": tone}


def _conviction_label(score: float) -> tuple[str, Tone]:
    if score >= 75:
        return "Very high conviction", "good"
    if score >= 60:
        return "High conviction", "good"
    if score >= 45:
        return "Moderate / mixed", "ok"
    if score >= 30:
        return "Low conviction", "warn"
    return "Avoid", "bad"


def _score_fundamentals(f: dict[str, Any]) -> tuple[float | None, str | None]:
    """Quality: ROE, margin, growth, leverage. Each maps to 0-100; averaged."""
    parts: list[float] = []
    notes: list[str] = []

    roe = f.get("roe")
    if roe is not None:
        s = _clamp(roe * 100 * 4)  # 25% ROE -> 100
        parts.append(s)
        if roe * 100 >= 18:
            notes.append(f"strong ROE {roe * 100:.0f}%")
        elif roe * 100 < 8:
            notes.append(f"weak ROE {roe * 100:.0f}%")

    margin = f.get("profit_margin")
    if margin is not None:
        parts.append(_clamp(margin * 100 * 4))  # 25% margin -> 100

    rev_g = f.get("revenue_growth")
    if rev_g is not None:
        s = _clamp(50 + rev_g * 100 * 2)  # +25% -> 100, -25% -> 0
        parts.append(s)
        if rev_g * 100 >= 15:
            notes.append(f"revenue +{rev_g * 100:.0f}% YoY")
        elif rev_g * 100 < 0:
            notes.append(f"revenue {rev_g * 100:.0f}% YoY")

    earn_g = f.get("earnings_growth")
    if earn_g is not None:
        parts.append(_clamp(50 + earn_g * 100 * 1.5))
        if earn_g * 100 < 0:
            notes.append(f"earnings {earn_g * 100:.0f}% YoY")

    de = f.get("debt_to_equity")
    if de is not None and de >= 0:
        # 0 debt -> 100, 2.0 D/E -> 0
        parts.append(_clamp(100 - de * 50))
        if de <= 0.5:
            notes.append("low debt")
        elif de >= 2.0:
            notes.append(f"high debt (D/E {de:.1f})")

    if not parts:
        return None, None
    score = sum(parts) / len(parts)
    reason = ("Fundamentals: " + ", ".join(notes)) if notes else None
    return score, reason


def _score_valuation(f: dict[str, Any]) -> tuple[float | None, str | None]:
    """Cheaper P/E and P/B -> higher score."""
    parts: list[float] = []
    notes: list[str] = []

    pe = f.get("pe_ttm")
    if pe is not None and pe > 0:
        # P/E 10 -> ~92, 25 -> ~50, 60 -> ~0
        parts.append(_clamp(100 - (pe - 10) * (100 / 50)))
        if pe < 15:
            notes.append(f"cheap P/E {pe:.0f}")
        elif pe >= 45:
            notes.append(f"expensive P/E {pe:.0f}")

    pb = f.get("pb")
    if pb is not None and pb > 0:
        # P/B 1 -> ~90, 5 -> ~50, 10 -> ~0
        parts.append(_clamp(100 - (pb - 1) * 10))
        if pb < 1.5:
            notes.append(f"low P/B {pb:.1f}")

    if not parts:
        return None, None
    score = sum(parts) / len(parts)
    reason = ("Valuation: " + ", ".join(notes)) if notes else None
    return score, reason


def _score_technicals(t: dict[str, Any]) -> tuple[float | None, str | None]:
    if not t.get("available"):
        return None, None
    score = 50.0
    notes: list[str] = []

    tone = t.get("trend_tone")
    if tone == "good":
        score += 22
        notes.append((t.get("trend_label") or "uptrend").lower())
    elif tone == "ok":
        score += 6
    elif tone == "warn":
        score -= 12
    elif tone == "bad":
        score -= 25
        notes.append("downtrend")

    if t.get("macd_hist") is not None:
        score += 10 if t["macd_hist"] > 0 else -10

    rsi = t.get("rsi")
    if rsi is not None:
        if rsi >= 70:
            score -= 8
            notes.append("RSI overbought")
        elif rsi < 30:
            score += 6
            notes.append("RSI oversold (mean-reversion)")
        elif 50 <= rsi < 70:
            score += 6

    if (t.get("adx") or 0) >= 25 and tone == "good":
        score += 6  # strong, confirmed trend

    reason = ("Technicals: " + ", ".join(notes)) if notes else None
    return _clamp(score), reason


def _score_momentum(t: dict[str, Any]) -> tuple[float | None, str | None]:
    if not t.get("available"):
        return None, None
    r20 = t.get("ret_20d")
    r60 = t.get("ret_60d")
    vals = [r for r in (r20, r60) if r is not None]
    if not vals:
        return None, None
    avg = sum(vals) / len(vals)
    score = _clamp(50 + avg * 2.5)  # +20% avg -> 100
    notes = None
    if avg >= 10:
        notes = f"Momentum: +{avg:.0f}% recent"
    elif avg <= -10:
        notes = f"Momentum: {avg:.0f}% recent"
    return score, notes


def _score_risk(t: dict[str, Any]) -> tuple[float | None, str | None]:
    """Lower volatility / drawdown -> higher (safer) score."""
    if not t.get("available"):
        return None, None
    atr_pct = t.get("atr_pct")
    if atr_pct is None:
        return None, None
    # 1% daily ATR -> ~92, 4% -> ~50, 7% -> ~8
    score = _clamp(100 - (atr_pct - 1.0) * (100 / 6.0))
    note = None
    if atr_pct >= 4:
        note = f"Risk: high volatility (~{atr_pct:.1f}%/day)"
    elif atr_pct <= 1.5:
        note = "Risk: low volatility"
    return score, note


# ---------------------------------------------------------------------------
# 3. Buy / Sell zones (ATR-based, volatility-aware)
# ---------------------------------------------------------------------------


def buy_sell_zones(last_close: float | None,
                   technicals: dict[str, Any] | None,
                   min_profit_pct: float = 5.0) -> dict[str, Any]:
    """Volatility-based entry/target/stop bands and risk:reward.

    Uses ATR (true daily range) so the zones scale with the stock's own
    volatility instead of arbitrary fixed percentages. Falls back to a 2%
    band if ATR is unavailable.

    ``min_profit_pct`` floors the first target so it always represents at
    least that much gain over the current price (default +5%).
    """
    if not last_close or last_close <= 0:
        return {"available": False}
    t = technicals or {}
    atr = t.get("atr")
    if not atr or atr <= 0:
        atr = last_close * 0.02  # sensible fallback

    support = t.get("support")
    resistance = t.get("resistance")

    # Stop first: just under structural support if it's nearby, otherwise an
    # ATR-based stop. Take the *closer* (higher) of the two so risk stays sane.
    atr_stop = last_close - 1.8 * atr
    if support is not None and support < last_close:
        structural_stop = support - 0.3 * atr
        stop = max(atr_stop, structural_stop)
    else:
        stop = atr_stop
    risk = max(last_close - stop, 0.01)

    # Targets as R-multiples of risk -> R:R is good by construction (1.5 / 3.0),
    # but floored so target 1 always clears the minimum profit goal (>= +5%)
    # and target 2 is at least double that.
    target1 = last_close + 1.5 * risk
    target2 = last_close + 3.0 * risk
    if min_profit_pct and min_profit_pct > 0:
        target1 = max(target1, last_close * (1.0 + min_profit_pct / 100.0))
        target2 = max(target2, last_close * (1.0 + 2.0 * min_profit_pct / 100.0))

    # Entry bands.
    buy_low = last_close - 0.4 * atr
    buy_high = last_close + 0.4 * atr
    strong_buy_below = last_close - 1.2 * atr
    if support is not None and support < last_close:
        strong_buy_below = max(strong_buy_below, support)
    avoid_above = last_close + 1.5 * atr
    if resistance is not None and resistance > last_close:
        avoid_above = min(avoid_above, resistance)

    rr = (target1 - last_close) / risk

    return {
        "available": True,
        "current": round(last_close, 2),
        "buy_low": round(buy_low, 2),
        "buy_high": round(buy_high, 2),
        "strong_buy_below": round(strong_buy_below, 2),
        "avoid_above": round(avoid_above, 2),
        "stop": round(stop, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "target1_pct": round((target1 / last_close - 1.0) * 100.0, 2),
        "target2_pct": round((target2 / last_close - 1.0) * 100.0, 2),
        "stop_pct": round((stop / last_close - 1.0) * 100.0, 2),
        "min_profit_pct": round(float(min_profit_pct), 1),
        "risk_reward": round(rr, 2),
    }


# ---------------------------------------------------------------------------
# 4. Rule-based verdict (for stocks outside the trained model's universe)
# ---------------------------------------------------------------------------


def rule_based_verdict(conviction: dict[str, Any] | None) -> dict[str, Any]:
    """Derive BUY/HOLD/SELL + confidence from the conviction score.

    This is a transparent rule, NOT the trained ML model. The UI labels it as
    such so stocks outside the daily model universe still get an honest call.
    """
    score = (conviction or {}).get("overall")
    if score is None:
        return {"verdict": None, "confidence": None, "source": "rule_based"}
    if score >= 65:
        verdict = "BUY"
    elif score >= 45:
        verdict = "HOLD"
    else:
        verdict = "SELL"
    return {"verdict": verdict, "confidence": float(score), "source": "rule_based"}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze(df: pd.DataFrame,
            fundamentals: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full analysis: technicals + conviction + zones (+ rule verdict)."""
    tech = compute_technicals(df)
    conv = conviction_score(fundamentals, tech)
    zones = buy_sell_zones(tech.get("last_close"), tech)
    verdict = rule_based_verdict(conv)
    return {
        "technicals": tech,
        "conviction": conv,
        "zones": zones,
        "rule_verdict": verdict,
    }
