"""Risk-management engine: position sizing + portfolio health.

Pure functions (no DB, no network):

* :func:`position_size` -- given account capital, the % you're willing to risk
  per trade, an entry and a stop, returns how many shares to buy so the loss at
  the stop equals your risk budget (the standard fixed-fractional rule), capped
  by available capital.

* :func:`portfolio_health` -- given current holdings (and per-name daily
  volatility), reports position/sector concentration, a diversification score,
  plain-language warnings, and a rough 1-day 95% Value-at-Risk.

All money is in rupees; volatility inputs are *daily* (std of 1-day returns),
matching ``feature_data.vol_20d``.
"""

from __future__ import annotations

import math
from typing import Any

# 95% one-tailed normal quantile (for a 1-day VaR proxy).
_Z_95 = 1.645

# Concentration thresholds (percent of portfolio).
_MAX_POSITION_PCT = 25.0
_MAX_SECTOR_PCT = 40.0
_MIN_POSITIONS = 5
_MAX_POSITIONS = 20
_DEFAULT_DAILY_VOL = 0.02  # fallback when a name has no stored volatility


def _f(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def position_size(
    capital: float,
    risk_pct: float,
    entry: float,
    stop: float,
    *,
    lot_size: int = 1,
    target: float | None = None,
) -> dict[str, Any]:
    """Fixed-fractional position size for a long trade.

    The loss if ``stop`` is hit should equal ``capital * risk_pct/100``. Shares
    are floored to ``lot_size`` and capped so the position never exceeds
    available capital. Returns a template-friendly dict including notes.
    """
    cap = _f(capital) or 0.0
    rp = _f(risk_pct) or 0.0
    e = _f(entry) or 0.0
    s = _f(stop)
    lot = max(int(lot_size or 1), 1)
    notes: list[str] = []

    out: dict[str, Any] = {
        "valid": False, "shares": 0, "position_value": 0.0,
        "pct_of_capital": 0.0, "risk_amount": 0.0, "risk_pct_actual": 0.0,
        "per_share_risk": None, "risk_reward": None, "notes": notes,
    }

    if cap <= 0 or rp <= 0 or e <= 0 or s is None:
        notes.append("Enter capital, risk %, entry and stop to size the trade.")
        return out
    if rp > 100:
        notes.append("Risk per trade above 100% is not meaningful.")
        return out
    if s >= e:
        notes.append("For a long trade the stop must be below the entry price.")
        return out

    per_share_risk = e - s
    risk_budget = cap * rp / 100.0
    shares_by_risk = math.floor((risk_budget / per_share_risk) / lot) * lot
    max_shares_by_capital = math.floor((cap / e) / lot) * lot
    shares = min(shares_by_risk, max_shares_by_capital)

    if shares <= 0:
        notes.append("Capital too small for one lot at this entry/stop.")
        return out

    if shares_by_risk > max_shares_by_capital:
        notes.append(
            "Stop is tight: capital caps the size before the full risk "
            "budget is used. Actual risk is below your limit."
        )

    position_value = shares * e
    risk_amount = shares * per_share_risk
    rr = None
    t = _f(target)
    if t is not None and t > e:
        rr = round((t - e) / per_share_risk, 2)

    pct_cap = position_value / cap * 100.0
    if pct_cap > 50:
        notes.append(
            f"This single position is {pct_cap:.0f}% of capital — large "
            "concentration in one name."
        )

    out.update({
        "valid": True,
        "shares": int(shares),
        "position_value": round(position_value, 2),
        "pct_of_capital": round(pct_cap, 1),
        "risk_amount": round(risk_amount, 2),
        "risk_pct_actual": round(risk_amount / cap * 100.0, 2),
        "per_share_risk": round(per_share_risk, 2),
        "risk_reward": rr,
    })
    return out


# ---------------------------------------------------------------------------
# Portfolio health
# ---------------------------------------------------------------------------


def portfolio_health(
    positions: list[dict[str, Any]],
    vol_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Concentration, diversification and a rough 1-day VaR for holdings.

    ``positions``: dicts with ``symbol``, ``sector``, ``qty``, ``price``.
    ``vol_map``: symbol -> daily volatility (e.g. feature_data.vol_20d).
    """
    vol_map = vol_map or {}
    rows: list[dict[str, Any]] = []
    for p in positions:
        qty = _f(p.get("qty")) or 0.0
        price = _f(p.get("price")) or 0.0
        value = qty * price
        if value <= 0:
            continue
        rows.append({
            "symbol": p.get("symbol", "?"),
            "sector": p.get("sector") or "UNKNOWN",
            "value": value,
        })

    if not rows:
        return {"has_positions": False}

    total = sum(r["value"] for r in rows)
    for r in rows:
        r["weight_pct"] = round(r["value"] / total * 100.0, 1)
    rows.sort(key=lambda r: r["value"], reverse=True)

    # Sector aggregation.
    sec: dict[str, float] = {}
    for r in rows:
        sec[r["sector"]] = sec.get(r["sector"], 0.0) + r["value"]
    sectors = sorted(
        ({"sector": k, "weight_pct": round(v / total * 100.0, 1)}
         for k, v in sec.items()),
        key=lambda d: d["weight_pct"], reverse=True,
    )

    n = len(rows)
    max_pos = rows[0]
    top_sector = sectors[0]

    # Warnings.
    warnings: list[dict[str, str]] = []
    if max_pos["weight_pct"] > _MAX_POSITION_PCT:
        warnings.append({"tone": "warn", "text":
            f"{max_pos['symbol']} is {max_pos['weight_pct']:.0f}% of the "
            f"portfolio — heavy single-name concentration."})
    if top_sector["weight_pct"] > _MAX_SECTOR_PCT:
        warnings.append({"tone": "warn", "text":
            f"Portfolio is {top_sector['weight_pct']:.0f}% in "
            f"{top_sector['sector']} — concentrated in one sector."})
    if n < _MIN_POSITIONS:
        warnings.append({"tone": "ok", "text":
            f"Only {n} position(s) — limited diversification. Consider "
            "spreading risk across more names/sectors."})
    if n > _MAX_POSITIONS:
        warnings.append({"tone": "ok", "text":
            f"{n} positions — possibly over-diversified (hard to monitor; "
            "returns dilute toward the index)."})
    if not warnings:
        warnings.append({"tone": "good", "text":
            "No major concentration flags — reasonably balanced."})

    # Diversification score (transparent deductions from 100).
    score = 100.0
    if max_pos["weight_pct"] > _MAX_POSITION_PCT:
        score -= (max_pos["weight_pct"] - _MAX_POSITION_PCT) * 1.5
    if top_sector["weight_pct"] > _MAX_SECTOR_PCT:
        score -= (top_sector["weight_pct"] - _MAX_SECTOR_PCT) * 1.2
    if n < _MIN_POSITIONS:
        score -= (_MIN_POSITIONS - n) * 8
    score = round(_clamp(score))

    # Rough 1-day 95% VaR. Conservative: assume positions move together
    # (correlation = 1), so the position VaRs simply add.
    var_rupees = 0.0
    for r in rows:
        v = vol_map.get(r["symbol"])
        daily_vol = v if (v and v > 0) else _DEFAULT_DAILY_VOL
        var_rupees += _Z_95 * daily_vol * r["value"]
    var_pct = var_rupees / total * 100.0

    return {
        "has_positions": True,
        "n_positions": n,
        "total_value": round(total, 2),
        "positions": rows,
        "sectors": sectors,
        "max_position": max_pos,
        "top_sector": top_sector,
        "warnings": warnings,
        "diversification_score": score,
        "var_95_1d_rupees": round(var_rupees, 0),
        "var_95_1d_pct": round(var_pct, 2),
    }
