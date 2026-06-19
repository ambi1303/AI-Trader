"""Rules-based factor strategy -- the automated 'algo to invest'.

This is a transparent alternative to the ML signal: every trading day it ranks
the whole universe by a blended **momentum + quality + value + trend** score,
keeps only names that pass hard trend/risk *gates*, and writes the top picks
into ``signal_outbox`` (side='BUY') exactly like ``signals.generator`` does --
so the existing paper-trade reconciler opens, trails, and exits them with no
extra plumbing.

Why factors? Momentum, quality (high ROE / low debt) and value (sane P/E) are
the most robust, well-documented equity premia. We require a confirmed
*uptrend* (above the 200-EMA, healthy RSI, not in a deep drawdown) so we never
"catch a falling knife". The score and the reasons behind it are stored on each
signal, so every trade is explainable.

Scale conventions match the rest of the app: ratios from ``fundamental_data``
are fractions (roe 0.18 == 18%); momentum / EMA-distance / drawdown from
``feature_data`` are fractions (mom_60d 0.12 == +12%).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.analysis import discovery, feasibility
from src.backtesting.risk import RiskConfig
from src.backtesting.sizing import SizingConfig, size_position
from src.utils.db import fetch_all, fetch_one, transaction
from src.utils.logger import get_logger

log = get_logger("signals.strategy")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyConfig:
    """Knobs for one strategy run. Defaults target a diversified 20-25 name
    position-trading book on daily entries."""
    target_holdings: int = 22
    max_per_sector: int = 6
    min_score: float = 55.0
    max_holding_days: int = 90          # position-trading, not 10-day swing
    equity: float = 1_000_000.0
    min_profit_pct: float = 5.0         # take-profit floored at >= +5% per trade
    # Conviction-scaled take-profit: a borderline (min_score) name targets
    # base_tp_atr_mult x ATR, while a top-conviction (score 100) name targets
    # strong_tp_atr_mult x ATR so winners the algo is confident in can run far
    # past +5%. A trailing stop protects the gains on the way up.
    base_tp_atr_mult: float = 3.0
    strong_tp_atr_mult: float = 8.0
    # Feasibility gate: only enter when the target has at least this touch
    # probability within the holding window, given the stock's volatility +
    # trend. If the conviction target is too rich it's trimmed down to the
    # highest feasible level (never below min_profit_pct); if even +5% isn't
    # reachable in time, the name is skipped. Set require_feasible=False to
    # disable the gate entirely.
    require_feasible: bool = True
    min_feasibility_prob: float = 0.50

    # Hard entry gates -----------------------------------------------------
    min_close: float = 50.0             # skip illiquid sub-50 names
    min_dist_ema_200: float = 0.0       # must be at/above the 200-EMA (uptrend)
    rsi_floor: float = 40.0             # avoid falling knives
    rsi_ceiling: float = 82.0           # avoid blow-off overbought
    min_mom_60d: float = 0.0            # 3-month momentum non-negative
    max_drawdown_252d: float = -0.30    # skip names >30% below their 1y high

    # Composite weights (need not sum to 1; normalised at use) -------------
    w_momentum: float = 0.35
    w_quality: float = 0.30
    w_value: float = 0.20
    w_trend: float = 0.15

    sizing: SizingConfig = field(default_factory=lambda: SizingConfig(
        max_position_pct=0.045,         # ~1/22 -> roughly equal weight
        min_trade_rupees=5_000.0,
    ))


def strategy_risk_config(cfg: StrategyConfig | None = None) -> RiskConfig:
    """RiskConfig that BOTH the strategy generator and the paper reconciler
    must share, so capacity / sector caps / holding period are consistent."""
    cfg = cfg or StrategyConfig()
    return RiskConfig(
        max_concurrent_positions=cfg.target_holdings,
        max_per_sector=cfg.max_per_sector,
        max_holding_days=cfg.max_holding_days,
        min_profit_pct=cfg.min_profit_pct,
    )


# ---------------------------------------------------------------------------
# Composite scorer (pure)
# ---------------------------------------------------------------------------


def _g(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ScoredCandidate:
    symbol: str
    sector: str
    score: float
    reasons: list[str]
    sub: dict[str, float]
    close: float | None


def score_row(row: dict[str, Any], cfg: StrategyConfig) -> ScoredCandidate | None:
    """Return a ScoredCandidate, or None if a hard gate excludes the name."""
    close = _g(row, "close")
    d200 = _g(row, "dist_ema_200_pct")
    d50 = _g(row, "dist_ema_50_pct")
    rsi = _g(row, "rsi_14")
    mom60 = _g(row, "mom_60d")
    ret20 = _g(row, "ret_20d")
    dd = _g(row, "dd_from_high_252d")
    macd = _g(row, "macd_hist")
    roe = _g(row, "roe")
    de = _g(row, "debt_to_equity")
    margin = _g(row, "profit_margin")
    pe = _g(row, "pe_ttm")

    # ---- hard gates ----
    if close is None or close < cfg.min_close:
        return None
    if d200 is None or d200 < cfg.min_dist_ema_200:        # uptrend filter
        return None
    if rsi is None or rsi < cfg.rsi_floor or rsi > cfg.rsi_ceiling:
        return None
    if mom60 is None or mom60 < cfg.min_mom_60d:
        return None
    if dd is not None and dd < cfg.max_drawdown_252d:
        return None

    reasons: list[str] = []

    # ---- momentum sub-score ----
    mom_s = _clamp(50 + mom60 * 120 + (ret20 or 0.0) * 60)
    reasons.append(f"3-month momentum {mom60 * 100:+.0f}%")

    # ---- quality sub-score (neutral 45 if fundamentals missing) ----
    if roe is not None:
        q = _clamp(roe * 250)                       # roe 0.20 -> 50; 0.40 -> 100
        if margin is not None:
            q = 0.65 * q + 0.35 * _clamp(margin * 100 * 4)
        if de is not None:
            q += _clamp(20 - de * 20, 0, 20) * 0.25  # reward low leverage
        quality_s = _clamp(q)
        reasons.append(f"ROE {roe * 100:.0f}%")
        if de is not None and de <= 0.5:
            reasons.append(f"low debt (D/E {de:.2f})")
    else:
        quality_s = 45.0

    # ---- value sub-score (neutral 45 if P/E unusable) ----
    if pe is not None and pe > 0:
        value_s = _clamp(110 - pe * 2.5)            # PE 10 -> 85, 30 -> 35
        if pe <= 25:
            reasons.append(f"reasonable P/E {pe:.0f}")
    else:
        value_s = 45.0

    # ---- trend-confirmation sub-score ----
    trend_s = 50.0
    if d50 is not None:
        trend_s += d50 * 120
    if macd is not None:
        trend_s += 8 if macd > 0 else -6
    if dd is not None:
        trend_s += dd * 100 * 1.2                   # closer to high -> higher
    trend_s = _clamp(trend_s)
    reasons.append(f"{d200 * 100:.0f}% above 200-EMA")
    reasons.append(f"RSI {rsi:.0f}")

    wsum = cfg.w_momentum + cfg.w_quality + cfg.w_value + cfg.w_trend
    composite = (
        cfg.w_momentum * mom_s + cfg.w_quality * quality_s
        + cfg.w_value * value_s + cfg.w_trend * trend_s
    ) / wsum

    return ScoredCandidate(
        symbol=row["symbol"],
        sector=row.get("sector", "UNKNOWN") or "UNKNOWN",
        score=round(composite, 1),
        reasons=reasons,
        sub={"momentum": round(mom_s, 1), "quality": round(quality_s, 1),
             "value": round(value_s, 1), "trend": round(trend_s, 1)},
        close=close,
    )


def score_universe(rows: list[dict[str, Any]],
                   cfg: StrategyConfig | None = None) -> list[ScoredCandidate]:
    """Score + filter every row, returning candidates sorted best-first."""
    cfg = cfg or StrategyConfig()
    out: list[ScoredCandidate] = []
    for r in rows:
        c = score_row(r, cfg)
        if c is not None and c.score >= cfg.min_score:
            out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# DB helpers (mirror signals.generator so behaviour is consistent)
# ---------------------------------------------------------------------------


def _open_position_count() -> int:
    row = fetch_one("SELECT COUNT(*) AS n FROM paper_trades WHERE status = 'open'")
    return int(row["n"]) if row else 0


def _open_per_sector() -> dict[str, int]:
    rows = fetch_all(
        "SELECT sector, COUNT(*) AS n FROM paper_trades "
        "WHERE status = 'open' AND sector IS NOT NULL GROUP BY sector"
    )
    return {r["sector"]: int(r["n"]) for r in rows}


def _existing_signals(signal_date: str) -> set[str]:
    rows = fetch_all(
        "SELECT symbol FROM signal_outbox WHERE signal_date = ?", (signal_date,)
    )
    return {r["symbol"] for r in rows}


def _held_symbols() -> set[str]:
    rows = fetch_all("SELECT symbol FROM paper_trades WHERE status = 'open'")
    return {r["symbol"] for r in rows}


def _price_atr(symbol: str, on_date: str) -> dict[str, float | None] | None:
    """Authoritative close + ATR(14) <= on_date (same source as the ML path),
    plus the daily volatility / momentum needed for the feasibility check."""
    feat = fetch_one(
        "SELECT close, atr_14, vol_20d, mom_20d, mom_60d FROM feature_data "
        "WHERE symbol = ? AND feature_date <= ? "
        "ORDER BY feature_date DESC LIMIT 1",
        (symbol, on_date),
    )
    if not feat or feat["atr_14"] is None or feat["close"] is None:
        return None
    close, atr = float(feat["close"]), float(feat["atr_14"])
    if close <= 0 or atr <= 0:
        return None

    def _opt(key: str) -> float | None:
        v = feat[key]
        return float(v) if v is not None else None

    return {"close": close, "atr_14": atr, "vol_20d": _opt("vol_20d"),
            "mom_20d": _opt("mom_20d"), "mom_60d": _opt("mom_60d")}


def _record_validation(conn: sqlite3.Connection, *, severity: str,
                       message: str, symbol: str | None = None) -> None:
    conn.execute(
        "INSERT INTO validation_failures (run_id, check_name, symbol, severity, message) "
        "VALUES ('strategy_gen', 'strategy_generator', ?, ?, ?)",
        (symbol, severity, message),
    )


def conviction_tp_atr_mult(score: float, cfg: StrategyConfig) -> float:
    """Take-profit ATR multiple that scales with conviction.

    Maps ``score`` linearly from ``cfg.min_score`` (-> base_tp_atr_mult) to
    100 (-> strong_tp_atr_mult). The stronger the algo's read, the higher the
    target, so confident winners are allowed to run well beyond +5%.
    """
    span = max(1e-9, 100.0 - cfg.min_score)
    frac = _clamp((score - cfg.min_score) / span, 0.0, 1.0)
    return cfg.base_tp_atr_mult + (cfg.strong_tp_atr_mult - cfg.base_tp_atr_mult) * frac


def _score_to_prob(score: float) -> float:
    """Map a 0-100 conviction score to a pseudo win-probability for Kelly.

    Deliberately conservative: even a perfect score caps at 0.70 so the
    fractional-Kelly leg never sizes aggressively on rules alone.
    """
    return max(0.5, min(0.70, 0.5 + (score - 50.0) / 100.0 * 0.4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    sector: str
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: int
    score: float
    reasons: list[str]


def generate_strategy_signals(
    *,
    signal_date: str | None = None,
    config: StrategyConfig | None = None,
    risk: RiskConfig | None = None,
) -> list[StrategySignal]:
    """Rank the universe and queue BUY signals for the best free slots.

    Idempotent: re-running the same day is a no-op (unique index on
    ``signal_outbox(symbol, signal_date)`` + we skip names already held).
    """
    cfg = config or StrategyConfig()
    risk = risk or strategy_risk_config(cfg)
    sd = signal_date or date.today().isoformat()

    candidates = score_universe(discovery._rows(force=True), cfg)
    if not candidates:
        log.info("strategy: no candidates passed the gates for {}", sd)
        return []

    n_open = _open_position_count()
    capacity = max(0, risk.max_concurrent_positions - n_open)
    if capacity == 0:
        log.info("strategy: portfolio full (open={}, cap={})",
                 n_open, risk.max_concurrent_positions)
        return []

    sector_open = _open_per_sector()
    already = _existing_signals(sd) | _held_symbols()
    kept: list[StrategySignal] = []

    with transaction() as conn:
        for cand in candidates:
            if len(kept) >= capacity:
                break
            sym = cand.symbol
            if sym in already:
                continue

            # Sector cap only applies to *known* sectors. Most of the broad
            # universe has no mapping in stock_sectors (sector='UNKNOWN'); capping
            # that catch-all would wrongly starve the book, so we exempt it.
            if cand.sector != "UNKNOWN":
                kept_in_sector = sum(1 for k in kept if k.sector == cand.sector)
                if (sector_open.get(cand.sector, 0) + kept_in_sector
                        >= risk.max_per_sector):
                    _record_validation(conn, severity="info", symbol=sym,
                                       message=f"sector cap reached ({cand.sector})")
                    continue

            quote = _price_atr(sym, sd)
            if quote is None:
                _record_validation(conn, severity="warning", symbol=sym,
                                   message="missing close/atr for strategy signal")
                continue

            entry = quote["close"]
            atr = quote["atr_14"]
            stop = risk.stop_for(entry, atr)
            # Conviction-scaled target: the more confident the score, the
            # higher the take-profit (winners run past +5%); the trailing stop
            # locks in gains as price climbs.
            tp_mult = conviction_tp_atr_mult(cand.score, cfg)
            target = risk.target_for(entry, atr, tp_mult)
            target_pct = (target / entry - 1.0) * 100.0 if entry else 0.0
            atr_pct = (atr / entry) if entry else None
            feas_prob: float | None = None

            # ---- feasibility gate -------------------------------------------
            # Only enter when the target is plausibly reachable within the
            # holding window given this stock's volatility + trend. Trim a too-
            # rich target down to the highest feasible level (never below the
            # +5% floor); skip the name if even +5% isn't reachable in time.
            vol_20d = quote.get("vol_20d")
            if cfg.require_feasible and vol_20d and vol_20d > 0:
                feasible_pct = feasibility.feasible_target_pct(
                    min_prob=cfg.min_feasibility_prob,
                    max_target_pct=target_pct,
                    floor_pct=cfg.min_profit_pct,
                    horizon_days=cfg.max_holding_days,
                    daily_vol=quote.get("vol_20d"),
                    mom_20d=quote.get("mom_20d"),
                    mom_60d=quote.get("mom_60d"),
                    atr_pct=atr_pct,
                )
                if feasible_pct is None:
                    _record_validation(
                        conn, severity="info", symbol=sym,
                        message=(f"target infeasible: <{cfg.min_feasibility_prob:.0%} "
                                 f"chance of +{cfg.min_profit_pct:.0f}% in "
                                 f"{cfg.max_holding_days}d"),
                    )
                    continue
                if feasible_pct < target_pct - 1e-9:
                    target_pct = feasible_pct
                    target = round(entry * (1.0 + target_pct / 100.0), 2)
                feas_prob = feasibility.touch_prob(
                    target_pct=target_pct, horizon_days=cfg.max_holding_days,
                    daily_vol=vol_20d, mom_20d=quote.get("mom_20d"),
                    mom_60d=quote.get("mom_60d"), atr_pct=atr_pct,
                )
                cand.reasons.append(
                    f"~{feas_prob * 100:.0f}% chance to hit +{target_pct:.0f}% "
                    f"in {cfg.max_holding_days}d"
                )
            else:
                cand.reasons.append(f"profit target +{target_pct:.0f}%")

            decision = size_position(
                prob_win=_score_to_prob(cand.score),
                entry_price=entry, atr=atr,
                stop_atr_mult=risk.stop_atr_mult,
                equity=cfg.equity, cfg=cfg.sizing,
            )
            if decision.qty <= 0:
                _record_validation(conn, severity="info", symbol=sym,
                                   message=f"sizer qty=0 ({decision.rationale})")
                continue

            payload = {
                "engine": "rules",
                "strategy": "factor_blend",
                "score": cand.score,
                "sub_scores": cand.sub,
                "reasons": cand.reasons,
                "atr_14": atr,
                "sector": cand.sector,
                "tp_atr_mult": round(tp_mult, 2),
                "target_pct": round(target_pct, 1),
                "feasibility_prob": (round(feas_prob, 3)
                                     if feas_prob is not None else None),
                "feasibility_horizon_days": cfg.max_holding_days,
            }
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO signal_outbox
                    (symbol, signal_date, side, entry_price, stop_loss,
                     take_profit, qty, confidence, status, payload_json)
                VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (sym, sd, entry, stop, target, int(decision.qty),
                 cand.score / 100.0, json.dumps(payload, default=str)),
            )
            if cur.rowcount == 0:
                continue

            kept.append(StrategySignal(
                symbol=sym, sector=cand.sector, entry_price=entry,
                stop_loss=stop, take_profit=target, qty=int(decision.qty),
                score=cand.score, reasons=cand.reasons,
            ))

    log.info(
        "strategy signal-gen | date={} | candidates={} | kept={} | "
        "capacity={} | open={}",
        sd, len(candidates), len(kept), capacity, n_open,
    )
    return kept
