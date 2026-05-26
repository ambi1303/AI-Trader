"""Position sizing.

We support two complementary approaches and pick the SMALLER of the two,
so risk is bounded by both information edge AND realized volatility:

1. Fractional Kelly
   - Optimal log-growth fraction: f* = p - (1-p)/R
       where p = calibrated probability of a win
             R = avg_win / avg_loss (reward-to-risk ratio)
   - Cap at quarter Kelly (0.25) to be robust to model error: full Kelly is
     mathematically optimal but assumes the probabilities are exact, which
     they are not. Quarter-Kelly preserves most of the long-run growth at
     ~6% of the variance.
   - Floor at 0 (never short or scale down to negative size).

2. Volatility targeting
   - Pick qty so that 1-stop-loss move costs at most `risk_per_trade_pct` of
     equity. With ATR(14) as our volatility proxy and a 2*ATR stop, this
     means qty = floor(risk_per_trade_pct * equity / (2 * ATR)).
   - This caps the per-trade tail risk, independent of model probability.

We also enforce hard caps:
- Max single-position notional <= max_position_pct of equity.
- Min trade notional >= min_trade_rupees (avoids paying flat broker fees on
  trivial sizes).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.logger import get_logger

log = get_logger("backtest.sizing")


@dataclass(frozen=True)
class SizingConfig:
    risk_per_trade_pct: float = 0.01      # 1% equity per trade max risk
    max_position_pct: float = 0.10         # any single name <= 10% of equity
    min_trade_rupees: float = 5_000.0      # smallest trade notional
    kelly_fraction: float = 0.25           # quarter Kelly
    kelly_reward_to_risk: float = 1.5      # default R when not supplied
    leverage: float = 1.0                  # 1.0 = cash account


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------


def kelly_fraction(prob_win: float, reward_to_risk: float) -> float:
    """Vanilla Kelly: f* = p - (1-p)/R. Floored at 0."""
    if reward_to_risk <= 0:
        return 0.0
    f = prob_win - (1.0 - prob_win) / reward_to_risk
    return max(0.0, float(f))


def fractional_kelly_qty(
    *,
    prob_win: float,
    entry_price: float,
    equity: float,
    cfg: SizingConfig,
    reward_to_risk: float | None = None,
) -> int:
    R = reward_to_risk if reward_to_risk is not None else cfg.kelly_reward_to_risk
    raw_f = kelly_fraction(prob_win, R)
    f = min(raw_f, 1.0) * cfg.kelly_fraction  # quarter / fractional Kelly
    notional = f * equity * cfg.leverage
    return int(notional // entry_price)


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------


def vol_target_qty(
    *,
    entry_price: float,
    atr: float,
    stop_atr_mult: float,
    equity: float,
    cfg: SizingConfig,
) -> int:
    """qty such that a stop-out costs ~risk_per_trade_pct of equity."""
    if atr <= 0 or stop_atr_mult <= 0 or entry_price <= 0:
        return 0
    risk_per_share = stop_atr_mult * atr
    rupee_risk = cfg.risk_per_trade_pct * equity
    qty = int(rupee_risk // risk_per_share)
    # Cap by max position notional too.
    cap_qty = int((cfg.max_position_pct * equity) // entry_price)
    return max(0, min(qty, cap_qty))


# ---------------------------------------------------------------------------
# Combined sizer (the one the engine actually calls)
# ---------------------------------------------------------------------------


@dataclass
class SizingDecision:
    qty: int
    rationale: str         # which leg was binding: "kelly" | "vol_target" | "min_size" | "skip"
    raw_kelly_qty: int
    raw_vol_qty: int
    notional: float


def size_position(
    *,
    prob_win: float,
    entry_price: float,
    atr: float,
    stop_atr_mult: float,
    equity: float,
    cfg: SizingConfig,
    reward_to_risk: float | None = None,
) -> SizingDecision:
    """Pick min(Kelly qty, vol-target qty) subject to hard caps."""
    if equity <= 0 or entry_price <= 0:
        return SizingDecision(qty=0, rationale="skip", raw_kelly_qty=0,
                              raw_vol_qty=0, notional=0.0)

    k_qty = fractional_kelly_qty(
        prob_win=prob_win, entry_price=entry_price, equity=equity,
        cfg=cfg, reward_to_risk=reward_to_risk,
    )
    v_qty = vol_target_qty(
        entry_price=entry_price, atr=atr, stop_atr_mult=stop_atr_mult,
        equity=equity, cfg=cfg,
    )

    qty = min(k_qty, v_qty)
    if qty <= 0:
        return SizingDecision(qty=0, rationale="skip", raw_kelly_qty=k_qty,
                              raw_vol_qty=v_qty, notional=0.0)

    # Cap by max position
    cap_qty = int((cfg.max_position_pct * equity) // entry_price)
    if qty > cap_qty:
        qty = cap_qty

    notional = qty * entry_price
    if notional < cfg.min_trade_rupees:
        return SizingDecision(qty=0, rationale="min_size", raw_kelly_qty=k_qty,
                              raw_vol_qty=v_qty, notional=notional)

    rationale = "kelly" if k_qty <= v_qty else "vol_target"
    return SizingDecision(qty=qty, rationale=rationale, raw_kelly_qty=k_qty,
                          raw_vol_qty=v_qty, notional=notional)
