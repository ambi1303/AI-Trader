"""Risk overlays applied at every bar of the simulation.

These are *guards*, not signals. They:
  - Translate a model's probability into an entry that has bounded downside
    via an ATR stop and a TP target.
  - Track a trailing high-water-mark stop on each open position so winners
    aren't given back fully.
  - Force exits after `max_holding_days` to limit time decay of edge.
  - Refuse new entries if the *daily* loss limit has been breached.
  - Refuse new entries if the open-position count in a sector is already at
    the cap (avoid concentration in a single sector).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    use_trailing_stop: bool = True
    trail_atr_mult: float = 2.0
    max_holding_days: int = 10
    daily_loss_limit_pct: float = 0.03      # halt new entries after -3% on the day
    max_concurrent_positions: int = 8
    max_per_sector: int = 3                  # positions per sector
    cooldown_days_after_loss: int = 1        # don't re-enter same symbol immediately

    def stop_for(self, entry_price: float, atr: float) -> float:
        return entry_price - self.stop_atr_mult * atr

    def target_for(self, entry_price: float, atr: float) -> float:
        return entry_price + self.take_profit_atr_mult * atr


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------


@dataclass
class OpenPosition:
    symbol: str
    sector: str
    side: str              # "LONG"
    qty: int
    entry_date: str        # ISO YYYY-MM-DD
    entry_price: float
    atr_at_entry: float
    stop: float
    target: float
    high_watermark: float  # for trailing stop
    entry_prob: float
    threshold: float

    def update_trailing_stop(self, today_high: float, cfg: RiskConfig) -> None:
        if not cfg.use_trailing_stop:
            return
        if today_high > self.high_watermark:
            self.high_watermark = today_high
            new_stop = self.high_watermark - cfg.trail_atr_mult * self.atr_at_entry
            if new_stop > self.stop:
                self.stop = new_stop


# ---------------------------------------------------------------------------
# Intraday hit logic
# ---------------------------------------------------------------------------


@dataclass
class HitResult:
    hit: bool
    fill_price: float
    reason: str            # "stop" | "target" | "trail" | "time" | "none"


def check_stop_or_target(
    pos: OpenPosition,
    *,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
) -> HitResult:
    """Decide whether today's bar triggers a stop or take-profit.

    Convention (intentionally PESSIMISTIC for backtest realism):
      1. If the bar OPEN already gaps below the stop -> exit at OPEN.
         (Realistic: a gap-down opens past your stop and you eat the gap.)
      2. If the bar OPEN gaps above the target -> exit at OPEN.
         (Realistic: a gap-up opens past your target.)
      3. Else if both stop and target are within today's range, ASSUME stop
         hits first. This is the textbook conservative assumption since we
         can't infer intraday sequencing from daily bars.
      4. Else if only stop within range -> exit at stop.
      5. Else if only target within range -> exit at target.
    """
    # 1. Gap-down past stop
    if bar_open <= pos.stop:
        return HitResult(True, fill_price=bar_open, reason="stop")
    # 2. Gap-up past target
    if bar_open >= pos.target:
        return HitResult(True, fill_price=bar_open, reason="target")

    stop_hit = bar_low <= pos.stop
    target_hit = bar_high >= pos.target

    # 3. Both within range -> assume stop first
    if stop_hit and target_hit:
        return HitResult(True, fill_price=pos.stop, reason="stop")
    if stop_hit:
        return HitResult(True, fill_price=pos.stop, reason="stop")
    if target_hit:
        return HitResult(True, fill_price=pos.target, reason="target")
    return HitResult(False, fill_price=bar_close, reason="none")


# ---------------------------------------------------------------------------
# Portfolio-level guards
# ---------------------------------------------------------------------------


def can_open_new_position(
    *,
    cfg: RiskConfig,
    open_positions: list[OpenPosition],
    sector: str,
    today_pnl_pct: float,
) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    if len(open_positions) >= cfg.max_concurrent_positions:
        return False, "max_concurrent_positions"
    sector_count = sum(1 for p in open_positions if p.sector == sector)
    if sector_count >= cfg.max_per_sector:
        return False, "max_per_sector"
    if today_pnl_pct <= -cfg.daily_loss_limit_pct:
        return False, "daily_loss_limit"
    return True, "ok"
