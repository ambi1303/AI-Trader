"""Indian equity cost model (Zerodha-style discount broker, EQ_DELIVERY by default).

Single source of truth for trading costs across the system: backtester,
walk-forward threshold tuning, and live signal sizing all call the same
function so we never disagree on what a trade actually costs.

Per-leg components for delivery (CNC):
    Brokerage           : min(flat_rupees, pct * notional) per leg
    Exchange transaction: pct * notional per leg
    SEBI                : flat per crore notional per leg (rounded paise)
    GST                 : 18% on (brokerage + exchange transaction) per leg
    Stamp duty (BUY)    : pct * notional, only on buy leg
    STT (SELL)          : pct * notional, only on sell leg (delivery 0.1%)
    Slippage            : pct * notional per leg (per-stock override, else default)

The output is BOTH the rupees breakdown (for audit) and a dictionary of
totals (for fast aggregation in the equity loop).

Why each component:
- Brokerage: discount brokers either charge 0 (free delivery) or a flat
  ₹20 (intraday). We model the lesser-of-flat-or-pct so we automatically
  match Zerodha's pricing when the user chooses the EQ_INTRADAY profile.
- STT: government tax. Flat 0.1% on sell side for delivery. Cannot be
  avoided.
- Stamp duty: paid on the buy side. Government revenue.
- GST: 18% on the brokerage + exchange transaction component.
- SEBI: tiny but real.
- Slippage: market-impact estimate. Per-stock overrides reflect bid-ask
  spreads; this is the largest *variable* cost component.

We deliberately do NOT include DP charges (₹13.5 per ISIN per day for
sells in delivery) -- that is a fixed-per-exit fee that varies by broker
and matters only for very small trades. Add it via `extra_per_exit_rupees`
if you need pessimistic accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("backtest.cost")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """Frozen cost configuration. Build once, reuse for every fill."""

    segment: str = "EQ_DELIVERY"
    brokerage_flat_rupees: float = 0.0
    brokerage_pct: float = 0.0003
    stt_buy_pct: float = 0.0
    stt_sell_pct: float = 0.001
    exchange_txn_pct: float = 0.0000325
    sebi_charges_per_crore_rupees: float = 10.0
    stamp_duty_buy_pct: float = 0.00015
    gst_pct_on_brokerage_and_txn: float = 0.18
    slippage_pct_default: float = 0.0005
    per_stock_slippage_pct: dict[str, float] = field(default_factory=dict)
    extra_per_exit_rupees: float = 0.0  # e.g., DP charges if you want them

    def slippage_for(self, symbol: str) -> float:
        return self.per_stock_slippage_pct.get(symbol.upper(),
                                               self.slippage_pct_default)


def load_cost_config(
    *,
    intraday: bool = False,
    path: str | Path | None = None,
) -> CostConfig:
    """Load CostConfig from config/cost_model.yaml (or a custom path)."""
    if path is None:
        path = project_root() / "config" / "cost_model.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    base = raw.get("defaults", {}) or {}
    if intraday:
        for k, v in (raw.get("intraday_overrides", {}) or {}).items():
            base[k] = v

    brokerage = base.get("brokerage", {}) or {}
    stt = base.get("stt", {}) or {}

    return CostConfig(
        segment=str(base.get("segment", "EQ_DELIVERY")),
        brokerage_flat_rupees=float(brokerage.get("flat_rupees", 0.0)),
        brokerage_pct=float(brokerage.get("pct", 0.0003)),
        stt_buy_pct=float(stt.get("buy_pct", 0.0)),
        stt_sell_pct=float(stt.get("sell_pct", 0.001)),
        exchange_txn_pct=float(base.get("exchange_txn_pct", 0.0000325)),
        sebi_charges_per_crore_rupees=float(
            base.get("sebi_charges_per_crore_rupees", 10.0)
        ),
        stamp_duty_buy_pct=float(base.get("stamp_duty_buy_pct", 0.00015)),
        gst_pct_on_brokerage_and_txn=float(
            base.get("gst_pct_on_brokerage_and_txn", 0.18)
        ),
        slippage_pct_default=float(base.get("slippage_pct_default", 0.0005)),
        per_stock_slippage_pct={
            str(k).upper(): float(v)
            for k, v in (raw.get("per_stock_slippage_pct", {}) or {}).items()
        },
    )


# ---------------------------------------------------------------------------
# Per-leg cost
# ---------------------------------------------------------------------------


@dataclass
class LegCost:
    side: str                   # "BUY" or "SELL"
    notional: float
    brokerage: float
    exchange_txn: float
    sebi: float
    gst: float
    stamp_duty: float
    stt: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage + self.exchange_txn + self.sebi + self.gst
            + self.stamp_duty + self.stt + self.slippage
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "side": self.side,
            "notional": round(self.notional, 4),
            "brokerage": round(self.brokerage, 4),
            "exchange_txn": round(self.exchange_txn, 4),
            "sebi": round(self.sebi, 4),
            "gst": round(self.gst, 4),
            "stamp_duty": round(self.stamp_duty, 4),
            "stt": round(self.stt, 4),
            "slippage": round(self.slippage, 4),
            "total": round(self.total, 4),
        }


def _brokerage(notional: float, cfg: CostConfig) -> float:
    if cfg.brokerage_flat_rupees > 0:
        return min(cfg.brokerage_flat_rupees, cfg.brokerage_pct * notional)
    # Many discount brokers charge 0 on delivery; pct still applies as a floor.
    return cfg.brokerage_pct * notional if cfg.brokerage_flat_rupees == 0 \
        else cfg.brokerage_flat_rupees


def _sebi(notional: float, cfg: CostConfig) -> float:
    return cfg.sebi_charges_per_crore_rupees * (notional / 1e7)


def compute_leg_cost(
    side: str,
    *,
    price: float,
    qty: int,
    symbol: str,
    cfg: CostConfig,
) -> LegCost:
    """Compute the cost (in rupees) for ONE leg of a trade."""
    if qty <= 0:
        raise ValueError("qty must be > 0")
    if price <= 0:
        raise ValueError("price must be > 0")
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")

    notional = price * qty

    brokerage = _brokerage(notional, cfg)
    exchange_txn = cfg.exchange_txn_pct * notional
    sebi = _sebi(notional, cfg)
    gst = cfg.gst_pct_on_brokerage_and_txn * (brokerage + exchange_txn)
    slippage = cfg.slippage_for(symbol) * notional

    if side == "BUY":
        stamp_duty = cfg.stamp_duty_buy_pct * notional
        stt = cfg.stt_buy_pct * notional
    else:
        stamp_duty = 0.0
        stt = cfg.stt_sell_pct * notional

    return LegCost(
        side=side, notional=notional,
        brokerage=brokerage, exchange_txn=exchange_txn, sebi=sebi,
        gst=gst, stamp_duty=stamp_duty, stt=stt, slippage=slippage,
    )


# ---------------------------------------------------------------------------
# Round-trip helpers (used by threshold tuning + reporting)
# ---------------------------------------------------------------------------


@dataclass
class RoundTripCost:
    buy: LegCost
    sell: LegCost
    extra: float = 0.0          # DP charges, etc.

    @property
    def total_rupees(self) -> float:
        return self.buy.total + self.sell.total + self.extra

    @property
    def total_pct(self) -> float:
        # Expressed as a fraction of the entry notional (so it slots into
        # return-space PnL math directly).
        if self.buy.notional == 0:
            return 0.0
        return self.total_rupees / self.buy.notional


def compute_round_trip(
    *,
    symbol: str,
    entry_price: float,
    exit_price: float,
    qty: int,
    cfg: CostConfig,
    extra_per_exit_rupees: float | None = None,
) -> RoundTripCost:
    buy = compute_leg_cost("BUY", price=entry_price, qty=qty, symbol=symbol, cfg=cfg)
    sell = compute_leg_cost("SELL", price=exit_price, qty=qty, symbol=symbol, cfg=cfg)
    extra = (
        cfg.extra_per_exit_rupees
        if extra_per_exit_rupees is None
        else float(extra_per_exit_rupees)
    )
    return RoundTripCost(buy=buy, sell=sell, extra=extra)


def round_trip_pct(symbol: str, cfg: CostConfig,
                   *, notional: float = 1_000_000.0) -> float:
    """Cheap helper: round-trip cost as a fraction of notional (for threshold
    tuning). Uses a representative notional so brokerage caps apply correctly.
    """
    qty = max(1, int(notional / 100.0))  # arbitrary; pct is scale-invariant
    rt = compute_round_trip(
        symbol=symbol, entry_price=100.0, exit_price=100.0,
        qty=qty, cfg=cfg,
    )
    return rt.total_pct
