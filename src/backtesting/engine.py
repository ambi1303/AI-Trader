"""Bar-by-bar long-only equity backtester.

Inputs:
  - `predictions`: per (symbol, prediction_date) calibrated probability +
                   threshold. Typically the Week-3 walk-forward output.
  - `prices`:      per (symbol, bar_date) OHLCV from BhavCopy/yfinance.
  - `atr_by_symbol_date`: per (symbol, bar_date) ATR(14) used for stops/sizing.
  - `sectors`:     per-symbol sector mapping (used for concentration cap).
  - SizingConfig + RiskConfig + CostConfig.

Conventions (auditable, documented):
  - A signal at PREDICTION_DATE = T fires using info available at end of day T.
  - We ENTER at the OPEN of day T+1 (next-day-open fill).
  - Stops/targets are checked intraday on day T+1 onwards using H/L bars.
  - We MARK-TO-MARKET equity at every bar's close for every open position.
  - On the last bar of the simulation, all open positions are CLOSED at close
    so we don't leave phantom equity on the books.

This engine is intentionally unvectorised: a Python loop over dates and
positions is fast enough at 250 days * 50 symbols * 5 years (~62k iterations)
and far easier to reason about for stop/target sequencing than a clever
vectorised version.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtesting.cost_model import CostConfig, compute_round_trip
from src.backtesting.metrics import BacktestMetrics, compute_metrics
from src.backtesting.regime_analysis import summarize_by_regime
from src.backtesting.risk import (
    OpenPosition,
    RiskConfig,
    can_open_new_position,
    check_stop_or_target,
)
from src.backtesting.sizing import SizingConfig, size_position
from src.utils.logger import get_logger

log = get_logger("backtest.engine")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    bt_run_id: str
    symbol: str
    side: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    cost_rupees: float
    net_pnl: float
    holding_days: int
    exit_reason: str
    entry_prob: float
    threshold: float
    entry_regime: str | None = None


@dataclass
class BacktestResult:
    bt_run_id: str
    name: str
    initial_capital: float
    start_date: str
    end_date: str
    config: dict
    metrics: BacktestMetrics
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_regime: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    initial_capital: float = 1_000_000.0
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    name: str = "default"
    require_atr: bool = True


def _new_bt_run_id(name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"bt-{name}-{ts}-{uuid.uuid4().hex[:6]}"


def _to_iso(d) -> str:
    if isinstance(d, str):
        return d
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _validate_inputs(predictions: pd.DataFrame, prices: pd.DataFrame,
                     atr: pd.DataFrame, sectors: dict[str, str]) -> None:
    req_pred = {"symbol", "feature_date", "calibrated_prob"}
    req_price = {"symbol", "bar_date", "open", "high", "low", "close"}
    req_atr = {"symbol", "bar_date", "atr"}
    if not req_pred.issubset(predictions.columns):
        raise ValueError(f"predictions missing cols: {req_pred - set(predictions.columns)}")
    if not req_price.issubset(prices.columns):
        raise ValueError(f"prices missing cols: {req_price - set(prices.columns)}")
    if not req_atr.issubset(atr.columns):
        raise ValueError(f"atr missing cols: {req_atr - set(atr.columns)}")
    if not sectors:
        log.warning("No sector map supplied; sector caps will be ineffective.")


def run_backtest(
    *,
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    atr: pd.DataFrame,
    sectors: dict[str, str],
    threshold: float,
    cfg: EngineConfig | None = None,
    regime_by_date: "dict | None" = None,
) -> BacktestResult:
    cfg = cfg or EngineConfig()
    _validate_inputs(predictions, prices, atr, sectors)

    bt_run_id = _new_bt_run_id(cfg.name)
    # Per-regime tagging: regime active on a given trading day (as-of lookup).
    # Empty/None mapping -> every tag is None (regime-agnostic backtest).
    regime_on = _make_regime_getter(regime_by_date)

    # ----- prep frames -----
    pred = predictions.copy()
    pred["feature_date"] = pd.to_datetime(pred["feature_date"]).dt.date
    pred = pred.sort_values(["feature_date", "symbol"]).reset_index(drop=True)

    pr = prices.copy()
    pr["bar_date"] = pd.to_datetime(pr["bar_date"]).dt.date
    pr = pr.sort_values(["bar_date", "symbol"]).reset_index(drop=True)

    a = atr.copy()
    a["bar_date"] = pd.to_datetime(a["bar_date"]).dt.date

    # Pre-compute per-symbol per-date lookup dicts for O(1) access.
    price_lookup: dict[tuple[str, "pd.Timestamp"], dict] = {}
    for r in pr.itertuples(index=False):
        price_lookup[(r.symbol, r.bar_date)] = {
            "open": float(r.open), "high": float(r.high),
            "low": float(r.low), "close": float(r.close),
        }
    atr_lookup: dict[tuple[str, "pd.Timestamp"], float] = {
        (r.symbol, r.bar_date): float(r.atr) for r in a.itertuples(index=False)
        if pd.notna(r.atr)
    }
    pred_lookup: dict[tuple[str, "pd.Timestamp"], float] = {
        (r.symbol, r.feature_date): float(r.calibrated_prob)
        for r in pred.itertuples(index=False)
        if pd.notna(r.calibrated_prob)
    }

    # The simulation runs over the union of all available trading dates.
    all_dates = sorted(pr["bar_date"].unique())
    if not all_dates:
        log.warning("No price data; nothing to simulate.")
        return BacktestResult(
            bt_run_id=bt_run_id, name=cfg.name,
            initial_capital=cfg.initial_capital,
            start_date="", end_date="",
            config=_config_to_json(cfg),
            metrics=compute_metrics(
                equity_curve=pd.DataFrame(columns=["bar_date", "equity"]),
                trades=pd.DataFrame(),
                initial_capital=cfg.initial_capital,
            ),
        )

    # ----- state -----
    cash = cfg.initial_capital
    open_positions: dict[str, OpenPosition] = {}
    cooldown_until: dict[str, "pd.Timestamp"] = {}
    trades: list[TradeRecord] = []
    equity_rows: list[dict] = []

    prev_equity = cfg.initial_capital

    # The signal queue: predictions on day T act on entries at OPEN of T+1.
    # Build a per-date list of (symbol, prob) once.
    signals_by_date: dict = {}
    for (sym, d), prob in pred_lookup.items():
        if prob >= threshold:
            signals_by_date.setdefault(d, []).append((sym, prob))

    for i, today in enumerate(all_dates):
        # 1. Process exits and trailing stops on today's bar for any open position.
        symbols_to_exit: list[tuple[str, float, str]] = []
        for sym, pos in list(open_positions.items()):
            bar = price_lookup.get((sym, today))
            if bar is None:
                # No bar today (holiday for that ticker?) -- skip.
                continue
            pos.update_trailing_stop(bar["high"], cfg.risk)
            hr = check_stop_or_target(
                pos,
                bar_open=bar["open"], bar_high=bar["high"],
                bar_low=bar["low"], bar_close=bar["close"],
            )
            if hr.hit:
                symbols_to_exit.append((sym, hr.fill_price, hr.reason))
                continue
            # Time stop
            holding = i - all_dates.index(_iso_to_date(pos.entry_date))
            if holding >= cfg.risk.max_holding_days:
                symbols_to_exit.append((sym, bar["close"], "time"))

        for sym, fill_price, reason in symbols_to_exit:
            pos = open_positions.pop(sym)
            rt = compute_round_trip(
                symbol=sym, entry_price=pos.entry_price, exit_price=fill_price,
                qty=pos.qty, cfg=cfg.cost,
            )
            gross = (fill_price - pos.entry_price) * pos.qty
            net = gross - rt.total_rupees
            cash += pos.entry_price * pos.qty + net  # return notional + net pnl
            holding_days = i - all_dates.index(_iso_to_date(pos.entry_date))
            trades.append(TradeRecord(
                bt_run_id=bt_run_id, symbol=sym, side=pos.side,
                entry_date=pos.entry_date, exit_date=_to_iso(today),
                entry_price=pos.entry_price, exit_price=fill_price,
                qty=pos.qty, gross_pnl=gross, cost_rupees=rt.total_rupees,
                net_pnl=net, holding_days=int(holding_days),
                exit_reason=reason, entry_prob=pos.entry_prob,
                threshold=pos.threshold,
                entry_regime=regime_on(_iso_to_date(pos.entry_date)),
            ))
            if net < 0 and cfg.risk.cooldown_days_after_loss > 0:
                cd_idx = min(i + cfg.risk.cooldown_days_after_loss,
                             len(all_dates) - 1)
                cooldown_until[sym] = all_dates[cd_idx]

        # 2. New entries triggered by YESTERDAY's signals, filled at TODAY's open.
        if i > 0:
            yday = all_dates[i - 1]
            today_pnl_pct = 0.0  # daily-loss limit checked against intraday cum pnl below
            for sym, prob in signals_by_date.get(yday, []):
                if sym in open_positions:
                    continue
                if cooldown_until.get(sym) and today < cooldown_until[sym]:
                    continue
                bar = price_lookup.get((sym, today))
                if bar is None:
                    continue
                atr_today = atr_lookup.get((sym, yday))
                if atr_today is None or atr_today <= 0:
                    if cfg.require_atr:
                        continue
                    atr_today = max(0.01, bar["open"] * 0.01)

                sector = sectors.get(sym, "UNKNOWN")
                equity_now = _equity(cash, open_positions, price_lookup, today)
                today_pnl_pct = (equity_now - prev_equity) / prev_equity \
                    if prev_equity > 0 else 0.0
                allowed, reason = can_open_new_position(
                    cfg=cfg.risk, open_positions=list(open_positions.values()),
                    sector=sector, today_pnl_pct=today_pnl_pct,
                )
                if not allowed:
                    continue

                entry_price = bar["open"]
                R = (
                    cfg.risk.take_profit_atr_mult / cfg.risk.stop_atr_mult
                    if cfg.risk.stop_atr_mult > 0 else 1.5
                )
                decision = size_position(
                    prob_win=prob, entry_price=entry_price, atr=atr_today,
                    stop_atr_mult=cfg.risk.stop_atr_mult,
                    equity=equity_now, cfg=cfg.sizing,
                    reward_to_risk=R,
                )
                if decision.qty <= 0:
                    continue

                buy_cost = compute_round_trip(
                    symbol=sym, entry_price=entry_price, exit_price=entry_price,
                    qty=decision.qty, cfg=cfg.cost,
                ).buy.total
                gross_outlay = entry_price * decision.qty + buy_cost
                if gross_outlay > cash:
                    continue
                cash -= gross_outlay
                open_positions[sym] = OpenPosition(
                    symbol=sym, sector=sector, side="LONG",
                    qty=decision.qty,
                    entry_date=_to_iso(today), entry_price=entry_price,
                    atr_at_entry=atr_today,
                    stop=cfg.risk.stop_for(entry_price, atr_today),
                    target=cfg.risk.target_for(entry_price, atr_today),
                    high_watermark=entry_price,
                    entry_prob=prob, threshold=threshold,
                )

        # 3. Mark-to-market and record equity row.
        equity_now = _equity(cash, open_positions, price_lookup, today)
        equity_rows.append({
            "bar_date": _to_iso(today),
            "cash": round(cash, 2),
            "equity": round(equity_now, 2),
            "open_count": len(open_positions),
            "daily_pnl": round(equity_now - prev_equity, 2),
            "regime": regime_on(today),
        })
        prev_equity = equity_now

    # 4. Force-close everything at the last bar's close.
    last_date = all_dates[-1]
    for sym, pos in list(open_positions.items()):
        bar = price_lookup.get((sym, last_date))
        if bar is None:
            continue
        rt = compute_round_trip(
            symbol=sym, entry_price=pos.entry_price, exit_price=bar["close"],
            qty=pos.qty, cfg=cfg.cost,
        )
        gross = (bar["close"] - pos.entry_price) * pos.qty
        net = gross - rt.total_rupees
        cash += pos.entry_price * pos.qty + net
        trades.append(TradeRecord(
            bt_run_id=bt_run_id, symbol=sym, side=pos.side,
            entry_date=pos.entry_date, exit_date=_to_iso(last_date),
            entry_price=pos.entry_price, exit_price=bar["close"],
            qty=pos.qty, gross_pnl=gross, cost_rupees=rt.total_rupees,
            net_pnl=net,
            holding_days=int(len(all_dates) - 1
                              - all_dates.index(_iso_to_date(pos.entry_date))),
            exit_reason="end", entry_prob=pos.entry_prob, threshold=pos.threshold,
            entry_regime=regime_on(_iso_to_date(pos.entry_date)),
        ))
    open_positions.clear()
    # Adjust last equity row to reflect closed-out portfolio.
    if equity_rows:
        equity_rows[-1]["equity"] = round(cash, 2)
        equity_rows[-1]["cash"] = round(cash, 2)
        equity_rows[-1]["open_count"] = 0

    eq_df = pd.DataFrame(equity_rows)
    tr_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame(
        columns=[f.name for f in TradeRecord.__dataclass_fields__.values()]
    )

    metrics = compute_metrics(
        equity_curve=eq_df, trades=tr_df, initial_capital=cfg.initial_capital,
    )
    by_regime = summarize_by_regime(tr_df, eq_df)

    log.info(
        "Backtest done | run_id={} trades={} equity_final={:.2f} "
        "Sharpe={:.2f} MaxDD={:.2f}% Hit={:.1f}%",
        bt_run_id, len(trades), eq_df["equity"].iloc[-1] if len(eq_df) else 0.0,
        metrics.sharpe, metrics.max_drawdown_pct, metrics.hit_rate_pct,
    )
    return BacktestResult(
        bt_run_id=bt_run_id, name=cfg.name,
        initial_capital=cfg.initial_capital,
        start_date=_to_iso(all_dates[0]), end_date=_to_iso(all_dates[-1]),
        config=_config_to_json(cfg),
        metrics=metrics,
        equity_curve=eq_df,
        trades=tr_df,
        by_regime=by_regime,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_to_date(s: str):
    return pd.Timestamp(s).date()


def _make_regime_getter(regime_by_date: dict | None):
    """Return a function date->regime label using as-of (latest <= date)
    semantics, so a trading day with no explicit entry inherits the last known
    regime. Empty/None mapping -> always None (regime-agnostic backtest)."""
    if not regime_by_date:
        return lambda _d: None

    import bisect

    norm: dict = {}
    for k, v in regime_by_date.items():
        kd = _iso_to_date(k) if isinstance(k, str) else k
        norm[kd] = v
    keys = sorted(norm)

    def get(d):
        dd = _iso_to_date(d) if isinstance(d, str) else d
        idx = bisect.bisect_right(keys, dd) - 1
        return norm[keys[idx]] if idx >= 0 else None

    return get


def _equity(cash: float, positions: dict[str, OpenPosition],
            price_lookup: dict, today) -> float:
    equity = cash
    for sym, pos in positions.items():
        bar = price_lookup.get((sym, today))
        if bar is not None:
            equity += pos.qty * bar["close"]
        else:
            equity += pos.qty * pos.entry_price
    return equity


def _config_to_json(cfg: EngineConfig) -> dict:
    """Make EngineConfig JSON-serializable for backtest_runs.config_json."""
    sizing = asdict(cfg.sizing)
    risk = asdict(cfg.risk)
    # CostConfig has a dict field per_stock_slippage_pct; convert frozen to dict
    cost = {
        "segment": cfg.cost.segment,
        "brokerage_flat_rupees": cfg.cost.brokerage_flat_rupees,
        "brokerage_pct": cfg.cost.brokerage_pct,
        "stt_buy_pct": cfg.cost.stt_buy_pct,
        "stt_sell_pct": cfg.cost.stt_sell_pct,
        "exchange_txn_pct": cfg.cost.exchange_txn_pct,
        "stamp_duty_buy_pct": cfg.cost.stamp_duty_buy_pct,
        "gst_pct_on_brokerage_and_txn": cfg.cost.gst_pct_on_brokerage_and_txn,
        "slippage_pct_default": cfg.cost.slippage_pct_default,
    }
    return {
        "name": cfg.name,
        "initial_capital": cfg.initial_capital,
        "sizing": sizing,
        "risk": risk,
        "cost": cost,
    }
