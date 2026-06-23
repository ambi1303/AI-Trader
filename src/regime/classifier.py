"""Rule-based market-regime classifier (pure, no IO).

Inputs are scalars derived from market context (NIFTY vs its 50/200-day MAs,
India VIX, and breadth from ``breadth_features``). Output is one of the five
labels in ``src.regime`` plus the human-readable reasons behind it.

Two design choices that matter:

* **Risk-off reacts immediately.** CRISIS / HIGH_VOLATILITY are driven by VIX
  and breadth collapse and bypass hysteresis -- when the market breaks you want
  the defensive switch *today*, not after a confirmation lag.
* **Trend transitions are sticky (Schmitt trigger).** BULL/RANGE/BEAR flips
  require breadth to cross a band, and ``prev`` widens the band in favour of the
  current label, so the router doesn't thrash strategies (and rack up costs) on
  a single borderline day.

All thresholds are module constants so they're easy to tune and unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.regime import (
    BEAR_TREND,
    BULL_TREND,
    CRISIS,
    HIGH_VOLATILITY,
    RANGE,
)

# --- VIX bands (India VIX points) ------------------------------------------
VIX_CRISIS = 30.0          # >= this -> CRISIS (risk-off, no new entries)
VIX_HIGH = 22.0            # >= this -> HIGH_VOLATILITY (trade smaller/breakout)

# --- Breadth bands (0..100 composite from compute_breadth) -----------------
BREADTH_BULL = 50.0        # need this much participation to call a bull trend
BREADTH_BEAR = 40.0        # below this (in a downtrend) -> bear trend
BREADTH_HYSTERESIS = 8.0   # band widened in favour of the current regime

# --- Crisis breadth floor (% of names above their 200-DMA) -----------------
CRISIS_PCT_ABOVE_200 = 20.0


@dataclass(frozen=True)
class RegimeInputs:
    """Scalar market-context snapshot for a single day."""
    nifty_ma50_gt_ma200: bool | None = None
    nifty_above_ma200: bool | None = None
    vix: float | None = None
    breadth_score: float | None = None       # 0..100
    pct_above_200dma: float | None = None     # 0..100


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    reasons: list[str] = field(default_factory=list)
    held_by_hysteresis: bool = False


def _trend_regime(x: RegimeInputs, prev: str | None) -> tuple[str, list[str], bool]:
    """BULL/BEAR/RANGE from trend + breadth, with prev-biased hysteresis."""
    reasons: list[str] = []
    up = bool(x.nifty_ma50_gt_ma200) and bool(x.nifty_above_ma200)
    down = (x.nifty_ma50_gt_ma200 is False) and (x.nifty_above_ma200 is False)
    breadth = x.breadth_score if x.breadth_score is not None else 50.0

    # Schmitt trigger: it's *easier to stay* in the current trend regime.
    bull_thr = BREADTH_BULL - (BREADTH_HYSTERESIS if prev == BULL_TREND else 0.0)
    bear_thr = BREADTH_BEAR + (BREADTH_HYSTERESIS if prev == BEAR_TREND else 0.0)

    if up and breadth >= bull_thr:
        reasons.append(
            f"NIFTY uptrend (50>200 EMA, above 200) + breadth {breadth:.0f}>={bull_thr:.0f}"
        )
        raw = BULL_TREND
    elif down and breadth <= bear_thr:
        reasons.append(
            f"NIFTY downtrend (50<200 EMA, below 200) + breadth {breadth:.0f}<={bear_thr:.0f}"
        )
        raw = BEAR_TREND
    else:
        reasons.append(
            f"no decisive trend (breadth {breadth:.0f}); ranging"
        )
        raw = RANGE

    held = False
    if prev in (BULL_TREND, BEAR_TREND, RANGE) and prev != raw:
        # The band above already biases toward `prev`; if despite that the raw
        # label still differs, the move is genuine -> allow the switch. (The
        # stickiness lives in the threshold, not in an extra veto.)
        held = False
    return raw, reasons, held


def classify_regime(x: RegimeInputs, prev: str | None = None) -> RegimeResult:
    """Classify the market into one of the five regimes.

    ``prev`` is yesterday's regime; pass it so trend transitions are sticky.
    """
    reasons: list[str] = []

    # 1) CRISIS -- immediate, overrides everything.
    if x.vix is not None and x.vix >= VIX_CRISIS:
        return RegimeResult(CRISIS, [f"VIX {x.vix:.1f} >= {VIX_CRISIS:.0f} (crisis)"])
    if (x.pct_above_200dma is not None
            and x.pct_above_200dma < CRISIS_PCT_ABOVE_200
            and x.nifty_above_ma200 is False):
        return RegimeResult(
            CRISIS,
            [f"breadth collapse: only {x.pct_above_200dma:.0f}% above 200-DMA "
             f"and NIFTY below 200-DMA"],
        )

    # 2) HIGH_VOLATILITY -- immediate.
    if x.vix is not None and x.vix >= VIX_HIGH:
        return RegimeResult(
            HIGH_VOLATILITY, [f"VIX {x.vix:.1f} >= {VIX_HIGH:.0f} (elevated)"]
        )

    # 3) Trend regimes (sticky).
    raw, trend_reasons, held = _trend_regime(x, prev)
    reasons.extend(trend_reasons)
    return RegimeResult(raw, reasons, held_by_hysteresis=held)
