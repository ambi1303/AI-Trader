"""Pre-defined stress windows for Indian markets.

These windows are well-known regime shifts where many naive long-only models
historically blew up. Running the backtest restricted to each window is the
single most useful sanity check: a model with positive in-sample Sharpe
but a -50% drawdown during March 2020 is not a model you want to live-trade.

Windows are loose; pad by 7 days each side so we capture entries placed
just before the regime kicked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class StressWindow:
    name: str
    start: date
    end: date
    description: str


STRESS_WINDOWS: tuple[StressWindow, ...] = (
    StressWindow(
        name="covid_2020",
        start=date(2020, 2, 1),
        end=date(2020, 5, 31),
        description="COVID-19 crash and recovery: 38% peak-to-trough drop in NIFTY 50.",
    ),
    StressWindow(
        name="rate_hike_2022",
        start=date(2022, 4, 1),
        end=date(2022, 7, 31),
        description="Aggressive RBI/Fed rate hikes; 12% NIFTY drawdown.",
    ),
    StressWindow(
        name="adani_2023",
        start=date(2023, 1, 24),
        end=date(2023, 3, 15),
        description="Hindenburg report on Adani group; sector-specific shock.",
    ),
    StressWindow(
        name="election_2024",
        start=date(2024, 5, 15),
        end=date(2024, 6, 15),
        description="Modi 3.0 election outcome; 8% intraday vol on result day.",
    ),
    StressWindow(
        name="oct_2024_correction",
        start=date(2024, 10, 1),
        end=date(2024, 11, 30),
        description="FII outflows + domestic-vs-FPI rotation; 11% NIFTY correction.",
    ),
)


def get_stress_window(name: str) -> StressWindow:
    for w in STRESS_WINDOWS:
        if w.name == name:
            return w
    raise KeyError(f"Unknown stress window: {name!r}. "
                   f"Choices: {[w.name for w in STRESS_WINDOWS]}")
