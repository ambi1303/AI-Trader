"""Circuit and liquidity features.

- `hit_upper_circuit`, `hit_lower_circuit`: from the circuit_flags table.
- `days_since_circuit`: number of trading days since the last circuit event.
- `low_volume_flag`: 1 when today's volume is below 20% of the 20d median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_circuit_flags(
    df: pd.DataFrame, circuit_df: pd.DataFrame | None
) -> pd.DataFrame:
    """`df` is the per-symbol price frame. `circuit_df` has columns
    [bar_date, hit_upper, hit_lower] for the same symbol.

    Returns df with three new columns: hit_upper_circuit, hit_lower_circuit,
    days_since_circuit.
    """
    out = df.copy()
    out["hit_upper_circuit"] = 0
    out["hit_lower_circuit"] = 0
    if circuit_df is not None and not circuit_df.empty:
        m = circuit_df.set_index("bar_date")
        out.loc[out.index.isin(m.index), "hit_upper_circuit"] = (
            m["hit_upper"].astype(int).reindex(out.index).fillna(0).astype(int)
        )
        out.loc[out.index.isin(m.index), "hit_lower_circuit"] = (
            m["hit_lower"].astype(int).reindex(out.index).fillna(0).astype(int)
        )

    any_circuit = (out["hit_upper_circuit"] | out["hit_lower_circuit"]).astype(int)
    # Distance to last 1, going forward only. When a stock has had NO circuit
    # event in the visible history we use a large sentinel (9999) -- this is
    # semantically "very far from a circuit", which is what we want the model
    # to interpret. Returning NaN here would unhelpfully hide every row of a
    # well-behaved stock from any model that drops NaNs.
    NO_CIRCUIT_SENTINEL = 9999
    days_since = []
    last = None
    for i, v in enumerate(any_circuit.tolist()):
        if v == 1:
            last = i
            days_since.append(0)
        elif last is None:
            days_since.append(NO_CIRCUIT_SENTINEL)
        else:
            days_since.append(i - last)
    out["days_since_circuit"] = pd.Series(days_since, index=out.index, dtype="float64")
    return out


def low_volume_flag(volume: pd.Series, window: int = 20, threshold: float = 0.20) -> pd.Series:
    """1 when today's volume < threshold * rolling median of last `window` days."""
    median = volume.rolling(window=window, min_periods=window).median()
    return ((volume < threshold * median).fillna(False).astype(int)).rename(
        "low_volume_flag"
    )
