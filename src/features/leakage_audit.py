"""The leakage audit.

The single most common quant-research bug is a feature that secretly looks
into the future. This module gives us a property-based test we can run
against any feature function:

    For a randomly-generated price frame X of length N, compute feature f(X).
    Construct X' which is identical to X for rows 0..K and arbitrary
    afterwards (rows K+1..N-1 are mutated). Compute f(X').
    For all rows i <= K, f(X)[i] must equal f(X')[i].

If a feature uses any future-looking slice (e.g. close.shift(-1), or some
rolling operation centered around `i`), values at row i will change when
we mutate rows j > i, and the audit fails.

Usage:
    from src.features import leakage_audit as la
    la.assert_no_future_dependence(lambda df: rsi(df['close']), n=200)

We also expose a `LeakageReport` you can print or persist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class LeakageReport:
    feature_name: str
    n_rows: int
    cut: int
    leak_rows: int
    max_abs_diff: float

    @property
    def passed(self) -> bool:
        return self.leak_rows == 0


def _synthetic_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """A plausible random-walk OHLCV frame for stress-testing features."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=n)
    close = 1000 * np.exp(np.cumsum(rets))
    intraday = rng.uniform(0.005, 0.02, size=n)
    high = close * (1 + intraday / 2)
    low = close * (1 - intraday / 2)
    open_ = close + rng.normal(0, 0.5, size=n)
    open_ = np.clip(open_, low, high)
    volume = rng.integers(50_000, 5_000_000, size=n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _to_frame(result) -> pd.DataFrame:
    if isinstance(result, pd.Series):
        return result.to_frame(name=result.name or "value")
    if isinstance(result, pd.DataFrame):
        return result
    raise TypeError(
        f"Feature function must return Series or DataFrame; got {type(result)!r}"
    )


def audit_feature(
    fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    *,
    feature_name: str,
    n: int = 200,
    cut: int | None = None,
    seed: int = 7,
    tol: float = 1e-9,
) -> LeakageReport:
    """Run the leakage audit on a feature function.

    `fn` takes a price frame and returns a Series or DataFrame whose index
    matches the input frame. We compute fn on (a) the original frame and
    (b) a copy in which rows after `cut` have been replaced with random
    noise. Any row at index <= cut whose value differs between the two
    runs indicates a future-data dependency.
    """
    df = _synthetic_frame(n=n, seed=seed)
    if cut is None:
        cut = n // 2

    df_mutated = df.copy()
    rng = np.random.default_rng(seed + 1)
    for col in ("open", "high", "low", "close"):
        df_mutated.loc[df.index[cut + 1 :], col] = rng.uniform(
            500, 2000, size=n - cut - 1
        )
    df_mutated.loc[df.index[cut + 1 :], "volume"] = rng.integers(
        10_000, 10_000_000, size=n - cut - 1
    )
    # Re-impose OHLC ordering after mutation
    df_mutated["high"] = df_mutated[["open", "high", "low", "close"]].max(axis=1)
    df_mutated["low"] = df_mutated[["open", "high", "low", "close"]].min(axis=1)

    out_a = _to_frame(fn(df))
    out_b = _to_frame(fn(df_mutated))

    a = out_a.iloc[: cut + 1]
    b = out_b.iloc[: cut + 1]

    diff = (a - b).abs()
    leak_mask = (diff > tol).fillna(False)
    leak_rows = int(leak_mask.any(axis=1).sum())
    max_abs = float(np.nan_to_num(diff.to_numpy()).max(initial=0.0))

    return LeakageReport(
        feature_name=feature_name,
        n_rows=n,
        cut=cut,
        leak_rows=leak_rows,
        max_abs_diff=max_abs,
    )


def assert_no_future_dependence(
    fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    *,
    feature_name: str,
    n: int = 200,
    cut: int | None = None,
    seed: int = 7,
    tol: float = 1e-9,
) -> None:
    rep = audit_feature(
        fn, feature_name=feature_name, n=n, cut=cut, seed=seed, tol=tol
    )
    if not rep.passed:
        raise AssertionError(
            f"LEAKAGE in {rep.feature_name}: "
            f"{rep.leak_rows} rows at index <= cut={rep.cut} changed "
            f"when only future rows were mutated. "
            f"max |diff| = {rep.max_abs_diff:.6g}"
        )
