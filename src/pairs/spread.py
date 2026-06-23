"""Spread standardisation + signal labelling (pure: NumPy only).

Given a cointegrated pair (Y, X) with hedge ratio ``beta``, the spread is
``Y - beta*X``. We standardise it to a z-score and map that to a desired action
on the *spread*:

* ``z >= +entry``  -> spread rich  -> SHORT_SPREAD (short Y, long X)
* ``z <= -entry``  -> spread cheap -> LONG_SPREAD  (long Y, short X)
* ``|z| <= exit``  -> reverted     -> EXIT (close any open pair position)
* ``|z| >= stop``  -> diverging    -> FLAT (relationship may be breaking; stand
  aside / stop out)
* otherwise        -> HOLD (in the band; no fresh action)

Signals describe the spread leg; the actual long-only execution caveat lives in
the package docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LONG_SPREAD = "LONG_SPREAD"
SHORT_SPREAD = "SHORT_SPREAD"
EXIT = "EXIT"
FLAT = "FLAT"
HOLD = "HOLD"


@dataclass(frozen=True)
class SpreadSignal:
    zscore: float
    signal: str
    spread: float
    mean: float
    std: float


def compute_spread(y: np.ndarray, x: np.ndarray, beta: float,
                   alpha: float = 0.0) -> np.ndarray:
    """Spread series ``Y - alpha - beta*X``."""
    return np.asarray(y, dtype=float) - alpha - beta * np.asarray(x, dtype=float)


def latest_zscore(spread: np.ndarray, window: int = 60) -> tuple[float, float, float]:
    """(z, mean, std) of the most recent spread point over a trailing ``window``.

    Falls back to the full series when shorter than ``window``. Returns
    ``(0.0, mean, 0.0)`` when the window has no dispersion.
    """
    s = np.asarray(spread, dtype=float)
    if s.size == 0:
        return 0.0, 0.0, 0.0
    w = s[-window:] if s.size > window else s
    mean = float(np.mean(w))
    std = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
    if std <= 0.0:
        return 0.0, mean, 0.0
    return (float(s[-1]) - mean) / std, mean, std


def classify(z: float, *, entry: float = 2.0, exit: float = 0.5,
             stop: float = 3.5) -> str:
    """Map a z-score to a desired spread action (see module docstring)."""
    az = abs(z)
    if az >= stop:
        return FLAT
    if az <= exit:
        return EXIT
    if z >= entry:
        return SHORT_SPREAD
    if z <= -entry:
        return LONG_SPREAD
    return HOLD


def signal_for(
    y: np.ndarray,
    x: np.ndarray,
    beta: float,
    alpha: float = 0.0,
    *,
    window: int = 60,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
) -> SpreadSignal:
    """End-to-end: spread -> latest z -> signal label for a pair."""
    spread = compute_spread(y, x, beta, alpha)
    z, mean, std = latest_zscore(spread, window=window)
    sig = classify(z, entry=entry, exit=exit, stop=stop)
    last = float(spread[-1]) if spread.size else 0.0
    return SpreadSignal(zscore=z, signal=sig, spread=last, mean=mean, std=std)
