"""Cointegration statistics (pure: NumPy only, no DB / network).

Engle-Granger two-step test for a pair (Y, X):

1. OLS ``Y = alpha + beta*X + resid`` -> ``beta`` is the hedge ratio and the
   residual ``Y - alpha - beta*X`` is the (mean-zero) spread.
2. Augmented Dickey-Fuller test on that residual. If the ADF t-statistic is
   below (more negative than) the Engle-Granger critical value, we reject a unit
   root: the spread is stationary and the pair is cointegrated.

Critical values are the Engle-Granger residual-based values for two series with
a constant (MacKinnon), NOT the plain ADF values -- residual-based tests need a
more negative threshold because beta was estimated. We default to the 5% value
(-3.34) and additionally require a sane mean-reversion half-life, which guards
against the approximation (no statsmodels dependency) being too permissive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Engle-Granger residual-based critical values (2 series, constant, large n).
EG_CRITICAL = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


@dataclass(frozen=True)
class CointResult:
    n_obs: int
    alpha: float
    beta: float
    adf_tstat: float
    half_life: float | None
    spread_mean: float
    spread_std: float
    cointegrated: bool
    pvalue_level: float | None  # tightest level passed (0.01/0.05/0.10) or None


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (coeffs, residuals) for ``y ~ X`` (X already includes a const)."""
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coeffs
    return coeffs, resid


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of ``y`` on ``x`` with intercept. Returns (alpha, beta, spread)."""
    X = np.column_stack([np.ones_like(x), x])
    coeffs, resid = _ols(y, X)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    return alpha, beta, resid


def adf_tstat(series: np.ndarray, lags: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic on ``series`` (with constant).

    Regress ``Δs_t = c + gamma*s_{t-1} + Σ delta_i Δs_{t-i} + e`` and return the
    t-stat on ``gamma``. More negative => stronger evidence of stationarity.
    Returns ``+inf`` (i.e. "definitely a unit root") when there isn't enough
    data to fit.
    """
    s = np.asarray(series, dtype=float)
    ds = np.diff(s)
    n = len(ds)
    if n <= lags + 2:
        return float("inf")

    # Rows align to t = lags .. n-1 of the differenced series.
    y = ds[lags:]
    cols = [s[lags:-1]]                       # s_{t-1} (level)
    for i in range(1, lags + 1):
        cols.append(ds[lags - i: -i])         # lagged differences
    X = np.column_stack([np.ones(len(y)), *cols])

    k = X.shape[1]
    if len(y) <= k:
        return float("inf")

    coeffs, resid = _ols(y, X)
    dof = len(y) - k
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("inf")

    se_gamma = math.sqrt(sigma2 * xtx_inv[1, 1])    # gamma is column index 1
    if se_gamma == 0.0 or not math.isfinite(se_gamma):
        return float("inf")
    return float(coeffs[1] / se_gamma)


def half_life(spread: np.ndarray) -> float | None:
    """Mean-reversion half-life (in observations) from an AR(1) fit:
    ``Δs_t = a + b*s_{t-1}``; half-life = -ln(2)/b. ``None`` if not mean-
    reverting (b >= 0) or unfittable."""
    s = np.asarray(spread, dtype=float)
    if len(s) < 3:
        return None
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([np.ones_like(lag), lag])
    coeffs, _ = _ols(ds, X)
    b = float(coeffs[1])
    # b must be meaningfully negative to mean-revert; values at/near zero
    # (flat/trending series, FP noise) imply no reversion within any sane window.
    if b >= -1e-8 or not math.isfinite(b):
        return None
    return -math.log(2.0) / b


def engle_granger(
    y: np.ndarray,
    x: np.ndarray,
    *,
    lags: int = 1,
    level: float = 0.05,
    max_half_life: float | None = 120.0,
    min_half_life: float = 1.0,
) -> CointResult:
    """Full two-step Engle-Granger screen for the pair (Y, X).

    ``cointegrated`` requires the ADF t-stat to clear the critical value at
    ``level`` AND (when ``max_half_life`` is set) a half-life within
    ``[min_half_life, max_half_life]`` -- a tradeable pair must revert within a
    sensible window, not over years.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = int(min(len(y), len(x)))
    if n < 30:
        return CointResult(n, 0.0, 0.0, float("inf"), None, 0.0, 0.0, False, None)

    y, x = y[-n:], x[-n:]
    alpha, beta, spread = hedge_ratio(y, x)
    t = adf_tstat(spread, lags=lags)
    hl = half_life(spread)
    s_mean = float(np.mean(spread))
    s_std = float(np.std(spread, ddof=1)) if n > 1 else 0.0

    # Tightest level the t-stat passes.
    passed: float | None = None
    for lv in (0.01, 0.05, 0.10):
        if t <= EG_CRITICAL[lv]:
            passed = lv
            break

    cointegrated = t <= EG_CRITICAL[level]
    if cointegrated and max_half_life is not None:
        cointegrated = hl is not None and min_half_life <= hl <= max_half_life

    return CointResult(
        n_obs=n, alpha=alpha, beta=beta, adf_tstat=t, half_life=hl,
        spread_mean=s_mean, spread_std=s_std,
        cointegrated=bool(cointegrated), pvalue_level=passed,
    )
