"""Training-matrix builder.

Convention (single source of truth, audited by tests):

  Features at row T   = info available at end of trading day T
                        (computed in src.features.feature_builder).
  Target  at row T    = (close[T+1] - close[T]) / close[T]  > THRESHOLD
                        i.e., we predict the next-day return.
  Action               = trade at the OPEN of T+1 based on the signal we
                        produced at the close of T. (Modelled as close-to-close
                        for v1; slippage absorbed by the cost model in Week 4.)

This means feature row T must NEVER use any data dated > T, and target row T
must use close[T+1] from the SAME source as features (we use BhavCopy raw
close throughout — that's what feature_builder also uses for `close`).

The last-date-per-symbol row is dropped because its target is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.utils.db import fetch_all
from src.utils.logger import get_logger

log = get_logger("models.dataset")


# ---------------------------------------------------------------------------
# Column lists. Must stay in sync with feature_data schema.
# ---------------------------------------------------------------------------

# These are NOT used as features even though they're in feature_data.
_NON_FEATURE_COLS = ("close", "volume")

# `feature_set_version` and `computed_at` are also persisted but excluded from X.
_META_COLS = ("symbol", "feature_date", "feature_set_version", "computed_at")

# Base features that get a cross-sectional (per-day, across-universe) percentile
# rank when cross_sectional_features=True. These are the columns that carry
# *relative* strength / value / risk information -- exactly the signal a purely
# per-symbol view throws away. Each produces an `xs_<col>` companion in [0,1]
# where 1.0 = highest in the universe that day.
_CROSS_SECTIONAL_BASE = (
    "ret_5d", "ret_20d",
    "mom_5d", "mom_20d", "mom_60d",
    "rsi_14",
    "dist_ema_20_pct", "dist_ema_50_pct", "dist_ema_200_pct",
    "macd_hist",
    "vol_z_20d", "vol_ratio_20d",
    "atr_pct",
    "dd_from_high_20d", "dd_from_high_60d",
    "bb_pct_b",
    "sector_rs_20d",
    "adx_14",
)


def add_cross_sectional_features(
    df: pd.DataFrame,
    base_cols: tuple[str, ...] = _CROSS_SECTIONAL_BASE,
    *,
    date_col: str = "feature_date",
) -> tuple[pd.DataFrame, list[str]]:
    """Append per-day, across-universe percentile-rank columns.

    For each base feature present in ``df`` we add ``xs_<feature>`` =
    rank within each ``date_col`` group, scaled to (0, 1] where 1.0 is the
    highest value in the universe that day. NaN inputs stay NaN (XGBoost
    handles them). This is the SINGLE definition of cross-sectional features
    used by BOTH training (dataset builder) and inference (predict), so the
    two can never silently diverge.

    Returns (df_with_xs, xs_column_names).
    """
    present = [c for c in base_cols if c in df.columns]
    xs_cols: list[str] = []
    if not present:
        return df, xs_cols
    grp = df.groupby(date_col)
    for col in present:
        xs_name = f"xs_{col}"
        df[xs_name] = grp[col].rank(pct=True, method="average")
        xs_cols.append(xs_name)
    return df, xs_cols


@dataclass
class TrainingMatrix:
    X: pd.DataFrame
    y: pd.Series
    forward_return: pd.Series  # the actual next-day return (for utility calcs)
    meta: pd.DataFrame  # symbol, feature_date — used for time-based splitting
    feature_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_feature_data(symbols: list[str] | None = None) -> pd.DataFrame:
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        rows = fetch_all(
            f"SELECT * FROM feature_data WHERE symbol IN ({placeholders})",  # noqa: S608
            tuple(symbols),
        )
    else:
        rows = fetch_all("SELECT * FROM feature_data")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["feature_date"] = pd.to_datetime(df["feature_date"]).dt.date
    return df.sort_values(["symbol", "feature_date"]).reset_index(drop=True)


def _next_day_returns_per_symbol(
    symbols: list[str], horizon: int = 1
) -> pd.DataFrame:
    """For each symbol we need close[T+h] / close[T] - 1 keyed at T.

    `horizon` (h) is the number of trading days ahead. h=1 is the
    next-day return; h=5 is roughly one trading week, h=20 ~ one month.
    Computed per-symbol from price_data (bhavcopy source) so it is the SAME
    raw close used in feature_data. Returns columns: symbol, feature_date,
    fwd_return.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "feature_date", "fwd_return"])
    h = max(1, int(horizon))
    placeholders = ",".join("?" * len(symbols))
    rows = fetch_all(
        f"""
        SELECT symbol, bar_date, close
        FROM   price_data
        WHERE  source = 'bhavcopy' AND symbol IN ({placeholders})
        ORDER BY symbol, bar_date
        """,  # noqa: S608
        tuple(symbols),
    )
    if not rows:
        return pd.DataFrame(columns=["symbol", "feature_date", "fwd_return"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date

    parts = []
    for sym, sub in df.groupby("symbol", sort=False):
        sub = sub.sort_values("bar_date").reset_index(drop=True)
        sub["next_close"] = sub["close"].shift(-h)
        sub["fwd_return"] = (sub["next_close"] - sub["close"]) / sub["close"]
        sub = sub.rename(columns={"bar_date": "feature_date"})
        parts.append(sub[["symbol", "feature_date", "fwd_return"]])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_matrix(
    symbols: list[str] | None = None,
    *,
    target_return_threshold: float = 0.005,
    label_mode: str = "absolute",
    label_quantile: float = 0.50,
    horizon: int = 1,
    cross_sectional_features: bool = False,
    drop_warmup: bool = False,
    min_non_nan_features: int | None = None,
) -> TrainingMatrix:
    """Build the training matrix.

    target_return_threshold: y = 1 if next-day return > this fraction.
                             Default 0.005 means > 0.5%. Used only when
                             label_mode == "absolute".

    label_mode: how the binary target is defined.
        "absolute"        -- y = 1 if next-day return > target_return_threshold.
                             This is a *directional* target dominated by
                             market beta (when the index rises, most stocks
                             rise), so on liquid names it is close to
                             unpredictable day-to-day.
        "cross_sectional" -- y = 1 if the stock's next-day return ranks in the
                             top (1 - label_quantile) fraction of ALL stocks on
                             that date. This is the standard quant-equity
                             framing: it strips out the common market move and
                             forces the model to learn *relative* strength,
                             which is where idiosyncratic edge actually lives.
                             `forward_return` is left untouched (still the real
                             return) so threshold tuning and backtest economics
                             remain in rupee-return space.
    label_quantile: cross-sectional cutoff. 0.50 -> label the top half (~50%
                 positive, balanced for learning); 0.70 -> top 30% (stricter,
                 fewer positives, closer to "clear outperformers").
    drop_warmup: drop rows where ANY required feature is NaN. Default False
                 because XGBoost handles missing values natively and can learn
                 from the missingness pattern (e.g., "first 200 days vs.
                 mature regime"). Set True if you plan to feed a model that
                 cannot handle NaN.
    min_non_nan_features: drop rows where the count of non-NaN features is
                 below this. Useful as a softer alternative to drop_warmup
                 (e.g., require at least 30 of 51 features non-NaN).
    """
    if label_mode not in ("absolute", "cross_sectional"):
        raise ValueError(
            f"label_mode must be 'absolute' or 'cross_sectional', got "
            f"{label_mode!r}"
        )
    feat = _load_feature_data(symbols)
    if feat.empty:
        log.warning("No feature_data rows found")
        return TrainingMatrix(
            X=pd.DataFrame(), y=pd.Series(dtype=int),
            forward_return=pd.Series(dtype=float), meta=pd.DataFrame(),
            feature_columns=[],
        )

    syms_present = feat["symbol"].unique().tolist()
    fwd = _next_day_returns_per_symbol(syms_present, horizon=horizon)

    df = feat.merge(fwd, on=["symbol", "feature_date"], how="inner")
    df = df.dropna(subset=["fwd_return"])  # last row per symbol has NaN
    log.info("Joined feature+target rows: {}", len(df))

    feature_cols = [
        c
        for c in df.columns
        if c not in _META_COLS
        and c not in _NON_FEATURE_COLS
        and c != "fwd_return"
    ]

    if drop_warmup:
        before = len(df)
        df = df.dropna(subset=feature_cols)
        log.info(
            "Dropped {} warmup rows with NaN features ({} -> {})",
            before - len(df),
            before,
            len(df),
        )
    elif min_non_nan_features is not None:
        before = len(df)
        non_nan_count = df[feature_cols].notna().sum(axis=1)
        df = df[non_nan_count >= min_non_nan_features].copy()
        log.info(
            "Dropped {} rows with < {} non-NaN features ({} -> {})",
            before - len(df),
            min_non_nan_features,
            before,
            len(df),
        )

    df = df.sort_values(["feature_date", "symbol"]).reset_index(drop=True)

    if cross_sectional_features:
        df, xs_cols = add_cross_sectional_features(df)
        feature_cols = feature_cols + xs_cols
        log.info(
            "Added {} cross-sectional rank features (universe/day ranks)",
            len(xs_cols),
        )

    if label_mode == "cross_sectional":
        # Percentile-rank the next-day return WITHIN each date (1.0 = best
        # performer that day). Label the top (1 - label_quantile) fraction.
        # method="average" handles ties; dates with a single symbol degenerate
        # to rank 1.0 (labelled 1) which is harmless and rare once the
        # universe is populated.
        csr = df.groupby("feature_date")["fwd_return"].rank(
            pct=True, method="average"
        )
        y = (csr >= label_quantile).astype(int)
    else:
        y = (df["fwd_return"] > target_return_threshold).astype(int)
    fwd_ret = df["fwd_return"].astype(float)
    meta = df[["symbol", "feature_date"]].copy()
    X = df[feature_cols].copy()

    log.info(
        "Built matrix: {} rows, {} features, label_mode={}, positive_rate={:.3f}",
        len(X),
        len(feature_cols),
        label_mode,
        y.mean() if len(y) else 0.0,
    )
    return TrainingMatrix(
        X=X, y=y, forward_return=fwd_ret, meta=meta, feature_columns=feature_cols
    )


def time_based_split(
    meta: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) by date order.

    Strict no-shuffle. Earlier dates -> train; middle -> val (calibration);
    latest -> test.
    """
    n = len(meta)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    sorted_df = meta.sort_values("feature_date").reset_index()
    cut1 = int(n * train_frac)
    cut2 = int(n * (train_frac + val_frac))
    train_idx = sorted_df.iloc[:cut1]["index"].to_numpy()
    val_idx = sorted_df.iloc[cut1:cut2]["index"].to_numpy()
    test_idx = sorted_df.iloc[cut2:]["index"].to_numpy()
    return train_idx, val_idx, test_idx


def boundary_dates(meta: pd.DataFrame, idx: np.ndarray) -> tuple[date, date]:
    if len(idx) == 0:
        return (date.min, date.min)
    sub = meta.iloc[idx]
    return (sub["feature_date"].min(), sub["feature_date"].max())
