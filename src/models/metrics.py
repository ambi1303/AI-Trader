"""Reporting metrics: Brier, ECE, ROC-AUC, PR-AUC, calibration curve.

We re-implement only what scikit-learn doesn't ship in 1.5 (ECE). Everything
else is delegated. All functions accept 1-D arrays and return scalars or
small dataclasses for easy JSON serialisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    n: int
    positive_rate: float
    brier: float
    log_loss: float
    roc_auc: float | None
    pr_auc: float | None
    ece: float

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average over bins of |bin_avg_pred - bin_actual_rate|.

    Lower is better. 0.0 means perfectly calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        # Inclusive on left, exclusive on right; last bin closes the interval.
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        bin_pred = y_prob[mask].mean()
        bin_actual = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_pred - bin_actual)
    return float(ece)


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> ClassificationReport:
    """Compute the standard classification report. Resilient to single-class slices."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return ClassificationReport(
            n=0,
            positive_rate=float("nan"),
            brier=float("nan"),
            log_loss=float("nan"),
            roc_auc=None,
            pr_auc=None,
            ece=float("nan"),
        )

    pos_rate = float(y_true.mean())

    # ROC/PR AUC undefined when only one class is present.
    n_classes = len(np.unique(y_true))
    roc = float(roc_auc_score(y_true, y_prob)) if n_classes == 2 else None
    pr = float(average_precision_score(y_true, y_prob)) if n_classes == 2 else None

    # log_loss requires both classes to be representable; clip prob to avoid 0/1.
    eps = 1e-7
    p = np.clip(y_prob, eps, 1.0 - eps)
    if n_classes == 2:
        ll = float(log_loss(y_true, p, labels=[0, 1]))
    else:
        ll = float("nan")

    return ClassificationReport(
        n=n,
        positive_rate=pos_rate,
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=ll,
        roc_auc=roc,
        pr_auc=pr,
        ece=expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    )


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Return list of (avg_predicted_prob, observed_rate, count) per bin.

    Useful for plotting calibration; we keep it data-only here.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            out.append((float("nan"), float("nan"), 0))
            continue
        out.append(
            (
                float(y_prob[mask].mean()),
                float(y_true[mask].mean()),
                int(mask.sum()),
            )
        )
    return out
