"""Compare labeling strategies on identical data/pipeline.

Trains the *same* XGBoost + isotonic stack under different target
definitions and prints held-out test metrics side by side, so we can
decide empirically whether cross-sectional (relative) labeling buys us
real edge over the absolute next-day-direction target.

This is a read-only experiment: it does NOT register a model or write to
predictions_log. Run it any time with:

    python -m scripts.experiment_label_mode
"""

from __future__ import annotations

import sys

from src.models.calibration import CalibratedXGB, fit_isotonic_calibrator
from src.models.dataset import build_training_matrix, time_based_split
from src.models.metrics import evaluate_classifier
from src.models.xgboost_classifier import DeterministicXGBClassifier, XGBParams
from src.utils.logger import get_logger

log = get_logger("script.exp_label")


def _run(label_mode: str, label_quantile: float = 0.50,
         horizon: int = 1, xsec: bool = False) -> dict | None:
    matrix = build_training_matrix(
        None, label_mode=label_mode, label_quantile=label_quantile,
        horizon=horizon, cross_sectional_features=xsec,
    )
    if matrix.X.empty:
        log.error("Empty matrix; run scripts.build_features first.")
        return None
    tr, val, te = time_based_split(matrix.meta)
    if min(len(tr), len(val), len(te)) < 50:
        log.error("Splits too small.")
        return None

    X_tr, y_tr = matrix.X.iloc[tr], matrix.y.iloc[tr]
    X_val, y_val = matrix.X.iloc[val], matrix.y.iloc[val]
    X_te, y_te = matrix.X.iloc[te], matrix.y.iloc[te]

    base = DeterministicXGBClassifier(params=XGBParams())
    base.fit(X_tr, y_tr)

    # Calibration needs both classes in val; cross-sectional is balanced so OK.
    try:
        cal = fit_isotonic_calibrator(base, X_val, y_val)
        cxgb = CalibratedXGB(base, cal)
        cal_te = cxgb.predict_calibrated(X_te)
    except ValueError as e:
        log.warning("calibration skipped ({}); using raw probs", e)
        cal_te = base.predict_proba(X_te)[:, 1]

    raw_te = base.predict_proba(X_te)[:, 1]
    rep_raw = evaluate_classifier(y_te.to_numpy(), raw_te)
    rep_cal = evaluate_classifier(y_te.to_numpy(), cal_te)
    return {
        "label_mode": label_mode,
        "label_quantile": label_quantile,
        "horizon": horizon,
        "xsec": xsec,
        "n_feat": matrix.X.shape[1],
        "n_rows": len(matrix.X),
        "pos_rate": float(matrix.y.mean()),
        "test_pos_rate": float(y_te.mean()),
        "raw_auc": rep_raw.roc_auc,
        "cal_auc": rep_cal.roc_auc,
        "raw_pr_auc": rep_raw.pr_auc,
        "cal_pr_auc": rep_cal.pr_auc,
        "cal_brier": rep_cal.brier,
    }


def main() -> int:
    # On the WIDER universe: test whether cross-sectional rank features +
    # relative labeling + longer horizon surface real edge.
    # Columns: (label_mode, quantile, horizon, xsec_features)
    configs = [
        ("absolute", 0.50, 1, False),
        ("cross_sectional", 0.70, 1, False),
        ("cross_sectional", 0.70, 1, True),
        ("cross_sectional", 0.70, 5, True),
        ("cross_sectional", 0.70, 10, True),
        ("cross_sectional", 0.70, 20, True),
        ("absolute", 0.50, 5, True),
        ("absolute", 0.50, 20, True),
    ]
    results = []
    for mode, q, h, xs in configs:
        log.info("=== {} (q={}, horizon={}, xsec={}) ===", mode, q, h, xs)
        r = _run(mode, q, h, xs)
        if r:
            results.append(r)

    if not results:
        return 1

    print("\n" + "=" * 112)
    print(f"{'label_mode':>16} {'q':>5} {'horiz':>6} {'xsec':>5} {'feat':>5} "
          f"{'rows':>7} {'pos%':>6} {'raw_AUC':>8} {'cal_AUC':>8} "
          f"{'raw_PR':>8} {'cal_PR':>8} {'brier':>7}")
    print("-" * 112)
    for r in results:
        tag = r["label_mode"] if r["label_mode"] == "absolute" else "cross_sec"
        print(f"{tag:>16} {r['label_quantile']:>5.2f} {r['horizon']:>6} "
              f"{str(r['xsec']):>5} {r['n_feat']:>5} "
              f"{r['n_rows']:>7} {r['pos_rate']*100:>5.1f}% "
              f"{r['raw_auc']:>8.4f} {r['cal_auc']:>8.4f} "
              f"{r['raw_pr_auc']:>8.4f} {r['cal_pr_auc']:>8.4f} "
              f"{r['cal_brier']:>7.4f}")
    print("=" * 112)
    print("AUC 0.50 = coin flip. Look for configs where AUC climbs above ~0.53 "
          "AND PR_AUC clears the pos% baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
