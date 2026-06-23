"""Train the learned multi-horizon price-target model.

Fits one XGBoost regressor per horizon (1W, 1M, 3M, 6M, 1Y) that predicts the
forward log-return from the engineered features + per-day cross-sectional
ranks, then saves the bundle to ``data/models/horizon/<run_id>/`` and registers
it in ``model_runs`` (model_name = ``horizon_forecaster``). The multi-year 3Y
horizon stays on the analytic projection and is not learned here.

Per-horizon diagnostics (held-out rank IC, RMSE, directional hit-rate) are
printed so you can judge where the model has skill -- short horizons typically
carry more signal than the 1-year target.

Usage:
    python -m scripts.train_horizon_models                  # all symbols
    python -m scripts.train_horizon_models --symbols TCS,INFY,RELIANCE
    python -m scripts.train_horizon_models --no-cross-sectional
"""

from __future__ import annotations

import argparse
import sys

from src.db.migrate import apply_schema
from src.models.horizon_forecaster import save_bundle, train_horizon_models
from src.utils.logger import get_logger

log = get_logger("script.train_horizon")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols. Default: all in feature_data.")
    p.add_argument("--no-cross-sectional", action="store_true",
                   help="Disable per-day cross-sectional rank features.")
    p.add_argument("--long-run-annual", type=float, default=0.11,
                   help="Baseline annual drift recorded for the analytic 3Y "
                        "horizon (default 0.11 = 11%%).")
    p.add_argument("--no-save", action="store_true",
                   help="Train and report metrics but do not persist the bundle.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    apply_schema()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else None
    )

    bundle = train_horizon_models(
        symbols,
        cross_sectional=not args.no_cross_sectional,
        long_run_annual=args.long_run_annual,
    )

    print("\n=== Learned multi-horizon model ===")
    print(f"run_id        : {bundle.run_id}")
    print(f"trained window: {bundle.trained_from} -> {bundle.trained_to}")
    print(f"features      : {len(bundle.feature_columns)} "
          f"(cross_sectional={bundle.cross_sectional})")
    print(f"\n{'horizon':<8}{'days':>6}{'rows':>9}{'IC':>8}"
          f"{'RMSE':>9}{'hit%':>8}{'band±%':>9}")
    print("-" * 57)
    for label, m in bundle.models.items():
        band_pct = (2.718281828 ** m.resid_std - 1.0) * 100.0
        print(f"{label:<8}{m.horizon_days:>6}{int(m.metrics['n_rows']):>9}"
              f"{m.metrics['ic']:>8.3f}{m.metrics['rmse']:>9.4f}"
              f"{m.metrics['hit_rate'] * 100:>8.1f}{band_pct:>9.1f}")
    print()

    if args.no_save:
        print("(--no-save) bundle not persisted.")
        return 0

    run_dir = save_bundle(bundle)
    print(f"Saved bundle -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
