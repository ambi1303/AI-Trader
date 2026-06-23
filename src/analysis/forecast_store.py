"""DB IO for the multi-horizon price forecaster.

Two engines feed the same ``price_forecasts`` table and UI:

* **ML** (:mod:`src.models.horizon_forecaster`) -- a learned XGBoost regressor
  per horizon (1W..1Y). Used automatically when a trained bundle exists.
* **Analytic drift** (:mod:`src.analysis.forecast`) -- the transparent
  drift+volatility projection. Always available, and the *only* source for the
  multi-year "3Y" horizon (too far to learn honestly), so the long-run target
  is consistent whether or not a model is trained.

Each persisted horizon records a ``method`` ('ml' | 'drift') and, for ML rows,
the ``model_run_id`` -- so the dashboard and audits can tell learned targets
apart from the projection. The ML path falls back to analytic for any symbol it
can't score, so there is never a regression versus the pre-ML behaviour.

Look-ahead safety: inputs are the most recent ``feature_date <= as_of``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.analysis.forecast import HORIZONS, forecast_stock
from src.utils.db import fetch_all, fetch_one, transaction
from src.utils.logger import get_logger

log = get_logger("analysis.forecast_store")

# Horizon labels the learned model covers; everything else (3Y) stays analytic.
_ML_LABELS = {"1W", "1M", "3M", "6M", "1Y"}


def _feature_row(symbol: str, as_of: str, db_path: str | None = None) -> dict | None:
    row = fetch_one(
        "SELECT close, vol_20d, mom_20d, mom_60d, atr_pct, feature_date "
        "FROM feature_data WHERE symbol = ? AND feature_date <= ? "
        "ORDER BY feature_date DESC LIMIT 1",
        (symbol, as_of), db_path=db_path,
    )
    return dict(row) if row else None


def _analytic_forecast(
    symbol: str, as_of: str, long_run_annual: float | None, db_path: str | None,
) -> dict[str, Any]:
    """Transparent drift+volatility projection (all horizons, method='drift')."""
    feat = _feature_row(symbol, as_of, db_path=db_path)
    if not feat:
        return {"available": False, "symbol": symbol}
    kwargs: dict[str, Any] = {
        "symbol": symbol,
        "last_close": feat.get("close"),
        "daily_vol": feat.get("vol_20d"),
        "mom_20d": feat.get("mom_20d"),
        "mom_60d": feat.get("mom_60d"),
    }
    if long_run_annual is not None:
        kwargs["long_run_annual"] = long_run_annual
    out = forecast_stock(**kwargs)
    out["as_of_date"] = feat.get("feature_date", as_of)
    out["method"] = "drift"
    if out.get("available"):
        for h in out["horizons"]:
            h.setdefault("method", "drift")
    return out


def _merge_ml_with_analytic(
    ml_out: dict[str, Any] | None, analytic: dict[str, Any],
) -> dict[str, Any]:
    """Prefer ML horizons (1W..1Y); always take 3Y (and any gaps) from analytic.

    If the ML output is missing/unavailable, the analytic projection is
    returned unchanged (graceful, no-regression fallback).
    """
    if not ml_out or not ml_out.get("available"):
        return analytic

    by_label: dict[str, dict[str, Any]] = {}
    for h in ml_out.get("horizons", []):
        by_label[h["label"]] = h
    # Fill non-ML horizons (notably 3Y) from analytic.
    if analytic.get("available"):
        for h in analytic.get("horizons", []):
            by_label.setdefault(h["label"], h)

    ordered = [by_label[lbl] for lbl, _ in HORIZONS if lbl in by_label]
    return {
        "available": True,
        "symbol": ml_out.get("symbol"),
        "method": "ml",
        "model_run_id": ml_out.get("model_run_id"),
        "last_close": ml_out.get("last_close", analytic.get("last_close")),
        "as_of_date": ml_out.get("as_of_date", analytic.get("as_of_date")),
        "horizons": ordered,
    }


# Cache the (potentially large) bundle per process so single-symbol web calls
# don't reload+re-validate it on every request. Keyed by db_path; value is
# (run_id, bundle). A cheap run_id probe lets a freshly-trained bundle activate
# without restarting a long-running web server, and avoids caching "None"
# (which would otherwise pin the analytic path until restart).
_BUNDLE_CACHE: dict[str, tuple[str, Any]] = {}


def _latest_run_id(db_path: str | None) -> str | None:
    row = fetch_one(
        "SELECT run_id FROM model_runs WHERE model_name = 'horizon_forecaster' "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        db_path=db_path,
    )
    return row["run_id"] if row else None


def _get_bundle(db_path: str | None):
    key = db_path or "__default__"
    try:
        run_id = _latest_run_id(db_path)
    except Exception:  # noqa: BLE001 -- no model_runs table yet, etc.
        return None
    if not run_id:
        return None
    cached = _BUNDLE_CACHE.get(key)
    if cached and cached[0] == run_id:
        return cached[1]
    try:
        from src.models.horizon_forecaster import latest_bundle
        bundle = latest_bundle(db_path=db_path)
        if bundle is not None:
            _BUNDLE_CACHE[key] = (run_id, bundle)
        return bundle
    except Exception as exc:  # noqa: BLE001 -- never break forecasting on ML load
        log.warning("Horizon bundle unavailable, using analytic: {}", exc)
        return None


def clear_bundle_cache() -> None:
    """Drop the cached horizon bundle (call after training a new one)."""
    _BUNDLE_CACHE.clear()


def forecast_symbol(
    symbol: str,
    as_of: str | None = None,
    *,
    long_run_annual: float | None = None,
    use_ml: bool = True,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Compute (but do not persist) the multi-horizon forecast for ``symbol``.

    Uses the learned per-horizon model when a trained bundle is available,
    falling back to the analytic projection otherwise. The 3Y horizon is always
    analytic.
    """
    as_of = as_of or date.today().isoformat()
    analytic = _analytic_forecast(symbol, as_of, long_run_annual, db_path)

    ml_out = None
    if use_ml:
        bundle = _get_bundle(db_path)
        if bundle is not None:
            try:
                from src.models.horizon_forecaster import predict_symbol
                ml_out = predict_symbol(bundle, symbol, as_of, db_path=db_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("ML forecast failed for {}, using analytic: {}",
                            symbol, exc)
                ml_out = None
    return _merge_ml_with_analytic(ml_out, analytic)


def _persist(out: dict[str, Any], symbol: str, db_path: str | None) -> None:
    as_of_date = out["as_of_date"]
    model_run_id = out.get("model_run_id")
    rows = [
        (
            symbol, as_of_date, h["label"], h["horizon_days"], out["last_close"],
            h["expected_price"], h["low_price"], h["high_price"],
            h["expected_return_pct"], h["annualized_return_pct"],
            h["prob_up_pct"], h["verdict"],
            h.get("method", out.get("method", "drift")),
            model_run_id if h.get("method") == "ml" else None,
        )
        for h in out["horizons"]
    ]
    with transaction(db_path=db_path) as conn:
        conn.executemany(
            """
            INSERT INTO price_forecasts
                (symbol, as_of_date, horizon_label, horizon_days, last_close,
                 expected_price, low_price, high_price, expected_return_pct,
                 annualized_return_pct, prob_up_pct, verdict, method, model_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, as_of_date, horizon_label) DO UPDATE SET
                horizon_days          = excluded.horizon_days,
                last_close            = excluded.last_close,
                expected_price        = excluded.expected_price,
                low_price             = excluded.low_price,
                high_price            = excluded.high_price,
                expected_return_pct   = excluded.expected_return_pct,
                annualized_return_pct = excluded.annualized_return_pct,
                prob_up_pct           = excluded.prob_up_pct,
                verdict               = excluded.verdict,
                method                = excluded.method,
                model_run_id          = excluded.model_run_id
            """,
            rows,
        )


def store_forecast(
    symbol: str,
    as_of: str | None = None,
    *,
    long_run_annual: float | None = None,
    use_ml: bool = True,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Compute and upsert the forecast for ``symbol`` into ``price_forecasts``."""
    out = forecast_symbol(symbol, as_of, long_run_annual=long_run_annual,
                          use_ml=use_ml, db_path=db_path)
    if not out.get("available"):
        log.info("forecast {}: no data as-of {}", symbol, as_of)
        return out
    _persist(out, symbol, db_path)
    return out


def store_universe(
    symbols: list[str],
    as_of: str | None = None,
    *,
    long_run_annual: float | None = None,
    use_ml: bool = True,
    db_path: str | None = None,
) -> int:
    """Forecast + persist every symbol; returns the count with data.

    When a learned bundle exists, the whole universe is scored in a single
    batched pass (cross-sectional ranks computed once) instead of reloading the
    universe per symbol -- the analytic 3Y horizon is still merged per symbol.
    """
    as_of = as_of or date.today().isoformat()
    ml_map: dict[str, dict[str, Any]] = {}
    if use_ml:
        bundle = _get_bundle(db_path)
        if bundle is not None:
            try:
                from src.models.horizon_forecaster import predict_universe
                ml_map = predict_universe(bundle, as_of, db_path=db_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Batch ML forecast failed, using analytic: {}", exc)
                ml_map = {}

    n = 0
    for sym in symbols:
        analytic = _analytic_forecast(sym, as_of, long_run_annual, db_path)
        out = _merge_ml_with_analytic(ml_map.get(sym), analytic)
        if out.get("available"):
            _persist(out, sym, db_path)
            n += 1
    method = "ml+drift" if ml_map else "drift"
    log.info("Stored forecasts for {}/{} symbols (as-of {}, method={}).",
             n, len(symbols), as_of, method)
    return n


def latest_forecast(
    symbol: str, as_of: str | None = None, db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Read back the most recent stored forecast horizons for ``symbol``."""
    row = fetch_one(
        "SELECT MAX(as_of_date) AS d FROM price_forecasts WHERE symbol = ? "
        + ("AND as_of_date <= ?" if as_of else ""),
        (symbol, as_of) if as_of else (symbol,), db_path=db_path,
    )
    d = row["d"] if row else None
    if not d:
        return []
    order = " ".join(f"WHEN '{lbl}' THEN {i}" for i, (lbl, _) in enumerate(HORIZONS))
    rows = fetch_all(
        f"SELECT * FROM price_forecasts WHERE symbol = ? AND as_of_date = ? "
        f"ORDER BY CASE horizon_label {order} ELSE 99 END",  # noqa: S608 - order is literal
        (symbol, d), db_path=db_path,
    )
    return [dict(r) for r in rows]
