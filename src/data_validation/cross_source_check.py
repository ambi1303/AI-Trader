"""Cross-source validator: compare yfinance vs bhavcopy for the same (symbol, date).

We compare unadjusted CLOSE within tolerance and VOLUME within tolerance.
The yfinance close, when auto_adjust=True, is split-adjusted; bhavcopy is
not adjusted. Therefore we only compare on dates AFTER the most recent
corporate action for that symbol (recorded in corporate_actions). This
catches data-quality issues without false-positives on splits.

Returns a ValidationIssue list and an aggregate "match rate".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from src.contracts import Severity, ValidationIssue
from src.utils.db import fetch_all
from src.utils.logger import get_logger
from src.utils.secrets import get_settings

log = get_logger("validation.cross_source")


@dataclass
class CrossSourceReport:
    run_id: str
    rows_compared: int
    rows_mismatched: int
    issues: list[ValidationIssue]

    @property
    def match_rate(self) -> float:
        if self.rows_compared == 0:
            return 1.0
        return 1.0 - (self.rows_mismatched / self.rows_compared)


def _last_action_dates() -> dict[str, date]:
    rows = fetch_all(
        """
        SELECT symbol, MAX(ex_date) AS last_ex
        FROM   corporate_actions
        WHERE  action_type IN ('split','bonus','demerger')
        GROUP BY symbol
        """
    )
    out: dict[str, date] = {}
    for r in rows:
        if r["last_ex"]:
            out[r["symbol"]] = datetime.fromisoformat(r["last_ex"]).date()
    return out


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return abs(a - b) / abs(b) * 100.0


def compare_sources(
    run_id: str,
    *,
    symbols: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> CrossSourceReport:
    settings = get_settings()
    close_tol = settings.cross_source_close_tolerance_pct
    vol_tol = settings.cross_source_volume_tolerance_pct

    where = ["yf.symbol = bc.symbol", "yf.bar_date = bc.bar_date"]
    params: list = []
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        where.append(f"yf.symbol IN ({placeholders})")
        params.extend(symbols)
    if start:
        where.append("yf.bar_date >= ?")
        params.append(start.isoformat())
    if end:
        where.append("yf.bar_date <= ?")
        params.append(end.isoformat())

    sql = f"""
        SELECT yf.symbol, yf.bar_date,
               yf.close AS yf_close, yf.volume AS yf_volume,
               bc.close AS bc_close, bc.volume AS bc_volume
        FROM   price_data yf
        JOIN   price_data bc ON {' AND '.join(where)}
        WHERE  yf.source = 'yfinance' AND bc.source = 'bhavcopy'
        ORDER BY yf.symbol, yf.bar_date
    """  # noqa: S608 (placeholders are bound)
    rows = fetch_all(sql, tuple(params))

    last_action = _last_action_dates()

    issues: list[ValidationIssue] = []
    mismatched = 0
    compared = 0

    for r in rows:
        bd = datetime.fromisoformat(r["bar_date"]).date()
        sym = r["symbol"]
        # Skip rows on/before the last split/bonus: yf is adjusted, bhavcopy is not.
        if sym in last_action and bd <= last_action[sym]:
            continue

        compared += 1
        close_diff = _pct_diff(r["yf_close"], r["bc_close"])
        vol_diff = _pct_diff(float(r["yf_volume"]), float(r["bc_volume"]))

        if close_diff > close_tol:
            mismatched += 1
            issues.append(
                ValidationIssue(
                    run_id=run_id,
                    check_name="cross_source.close_mismatch",
                    symbol=sym,
                    issue_date=bd,
                    severity=Severity.ERROR if close_diff > close_tol * 5 else Severity.WARNING,
                    message=(
                        f"close diff {close_diff:.3f}% > {close_tol:.2f}% "
                        f"(yf={r['yf_close']}, bc={r['bc_close']})"
                    ),
                    details={
                        "yf_close": float(r["yf_close"]),
                        "bc_close": float(r["bc_close"]),
                        "diff_pct": float(close_diff),
                    },
                )
            )
        if vol_diff > vol_tol:
            issues.append(
                ValidationIssue(
                    run_id=run_id,
                    check_name="cross_source.volume_mismatch",
                    symbol=sym,
                    issue_date=bd,
                    severity=Severity.WARNING,
                    message=(
                        f"volume diff {vol_diff:.2f}% > {vol_tol:.2f}% "
                        f"(yf={r['yf_volume']}, bc={r['bc_volume']})"
                    ),
                    details={
                        "yf_volume": int(r["yf_volume"]),
                        "bc_volume": int(r["bc_volume"]),
                        "diff_pct": float(vol_diff),
                    },
                )
            )

    rep = CrossSourceReport(
        run_id=run_id,
        rows_compared=compared,
        rows_mismatched=mismatched,
        issues=issues,
    )
    log.info(
        "cross_source: rows_compared={} mismatched={} match_rate={:.4f}",
        rep.rows_compared,
        rep.rows_mismatched,
        rep.match_rate,
    )
    return rep
