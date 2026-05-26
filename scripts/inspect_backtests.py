"""Quick inspection of persisted backtests."""
from __future__ import annotations

import json
import sys

from src.utils.db import fetch_all


def main(argv: list[str] | None = None) -> int:
    rows = fetch_all(
        "SELECT bt_run_id, name, start_date, end_date, metrics_json "
        "FROM backtest_runs ORDER BY created_at DESC, rowid DESC"
    )
    print(f"=== backtest_runs ({len(rows)} total) ===")
    for r in rows:
        m = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
        print(
            f"  {r['name']:25s} | {r['start_date']} -> {r['end_date']} "
            f"| trades={m.get('n_trades', 0):3d} "
            f"| Sharpe={m.get('sharpe', 0):6.2f} "
            f"| MaxDD={m.get('max_drawdown_pct', 0):6.2f}% "
            f"| TotRet={m.get('total_return_pct', 0):6.2f}%"
        )
    trades = fetch_all("SELECT COUNT(*) AS n FROM backtest_trades")[0]["n"]
    equity = fetch_all("SELECT COUNT(*) AS n FROM backtest_equity")[0]["n"]
    print(f"  total trades={trades} equity_rows={equity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
