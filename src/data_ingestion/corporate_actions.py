"""Seed loader for corporate actions used by split_audit.

Future work: scrape NSE / BSE corporate actions feeds. For Week 1 we only
need a small known set to verify yfinance adjustment correctness.
"""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.contracts import CorporateAction, CorporateActionType
from src.utils.db import transaction
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("ingest.corp_actions")

_DEFAULT_SEED = project_root() / "data" / "seed" / "known_corporate_actions.csv"


def load_seed(csv_path: Path | None = None) -> list[CorporateAction]:
    csv_path = csv_path or _DEFAULT_SEED
    out: list[CorporateAction] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda line: not line.startswith("#"), f))
        for row in reader:
            try:
                out.append(
                    CorporateAction(
                        symbol=row["symbol"].strip().upper(),
                        ex_date=datetime.fromisoformat(row["ex_date"]).date(),
                        action_type=CorporateActionType(row["action_type"].strip().lower()),
                        ratio_from=int(row["ratio_from"]) if row.get("ratio_from") else None,
                        ratio_to=int(row["ratio_to"]) if row.get("ratio_to") else None,
                        amount=Decimal(row["amount"]) if row.get("amount") else None,
                        notes=(row.get("notes") or None) or None,
                        source="seed",
                    )
                )
            except Exception as e:
                log.warning("Skipping corp action row {}: {}", row, str(e))
    return out


def upsert(actions: list[CorporateAction]) -> int:
    if not actions:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO corporate_actions
                (symbol, ex_date, action_type, ratio_from, ratio_to, amount, notes, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, ex_date, action_type) DO UPDATE SET
                ratio_from = excluded.ratio_from,
                ratio_to   = excluded.ratio_to,
                amount     = excluded.amount,
                notes      = excluded.notes,
                source     = excluded.source
            """,
            [
                (
                    a.symbol,
                    a.ex_date.isoformat(),
                    a.action_type.value,
                    a.ratio_from,
                    a.ratio_to,
                    float(a.amount) if a.amount is not None else None,
                    a.notes,
                    a.source,
                )
                for a in actions
            ],
        )
    log.info("Upserted {} corporate actions", len(actions))
    return len(actions)
