"""Load stock_to_sector mapping from CSV into the stock_sectors table."""

from __future__ import annotations

import csv
from pathlib import Path

from src.utils.db import transaction
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("ingest.sectors")

_DEFAULT_CSV = project_root() / "config" / "stock_to_sector.csv"


def load_seed(csv_path: Path | None = None) -> int:
    csv_path = csv_path or _DEFAULT_CSV
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda line: not line.startswith("#"), f))
        for row in reader:
            sym = (row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            rows.append(
                (
                    sym,
                    (row.get("sector") or "OTHER").strip().upper(),
                    (row.get("sector_index") or "^NSEI").strip(),
                    (row.get("notes") or None) or None,
                )
            )
    if not rows:
        return 0
    with transaction() as conn:
        conn.executemany(
            """
            INSERT INTO stock_sectors (symbol, sector, sector_index, notes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
              sector       = excluded.sector,
              sector_index = excluded.sector_index,
              notes        = excluded.notes
            """,
            rows,
        )
    log.info("Upserted {} stock_sectors rows", len(rows))
    return len(rows)
