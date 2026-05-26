"""Load Nifty 50 historical constituency from a curated CSV.

This is a *seed* loader. The CSV at data/seed/historical_constituents.csv
must be extended manually from NSE quarterly index review circulars to
achieve full survivorship-bias-free coverage. The loader is idempotent.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from src.contracts import ConstituencyEntry
from src.utils.db import transaction
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("ingest.constituents")

_DEFAULT_SEED = project_root() / "data" / "seed" / "historical_constituents.csv"


def _parse_date(s: str) -> date | None:
    s = s.strip()
    if not s:
        return None
    return datetime.fromisoformat(s).date()


def load_seed(csv_path: Path | None = None) -> list[ConstituencyEntry]:
    csv_path = csv_path or _DEFAULT_SEED
    entries: list[ConstituencyEntry] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda line: not line.startswith("#"), f))
        for row in reader:
            try:
                entries.append(
                    ConstituencyEntry(
                        symbol=row["symbol"].strip().upper(),
                        start_date=_parse_date(row["start_date"]),  # type: ignore[arg-type]
                        end_date=_parse_date(row.get("end_date", "")),
                        index_name=(row.get("index_name") or "NIFTY50").strip(),
                        notes=(row.get("notes") or None) or None,
                    )
                )
            except Exception as e:
                log.warning("Skipping constituents row {}: {}", row, str(e))
    return entries


def upsert_constituents(entries: list[ConstituencyEntry]) -> int:
    """Insert or replace constituency rows. Returns count written."""
    if not entries:
        return 0
    with transaction() as conn:
        rows = [
            (
                e.symbol,
                e.start_date.isoformat(),
                e.end_date.isoformat() if e.end_date else None,
                e.index_name,
                e.notes,
            )
            for e in entries
        ]
        conn.executemany(
            """
            INSERT INTO nifty_constituents
                (symbol, start_date, end_date, index_name, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, start_date, index_name) DO UPDATE SET
                end_date = excluded.end_date,
                notes    = excluded.notes
            """,
            rows,
        )
    log.info("Upserted {} constituency rows", len(rows))
    return len(rows)


def universe_as_of(d: date, index_name: str = "NIFTY50") -> list[str]:
    """Return symbols that were in the index on date d."""
    from src.utils.db import fetch_all

    rows = fetch_all(
        """
        SELECT symbol FROM nifty_constituents
        WHERE  index_name = ?
          AND  start_date <= ?
          AND  (end_date IS NULL OR end_date >= ?)
        ORDER BY symbol
        """,
        (index_name, d.isoformat(), d.isoformat()),
    )
    return [r["symbol"] for r in rows]
