"""Rebuild a local SQLite database from the Neon Postgres mirror.

This is the inverse of ``src.cloud.publish``. It exists for the cloud daily
pipeline (GitHub Actions): the working SQLite DB is kept in the Actions cache
between runs, but on a *cache miss* (first run ever, or after cache eviction)
there is no DB to start from. Rather than re-ingesting years of history from
scratch, we reconstruct a working DB from the Neon mirror -- which already holds
~1.5 years of prices, the reference tables, fundamentals, the model registry
rows, and the paper-trade / prediction history.

What it does:
  1. applies the full SQLite schema (``src/db/schema.sql``) to a fresh DB,
  2. copies every mirrored table from Neon into SQLite (FKs disabled during the
     bulk load so insertion order never matters),
  3. leaves ``feature_data`` empty on purpose -- ``build_features`` regenerates
     it from ``price_data`` on the next pipeline step.

Idempotent and safe: if the target DB already has price rows we skip (so a warm
cache is never clobbered). The model *artifact files* are NOT stored in Neon;
they ship in the repo under ``data/models/`` so ``predict`` can load them.

    python -m src.cloud.seed_from_neon          # seed if empty
    python -m src.cloud.seed_from_neon --force   # always rebuild
"""

from __future__ import annotations

import argparse
import os
import sqlite3

from src.cloud.publish import _SPECS
from src.db.migrate import apply_schema
from src.utils.db import resolve_db_path
from src.utils.logger import get_logger

log = get_logger("cloud.seed")

# feature_data is regenerated locally by build_features, so we never seed it
# (the Neon mirror only carries a narrow overlay subset anyway).
_SKIP_TABLES = {"feature_data"}

# Insert smaller bites so a multi-hundred-thousand-row price_data load keeps
# memory flat and gives progress logs.
_BATCH = 5_000


def _database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. The Neon connection string is required to "
            "seed the local DB from the cloud mirror."
        )
    return url


def _already_seeded(db_path) -> bool:
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()
            return bool(row and row[0] > 0)
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _copy_table(pg, sl: sqlite3.Connection, table: str, columns: list[str]) -> int:
    cols = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"

    total = 0
    with pg.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM {table}")
        while True:
            batch = cur.fetchmany(_BATCH)
            if not batch:
                break
            sl.executemany(insert_sql, [tuple(r) for r in batch])
            total += len(batch)
    sl.commit()
    return total


def seed(*, force: bool = False) -> dict[str, int]:
    """Create a local SQLite DB from the Neon mirror. Returns {table: rows}."""
    import psycopg

    db_path = resolve_db_path()

    if _already_seeded(db_path) and not force:
        log.info("local DB already populated ({}); skipping seed", db_path)
        return {}

    url = _database_url()
    log.info("seeding local SQLite {} from Neon mirror", db_path)

    # Build the full schema first (idempotent -- creates all tables/indices).
    apply_schema()

    summary: dict[str, int] = {}
    sl = sqlite3.connect(str(db_path))
    try:
        # Disable FK enforcement for the bulk load: we copy a consistent
        # snapshot, but parent rows (model_runs) and children (predictions_log,
        # paper_trades) arrive in spec order, and signal_id links are dropped in
        # the mirror -- so we don't want FK errors mid-load.
        sl.execute("PRAGMA foreign_keys=OFF;")
        with psycopg.connect(url, connect_timeout=20) as pg:
            for spec in _SPECS:
                if spec.table in _SKIP_TABLES:
                    continue
                n = _copy_table(pg, sl, spec.table, spec.columns)
                summary[spec.table] = n
                log.info("  {:<20} {:>8,} rows", spec.table, n)
    finally:
        sl.close()

    total = sum(summary.values())
    log.info("seed complete: {:,} rows across {} tables", total, len(summary))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed local SQLite from Neon mirror")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the local DB already has data")
    args = ap.parse_args()
    try:
        summary = seed(force=args.force)
    except Exception as exc:  # noqa: BLE001
        log.error("seed failed: {}", exc)
        print(f"SEED FAILED: {exc}")
        return 1
    if not summary:
        print("Local DB already populated; nothing to do.")
        return 0
    print("Seeded local SQLite from Neon:")
    for table, n in summary.items():
        print(f"  {table:<20} {n:>8,} rows")
    print(f"  {'TOTAL':<20} {sum(summary.values()):>8,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
