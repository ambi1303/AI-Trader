"""Initialise / migrate the SQLite database."""

from __future__ import annotations

from src.db.migrate import apply_schema
from src.utils.db import resolve_db_path
from src.utils.logger import get_logger


def main() -> int:
    log = get_logger("scripts.init_db")
    path = resolve_db_path()
    log.info("Initialising DB at {}", path)
    version = apply_schema()
    log.info("Schema applied. version={}", version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
