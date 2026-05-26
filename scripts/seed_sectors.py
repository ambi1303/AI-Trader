"""Seed stock_sectors from config/stock_to_sector.csv."""

from __future__ import annotations

from src.data_ingestion.sector_loader import load_seed
from src.utils.logger import get_logger


def main() -> int:
    log = get_logger("scripts.seed_sectors")
    n = load_seed()
    log.info("Stock sectors seeded: {}", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
