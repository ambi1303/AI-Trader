"""Seed: constituency, corporate actions, trading calendar."""

from __future__ import annotations

from src.data_ingestion.constituents_loader import load_seed as load_constituents
from src.data_ingestion.constituents_loader import upsert_constituents
from src.data_ingestion.corporate_actions import load_seed as load_actions
from src.data_ingestion.corporate_actions import upsert as upsert_actions
from src.data_validation.calendar_check import calendar_summary, upsert_calendar_to_db
from src.utils.logger import get_logger


def main() -> int:
    log = get_logger("scripts.seed_data")

    entries = load_constituents()
    n_const = upsert_constituents(entries)
    log.info("Constituency rows seeded: {}", n_const)

    actions = load_actions()
    n_act = upsert_actions(actions)
    log.info("Corporate action rows seeded: {}", n_act)

    n_cal = upsert_calendar_to_db()
    log.info(
        "Calendar rows seeded: {} (summary={})",
        n_cal,
        calendar_summary(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
