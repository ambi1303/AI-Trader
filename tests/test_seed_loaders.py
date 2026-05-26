"""Seeders parse and persist correctly."""

from __future__ import annotations

from datetime import date

from src.data_ingestion.constituents_loader import (
    load_seed as load_constituents,
)
from src.data_ingestion.constituents_loader import (
    universe_as_of,
    upsert_constituents,
)
from src.data_ingestion.corporate_actions import load_seed as load_actions
from src.data_ingestion.corporate_actions import upsert as upsert_actions
from src.db.migrate import apply_schema


def test_constituents_seed_loads_and_universe_resolves() -> None:
    apply_schema()
    entries = load_constituents()
    assert len(entries) > 30
    upsert_constituents(entries)

    today_universe = universe_as_of(date.today())
    assert len(today_universe) >= 40
    assert "RELIANCE" in today_universe
    assert "TCS" in today_universe


def test_corp_actions_seed_persists() -> None:
    apply_schema()
    actions = load_actions()
    assert len(actions) >= 5
    n = upsert_actions(actions)
    assert n == len(actions)


def test_pit_universe_excludes_post_removal_dates() -> None:
    """As-of historical universe should reflect index changes."""
    apply_schema()
    upsert_constituents(load_constituents())

    pre_universe = universe_as_of(date(2023, 1, 1))
    # HDFC was still listed and in Nifty 50 on this date.
    assert "HDFC" in pre_universe

    post_universe = universe_as_of(date(2024, 1, 1))
    # HDFC merged into HDFCBANK on 2023-07-13 per seed.
    assert "HDFC" not in post_universe
