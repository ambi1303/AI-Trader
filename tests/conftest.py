"""Test config: each test gets an isolated SQLite DB under tmp_path."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force DB_PATH to a fresh file per test."""
    db_file = tmp_path / "test_trading.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # Force re-creation of cached AppSettings between tests
    from src.utils import secrets as secrets_mod
    if hasattr(secrets_mod, "_cached_settings"):
        secrets_mod._cached_settings = None  # type: ignore[attr-defined]

    return db_file
