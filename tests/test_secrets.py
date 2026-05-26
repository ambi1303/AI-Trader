"""Secret loader behaviour."""

from __future__ import annotations

import pytest

from src.utils.secrets import MissingSecretError, get_secret


def test_get_secret_raises_when_required_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOME_FAKE_SECRET", raising=False)
    with pytest.raises(MissingSecretError):
        get_secret("SOME_FAKE_SECRET")


def test_get_secret_returns_none_when_optional_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOME_FAKE_SECRET", raising=False)
    assert get_secret("SOME_FAKE_SECRET", required=False) is None


def test_get_secret_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_FAKE_SECRET", "hello")
    assert get_secret("SOME_FAKE_SECRET") == "hello"


def test_empty_string_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_FAKE_SECRET", "")
    with pytest.raises(MissingSecretError):
        get_secret("SOME_FAKE_SECRET")
