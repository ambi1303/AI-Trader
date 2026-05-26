"""Logger redacts known sensitive keys and strips CR/LF."""

from __future__ import annotations

import io

from loguru import logger as base_logger

from src.utils import logger as logger_mod


def test_redacts_token_and_key(monkeypatch) -> None:
    logger_mod._INITIALISED = False
    logger_mod.setup_logging()

    sink = io.StringIO()
    sink_id = base_logger.add(sink, level="INFO", serialize=True)
    try:
        log = logger_mod.get_logger("redaction_test")
        log.bind(telegram_bot_token="123:ABCDEF").info("login event")
        log.bind(angel_one_api_key="SUPERSECRET").info("auth ok")
    finally:
        base_logger.remove(sink_id)

    out = sink.getvalue()
    assert "123:ABCDEF" not in out
    assert "SUPERSECRET" not in out
    assert "REDACTED" in out


def test_strips_crlf_in_message(monkeypatch) -> None:
    logger_mod._INITIALISED = False
    logger_mod.setup_logging()

    sink = io.StringIO()
    sink_id = base_logger.add(sink, level="INFO", serialize=True)
    try:
        log = logger_mod.get_logger("crlf_test")
        log.info("hello\r\nFAKE LOG ENTRY level=ERROR boom")
    finally:
        base_logger.remove(sink_id)

    out = sink.getvalue()
    # CR/LF should not appear within the message portion of the JSON record
    # (they may exist as record terminators only).
    assert "hello FAKE LOG ENTRY" in out or "hello  FAKE LOG ENTRY" in out
