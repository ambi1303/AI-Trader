"""Tests for the news-sentiment overlay in signal generation.

Verifies that strongly negative news vetoes an otherwise-passing candidate,
that positive news scales size up, and that the overlay can be disabled.
Sentiment scores are pre-seeded into ``news_headlines`` so no FinBERT/torch
is needed.
"""

from __future__ import annotations

import json

import pytest

from src.db.migrate import apply_schema
from src.signals.generator import SignalGenConfig, generate_signals
from src.utils import db as db_mod


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    db_file = tmp_path / "sig.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    apply_schema(db_path=db_file)

    sd = "2026-06-19"
    with db_mod.transaction() as conn:
        conn.execute(
            "INSERT INTO model_runs (run_id, model_name, metrics_json) "
            "VALUES ('r1', 'test', ?)",
            (json.dumps({"threshold": 0.5}),),
        )
        # Two strong candidates above threshold.
        conn.executemany(
            "INSERT INTO predictions_log (run_id, symbol, prediction_date, "
            "raw_prob, calibrated_prob) VALUES ('r1', ?, ?, ?, ?)",
            [("GOOD", sd, 0.70, 0.70), ("BADNEWS", sd, 0.72, 0.72)],
        )
        # Price + ATR for both.
        conn.executemany(
            "INSERT INTO feature_data (symbol, feature_date, close, atr_14) "
            "VALUES (?, ?, ?, ?)",
            [("GOOD", sd, 100.0, 2.0), ("BADNEWS", sd, 100.0, 2.0)],
        )
        # Different sectors so the per-sector cap never interferes.
        conn.executemany(
            "INSERT INTO stock_sectors (symbol, sector, sector_index) VALUES (?, ?, ?)",
            [("GOOD", "IT", "^CNXIT"), ("BADNEWS", "BANK", "^NSEBANK")],
        )
        # Pre-scored headlines within the 7d window before the signal date.
        rows = [
            ("GOOD", "2026-06-18T09:00:00+00:00", "s:a", "GOOD beats", "g1", 0.80, "positive"),
            ("GOOD", "2026-06-17T09:00:00+00:00", "s:b", "GOOD wins order", "g2", 0.70, "positive"),
            ("BADNEWS", "2026-06-18T09:00:00+00:00", "s:c", "BADNEWS fraud", "b1", -0.90, "negative"),
            ("BADNEWS", "2026-06-17T09:00:00+00:00", "s:d", "BADNEWS probe", "b2", -0.85, "negative"),
        ]
        conn.executemany(
            "INSERT INTO news_headlines (symbol, published_at, source, title, url, "
            "sentiment, sentiment_label) VALUES (?, ?, ?, ?, ?, ?, ?)", rows,
        )
    return db_file, sd


def test_negative_news_vetoes_entry(seeded_db):
    _, sd = seeded_db
    kept = generate_signals(signal_date=sd, run_id="r1", config=SignalGenConfig())
    syms = {k.symbol for k in kept}
    assert "GOOD" in syms
    assert "BADNEWS" not in syms          # vetoed by negative news

    # A veto reason was logged.
    vf = db_mod.fetch_all(
        "SELECT message FROM validation_failures WHERE symbol = 'BADNEWS'")
    assert any("negative-news veto" in (r["message"] or "") for r in vf)


def test_positive_news_scales_size_up(seeded_db):
    _, sd = seeded_db
    kept = generate_signals(signal_date=sd, run_id="r1", config=SignalGenConfig())
    good = next(k for k in kept if k.symbol == "GOOD")
    assert good.news_label == "positive"
    assert good.size_factor > 1.0          # positive sentiment boost
    # Persisted payload carries the audit trail.
    row = db_mod.fetch_one(
        "SELECT payload_json FROM signal_outbox WHERE symbol = 'GOOD'")
    payload = json.loads(row["payload_json"])
    assert payload["news_label"] == "positive"
    assert payload["sentiment_size_factor"] > 1.0


def test_overlay_can_be_disabled(seeded_db):
    _, sd = seeded_db
    cfg = SignalGenConfig(use_news_sentiment=False)
    kept = generate_signals(signal_date=sd, run_id="r1", config=cfg)
    syms = {k.symbol for k in kept}
    # With the overlay off, the negative-news name is no longer vetoed.
    assert {"GOOD", "BADNEWS"} <= syms
    for k in kept:
        assert k.size_factor == 1.0
