"""Tests for the news sentiment scorer (lexical backend, offline/deterministic).

The transformer (FinBERT) path needs torch + a model download, so these tests
pin the lexical fallback backend, which is the same code path used whenever
transformers/torch are unavailable.
"""

from __future__ import annotations

import pytest

from src.data_ingestion import finbert_scorer as fb
from src.db.migrate import apply_schema
from src.utils import db as db_mod


@pytest.fixture(autouse=True)
def _force_lexical(monkeypatch):
    # Pin the deterministic lexical backend regardless of installed deps.
    monkeypatch.setattr(fb, "_backend", "lexical")
    monkeypatch.setattr(fb, "_pipeline", None)
    yield


# --------------------------------------------------------------------------
# Lexical scoring
# --------------------------------------------------------------------------


def test_positive_headline_scores_positive():
    score, label = fb.score_text("Company profit surges, beats estimates; shares rally")
    assert score > 0 and label == "positive"


def test_negative_headline_scores_negative():
    score, label = fb.score_text("Stock plunges on fraud probe and downgrade")
    assert score < 0 and label == "negative"


def test_neutral_headline_is_neutral():
    score, label = fb.score_text("Company to hold annual general meeting next week")
    assert score == 0.0 and label == "neutral"


def test_score_texts_aligns_and_bounds():
    out = fb.score_texts(["profit jumps", "", "loss widens"])
    assert len(out) == 3
    for s, lbl in out:
        assert -1.0 <= s <= 1.0
        assert lbl in {"positive", "negative", "neutral"}
    assert out[0][0] > 0 and out[2][0] < 0


def test_active_backend_is_lexical():
    assert fb.active_backend() == "lexical"


# --------------------------------------------------------------------------
# Persistence + aggregation
# --------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "news.db"
    monkeypatch.setattr(db_mod, "resolve_db_path", lambda *a, **kw: db_file)
    apply_schema(db_path=db_file)
    rows = [
        ("TCS", "2026-06-19T09:00:00+00:00", "src:a", "TCS profit surges, beats", "u1"),
        ("TCS", "2026-06-18T09:00:00+00:00", "src:b", "TCS wins record order", "u2"),
        ("TCS", "2026-06-17T09:00:00+00:00", "src:c", "TCS stock plunges on probe", "u3"),
    ]
    with db_mod.transaction() as conn:
        conn.executemany(
            "INSERT INTO news_headlines (symbol, published_at, source, title, url) "
            "VALUES (?, ?, ?, ?, ?)", rows,
        )
    return db_file


def test_score_unscored_persists_and_is_idempotent(temp_db):
    n1 = fb.score_unscored_headlines()
    assert n1 == 3
    # Every row now has a score + label.
    cnt = db_mod.fetch_one(
        "SELECT COUNT(*) AS c FROM news_headlines WHERE sentiment IS NOT NULL")["c"]
    assert cnt == 3
    # Re-running scores nothing new (idempotent on the NULL filter).
    assert fb.score_unscored_headlines() == 0
    # Rescore touches all rows again.
    assert fb.score_unscored_headlines(rescore=True) == 3


def test_symbol_sentiment_aggregate(temp_db):
    fb.score_unscored_headlines()
    agg = fb.symbol_sentiment("TCS", as_of="2026-06-19", window_days=7)
    assert agg["available"] is True
    assert agg["n"] == 3
    assert -1.0 <= agg["score"] <= 1.0
    # Two positive + one negative headline -> net positive mean.
    assert agg["score"] > 0 and agg["label"] == "positive"


def test_symbol_sentiment_unavailable_when_empty(temp_db):
    agg = fb.symbol_sentiment("NOPE", as_of="2026-06-19")
    assert agg["available"] is False and agg["n"] == 0
