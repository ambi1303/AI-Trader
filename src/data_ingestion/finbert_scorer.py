"""FinBERT sentiment scoring for news headlines.

Replaces the prior "headlines are unscored" gap with a real polarity score per
headline, persisted into ``news_headlines.sentiment`` (signed, in [-1, 1]) and
``news_headlines.sentiment_label`` ('positive' | 'negative' | 'neutral').

Two backends, chosen automatically:

* **FinBERT** (preferred) -- the ``ProsusAI/finbert`` transformer, a BERT model
  fine-tuned on financial text. Loaded lazily on first use (CPU), cached as a
  process singleton. Signed score = P(positive) - P(negative); label = argmax.
* **Lexical fallback** -- a small, deterministic finance sentiment lexicon used
  when ``transformers``/``torch`` aren't installed or the model can't load. It
  keeps the whole pipeline working offline and makes the unit tests dependency
  free. ``active_backend()`` reports which is live.

Design notes / safety:
* The model is downloaded from Hugging Face on first run and cached under the
  HF cache dir; no secrets, network only to the model hub. We pin the model id
  and revision so a renamed/poisoned upstream tag can't silently change scores.
* Scoring is read-only w.r.t. trading; it only annotates headlines. Failures
  degrade to the lexical backend and never raise into the daily pipeline.
* All inputs are short, untrusted headline strings -- treated as plain text,
  truncated, never executed or interpolated into queries.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

from src.utils.db import execute, fetch_all
from src.utils.logger import get_logger

log = get_logger("ingest.finbert")

# Pin model id + revision so the artifact we score with can't change under us.
# ``FINBERT_MODEL_PATH`` lets you point at a locally-downloaded model directory
# (useful behind a corporate proxy that intercepts TLS to huggingface.co, or for
# fully offline runs) -- when it's an existing path we load it with
# ``local_files_only`` so transformers never touches the network.
_MODEL_ID = os.environ.get("FINBERT_MODEL_PATH") or "ProsusAI/finbert"
_MODEL_REVISION = os.environ.get("FINBERT_MODEL_REVISION", "main")
_LOCAL_ONLY = os.path.isdir(_MODEL_ID)
_MAX_CHARS = 400              # headlines are short; bound tokenizer work
_NEUTRAL_BAND = 0.15         # |score| <= this -> 'neutral' for the lexical path

# Label set FinBERT emits (lower-cased on read).
_POS, _NEG, _NEU = "positive", "negative", "neutral"

_lock = threading.Lock()
_pipeline: Any = None
_backend: str | None = None       # 'finbert' | 'lexical' (resolved on first use)


# ---------------------------------------------------------------------------
# Lexical fallback
# ---------------------------------------------------------------------------

_POS_WORDS = frozenset({
    "surge", "surges", "surged", "jump", "jumps", "jumped", "gain", "gains",
    "gained", "rise", "rises", "rose", "rally", "rallies", "rallied", "soar",
    "soars", "soared", "profit", "profits", "beat", "beats", "beated",
    "upgrade", "upgraded", "record", "growth", "grow", "grows", "bullish",
    "outperform", "outperforms", "win", "wins", "won", "approval", "approved",
    "expansion", "expand", "expands", "strong", "stronger", "boost", "boosts",
    "boosted", "high", "highs", "buy", "positive", "robust", "wins", "bags",
    "secures", "secured", "order", "orders", "dividend", "bonus", "deal",
})
_NEG_WORDS = frozenset({
    "plunge", "plunges", "plunged", "fall", "falls", "fell", "loss", "losses",
    "miss", "misses", "missed", "downgrade", "downgraded", "fraud", "probe",
    "decline", "declines", "declined", "slump", "slumps", "slumped", "crash",
    "crashes", "crashed", "bearish", "cut", "cuts", "lawsuit", "default",
    "defaults", "ban", "banned", "resign", "resigns", "resigned", "weak",
    "weaker", "drop", "drops", "dropped", "slip", "slips", "slipped", "low",
    "lows", "sell", "selloff", "negative", "warning", "warns", "warned",
    "scam", "penalty", "fine", "fined", "downturn", "halt", "halts", "recall",
})

_TOKEN_RE = re.compile(r"[a-z']+")


def _lexical_score(text: str) -> tuple[float, str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    if pos == 0 and neg == 0:
        return 0.0, _NEU
    score = (pos - neg) / (pos + neg)
    if score > _NEUTRAL_BAND:
        return score, _POS
    if score < -_NEUTRAL_BAND:
        return score, _NEG
    return score, _NEU


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _ensure_backend() -> str:
    """Resolve the scoring backend once, loading FinBERT if available."""
    global _pipeline, _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend
        # On networks that intercept TLS (corporate proxies), Python's bundled
        # CA set won't trust the proxy's root, so the model download fails. If
        # `truststore` is installed, route SSL through the OS trust store (which
        # *does* include the corporate root) -- this keeps full cert validation
        # on, we never disable verification.
        try:
            import truststore  # type: ignore

            truststore.inject_into_ssl()
        except Exception:  # noqa: BLE001 -- optional; absence just means default CAs
            pass
        try:
            from transformers import pipeline  # type: ignore

            kwargs: dict[str, Any] = {
                "top_k": None,         # return all class scores
                "truncation": True,
                "device": -1,          # CPU
            }
            if _LOCAL_ONLY:
                kwargs["model"] = _MODEL_ID
                kwargs["model_kwargs"] = {"local_files_only": True}
            else:
                kwargs["model"] = _MODEL_ID
                kwargs["revision"] = _MODEL_REVISION
            _pipeline = pipeline("sentiment-analysis", **kwargs)
            _backend = "finbert"
            log.info("FinBERT backend active ({} @ {})", _MODEL_ID, _MODEL_REVISION)
        except Exception as exc:  # noqa: BLE001 -- any import/load failure -> lexical
            _pipeline = None
            _backend = "lexical"
            log.warning("FinBERT unavailable, using lexical sentiment: {}", exc)
    return _backend


def active_backend() -> str:
    """Return the live backend ('finbert' or 'lexical'), resolving if needed."""
    return _ensure_backend()


def _finbert_to_signed(scores: list[dict[str, Any]]) -> tuple[float, str]:
    """Map FinBERT's per-class probabilities to a signed score + label."""
    by = {str(s["label"]).lower(): float(s["score"]) for s in scores}
    p_pos = by.get(_POS, 0.0)
    p_neg = by.get(_NEG, 0.0)
    p_neu = by.get(_NEU, 0.0)
    signed = p_pos - p_neg
    label = max(((p_pos, _POS), (p_neg, _NEG), (p_neu, _NEU)), key=lambda x: x[0])[1]
    return signed, label


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


def score_texts(texts: list[str]) -> list[tuple[float, str]]:
    """Score a batch of headline strings.

    Returns a list of ``(signed_score, label)`` aligned with ``texts``, where
    ``signed_score`` is in [-1, 1] (positive = bullish) and ``label`` is one of
    'positive' / 'negative' / 'neutral'.
    """
    if not texts:
        return []
    clean = [(t or "")[:_MAX_CHARS] for t in texts]
    backend = _ensure_backend()
    if backend == "finbert" and _pipeline is not None:
        try:
            raw = _pipeline(clean)
            # Normalise to list-of-lists (top_k=None yields per-item class lists).
            out: list[tuple[float, str]] = []
            for item in raw:
                scores = item if isinstance(item, list) else [item]
                out.append(_finbert_to_signed(scores))
            return out
        except Exception as exc:  # noqa: BLE001 -- fall back per-batch on failure
            log.warning("FinBERT scoring failed, falling back to lexical: {}", exc)
    return [_lexical_score(t) for t in clean]


def score_text(text: str) -> tuple[float, str]:
    """Convenience single-headline scorer."""
    return score_texts([text])[0]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def score_unscored_headlines(
    *, limit: int | None = None, batch_size: int = 64,
    rescore: bool = False, db_path: str | None = None,
) -> int:
    """Score headlines whose sentiment is NULL and persist the result.

    ``rescore=True`` re-scores every headline (e.g. after switching backend).
    Returns the number of rows updated.
    """
    where = "" if rescore else "WHERE sentiment IS NULL"
    sql = f"SELECT id, title FROM news_headlines {where} ORDER BY id"  # noqa: S608
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = fetch_all(sql, db_path=db_path)
    if not rows:
        return 0

    updated = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        scored = score_texts([r["title"] for r in chunk])
        for r, (score, label) in zip(chunk, scored):
            execute(
                "UPDATE news_headlines SET sentiment = ?, sentiment_label = ? "
                "WHERE id = ?",
                (round(float(score), 4), label, r["id"]), db_path=db_path,
            )
            updated += 1
    log.info("Scored {} headline(s) [backend={}]", updated, active_backend())
    return updated


def symbol_sentiment(
    symbol: str, *, as_of: str | None = None, window_days: int = 7,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Aggregate recent headline sentiment for a symbol.

    Returns ``{available, score, label, n, window_days}`` where ``score`` is the
    mean signed sentiment over scored headlines in the trailing window (only
    headlines with ``published_at <= as_of`` are used, so it's look-ahead safe).
    """
    day = (as_of or "")[:10]
    # Anchor the trailing window at end-of-day(as_of), or "now" when unspecified.
    anchor = (day + "T23:59:59") if day else "now"
    upper_clause = "AND published_at <= ? " if day else ""
    params: list[Any] = [symbol.upper()]
    if day:
        params.append(day + "T23:59:59")
    params.extend([anchor, f"-{int(window_days)} days"])
    rows = fetch_all(
        f"""
        SELECT sentiment FROM news_headlines
        WHERE symbol = ? {upper_clause}
          AND sentiment IS NOT NULL
          AND published_at >= datetime(?, ?)
        """,  # noqa: S608 - clauses are literal; values are bound
        tuple(params),
        db_path=db_path,
    )
    scores = [float(r["sentiment"]) for r in rows if r["sentiment"] is not None]
    if not scores:
        return {"available": False, "n": 0, "window_days": window_days}
    mean = sum(scores) / len(scores)
    if mean > _NEUTRAL_BAND:
        label = _POS
    elif mean < -_NEUTRAL_BAND:
        label = _NEG
    else:
        label = _NEU
    return {
        "available": True,
        "score": round(mean, 4),
        "label": label,
        "n": len(scores),
        "window_days": window_days,
    }
