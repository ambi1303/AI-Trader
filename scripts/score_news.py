"""Score stored news headlines with FinBERT (or the lexical fallback).

Fills ``news_headlines.sentiment`` (signed, -1..1) and ``sentiment_label``
('positive' | 'negative' | 'neutral') for rows that don't have a score yet.

Usage:
    python -m scripts.score_news                  # score all unscored rows
    python -m scripts.score_news --limit 500      # cap work this run
    python -m scripts.score_news --rescore        # re-score everything
"""

from __future__ import annotations

import argparse
import sys

from src.data_ingestion.finbert_scorer import active_backend, score_unscored_headlines
from src.db.migrate import apply_schema
from src.utils.logger import get_logger

log = get_logger("script.score_news")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None,
                   help="Max headlines to score this run (default: all unscored).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rescore", action="store_true",
                   help="Re-score every headline, not just unscored ones.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    apply_schema()
    n = score_unscored_headlines(
        limit=args.limit, batch_size=args.batch_size, rescore=args.rescore,
    )
    print(f"Scored {n} headline(s) using backend='{active_backend()}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
