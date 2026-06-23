"""Tests for pure market-breadth aggregation."""

from __future__ import annotations

from src.features.breadth_features import compute_breadth


def _row(d50=0.05, d200=0.05, dd=-0.02, ret=0.01):
    return {
        "dist_ema_50_pct": d50,
        "dist_ema_200_pct": d200,
        "dd_from_high_252d": dd,
        "ret_1d": ret,
    }


def test_empty_universe_is_unavailable():
    out = compute_breadth([])
    assert out["available"] is False
    assert out["universe_count"] == 0
    assert out["breadth_score"] is None


def test_all_above_mas_gives_full_breadth():
    rows = [_row(d50=0.1, d200=0.1) for _ in range(10)]
    out = compute_breadth(rows)
    assert out["available"] is True
    assert out["pct_above_50dma"] == 100.0
    assert out["pct_above_200dma"] == 100.0
    assert out["breadth_score"] == 100.0


def test_half_above_mas():
    rows = [_row(d50=0.1, d200=0.1) for _ in range(5)]
    rows += [_row(d50=-0.1, d200=-0.1) for _ in range(5)]
    out = compute_breadth(rows)
    assert out["pct_above_50dma"] == 50.0
    assert out["pct_above_200dma"] == 50.0
    assert out["breadth_score"] == 50.0


def test_advance_decline_ratio():
    rows = [_row(ret=0.02) for _ in range(8)]       # 8 advancers
    rows += [_row(ret=-0.02) for _ in range(2)]     # 2 decliners
    out = compute_breadth(rows)
    assert out["adv_decl_ratio"] == 4.0


def test_new_high_pct_counts_names_at_their_high():
    rows = [_row(dd=-0.001) for _ in range(3)]       # at new highs
    rows += [_row(dd=-0.20) for _ in range(7)]       # well off highs
    out = compute_breadth(rows)
    assert out["new_high_pct"] == 30.0


def test_none_values_are_skipped_per_metric():
    rows = [
        {"dist_ema_50_pct": 0.1, "dist_ema_200_pct": None,
         "dd_from_high_252d": None, "ret_1d": None},
        {"dist_ema_50_pct": -0.1, "dist_ema_200_pct": 0.1,
         "dd_from_high_252d": -0.02, "ret_1d": 0.01},
    ]
    out = compute_breadth(rows)
    # 50-DMA: 1 of 2 positive -> 50%; 200-DMA: 1 of 1 usable positive -> 100%.
    assert out["pct_above_50dma"] == 50.0
    assert out["pct_above_200dma"] == 100.0
    assert out["available"] is True
