from __future__ import annotations

from pathlib import Path

import pandas as pd

from money_strategy.news import append_candidates_to_sentiment, score_news_item


def test_score_news_item_detects_policy_and_themes() -> None:
    candidate = score_news_item(
        "2026-05-12",
        "持续推进人工智能+行动并支持集成电路产业发展",
        "政策支持大模型、芯片和智能机器人等未来产业",
        "https://example.com/a",
        source="测试来源",
    )

    assert candidate is not None
    assert candidate.policy_score >= 62
    assert candidate.theme_score_159819 > 50
    assert candidate.theme_score_512480 > 50
    assert candidate.theme_score_159770 > 50


def test_append_candidates_to_sentiment_deduplicates_by_source_url(tmp_path: Path) -> None:
    path = tmp_path / "policy_events.csv"
    candidates = pd.DataFrame(
        [
            {
                "date": "2026-05-12",
                "news_score": 60,
                "policy_score": 70,
                "theme_score_159819": 75,
                "theme_score_512480": 50,
                "theme_score_159770": 50,
                "theme_score_516160": 50,
                "theme_score_561560": 50,
                "theme_score_563530": 50,
                "note": "政策事件",
                "source_url": "https://example.com/a",
            },
            {
                "date": "2026-05-12",
                "news_score": 60,
                "policy_score": 70,
                "theme_score_159819": 75,
                "theme_score_512480": 50,
                "theme_score_159770": 50,
                "theme_score_516160": 50,
                "theme_score_561560": 50,
                "theme_score_563530": 50,
                "note": "重复事件",
                "source_url": "https://example.com/a",
            },
        ]
    )

    result = append_candidates_to_sentiment(candidates, path)

    assert len(result) == 1
    assert path.exists()
