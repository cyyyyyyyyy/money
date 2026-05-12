from __future__ import annotations

from pathlib import Path

import pandas as pd

from money_strategy.sentiment import load_sentiment_scores


def test_load_sentiment_scores_preserves_theme_columns(tmp_path: Path) -> None:
    path = tmp_path / "sentiment.csv"
    path.write_text(
        "date,news_score,policy_score,theme_score_159819\n"
        "2024-01-01,60,70,80\n",
        encoding="utf-8",
    )
    weekly_index = pd.date_range("2024-01-05", periods=2, freq="W-FRI")

    result = load_sentiment_scores(path, weekly_index)

    assert result.loc[weekly_index[0], "news_score"] == 60
    assert result.loc[weekly_index[0], "policy_score"] == 70
    assert result.loc[weekly_index[0], "theme_score_159819"] == 80


def test_load_sentiment_scores_decays_event_impact(tmp_path: Path) -> None:
    path = tmp_path / "sentiment.csv"
    path.write_text("date,policy_score\n2024-01-01,90\n", encoding="utf-8")
    weekly_index = pd.to_datetime(["2024-01-05", "2024-01-26", "2024-03-01"])

    result = load_sentiment_scores(path, weekly_index, decay_weeks=4)

    assert result.loc[weekly_index[0], "policy_score"] == 90
    assert result.loc[weekly_index[1], "policy_score"] == 60
    assert result.loc[weekly_index[2], "policy_score"] == 50
