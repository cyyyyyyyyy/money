from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sentiment_scores(path: Path, weekly_index: pd.DatetimeIndex, *, decay_weeks: int = 8) -> pd.DataFrame:
    """Load optional weekly news and policy sentiment scores.

    Expected CSV columns:
    - date: any parseable date
    - news_score: optional 0-100, where 50 is neutral
    - policy_score: optional 0-100, where 50 is neutral
    """
    frame = pd.read_csv(path, parse_dates=["date"])
    score_columns = [
        column
        for column in frame.columns
        if column in {"news_score", "policy_score"} or column.startswith("theme_score_")
    ]
    if not score_columns:
        raise ValueError("sentiment file must contain news_score or policy_score")

    frame = frame[["date", *score_columns]].sort_values("date").set_index("date")
    for column in score_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(0, 100)

    weekly = _decay_events_to_weekly(frame, weekly_index, decay_weeks=decay_weeks)
    for column in ("news_score", "policy_score"):
        if column not in weekly.columns:
            weekly[column] = 50.0
    ordered = ["news_score", "policy_score", *sorted(column for column in weekly.columns if column.startswith("theme_score_"))]
    return weekly[ordered].fillna(50.0)


def load_news_sentiment(path: Path, weekly_index: pd.DatetimeIndex) -> pd.Series:
    return load_sentiment_scores(path, weekly_index)["news_score"]


def _decay_events_to_weekly(
    frame: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    *,
    decay_weeks: int,
) -> pd.DataFrame:
    if decay_weeks <= 0:
        raise ValueError("decay_weeks must be positive")

    rows: list[pd.Series] = []
    for date in weekly_index:
        history = frame.loc[frame.index <= date]
        if history.empty:
            rows.append(pd.Series(50.0, index=frame.columns, name=date))
            continue

        event_date = history.index[-1]
        event_scores = history.iloc[-1]
        weeks_elapsed = max(0, int((date - event_date).days // 7))
        if weeks_elapsed >= decay_weeks:
            rows.append(pd.Series(50.0, index=frame.columns, name=date))
            continue

        decay = 1.0 - weeks_elapsed / decay_weeks
        rows.append((50.0 + (event_scores - 50.0) * decay).rename(date))

    return pd.DataFrame(rows, index=weekly_index)
