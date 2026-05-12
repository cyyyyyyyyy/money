from __future__ import annotations

import numpy as np
import pandas as pd
from money_strategy.signals import (
    _smooth_bucket,
    attach_target_weights,
    build_signal_frame,
    defensive_factor_score,
    growth_expansion_score,
    rotation_factor_score,
    apply_theme_sentiment,
    support_confirmation_score,
)


def test_smooth_bucket_requires_confirmation_and_moves_one_step() -> None:
    index = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    raw = pd.Series(["balanced", "offense", "offense", "defense", "defense", "defense"], index=index)

    smoothed = _smooth_bucket(raw)

    assert smoothed.tolist() == ["balanced", "balanced", "offense", "offense", "balanced", "balanced"]


def test_attach_target_weights_holds_one_etf() -> None:
    index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    signals = pd.DataFrame(
        {
            "bucket": ["offense", "balanced", "defense"],
            "market_score": [70.0, 45.0, 30.0],
            "rotation_choice": ["159819", "512480", "512480"],
            "defensive_first": ["515100", "512800", "515100"],
            "defensive_second": ["512800", "515100", "512800"],
            "rotation_score_159819": [0.8, 0.1, 0.1],
            "rotation_score_512480": [0.2, 0.8, 0.2],
            "rotation_score_159770": [0.1, 0.2, 0.3],
            "rotation_score_516160": [0.1, 0.2, 0.3],
            "rotation_score_561560": [0.1, 0.2, 0.3],
            "defensive_score_515100": [0.8, 0.2, 0.8],
            "defensive_score_512800": [0.2, 0.8, 0.2],
        },
        index=index,
    )

    result = attach_target_weights(signals)
    weight_cols = [column for column in result.columns if column.startswith("weight_")]

    assert result.loc[index[0], "weight_159819"] == 1.0
    assert result.loc[index[1], "weight_512800"] == 1.0
    assert result.loc[index[2], "weight_515100"] == 1.0
    assert np.allclose(result[weight_cols].sum(axis=1), 1.0)


def test_single_holding_requires_score_advantage_within_same_sleeve() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    signals = pd.DataFrame(
        {
            "bucket": ["offense", "offense"],
            "market_score": [70.0, 70.0],
            "rotation_choice": ["159819", "512480"],
            "defensive_first": ["515100", "515100"],
            "defensive_second": ["512800", "512800"],
            "rotation_score_159819": [0.70, 0.70],
            "rotation_score_512480": [0.60, 0.75],
            "rotation_score_159770": [0.10, 0.10],
            "rotation_score_516160": [0.10, 0.10],
            "rotation_score_561560": [0.10, 0.10],
            "defensive_score_515100": [0.80, 0.80],
            "defensive_score_512800": [0.20, 0.20],
        },
        index=index,
    )

    result = attach_target_weights(signals, switch_threshold=0.10)

    assert result.loc[index[0], "weight_159819"] == 1.0
    assert result.loc[index[1], "weight_159819"] == 1.0
    assert result.loc[index[1], "weight_512480"] == 0.0


def test_build_signal_frame_outputs_weights_that_sum_to_one() -> None:
    dates = pd.date_range("2022-01-03", periods=620, freq="B")
    base = pd.Series(np.linspace(100, 160, len(dates)), index=dates)
    daily_close = pd.DataFrame(
        {
            "000300": base,
            "511010": 100 + np.sin(np.linspace(0, 5, len(dates))),
            "588000": base * np.linspace(1.0, 1.35, len(dates)),
            "159949": base * np.linspace(1.0, 1.25, len(dates)),
            "510300": base * np.linspace(1.0, 1.08, len(dates)),
            "510050": base * np.linspace(1.0, 1.07, len(dates)),
            "510500": base * np.linspace(1.0, 1.09, len(dates)),
            "159915": base * np.linspace(1.0, 1.12, len(dates)),
            "159819": base * np.linspace(1.0, 1.30, len(dates)),
            "512480": base * np.linspace(1.0, 1.15, len(dates)),
            "159770": base * np.linspace(1.0, 1.24, len(dates)),
            "516160": base * np.linspace(1.0, 1.22, len(dates)),
            "561560": base * np.linspace(1.0, 1.10, len(dates)),
            "563530": base * np.linspace(1.0, 1.26, len(dates)),
            "515100": base * np.linspace(1.0, 1.08, len(dates)),
            "512800": base * np.linspace(1.0, 1.02, len(dates)),
        }
    )
    daily_amount = pd.DataFrame(1_000_000.0, index=dates, columns=daily_close.columns)
    weekly_close = daily_close.resample("W-FRI").last()
    weekly_amount = daily_amount.resample("W-FRI").sum()

    signals = build_signal_frame(daily_close, daily_amount, weekly_close, weekly_amount)
    weight_cols = [column for column in signals.columns if column.startswith("weight_")]

    assert not signals.empty
    assert np.allclose(signals[weight_cols].sum(axis=1).tail(20), 1.0)
    assert set(signals["bucket"].dropna().unique()) <= {"offense", "balanced", "defense"}
    assert "breadth_score" in signals.columns
    assert "growth_score" in signals.columns
    assert "support_score" in signals.columns
    assert "watch_score_563530" in signals.columns


def test_news_sentiment_changes_market_score() -> None:
    index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    daily_dates = pd.date_range("2024-01-01", periods=15, freq="B")
    columns = ["000300", "511010", "159819", "512480", "159770", "516160", "561560", "563530", "515100", "512800"]
    daily_close = pd.DataFrame(100.0, index=daily_dates, columns=columns)
    daily_amount = pd.DataFrame(1_000_000.0, index=daily_dates, columns=daily_close.columns)
    weekly_close = pd.DataFrame(100.0, index=index, columns=daily_close.columns)
    weekly_amount = pd.DataFrame(5_000_000.0, index=index, columns=daily_close.columns)

    neutral = build_signal_frame(daily_close, daily_amount, weekly_close, weekly_amount)
    positive = build_signal_frame(
        daily_close,
        daily_amount,
        weekly_close,
        weekly_amount,
        pd.Series(80.0, index=index),
    )

    assert (positive["market_score"] > neutral["market_score"]).all()


def test_policy_sentiment_changes_market_score() -> None:
    index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    daily_dates = pd.date_range("2024-01-01", periods=15, freq="B")
    columns = ["000300", "511010", "159819", "512480", "159770", "516160", "561560", "563530", "515100", "512800"]
    daily_close = pd.DataFrame(100.0, index=daily_dates, columns=columns)
    daily_amount = pd.DataFrame(1_000_000.0, index=daily_dates, columns=daily_close.columns)
    weekly_close = pd.DataFrame(100.0, index=index, columns=daily_close.columns)
    weekly_amount = pd.DataFrame(5_000_000.0, index=index, columns=daily_close.columns)

    neutral = build_signal_frame(daily_close, daily_amount, weekly_close, weekly_amount)
    policy_positive = build_signal_frame(
        daily_close,
        daily_amount,
        weekly_close,
        weekly_amount,
        pd.DataFrame({"news_score": 50.0, "policy_score": 85.0}, index=index),
    )

    assert (policy_positive["market_score"] > neutral["market_score"]).all()


def test_theme_sentiment_changes_rotation_score() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    scores = pd.DataFrame({"159819": [0.5, 0.5], "512480": [0.5, 0.5]}, index=index)
    sentiment = pd.DataFrame({"theme_score_159819": [90, 90]}, index=index)

    adjusted = apply_theme_sentiment(scores, sentiment, index)

    assert adjusted["159819"].iloc[-1] > adjusted["512480"].iloc[-1]


def test_neutral_theme_sentiment_does_not_change_rotation_score() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    scores = pd.DataFrame({"159819": [0.2, 0.8], "512480": [0.7, 0.3]}, index=index)
    sentiment = pd.DataFrame({"theme_score_159819": [50, 50], "theme_score_512480": [50, 50]}, index=index)

    adjusted = apply_theme_sentiment(scores, sentiment, index)

    pd.testing.assert_frame_equal(adjusted, scores)


def test_growth_expansion_score_rewards_growth_leadership() -> None:
    index = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
    weekly_close = pd.DataFrame(
        {
            "000300": np.linspace(100, 110, len(index)),
            "588000": np.linspace(100, 150, len(index)),
            "159949": np.linspace(100, 145, len(index)),
        },
        index=index,
    )

    score = growth_expansion_score(weekly_close, ["588000", "159949"], "000300")

    assert score.iloc[-1] > 50


def test_support_confirmation_score_detects_weak_market_absorption() -> None:
    index = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
    weekly_close = pd.DataFrame(
        {
            "000300": np.linspace(110, 100, len(index)),
            "510300": np.linspace(110, 101, len(index)),
            "510050": np.linspace(110, 102, len(index)),
            "515100": np.linspace(100, 105, len(index)),
            "512800": np.linspace(100, 106, len(index)),
        },
        index=index,
    )
    weekly_amount = pd.DataFrame(1_000_000.0, index=index, columns=weekly_close.columns)
    weekly_amount.loc[index[-1], ["510300", "510050"]] = 3_000_000.0

    score = support_confirmation_score(weekly_close, weekly_amount, ["510300", "510050"], ["515100", "512800"], "000300")

    assert score.iloc[-1] > 50


def test_rotation_factor_score_prefers_strong_confirmed_asset() -> None:
    index = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
    weekly_close = pd.DataFrame(
        {
            "a": np.linspace(100, 150, len(index)),
            "b": np.linspace(100, 110, len(index)),
        },
        index=index,
    )
    weekly_amount = pd.DataFrame(1_000_000.0, index=index, columns=weekly_close.columns)

    scores = rotation_factor_score(weekly_close, weekly_amount, ["a", "b"])

    assert scores["a"].iloc[-1] > scores["b"].iloc[-1]


def test_defensive_factor_score_rewards_lower_volatility() -> None:
    index = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
    weekly_close = pd.DataFrame(
        {
            "stable": np.linspace(100, 112, len(index)),
            "volatile": [100, 115, 95, 116, 96, 117, 97, 118, 98, 119, 99, 120, 100, 121, 101, 122, 102, 123, 103, 124],
        },
        index=index,
    )

    scores = defensive_factor_score(weekly_close, ["stable", "volatile"])

    assert scores["stable"].iloc[-1] > scores["volatile"].iloc[-1]
