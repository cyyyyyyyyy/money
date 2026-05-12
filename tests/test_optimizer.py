from __future__ import annotations

import numpy as np
import pandas as pd

from money_strategy.optimizer import StrategyParams, evaluate_grid, walk_forward_optimize


def _sample_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-01-03", periods=420, freq="B")
    base = pd.Series(np.linspace(100, 145, len(dates)), index=dates)
    close = pd.DataFrame(
        {
            "000300": base,
            "511010": 100 + np.sin(np.linspace(0, 5, len(dates))),
            "159819": base * np.linspace(1.0, 1.25, len(dates)),
            "512480": base * np.linspace(1.0, 1.15, len(dates)),
            "159770": base * np.linspace(1.0, 1.12, len(dates)),
            "516160": base * np.linspace(1.0, 1.18, len(dates)),
            "561560": base * np.linspace(1.0, 1.08, len(dates)),
            "563530": base * np.linspace(1.0, 1.10, len(dates)),
            "515100": base * np.linspace(1.0, 1.05, len(dates)),
            "512800": base * np.linspace(1.0, 1.02, len(dates)),
        }
    )
    amount = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    weekly_close = close.resample("W-FRI").last()
    weekly_amount = amount.resample("W-FRI").sum()
    return close, amount, weekly_close, weekly_amount


def test_evaluate_grid_returns_sorted_results() -> None:
    close, amount, weekly_close, weekly_amount = _sample_panels()
    params = [
        StrategyParams(offense_threshold=60.0),
        StrategyParams(offense_threshold=63.0),
    ]

    grid, cached = evaluate_grid(close, amount, weekly_close, weekly_amount, params_grid=params)

    assert len(grid) == 2
    assert len(cached) == 2
    assert grid["score"].is_monotonic_decreasing


def test_walk_forward_returns_folds() -> None:
    close, amount, weekly_close, weekly_amount = _sample_panels()
    params = [
        StrategyParams(offense_threshold=60.0),
        StrategyParams(offense_threshold=63.0),
    ]

    result = walk_forward_optimize(
        close,
        amount,
        weekly_close,
        weekly_amount,
        params_grid=params,
        train_weeks=26,
        test_weeks=8,
    )

    assert not result.empty
    assert {"param_offense_threshold", "test_annual_return", "test_score"} <= set(result.columns)
