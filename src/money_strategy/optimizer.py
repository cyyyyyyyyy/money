from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import pandas as pd

from .backtest import run_backtest, summarize_performance
from .signals import build_signal_frame


@dataclass(frozen=True)
class StrategyParams:
    switch_threshold: float = 0.15
    offense_threshold: float = 60.0
    defense_threshold: float = 35.0
    balanced_defensive_threshold: float = 45.0
    confirm_weeks: int = 1


def default_param_grid() -> list[StrategyParams]:
    return [
        StrategyParams(
            switch_threshold=switch,
            offense_threshold=offense,
            defense_threshold=defense,
            balanced_defensive_threshold=balanced,
            confirm_weeks=confirm,
        )
        for offense, defense, balanced, switch, confirm in product(
            [58.0, 60.0, 62.0, 63.0],
            [30.0, 35.0, 40.0],
            [40.0, 45.0, 50.0],
            [0.05, 0.10, 0.15],
            [1, 2],
        )
        if defense < offense
    ]


def evaluate_grid(
    daily_close: pd.DataFrame,
    daily_amount: pd.DataFrame,
    weekly_close: pd.DataFrame,
    weekly_amount: pd.DataFrame,
    *,
    params_grid: Iterable[StrategyParams] | None = None,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, dict[StrategyParams, tuple[pd.DataFrame, pd.DataFrame]]]:
    rows: list[dict[str, float]] = []
    cached: dict[StrategyParams, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for params in params_grid or default_param_grid():
        signals = build_signal_frame(daily_close, daily_amount, weekly_close, weekly_amount, **asdict(params))
        result = run_backtest(weekly_close, signals, cost_bps=cost_bps)
        metrics = result.performance.loc["strategy"].to_dict()
        row = {**asdict(params), **metrics}
        row["score"] = score_row(row)
        rows.append(row)
        cached[params] = (result.equity_curve, result.signals)

    frame = pd.DataFrame(rows).sort_values(["score", "annual_return"], ascending=False).reset_index(drop=True)
    return frame, cached


def walk_forward_optimize(
    daily_close: pd.DataFrame,
    daily_amount: pd.DataFrame,
    weekly_close: pd.DataFrame,
    weekly_amount: pd.DataFrame,
    *,
    params_grid: Iterable[StrategyParams] | None = None,
    train_weeks: int = 52,
    test_weeks: int = 13,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    grid = list(params_grid or default_param_grid())
    _, cached = evaluate_grid(
        daily_close,
        daily_amount,
        weekly_close,
        weekly_amount,
        params_grid=grid,
        cost_bps=cost_bps,
    )

    valid_index = next(iter(cached.values()))[0].index
    rows: list[dict[str, float | str]] = []
    start = 0
    fold = 1
    while start + train_weeks + test_weeks <= len(valid_index):
        train_index = valid_index[start : start + train_weeks]
        test_index = valid_index[start + train_weeks : start + train_weeks + test_weeks]

        train_rows = []
        for params in grid:
            equity, signals = cached[params]
            train_metrics = period_metrics(equity, signals, train_index)
            train_rows.append((score_row(train_metrics), params, train_metrics))

        _, best_params, best_train = max(train_rows, key=lambda item: item[0])
        best_equity, best_signals = cached[best_params]
        test_metrics = period_metrics(best_equity, best_signals, test_index)

        rows.append(
            {
                "fold": fold,
                "train_start": str(train_index[0].date()),
                "train_end": str(train_index[-1].date()),
                "test_start": str(test_index[0].date()),
                "test_end": str(test_index[-1].date()),
                **{f"param_{key}": value for key, value in asdict(best_params).items()},
                **{f"train_{key}": value for key, value in best_train.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "test_score": score_row(test_metrics),
            }
        )
        fold += 1
        start += test_weeks

    return pd.DataFrame(rows)


def period_metrics(equity: pd.DataFrame, signals: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, float]:
    period_equity = equity.reindex(index).dropna()
    period_signals = signals.reindex(period_equity.index).dropna(how="all")
    if period_equity.empty:
        return {}
    performance = summarize_performance(period_equity, period_signals).loc["strategy"]
    return performance.to_dict()


def score_row(row: dict[str, float]) -> float:
    annual = float(row.get("annual_return", 0.0) or 0.0)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    max_drawdown = float(row.get("max_drawdown", 0.0) or 0.0)
    turnover = float(row.get("annual_turnover", 0.0) or 0.0)
    return annual + 0.10 * sharpe + 0.50 * max_drawdown - 0.001 * turnover
