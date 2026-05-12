from __future__ import annotations

import pandas as pd

from money_strategy.backtest import run_backtest


def test_run_backtest_produces_equity_and_metrics() -> None:
    index = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    weekly_close = pd.DataFrame(
        {
            "000300": [100, 101, 102, 101, 103, 104, 103, 105],
            "159819": [12, 13, 14, 13, 14, 15, 15, 16],
            "512480": [20, 20, 21, 21, 22, 22, 23, 24],
            "159770": [8, 8.1, 8.3, 8.2, 8.5, 8.7, 8.6, 8.9],
            "516160": [11, 11.2, 11.1, 11.4, 11.6, 11.8, 11.7, 12.0],
            "561560": [6, 6.1, 6.2, 6.2, 6.3, 6.4, 6.4, 6.5],
            "515100": [5, 5.05, 5.1, 5.15, 5.2, 5.22, 5.24, 5.3],
            "512800": [4, 4.1, 4.0, 4.1, 4.2, 4.2, 4.3, 4.3],
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "bucket": ["balanced"] * len(index),
            "weight_159819": [0.5] * len(index),
            "weight_512480": [0.0] * len(index),
            "weight_159770": [0.0] * len(index),
            "weight_516160": [0.0] * len(index),
            "weight_561560": [0.0] * len(index),
            "weight_515100": [0.0] * len(index),
            "weight_512800": [0.5] * len(index),
        },
        index=index,
    )

    result = run_backtest(weekly_close, signals)

    assert result.equity_curve["strategy"].iloc[-1] > 0
    assert "strategy" in result.performance.index
    assert "max_drawdown" in result.performance.columns
    assert not result.trades.empty
    assert {"buy_date", "sell_date", "sell_reason"}.issubset(result.trades.columns)
