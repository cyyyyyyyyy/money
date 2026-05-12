from __future__ import annotations

import pandas as pd

from money_strategy.data import _normalize_akshare_etf, _normalize_akshare_index


def test_normalize_akshare_etf_columns() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2026-05-11"],
            "开盘": [1.0],
            "收盘": [1.1],
            "最高": [1.2],
            "最低": [0.9],
            "成交量": [1000],
            "成交额": [10000],
            "振幅": [1.0],
            "涨跌幅": [2.0],
            "涨跌额": [0.1],
            "换手率": [3.0],
        }
    )

    frame = _normalize_akshare_etf(raw)

    assert frame.index[0] == pd.Timestamp("2026-05-11")
    assert frame.loc[pd.Timestamp("2026-05-11"), "close"] == 1.1


def test_normalize_akshare_index_columns() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-05-11", "2026-05-12"],
            "open": [100, 101],
            "close": [101, 102],
            "high": [102, 103],
            "low": [99, 100],
            "volume": [1000, 1100],
            "amount": [10000, 12000],
        }
    )

    frame = _normalize_akshare_index(raw)

    assert frame.index[-1] == pd.Timestamp("2026-05-12")
    assert frame.loc[pd.Timestamp("2026-05-12"), "close"] == 102
