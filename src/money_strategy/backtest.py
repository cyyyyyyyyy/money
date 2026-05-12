from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ALL_INSTRUMENTS, DEFENSIVE, MARKET, ROTATION


@dataclass(frozen=True)
class BacktestResult:
    signals: pd.DataFrame
    equity_curve: pd.DataFrame
    performance: pd.DataFrame
    trades: pd.DataFrame


def run_backtest(
    weekly_close: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
) -> BacktestResult:
    configured_codes = list(dict.fromkeys(item.code for item in (*ROTATION, *DEFENSIVE)))
    asset_codes = [code for code in configured_codes if code in weekly_close.columns and f"weight_{code}" in signals.columns]
    required_codes = [MARKET.code, *asset_codes]
    valid_index = weekly_close[required_codes].dropna().index.intersection(signals.index)
    weekly_close = weekly_close.reindex(valid_index)
    signals = signals.reindex(valid_index)

    weight_cols = [f"weight_{code}" for code in asset_codes]
    weights = signals[weight_cols].copy()
    weights.columns = asset_codes

    weekly_returns = weekly_close[asset_codes].pct_change().reindex(weights.index).fillna(0.0)
    execution_weights = weights.shift(1).fillna(0.0)
    turnover = execution_weights.diff().abs().sum(axis=1).fillna(execution_weights.abs().sum(axis=1))
    cost = turnover * (cost_bps / 10000.0)
    strategy_ret = (execution_weights * weekly_returns).sum(axis=1) - cost

    market_ret = weekly_close[MARKET.code].pct_change().reindex(weights.index).fillna(0.0)
    defensive_codes = [item.code for item in DEFENSIVE if item.code in weekly_close.columns]
    defensive_static_ret = (
        weekly_close[defensive_codes].pct_change().reindex(weights.index).fillna(0.0).mean(axis=1)
    )

    equity = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "benchmark_return": market_ret,
            "defensive_static_return": defensive_static_ret,
            "turnover": turnover,
            "cost": cost,
        },
        index=weights.index,
    )
    equity["strategy"] = (1.0 + equity["strategy_return"]).cumprod()
    equity["benchmark_hs300"] = (1.0 + equity["benchmark_return"]).cumprod()
    equity["defensive_static"] = (1.0 + equity["defensive_static_return"]).cumprod()

    performance = summarize_performance(equity, signals)
    trades = build_trade_ledger(weekly_close, signals, asset_codes, cost_bps=cost_bps)
    return BacktestResult(signals=signals, equity_curve=equity, performance=performance, trades=trades)


def build_trade_ledger(
    weekly_close: pd.DataFrame,
    signals: pd.DataFrame,
    asset_codes: list[str],
    *,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    names = {instrument.code: instrument.name for instrument in ALL_INSTRUMENTS}
    weight_cols = [f"weight_{code}" for code in asset_codes if f"weight_{code}" in signals.columns]
    if not weight_cols:
        return pd.DataFrame()

    targets = signals[weight_cols].idxmax(axis=1).str.replace("weight_", "", regex=False)
    rows: list[dict[str, object]] = []
    current_code: str | None = None
    buy_date: pd.Timestamp | None = None
    buy_price: float | None = None
    entry_bucket: str | None = None
    entry_market_score: float | None = None
    one_side_cost = cost_bps / 10000.0

    for date, target_code in targets.items():
        target_code = str(target_code)
        if current_code is None:
            current_code = target_code
            buy_date = pd.Timestamp(date)
            buy_price = _price_at(weekly_close, date, current_code)
            entry_bucket = str(signals.loc[date].get("bucket", ""))
            entry_market_score = _float_or_none(signals.loc[date].get("market_score"))
            continue

        if target_code == current_code:
            continue

        sell_date = pd.Timestamp(date)
        sell_price = _price_at(weekly_close, date, current_code)
        gross_return = sell_price / buy_price - 1.0 if buy_price and sell_price else None
        net_return = (
            sell_price / buy_price * (1.0 - one_side_cost) ** 2 - 1.0
            if buy_price and sell_price
            else None
        )
        rows.append(
            {
                "status": "closed",
                "code": current_code,
                "name": names.get(current_code, current_code),
                "buy_date": buy_date,
                "sell_date": sell_date,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "holding_days": (sell_date - buy_date).days if buy_date is not None else None,
                "holding_weeks": round((sell_date - buy_date).days / 7, 2) if buy_date is not None else None,
                "gross_return": gross_return,
                "estimated_net_return": net_return,
                "entry_bucket": entry_bucket,
                "exit_bucket": str(signals.loc[date].get("bucket", "")),
                "entry_market_score": entry_market_score,
                "exit_market_score": _float_or_none(signals.loc[date].get("market_score")),
                "sell_reason": f"switch_to_{target_code}",
                "next_code": target_code,
                "next_name": names.get(target_code, target_code),
            }
        )

        current_code = target_code
        buy_date = sell_date
        buy_price = _price_at(weekly_close, date, current_code)
        entry_bucket = str(signals.loc[date].get("bucket", ""))
        entry_market_score = _float_or_none(signals.loc[date].get("market_score"))

    if current_code is not None and buy_date is not None:
        last_date = pd.Timestamp(signals.index[-1])
        last_price = _price_at(weekly_close, last_date, current_code)
        gross_return = last_price / buy_price - 1.0 if buy_price and last_price else None
        rows.append(
            {
                "status": "open",
                "code": current_code,
                "name": names.get(current_code, current_code),
                "buy_date": buy_date,
                "sell_date": pd.NaT,
                "buy_price": buy_price,
                "sell_price": None,
                "holding_days": (last_date - buy_date).days,
                "holding_weeks": round((last_date - buy_date).days / 7, 2),
                "gross_return": gross_return,
                "estimated_net_return": (
                    last_price / buy_price * (1.0 - one_side_cost) - 1.0
                    if buy_price and last_price
                    else None
                ),
                "entry_bucket": entry_bucket,
                "exit_bucket": None,
                "entry_market_score": entry_market_score,
                "exit_market_score": None,
                "sell_reason": "still_holding",
                "next_code": None,
                "next_name": None,
            }
        )

    return pd.DataFrame(rows)


def _price_at(weekly_close: pd.DataFrame, date: pd.Timestamp, code: str) -> float | None:
    value = weekly_close.loc[date, code] if code in weekly_close.columns and date in weekly_close.index else np.nan
    return _float_or_none(value)


def _float_or_none(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def summarize_performance(equity: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _metric_row("strategy", equity["strategy_return"], equity["strategy"], equity["turnover"], signals),
        _metric_row("benchmark_hs300", equity["benchmark_return"], equity["benchmark_hs300"], None, signals),
        _metric_row(
            "defensive_static",
            equity["defensive_static_return"],
            equity["defensive_static"],
            None,
            signals,
        ),
    ]
    return pd.DataFrame(rows).set_index("name")


def _metric_row(
    name: str,
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series | None,
    signals: pd.DataFrame,
) -> dict[str, float | str]:
    periods_per_year = 52
    clean_returns = returns.dropna()
    years = max(len(clean_returns) / periods_per_year, 1e-9)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else np.nan
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    annual_vol = float(clean_returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    win_rate = float((clean_returns > 0).mean())

    row: dict[str, float | str] = {
        "name": name,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
    }
    if turnover is not None:
        row["annual_turnover"] = float(turnover.mean() * periods_per_year)
        for bucket in ("offense", "balanced", "defense"):
            row[f"{bucket}_ratio"] = float((signals["bucket"] == bucket).mean())
    return row
