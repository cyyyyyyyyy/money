from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestResult
from .optimizer import period_metrics


def build_validation_report(
    result: BacktestResult,
    weekly_close: pd.DataFrame,
    *,
    requested_start: str,
    requested_end: str,
    cost_bps: float,
    sentiment_file: str | None,
    params: Any,
    min_years: float = 5.0,
    forward_split: float = 0.60,
) -> dict[str, Any]:
    equity = result.equity_curve
    if equity.empty:
        return {
            "method": "rule_based_no_training",
            "warnings": ["empty_backtest_result"],
        }

    effective_start = pd.Timestamp(equity.index[0])
    effective_end = pd.Timestamp(equity.index[-1])
    requested_start_ts = pd.Timestamp(requested_start)
    weeks = int(len(equity))
    years = weeks / 52.0
    weight_cols = [column for column in result.signals.columns if column.startswith("weight_")]
    traded_codes = [column.replace("weight_", "") for column in weight_cols if result.signals[column].sum() > 0]
    required_codes = [code for code in [*traded_codes, "000300"] if code in weekly_close.columns]
    first_valid_by_code = {
        code: str(weekly_close[code].dropna().index[0].date())
        for code in required_codes
        if not weekly_close[code].dropna().empty
    }

    warnings: list[str] = []
    if years < min_years:
        warnings.append(
            f"effective_history_short:{years:.2f}y<{min_years:.2f}y; current ETF pool is not enough for long-cycle proof"
        )
    if effective_start > requested_start_ts + pd.Timedelta(days=180):
        warnings.append(
            f"start_shifted_by_data_availability:{requested_start}->{effective_start.date()}; recent ETF listings may create survivorship/selection bias"
        )
    if sentiment_file:
        warnings.append("sentiment_file_used; verify every event timestamp is point-in-time before using it as historical evidence")
    if result.performance.loc["strategy", "annual_turnover"] > 20:
        warnings.append("high_turnover; live slippage and execution quality need separate tracking")

    split_report = _forward_split_report(result, forward_split=forward_split)

    return {
        "method": "rule_based_no_training",
        "execution_assumption": "weekly close signal; next-period weights are used in returns via weights.shift(1)",
        "cost_bps": float(cost_bps),
        "sentiment_file": str(sentiment_file) if sentiment_file else None,
        "uses_sentiment_in_backtest": bool(sentiment_file),
        "requested_period": {
            "start": requested_start,
            "end": requested_end,
        },
        "effective_period": {
            "start": str(effective_start.date()),
            "end": str(effective_end.date()),
            "weeks": weeks,
            "years": round(years, 4),
        },
        "first_valid_by_code": first_valid_by_code,
        "params": _params_to_dict(params),
        "full_period": _metric_payload(result.performance.loc["strategy"].to_dict()),
        "forward_split": split_report,
        "warnings": warnings,
    }


def validation_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    full = report.get("full_period", {})
    rows.append({"period": "full", **full})
    split = report.get("forward_split", {})
    for key in ("early", "late"):
        metrics = split.get(key, {})
        if metrics:
            rows.append({"period": key, **metrics})
    return pd.DataFrame(rows)


def write_validation_outputs(report: dict[str, Any], output_dir: Path) -> None:
    import json

    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_summary_frame(report).to_csv(output_dir / "validation_summary.csv", index=False)


def _forward_split_report(result: BacktestResult, *, forward_split: float) -> dict[str, Any]:
    equity = result.equity_curve
    if len(equity) < 104:
        return {
            "enabled": False,
            "reason": "need_at_least_104_weekly_points",
        }

    split_pos = max(52, min(len(equity) - 26, int(len(equity) * forward_split)))
    early_index = pd.DatetimeIndex(equity.index[:split_pos])
    late_index = pd.DatetimeIndex(equity.index[split_pos:])
    early = period_metrics(equity, result.signals, early_index)
    late = period_metrics(equity, result.signals, late_index)
    return {
        "enabled": True,
        "note": "fixed-parameter forward split; not a trained model, but separates earlier and later market regimes",
        "early": {
            "start": str(early_index[0].date()),
            "end": str(early_index[-1].date()),
            **_metric_payload(early),
        },
        "late": {
            "start": str(late_index[0].date()),
            "end": str(late_index[-1].date()),
            **_metric_payload(late),
        },
    }


def _metric_payload(metrics: dict[str, Any]) -> dict[str, float]:
    keys = [
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "win_rate",
        "annual_turnover",
        "offense_ratio",
        "balanced_ratio",
        "defense_ratio",
    ]
    return {key: round(float(metrics[key]), 6) for key in keys if key in metrics and pd.notna(metrics[key])}


def _params_to_dict(params: Any) -> dict[str, Any]:
    if is_dataclass(params):
        return asdict(params)
    if isinstance(params, dict):
        return dict(params)
    return {}
