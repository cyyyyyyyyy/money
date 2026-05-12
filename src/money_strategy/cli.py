from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from .backtest import run_backtest
from .config import ALL_INSTRUMENTS, DEFAULT_COST_BPS
from .data import load_universe, make_amount_panel, make_close_panel, to_weekly
from .news import append_candidates_to_sentiment, refresh_news_candidates, write_news_candidates
from .optimizer import evaluate_grid, walk_forward_optimize
from .sentiment import load_sentiment_scores
from .signals import build_signal_frame


def main() -> None:
    parser = argparse.ArgumentParser(prog="money-strategy")
    subparsers = parser.add_subparsers(dest="command")
    default_end = date.today().isoformat()

    backtest = subparsers.add_parser("backtest", help="Run ETF rotation backtest")
    backtest.add_argument("--start", default="2018-01-01")
    backtest.add_argument("--end", default=default_end)
    backtest.add_argument("--cache-dir", default="data/cache")
    backtest.add_argument("--output-dir", default="output")
    backtest.add_argument("--refresh", action="store_true", help="Refresh cached Eastmoney data")
    backtest.add_argument("--strict-data", action="store_true", help="Fail if any instrument cannot be loaded")
    backtest.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    backtest.add_argument(
        "--switch-threshold",
        type=float,
        default=0.15,
        help="Same-sleeve switch threshold on factor score scale",
    )
    backtest.add_argument("--offense-threshold", type=float, default=60.0)
    backtest.add_argument("--defense-threshold", type=float, default=35.0)
    backtest.add_argument("--balanced-defensive-threshold", type=float, default=45.0)
    backtest.add_argument("--confirm-weeks", type=int, default=1)
    backtest.add_argument("--news-weight", type=float, default=0.03)
    backtest.add_argument("--policy-weight", type=float, default=0.05)
    backtest.add_argument("--theme-weight", type=float, default=0.06)
    backtest.add_argument(
        "--sentiment-file",
        help="Optional CSV with date,news_score,policy_score columns. Scores are 0-100, 50 neutral.",
    )
    backtest.add_argument("--refresh-news", action="store_true", help="Fetch latest official policy/news candidates")
    backtest.add_argument("--apply-news", action="store_true", help="Append fetched candidates to sentiment file")
    backtest.add_argument("--news-days", type=int, default=30)

    signal = subparsers.add_parser("signal", help="Print latest stable single-position signal")
    signal.add_argument("--start", default="2018-01-01")
    signal.add_argument("--end", default=default_end)
    signal.add_argument("--cache-dir", default="data/cache")
    signal.add_argument("--output-dir", default="output/stable_signal")
    signal.add_argument("--refresh", action="store_true", help="Refresh cached Eastmoney data")
    signal.add_argument("--strict-data", action="store_true", help="Fail if any instrument cannot be loaded")
    signal.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    signal.add_argument("--switch-threshold", type=float, default=0.15)
    signal.add_argument("--offense-threshold", type=float, default=60.0)
    signal.add_argument("--defense-threshold", type=float, default=35.0)
    signal.add_argument("--balanced-defensive-threshold", type=float, default=45.0)
    signal.add_argument("--confirm-weeks", type=int, default=1)
    signal.add_argument("--news-weight", type=float, default=0.03)
    signal.add_argument("--policy-weight", type=float, default=0.05)
    signal.add_argument("--theme-weight", type=float, default=0.06)
    signal.add_argument(
        "--sentiment-file",
        default="data/policy_events.real.csv",
        help="CSV with date,news_score,policy_score and optional theme scores.",
    )
    signal.add_argument("--refresh-news", action="store_true", help="Fetch latest official policy/news candidates")
    signal.add_argument("--apply-news", action="store_true", help="Append fetched candidates to sentiment file")
    signal.add_argument("--news-days", type=int, default=30)

    optimize = subparsers.add_parser("optimize", help="Run parameter grid and walk-forward optimization")
    optimize.add_argument("--start", default="2018-01-01")
    optimize.add_argument("--end", default=default_end)
    optimize.add_argument("--cache-dir", default="data/cache")
    optimize.add_argument("--output-dir", default="output/optimize")
    optimize.add_argument("--refresh", action="store_true", help="Refresh cached market data")
    optimize.add_argument("--strict-data", action="store_true", help="Fail if any instrument cannot be loaded")
    optimize.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    optimize.add_argument("--train-weeks", type=int, default=52)
    optimize.add_argument("--test-weeks", type=int, default=13)

    news = subparsers.add_parser("refresh-news", help="Fetch official policy/news candidates")
    news.add_argument("--sentiment-file", default="data/policy_events.real.csv")
    news.add_argument("--candidates-file", default="data/news_candidates.csv")
    news.add_argument("--days", type=int, default=30)
    news.add_argument("--apply", action="store_true", help="Append candidates to sentiment file")

    args = parser.parse_args()
    if args.command not in {"backtest", "optimize", "signal", "refresh-news"}:
        parser.print_help()
        return

    if args.command == "refresh-news":
        candidates = refresh_news_candidates(days=args.days)
        write_news_candidates(candidates, Path(args.candidates_file))
        print(f"Fetched {len(candidates)} candidates -> {Path(args.candidates_file).resolve()}")
        if args.apply:
            combined = append_candidates_to_sentiment(candidates, Path(args.sentiment_file))
            print(f"Updated {Path(args.sentiment_file).resolve()} rows={len(combined)}")
        if not candidates.empty:
            print(candidates[["date", "policy_score", "news_score", "title", "source_url"]].head(20).to_string(index=False))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = load_universe(args.start, args.end, Path(args.cache_dir), refresh=args.refresh, strict=args.strict_data)
    daily_close = make_close_panel(daily)
    daily_amount = make_amount_panel(daily)
    weekly_close, weekly_amount = to_weekly(daily_close, daily_amount)

    if args.command == "optimize":
        grid, _ = evaluate_grid(
            daily_close,
            daily_amount,
            weekly_close,
            weekly_amount,
            cost_bps=args.cost_bps,
        )
        walk = walk_forward_optimize(
            daily_close,
            daily_amount,
            weekly_close,
            weekly_amount,
            train_weeks=args.train_weeks,
            test_weeks=args.test_weeks,
            cost_bps=args.cost_bps,
        )
        grid.to_csv(output_dir / "grid_results.csv", index=False)
        walk.to_csv(output_dir / "walk_forward.csv", index=False)
        print("Top grid results:")
        print(grid.head(10).round(4).to_string(index=False))
        if not walk.empty:
            print("\nWalk-forward test summary:")
            print(walk.filter(regex="^(fold|test_start|test_end|param_|test_annual_return|test_max_drawdown|test_sharpe|test_score)").round(4).to_string(index=False))
        print(f"\nWrote optimization outputs to {output_dir.resolve()}")
        return

    if getattr(args, "refresh_news", False):
        sentiment_file = Path(args.sentiment_file) if getattr(args, "sentiment_file", None) else Path("data/policy_events.real.csv")
        candidates = refresh_news_candidates(days=args.news_days)
        candidates_file = output_dir / "news_candidates.csv"
        write_news_candidates(candidates, candidates_file)
        if getattr(args, "apply_news", False):
            append_candidates_to_sentiment(candidates, sentiment_file)
        print(f"Fetched {len(candidates)} official news candidates -> {candidates_file.resolve()}")

    news_sentiment = (
        load_sentiment_scores(Path(args.sentiment_file), weekly_close.index)
        if getattr(args, "sentiment_file", None)
        else None
    )
    signals = build_signal_frame(
        daily_close,
        daily_amount,
        weekly_close,
        weekly_amount,
        news_sentiment,
        switch_threshold=args.switch_threshold,
        offense_threshold=args.offense_threshold,
        defense_threshold=args.defense_threshold,
        balanced_defensive_threshold=args.balanced_defensive_threshold,
        confirm_weeks=args.confirm_weeks,
        news_weight=args.news_weight,
        policy_weight=args.policy_weight,
        theme_weight=args.theme_weight,
    )
    result = run_backtest(weekly_close, signals, cost_bps=args.cost_bps)

    if args.command == "signal":
        payload = _latest_signal_payload(result.signals, result.performance)
        (output_dir / "latest_signal.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nWrote latest signal to {(output_dir / 'latest_signal.json').resolve()}")
        return

    result.signals.to_csv(output_dir / "weekly_signals.csv", index_label="date")
    result.equity_curve.to_csv(output_dir / "equity_curve.csv", index_label="date")
    result.performance.to_csv(output_dir / "performance.csv")
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    _plot_equity(result.equity_curve, output_dir / "equity_curve.png")

    print(result.performance.round(4).to_string())
    print(f"\nWrote outputs to {output_dir.resolve()}")


def _latest_signal_payload(signals, performance) -> dict[str, object]:
    names = {instrument.code: instrument.name for instrument in ALL_INSTRUMENTS}
    latest = signals.dropna(subset=["market_score"]).tail(1).iloc[0]
    weight_cols = [column for column in signals.columns if column.startswith("weight_")]
    weights = {
        column.replace("weight_", ""): _as_float(latest[column])
        for column in weight_cols
        if _as_float(latest[column]) > 0
    }
    target_code = max(weights, key=weights.get)

    return {
        "date": str(latest.name.date()),
        "bucket": str(latest["bucket"]),
        "target": {
            "code": target_code,
            "name": names.get(target_code, target_code),
            "weight": weights[target_code],
        },
        "weights": {
            code: {
                "name": names.get(code, code),
                "weight": weight,
            }
            for code, weight in weights.items()
        },
        "scores": {
            "market": _as_float(latest["market_score"]),
            "technical": _as_float(latest["technical_score"]),
            "trend": _as_float(latest["trend_score"]),
            "growth": _as_float(latest["growth_score"]),
            "support": _as_float(latest["support_score"]),
            "breadth": _as_float(latest["breadth_score"]),
            "bond": _as_float(latest["bond_score"]),
            "news": _as_float(latest["news_score"]),
            "policy": _as_float(latest["policy_score"]),
        },
        "choices": {
            "rotation": {
                "code": str(latest["rotation_choice"]),
                "name": names.get(str(latest["rotation_choice"]), str(latest["rotation_choice"])),
            },
            "defensive_first": {
                "code": str(latest["defensive_first"]),
                "name": names.get(str(latest["defensive_first"]), str(latest["defensive_first"])),
            },
        },
        "performance": {
            "annual_return": _as_float(performance.loc["strategy", "annual_return"]),
            "max_drawdown": _as_float(performance.loc["strategy", "max_drawdown"]),
            "sharpe": _as_float(performance.loc["strategy", "sharpe"]),
            "annual_turnover": _as_float(performance.loc["strategy", "annual_turnover"]),
        },
    }


def _as_float(value) -> float:
    return round(float(value), 6)


def _plot_equity(equity, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    equity[["strategy", "benchmark_hs300", "defensive_static"]].plot(ax=ax)
    ax.set_title("ETF Rotation Backtest")
    ax.set_ylabel("Net Value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
