from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BOND_SIGNAL, DEFENSIVE, GROWTH_SIGNALS, MARKET, ROTATION, SUPPORT_SIGNALS, WATCHLIST


def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        last = values[-1]
        valid = values[~np.isnan(values)]
        if len(valid) < 2 or np.isnan(last):
            return np.nan
        return float((valid <= last).sum() / len(valid))

    return series.rolling(window, min_periods=max(5, window // 4)).apply(rank_last, raw=True)


def _weekly_last(series: pd.Series, weekly_index: pd.DatetimeIndex) -> pd.Series:
    values = series.resample("W-FRI").last()
    actual_dates = series.index.to_series().resample("W-FRI").max()
    valid = values.notna() & actual_dates.notna()
    values = values.loc[valid]
    values.index = pd.DatetimeIndex(actual_dates.loc[valid].to_numpy(), name="date")
    return values.reindex(weekly_index)


def risk_adjusted_momentum(weekly_close: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    returns = weekly_close.pct_change()
    momentum = weekly_close.pct_change(lookback)
    volatility = returns.rolling(lookback, min_periods=max(4, lookback // 2)).std()
    return momentum / volatility.replace(0, np.nan)


def rotation_factor_score(
    weekly_close: pd.DataFrame,
    weekly_amount: pd.DataFrame,
    codes: list[str],
) -> pd.DataFrame:
    ret8 = weekly_close[codes].pct_change(8)
    ret4 = weekly_close[codes].pct_change(4)
    vol12 = weekly_close[codes].pct_change().rolling(12, min_periods=6).std()
    drawdown12 = weekly_close[codes] / weekly_close[codes].rolling(12, min_periods=6).max() - 1.0
    amount_ratio = weekly_amount[codes] / weekly_amount[codes].rolling(12, min_periods=6).mean()

    return (
        ret4.rank(axis=1, pct=True) * 0.30
        + (-vol12).rank(axis=1, pct=True) * 0.25
        + drawdown12.rank(axis=1, pct=True) * 0.20
        + ret8.rank(axis=1, pct=True) * 0.15
        + amount_ratio.rank(axis=1, pct=True) * 0.10
    )


def defensive_factor_score(
    weekly_close: pd.DataFrame,
    codes: list[str],
) -> pd.DataFrame:
    ret12 = weekly_close[codes].pct_change(12)
    vol12 = weekly_close[codes].pct_change().rolling(12, min_periods=6).std()
    drawdown12 = weekly_close[codes] / weekly_close[codes].rolling(12, min_periods=6).max() - 1.0

    return (
        ret12.rank(axis=1, pct=True) * 0.45
        + (-vol12).rank(axis=1, pct=True) * 0.30
        + drawdown12.rank(axis=1, pct=True) * 0.25
    )


def growth_expansion_score(weekly_close: pd.DataFrame, growth_codes: list[str], market: str) -> pd.Series:
    if not growth_codes:
        return pd.Series(50.0, index=weekly_close.index)

    growth_close = weekly_close[growth_codes]
    market_ret12 = weekly_close[market].pct_change(12)
    growth_ret12 = growth_close.pct_change(12).mean(axis=1)
    growth_breadth = (growth_close > growth_close.rolling(20, min_periods=10).mean()).mean(axis=1)
    growth_vs_market = growth_ret12 - market_ret12

    score = pd.Series(50.0, index=weekly_close.index)
    score = score.mask((growth_vs_market > 0.04) & (growth_breadth >= 0.50), 72.0)
    score = score.mask((growth_vs_market < -0.04) | (growth_breadth <= 0.25), 35.0)
    return score.fillna(50.0)


def support_confirmation_score(
    weekly_close: pd.DataFrame,
    weekly_amount: pd.DataFrame,
    support_codes: list[str],
    defensive_codes: list[str],
    market: str,
) -> pd.Series:
    if not support_codes:
        return pd.Series(50.0, index=weekly_close.index)

    market_week_ret = weekly_close[market].pct_change()
    market_ret4 = weekly_close[market].pct_change(4)
    support_ret = weekly_close[support_codes].pct_change().mean(axis=1)
    support_amount = weekly_amount[support_codes].sum(axis=1)
    support_amount_ratio = support_amount / support_amount.rolling(20, min_periods=8).mean()

    defensive_ret4 = (
        weekly_close[defensive_codes].pct_change(4).mean(axis=1)
        if defensive_codes
        else pd.Series(0.0, index=weekly_close.index)
    )
    defensive_relative = defensive_ret4 - market_ret4

    weak_market = (market_week_ret < 0) | (market_ret4 < 0)
    broad_absorption = (support_amount_ratio > 1.35) & (support_ret >= market_week_ret)
    defensive_bid = defensive_relative > 0.02

    score = pd.Series(50.0, index=weekly_close.index)
    score = score.mask(weak_market & broad_absorption & defensive_bid, 68.0)
    score = score.mask(weak_market & broad_absorption & ~defensive_bid, 60.0)
    score = score.mask(weak_market & (support_amount_ratio < 0.75) & (support_ret < market_week_ret), 42.0)
    return score.fillna(50.0)


def _safe_idxmax(frame: pd.DataFrame, fallback: str) -> pd.Series:
    filled = frame.copy()
    filled[fallback] = filled[fallback].fillna(-np.inf)
    result = filled.idxmax(axis=1)
    all_missing = frame.isna().all(axis=1)
    return result.mask(all_missing, fallback)


def _ranked_codes(row: pd.Series, fallbacks: list[str]) -> list[str]:
    ranked = [str(code) for code in row.dropna().sort_values(ascending=False).index]
    for code in fallbacks:
        if code not in ranked:
            ranked.append(code)
    return ranked


def build_signal_frame(
    daily_close: pd.DataFrame,
    daily_amount: pd.DataFrame,
    weekly_close: pd.DataFrame,
    weekly_amount: pd.DataFrame,
    news_sentiment: pd.Series | pd.DataFrame | None = None,
    switch_threshold: float = 0.15,
    offense_threshold: float = 60.0,
    defense_threshold: float = 35.0,
    balanced_defensive_threshold: float = 45.0,
    confirm_weeks: int = 1,
    news_weight: float = 0.03,
    policy_weight: float = 0.05,
    theme_weight: float = 0.06,
) -> pd.DataFrame:
    market = MARKET.code
    bond = BOND_SIGNAL.code
    rotation_codes = [item.code for item in ROTATION if item.code in weekly_close.columns]
    defensive_codes = [item.code for item in DEFENSIVE if item.code in weekly_close.columns]
    growth_codes = [item.code for item in GROWTH_SIGNALS if item.code in weekly_close.columns]
    support_codes = [item.code for item in SUPPORT_SIGNALS if item.code in weekly_close.columns]
    if market not in weekly_close.columns:
        raise ValueError(f"missing market data: {market}")
    if bond not in weekly_close.columns:
        raise ValueError(f"missing bond signal data: {bond}")
    if not rotation_codes:
        raise ValueError("no rotation instruments available")
    if len(defensive_codes) < 2:
        raise ValueError("at least two defensive instruments are required")

    market_close = weekly_close[market]
    market_ma20 = market_close.rolling(20, min_periods=10).mean()
    market_ret12 = market_close.pct_change(12)
    market_above_ma = market_close > market_ma20
    trend_raw = market_above_ma & (market_ret12 > 0)
    trend_confirmed = trend_raw.rolling(2, min_periods=2).sum().eq(2)
    clear_breakdown = market_close < market_ma20 * 0.98

    daily_market_ret = daily_close[market].pct_change()
    vol20 = daily_market_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    vol_rank_daily = _percentile_rank(vol20, 504)
    vol_rank = _weekly_last(vol_rank_daily, weekly_close.index)
    high_vol = vol_rank >= 0.80

    market_week_ret = weekly_close[market].pct_change()
    amount_ratio = weekly_amount[market] / weekly_amount[market].rolling(20, min_periods=8).mean()
    heat_score = pd.Series(50.0, index=weekly_close.index)
    heat_score = heat_score.mask((amount_ratio > 1.2) & (market_week_ret > 0), 70.0)
    heat_score = heat_score.mask((amount_ratio > 1.2) & (market_week_ret < 0), 30.0)

    watch_codes = [item.code for item in WATCHLIST if item.code in weekly_close.columns]
    rotation_scores = rotation_factor_score(weekly_close, weekly_amount, rotation_codes)
    rotation_scores = apply_theme_sentiment(rotation_scores, news_sentiment, weekly_close.index, weight=theme_weight)
    relative_scores = (
        rotation_factor_score(weekly_close, weekly_amount, [*rotation_codes, *watch_codes])
        if watch_codes
        else pd.DataFrame(index=weekly_close.index)
    )
    rotation_choice = _safe_idxmax(rotation_scores, rotation_codes[0])
    rotation_vs_market = weekly_close[rotation_codes].pct_change(12).max(axis=1) - market_ret12
    offensive_score = pd.Series(50.0, index=weekly_close.index)
    offensive_score = offensive_score.mask(rotation_vs_market > 0.03, 70.0)
    offensive_score = offensive_score.mask(rotation_vs_market < -0.03, 30.0)

    breadth = (weekly_close[rotation_codes] > weekly_close[rotation_codes].rolling(20, min_periods=10).mean()).mean(axis=1)
    breadth_score = pd.Series(50.0, index=weekly_close.index)
    breadth_score = breadth_score.mask(breadth >= 0.65, 70.0)
    breadth_score = breadth_score.mask(breadth <= 0.35, 30.0)
    growth_score = growth_expansion_score(weekly_close, growth_codes, market)
    support_score = support_confirmation_score(weekly_close, weekly_amount, support_codes, defensive_codes, market)

    defensive_scores = defensive_factor_score(weekly_close, defensive_codes)
    defensive_ranked = defensive_scores.apply(lambda row: _ranked_codes(row, defensive_codes), axis=1)
    defensive_first = defensive_ranked.map(lambda codes: codes[0])
    defensive_second = defensive_ranked.map(lambda codes: codes[1])

    bond_ret12 = weekly_close[bond].pct_change(12)
    bond_vs_market = bond_ret12 - market_ret12
    bond_score = pd.Series(50.0, index=weekly_close.index)
    bond_score = bond_score.mask(bond_vs_market > 0.05, 30.0)
    bond_score = bond_score.mask(bond_vs_market < -0.05, 60.0)

    trend_score = pd.Series(50.0, index=weekly_close.index)
    trend_score = trend_score.mask(trend_confirmed, 80.0)
    trend_score = trend_score.mask(clear_breakdown, 20.0)
    trend_score = trend_score.mask(~trend_confirmed & ~clear_breakdown, 45.0)

    volatility_score = pd.Series(50.0, index=weekly_close.index)
    volatility_score = volatility_score.mask(vol_rank < 0.50, 65.0)
    volatility_score = volatility_score.mask((vol_rank >= 0.50) & (vol_rank < 0.80), 50.0)
    volatility_score = volatility_score.mask(vol_rank >= 0.80, 25.0)

    technical_score = (
        trend_score * 0.30
        + volatility_score * 0.15
        + heat_score * 0.10
        + offensive_score * 0.20
        + bond_score * 0.10
        + breadth_score * 0.15
    )
    news_score = pd.Series(50.0, index=weekly_close.index)
    policy_score = pd.Series(50.0, index=weekly_close.index)
    if news_sentiment is not None:
        if isinstance(news_sentiment, pd.DataFrame):
            if "news_score" in news_sentiment.columns:
                news_score = news_sentiment["news_score"].reindex(weekly_close.index).ffill().fillna(50.0).clip(0, 100)
            if "policy_score" in news_sentiment.columns:
                policy_score = news_sentiment["policy_score"].reindex(weekly_close.index).ffill().fillna(50.0).clip(0, 100)
        else:
            news_score = news_sentiment.reindex(weekly_close.index).ffill().fillna(50.0).clip(0, 100)

    technical_weight = max(0.0, 1.0 - news_weight - policy_weight)
    market_score = technical_score * technical_weight + news_score * news_weight + policy_score * policy_weight

    raw_bucket = pd.Series("balanced", index=weekly_close.index, dtype="object")
    raw_bucket = raw_bucket.mask(market_score >= offense_threshold, "offense")
    raw_bucket = raw_bucket.mask(market_score < defense_threshold, "defense")
    raw_bucket = raw_bucket.mask(high_vol & ~trend_confirmed & (market_score < 45), "defense")

    bucket = _smooth_bucket(raw_bucket, confirm_weeks=confirm_weeks)

    frame = pd.DataFrame(
        {
            "market_close": market_close,
            "market_ma20": market_ma20,
            "market_ret12": market_ret12,
            "vol20_rank": vol_rank,
            "amount_ratio": amount_ratio,
            "trend_score": trend_score,
            "volatility_score": volatility_score,
            "heat_score": heat_score,
            "offensive_score": offensive_score,
            "breadth": breadth,
            "breadth_score": breadth_score,
            "growth_score": growth_score,
            "support_score": support_score,
            "bond_score": bond_score,
            "news_score": news_score,
            "policy_score": policy_score,
            "technical_score": technical_score,
            "market_score": market_score,
            "raw_bucket": raw_bucket,
            "bucket": bucket,
            "rotation_choice": rotation_choice,
            "defensive_first": defensive_first,
            "defensive_second": defensive_second,
        }
    )
    frame = frame.join(rotation_scores.add_prefix("rotation_score_"))
    if watch_codes:
        watch_scores = apply_theme_sentiment(relative_scores[watch_codes], news_sentiment, weekly_close.index, weight=theme_weight)
        frame = frame.join(watch_scores.add_prefix("watch_score_"))
    frame = frame.join(defensive_scores.add_prefix("defensive_score_"))
    return attach_target_weights(
        frame,
        switch_threshold=switch_threshold,
        balanced_defensive_threshold=balanced_defensive_threshold,
    )


def apply_theme_sentiment(
    factor_scores: pd.DataFrame,
    sentiment: pd.Series | pd.DataFrame | None,
    weekly_index: pd.DatetimeIndex,
    *,
    weight: float = 0.12,
) -> pd.DataFrame:
    if not isinstance(sentiment, pd.DataFrame):
        return factor_scores

    adjusted = factor_scores.copy()
    for code in factor_scores.columns:
        column = f"theme_score_{code}"
        if column not in sentiment.columns:
            continue
        theme_delta = (sentiment[column].reindex(weekly_index).ffill().fillna(50.0).clip(0, 100) - 50.0) / 100.0
        adjusted[code] = (factor_scores[code] + theme_delta * weight).clip(0.0, 1.0)
    return adjusted


def _smooth_bucket(raw_bucket: pd.Series, *, confirm_weeks: int = 2) -> pd.Series:
    order = {"defense": 0, "balanced": 1, "offense": 2}
    inverse = {value: key for key, value in order.items()}
    current = "balanced"
    pending: str | None = None
    pending_count = 0
    smoothed: list[str] = []

    for raw in raw_bucket.fillna("balanced"):
        if raw == current:
            pending = None
            pending_count = 0
        elif raw == pending:
            pending_count += 1
        else:
            pending = raw
            pending_count = 1

        if pending_count >= confirm_weeks and pending is not None:
            current_level = order[current]
            target_level = order[pending]
            if target_level > current_level:
                current = inverse[current_level + 1]
            elif target_level < current_level:
                current = inverse[current_level - 1]
            pending = None
            pending_count = 0

        smoothed.append(current)

    return pd.Series(smoothed, index=raw_bucket.index, dtype="object")


def attach_target_weights(
    signals: pd.DataFrame,
    *,
    switch_threshold: float = 0.10,
    balanced_defensive_threshold: float = 50.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    rotation_codes = [item.code for item in ROTATION]
    defensive_codes = [item.code for item in DEFENSIVE]
    all_asset_codes = list(dict.fromkeys([*rotation_codes, *defensive_codes]))

    previous_single: str | None = None

    for date, row in signals.iterrows():
        rotation_choice = row.get("rotation_choice", row.get("offensive_choice"))
        if rotation_choice not in rotation_codes:
            rotation_choice = rotation_codes[0]

        defensive_first = row.get("defensive_first", defensive_codes[0])
        defensive_second = row.get("defensive_second", defensive_codes[1])
        if defensive_first not in defensive_codes:
            defensive_first = defensive_codes[0]
        if defensive_second not in defensive_codes or defensive_second == defensive_first:
            defensive_second = next(code for code in defensive_codes if code != defensive_first)

        weights = {code: 0.0 for code in all_asset_codes}
        if row["bucket"] == "defense" or (
            row["bucket"] == "balanced" and float(row["market_score"]) < balanced_defensive_threshold
        ):
            desired = str(defensive_first)
            desired_sleeve = "defensive"
        else:
            desired = str(rotation_choice)
            desired_sleeve = "rotation"

        chosen = _apply_single_holding_hysteresis(
            row,
            desired,
            desired_sleeve,
            previous_single,
            defensive_codes,
            rotation_codes,
            switch_threshold,
        )
        weights[chosen] = 1.0
        previous_single = chosen
        weights["date"] = date
        rows.append(weights)

    weight_frame = pd.DataFrame(rows).set_index("date")
    return signals.join(weight_frame.add_prefix("weight_"))


def _apply_single_holding_hysteresis(
    row: pd.Series,
    desired: str,
    desired_sleeve: str,
    previous: str | None,
    defensive_codes: list[str],
    rotation_codes: list[str],
    switch_threshold: float,
) -> str:
    if previous is None or previous == desired:
        return desired

    previous_sleeve = "defensive" if previous in defensive_codes else "rotation"
    if previous_sleeve != desired_sleeve:
        return desired

    prefix = "defensive_score_" if desired_sleeve == "defensive" else "rotation_score_"
    desired_score = float(row.get(f"{prefix}{desired}", 0.0) or 0.0)
    previous_score = float(row.get(f"{prefix}{previous}", 0.0) or 0.0)
    if desired_score >= previous_score + switch_threshold:
        return desired
    return previous
