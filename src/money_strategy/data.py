from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from .config import ALL_INSTRUMENTS, Instrument


EASTMONEY_KLINE_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
KLINE_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
]


class DataError(RuntimeError):
    pass


def fetch_eastmoney_daily(
    instrument: Instrument,
    start: str,
    end: str,
    *,
    timeout: int = 20,
) -> pd.DataFrame:
    params = {
        "secid": instrument.secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    response = _get_with_direct_fallback(EASTMONEY_KLINE_URL, params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    klines = (payload.get("data") or {}).get("klines") or []
    if not klines:
        raise DataError(f"No kline data returned for {instrument.code}: {json.dumps(payload)[:300]}")

    rows = [line.split(",") for line in klines]
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in KLINE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["close"]).sort_values("date").set_index("date")
    frame["code"] = instrument.code
    frame["name"] = instrument.name
    return frame


def fetch_tencent_daily(
    instrument: Instrument,
    start: str,
    end: str,
    *,
    timeout: int = 20,
) -> pd.DataFrame:
    symbol = _tencent_symbol(instrument)
    params = {
        "param": f"{symbol},day,{start},{end},2000,qfq",
    }
    response = _get_with_direct_fallback(TENCENT_KLINE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    data = (payload.get("data") or {}).get(symbol) or {}
    rows = data.get("qfqday") or data.get("day") or []
    if not rows:
        raise DataError(f"No Tencent kline data returned for {instrument.code}: {str(payload)[:300]}")

    frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["open", "close", "high", "low", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("date").set_index("date")
    frame["amount"] = frame["close"] * frame["volume"] * 100
    frame["amplitude"] = (frame["high"] - frame["low"]) / frame["close"].shift(1) * 100
    frame["pct_chg"] = frame["close"].pct_change() * 100
    frame["change"] = frame["close"].diff()
    frame["turnover"] = np.nan
    frame["code"] = instrument.code
    frame["name"] = instrument.name
    return frame[KLINE_COLUMNS[1:] + ["code", "name"]]


def _tencent_symbol(instrument: Instrument) -> str:
    market, code = instrument.secid.split(".", maxsplit=1)
    prefix = "sh" if market == "1" else "sz"
    return f"{prefix}{code}"


def _get_with_direct_fallback(url: str, *, params: dict[str, str], timeout: int) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }
    try:
        return requests.get(url, params=params, timeout=timeout, headers=headers)
    except RequestException:
        session = requests.Session()
        session.trust_env = False
        return session.get(url, params=params, timeout=timeout, headers=headers)


def load_or_fetch_daily(
    instrument: Instrument,
    start: str,
    end: str,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{instrument.code}.csv"
    cached = _read_cached_frame(path, start, end) if path.exists() else None

    if cached is not None and not refresh:
        return cached

    try:
        try:
            frame = fetch_eastmoney_daily(instrument, start, end)
        except Exception:
            frame = fetch_tencent_daily(instrument, start, end)
        frame = _merge_with_cache(cached, frame)
        frame.reset_index().to_csv(path, index=False)
        return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    except Exception:
        if cached is not None:
            return cached
        raise


def _read_cached_frame(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _merge_with_cache(cached: pd.DataFrame | None, fetched: pd.DataFrame) -> pd.DataFrame:
    if cached is None or cached.empty:
        return fetched

    overlap = cached.index.intersection(fetched.index)
    if len(overlap) >= 20:
        ratio = (fetched.loc[overlap, "close"] / cached.loc[overlap, "close"]).replace([np.inf, -np.inf], np.nan)
        median_ratio = ratio.dropna().median()
        if not 0.98 <= median_ratio <= 1.02:
            return fetched if len(fetched) > len(cached) else cached

    combined = pd.concat([cached, fetched]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def load_universe(
    start: str,
    end: str,
    cache_dir: Path,
    *,
    refresh: bool = False,
    instruments: Iterable[Instrument] = ALL_INSTRUMENTS,
    strict: bool = False,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for instrument in instruments:
        try:
            data[instrument.code] = load_or_fetch_daily(instrument, start, end, cache_dir, refresh=refresh)
        except Exception as exc:
            if strict:
                raise
            failures.append(f"{instrument.code} {instrument.name}: {exc}")

    if failures:
        print("Skipped instruments with unavailable data:")
        for failure in failures:
            print(f"  - {failure}")
    return data


def make_close_panel(daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        {code: frame["close"].rename(code) for code, frame in daily.items()},
        axis=1,
    ).sort_index()


def make_amount_panel(daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        {code: frame["amount"].rename(code) for code, frame in daily.items()},
        axis=1,
    ).sort_index()


def to_weekly(close: pd.DataFrame, amount: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_close = close.resample("W-FRI").last()
    weekly_amount = amount.resample("W-FRI").sum()
    actual_dates = close.index.to_series().resample("W-FRI").max()
    valid = weekly_close.notna().any(axis=1) & actual_dates.notna()
    weekly_close = weekly_close.loc[valid]
    weekly_amount = weekly_amount.loc[valid]
    actual_index = pd.DatetimeIndex(actual_dates.loc[valid].to_numpy(), name="date")
    weekly_close.index = actual_index
    weekly_amount.index = actual_index
    return weekly_close, weekly_amount
