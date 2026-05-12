from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    code: str
    name: str
    secid: str


MARKET = Instrument("000300", "沪深300", "1.000300")
BOND_SIGNAL = Instrument("511010", "国债ETF", "1.511010")

GROWTH_SIGNALS = (
    Instrument("588000", "科创50ETF", "1.588000"),
    Instrument("159949", "创业板50ETF", "0.159949"),
)

SUPPORT_SIGNALS = (
    Instrument("510300", "沪深300ETF", "1.510300"),
    Instrument("510050", "上证50ETF", "1.510050"),
    Instrument("510500", "中证500ETF", "1.510500"),
    Instrument("159915", "创业板ETF", "0.159915"),
)

ROTATION = (
    Instrument("159819", "人工智能ETF", "0.159819"),
    Instrument("512480", "半导体ETF", "1.512480"),
    Instrument("159770", "机器人ETF", "0.159770"),
    Instrument("516160", "新能源ETF", "1.516160"),
    Instrument("561560", "电力ETF", "1.561560"),
)

WATCHLIST = (
    Instrument("563530", "卫星ETF", "1.563530"),
)

DEFENSIVE = (
    Instrument("515100", "红利低波100ETF", "1.515100"),
    Instrument("512800", "银行ETF", "1.512800"),
)

ALL_INSTRUMENTS = tuple(
    {
        instrument.code: instrument
        for instrument in (MARKET, BOND_SIGNAL, *GROWTH_SIGNALS, *ROTATION, *DEFENSIVE, *WATCHLIST)
        + SUPPORT_SIGNALS
    }.values()
)

DEFAULT_COST_BPS = 10.0
