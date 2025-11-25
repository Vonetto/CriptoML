"""Helpers to build point-in-time universes for backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


STABLECOIN_TICKERS = ("USDT", "USDC", "BUSD", "DAI", "USDP", "TUSD")


@dataclass
class UniverseSelection:
    """Result of a universe selection step."""

    as_of: pd.Timestamp
    symbols: List[str]


def _dollar_volume(df: pd.DataFrame) -> pd.Series:
    if "volume_usd" in df.columns:
        return df["volume_usd"]
    if "turnover" in df.columns:
        return df["turnover"]
    return df["close"] * df.get("volume_quote", df["volume"])


def _apply_exclusions(symbols: Iterable[str], config: dict) -> List[str]:
    excludes = config.get("exclude", {})
    result: List[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        if excludes.get("stablecoins"):
            if any(upper.endswith(s) for s in STABLECOIN_TICKERS):
                continue
        if excludes.get("wrapped_assets") and upper.startswith("W"):
            continue
        result.append(symbol)
    return result


def select_universe(df: pd.DataFrame, as_of: pd.Timestamp, config: dict) -> UniverseSelection:
    """Select top-N symbols by liquidity at a given date.

    Parameters
    ----------
    df:
        DataFrame with at least ``date``, ``symbol``, ``close`` and ``volume``.
    as_of:
        Date used for the point-in-time selection (inclusive).
    config:
        Section ``strategy.universe`` from the YAML config.
    """

    liquidity_lookback = int(config.get("liquidity_lookback_days", 30))
    window_start = as_of - pd.Timedelta(days=liquidity_lookback)
    window = df[(df["date"] > window_start) & (df["date"] <= as_of)].copy()
    if window.empty:
        return UniverseSelection(as_of=as_of, symbols=[])

    window["dollar_volume"] = _dollar_volume(window)
    grouped = window.groupby("symbol")["dollar_volume"].mean()
    liquidity = grouped.sort_values(ascending=False)

    min_volume = config.get("min_volume_usd")
    if min_volume is not None:
        liquidity = liquidity[liquidity >= float(min_volume)]

    ordered_symbols = list(liquidity.index)
    ordered_symbols = _apply_exclusions(ordered_symbols, config)
    top_n = int(config.get("top_n", len(ordered_symbols)))
    symbols = ordered_symbols[:top_n]
    return UniverseSelection(as_of=as_of, symbols=symbols)
