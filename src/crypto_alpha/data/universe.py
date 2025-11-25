"""Helpers to build point-in-time universes for backtests."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

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


def _pit_csv_selection(as_of: pd.Timestamp, config: dict) -> UniverseSelection:
    universe_dir = config.get("path")
    if not universe_dir:
        raise ValueError("Universe config missing 'path' for pit_csv mode.")
    files = _list_universe_files(Path(universe_dir))
    target: tuple[pd.Timestamp, Path] | None = None
    for file_date, file_path in files:
        if file_date <= as_of:
            target = (file_date, file_path)
        else:
            break
    if target is None:
        raise ValueError(f"No universe CSV available on/before {as_of.date()} in {universe_dir}.")
    target_date, target_path = target
    df = pd.read_csv(target_path)
    if df.empty:
        return UniverseSelection(as_of=target_date, symbols=[])
    if "rank" in df.columns:
        df = df.sort_values("rank")
    symbols = df["symbol"].astype(str).tolist()
    top_n = config.get("top_n")
    if top_n is not None:
        symbols = symbols[: int(top_n)]
    symbols = _apply_exclusions(symbols, config)
    return UniverseSelection(as_of=target_date, symbols=symbols)


@lru_cache(maxsize=8)
def _list_universe_files(directory: Path) -> Sequence[tuple[pd.Timestamp, Path]]:
    if not directory.exists():
        raise FileNotFoundError(f"Universe directory not found: {directory}")
    files: List[tuple[pd.Timestamp, Path]] = []
    for path in sorted(directory.glob("universe_*.csv")):
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 2:
            continue
        date_str = stem_parts[1]
        try:
            file_date = pd.Timestamp(date_str)
        except ValueError:
            continue
        files.append((file_date, path))
    if not files:
        raise FileNotFoundError(f"No universe CSV files found in {directory}")
    return files


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

    mode = config.get("type", "liquidity")
    if mode == "pit_csv":
        return _pit_csv_selection(as_of, config)

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
