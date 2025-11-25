"""ETL helpers to source Binance Futures market data."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from tqdm import tqdm

from crypto_alpha.data.exchanges import BinanceFuturesClient


def _ensure_iterable(symbols: Iterable[str]) -> List[str]:
    return [s.upper() for s in symbols]


def _default_output(path: Path | str, name: str) -> Path:
    base = Path(path)
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def download_ohlcv(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    interval: str = "1d",
    output_dir: Path | str = "data/raw/binance_futures/ohlcv",
    output_file: Path | str | None = None,
    client: BinanceFuturesClient | None = None,
) -> pd.DataFrame:
    """Download OHLCV candles for a list of symbols."""

    own_client = client is None
    client = client or BinanceFuturesClient()
    symbols = _ensure_iterable(symbols)
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []

    for symbol in tqdm(symbols, desc="Binance OHLCV"):
        df = client.fetch_klines(symbol, interval=interval, start=start, end=end)
        if df.empty:
            continue
        frames.append(df)
        df.to_parquet(base_dir / f"{symbol}_{interval}.parquet", index=False)

    if own_client:
        client.close()

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    outfile = output_file or _default_output(base_dir.parent, f"ohlcv_{interval}.parquet")
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(outfile, index=False)
    return dataset


def download_open_interest(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    period: str = "1d",
    output_dir: Path | str = "data/raw/binance_futures/open_interest",
    output_file: Path | str | None = None,
    client: BinanceFuturesClient | None = None,
) -> pd.DataFrame:
    """Download open-interest history for Binance USDⓈ-M perps."""

    own_client = client is None
    client = client or BinanceFuturesClient()
    symbols = _ensure_iterable(symbols)
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []

    for symbol in tqdm(symbols, desc="Binance OI"):
        df = client.fetch_open_interest_history(symbol, start=start, end=end, period=period)
        if df.empty:
            continue
        frames.append(df)
        df.to_parquet(base_dir / f"{symbol}_{period}.parquet", index=False)

    if own_client:
        client.close()

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    outfile = output_file or _default_output(base_dir.parent, f"open_interest_{period}.parquet")
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(outfile, index=False)
    return dataset


__all__ = ["download_ohlcv", "download_open_interest"]
