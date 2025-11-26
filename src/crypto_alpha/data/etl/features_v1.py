"""Feature engineering pipeline for the V1 cross-sectional model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from crypto_alpha.utils import dates as date_utils


logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "ret_30d",
    "vol_7d",
    "vol_30d",
    "sma_gap_10d",
    "sma_gap_30d",
    "sma_gap_60d",
    "rsi_14",
    "volume_ratio_7d",
    "volume_ratio_30d",
    "avg_volume_quote_30d",
    "avg_num_trades_30d",
    "volume_persistence_30d",
]


@dataclass
class PITUniverseEntry:
    date: pd.Timestamp
    symbols: List[str]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _load_universe(universe_dir: Path) -> Sequence[PITUniverseEntry]:
    entries: List[PITUniverseEntry] = []
    for path in sorted(universe_dir.glob("universe_*.csv")):
        stem = path.stem.split("_")
        if len(stem) < 2:
            continue
        try:
            as_of = pd.Timestamp(stem[1]).tz_localize(None)
        except ValueError:
            continue
        df = pd.read_csv(path)
        if df.empty or "symbol" not in df.columns:
            continue
        symbols = df["symbol"].astype(str).str.upper().tolist()
        entries.append(PITUniverseEntry(date=as_of.normalize(), symbols=symbols))
    if not entries:
        raise FileNotFoundError(f"No universe files found under {universe_dir}")
    return entries


def _filter_by_universe(
    df: pd.DataFrame,
    entries: Sequence[PITUniverseEntry],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    anchors = [entry.date for entry in entries]
    symbols = [set(entry.symbols) for entry in entries]
    selection = []
    for idx, anchor in enumerate(anchors):
        if anchor > end:
            break
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else end + pd.Timedelta(days=1)
        window_start = max(anchor, start)
        mask = (df["date"] >= window_start) & (df["date"] < next_anchor)
        if not mask.any():
            continue
        allowed = df.loc[mask & df["symbol"].isin(symbols[idx])].copy()
        if allowed.empty:
            continue
        allowed["universe_as_of"] = anchor
        selection.append(allowed)
    if not selection:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(selection, ignore_index=True)


def _compute_features(
    df: pd.DataFrame,
    active_vol_threshold: float,
) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = df.groupby("symbol", group_keys=False)

    df["log_close"] = np.log(df["close"])
    for window in (1, 3, 7, 30):
        df[f"ret_{window}d"] = grouped["log_close"].diff(window)

    daily_log_ret = grouped["log_close"].diff()
    for window in (7, 30):
        df[f"vol_{window}d"] = daily_log_ret.rolling(window, min_periods=window).std()

    for window in (10, 30, 60):
        sma = grouped["close"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        df[f"sma_gap_{window}d"] = df["close"] / sma - 1

    rsi = grouped["close"].transform(lambda s: _rsi(s, window=14))
    df["rsi_14"] = rsi

    vol_ma7 = grouped["volume_quote"].transform(
        lambda s: s.rolling(7, min_periods=7).mean()
    )
    vol_ma30 = grouped["volume_quote"].transform(
        lambda s: s.rolling(30, min_periods=30).mean()
    )
    df["volume_ratio_7d"] = df["volume_quote"] / vol_ma7
    df["volume_ratio_30d"] = df["volume_quote"] / vol_ma30
    df["avg_volume_quote_30d"] = vol_ma30

    if "num_trades" in df.columns:
        df["avg_num_trades_30d"] = grouped["num_trades"].transform(
            lambda s: s.rolling(30, min_periods=30).mean()
        )
    else:
        df["avg_num_trades_30d"] = np.nan

    persistence = grouped["volume_quote"].transform(
        lambda s: s.rolling(30, min_periods=30).apply(
            lambda x: np.count_nonzero(x >= active_vol_threshold), raw=True
        )
    )
    df["volume_persistence_30d"] = persistence

    forward = grouped["log_close"].shift(-5) - df["log_close"]
    df["forward_return_5d"] = forward
    return df


def build_features_v1(
    ohlcv_path: Path | str,
    universe_dir: Path | str,
    output_path: Path | str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rebalance_freq: str = "1W",
    active_vol_threshold: float = 5_000_000,
) -> Path:
    ohlcv_path = Path(ohlcv_path)
    universe_dir = Path(universe_dir)
    output_path = Path(output_path)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is not None:
        start = start.tz_convert(None)
    if end.tzinfo is not None:
        end = end.tz_convert(None)
    start = start.normalize()
    end = end.normalize()

    logger.info("Loading OHLCV from %s", ohlcv_path)
    df = pd.read_parquet(ohlcv_path)
    if df.empty:
        raise ValueError("OHLCV dataset is empty.")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        raise ValueError("No OHLCV data in requested interval.")

    logger.info("Computing technical features and liquidity proxies.")
    df = _compute_features(df, active_vol_threshold=active_vol_threshold)

    entries = _load_universe(universe_dir)
    logger.info("Applying PIT universe filter from %s", universe_dir)
    df = _filter_by_universe(df, entries, start=start, end=end)
    if df.empty:
        raise ValueError("Universe filtering removed all rows.")

    calendar = date_utils.normalize_calendar(df["date"])
    rebalance_dates = date_utils.generate_schedule(start, end, rebalance_freq, calendar)
    if not rebalance_dates:
        raise ValueError("No rebalance dates available for feature dataset.")
    df = df[df["date"].isin(rebalance_dates)].copy()

    df["target_excess_return_5d"] = df.groupby("date")["forward_return_5d"].transform(
        lambda s: s - s.median(skipna=True)
    )

    cols = ["date", "symbol", "universe_as_of", "forward_return_5d", "target_excess_return_5d"]
    cols.extend(FEATURE_COLUMNS)
    df = df[cols].dropna()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved V1 features to %s (%d rows, %d dates, %d symbols)",
        output_path,
        len(df),
        df["date"].nunique(),
        df["symbol"].nunique(),
    )
    return output_path


__all__ = ["build_features_v1", "FEATURE_COLUMNS"]
