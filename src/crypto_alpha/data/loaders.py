"""Loader helpers for OHLCV datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Dataset missing required columns: " + ", ".join(sorted(missing))
        )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


def load_ohlcv(config: Dict[str, object], base_path: Path | None = None) -> pd.DataFrame:
    """Load OHLCV data according to a YAML config.

    The loader expects a Parquet/CSV file with at least the canonical columns in
    ``REQUIRED_COLUMNS`` plus optional ``turnover`` or ``volume_usd``. The
    function only normalizes timestamps; the heavy ETL (downloading, merging,
    handling missing exchanges) should happen offline before running the
    backtests.
    """

    base = base_path or Path.cwd()
    storage = config.get("storage", {})
    path = base / Path(storage.get("processed_path", ""))
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Populate it before running backtests."
        )

    if path.is_dir():
        raise ValueError(f"processed_path must be a file, got directory: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df = _ensure_columns(df)

    funding_path = storage.get("funding_path")
    if funding_path:
        fpath = base / Path(funding_path)
        if not fpath.exists():
            raise FileNotFoundError(f"Funding dataset not found at {fpath}")
        fdf = pd.read_parquet(fpath)
        fdf["date"] = pd.to_datetime(fdf["date"]).dt.normalize()
        fdf["symbol"] = fdf["symbol"].astype(str).str.upper()
        df["date"] = df["timestamp"].dt.normalize()
        df = df.merge(fdf, on=["date", "symbol"], how="left")
        df = df.drop(columns=["date"])

    return df
