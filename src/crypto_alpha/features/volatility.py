"""Volatility estimators used by the backtester."""
from __future__ import annotations

import pandas as pd


def realized_volatility(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
) -> pd.Series:
    """Rolling standard deviation of simple returns over ``window`` days."""

    if window <= 1:
        raise ValueError("window must be > 1")
    returns = df.groupby("symbol")[price_col].pct_change()
    rolling = returns.groupby(df["symbol"]).rolling(window, min_periods=window).std()
    return rolling.reset_index(level=0, drop=True)
