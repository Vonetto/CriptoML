"""Momentum-related feature helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_momentum(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.Series:
    """Return log-price momentum over ``window`` observations."""

    if window <= 0:
        raise ValueError("window must be positive")
    log_price = np.log(df[price_col])
    return log_price.groupby(df["symbol"]).diff(window)
