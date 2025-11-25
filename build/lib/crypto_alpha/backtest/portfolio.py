"""Portfolio construction helpers for the V0 baseline."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _select_top(df: pd.DataFrame, selection_top_pct: float) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("signal", ascending=False)
    n_assets = max(int(len(df) * selection_top_pct), 1)
    return df.head(n_assets)


def compute_weights(
    df: pd.DataFrame,
    selection_top_pct: float,
    weighting: str = "equal",
    volatility_col: str = "volatility",
    cash_buffer_pct: float = 0.0,
) -> Dict[str, float]:
    """Return normalized weights according to the selected scheme."""

    keep = df.dropna(subset=["signal"])
    selection = _select_top(keep, selection_top_pct)
    if selection.empty:
        return {}

    if weighting == "inverse_vol" and volatility_col in selection:
        selection = selection.dropna(subset=[volatility_col])
        if selection.empty:
            return {}
        inv_vol = 1.0 / selection[volatility_col].replace(0, np.nan)
        weights = inv_vol.fillna(0.0)
    else:
        weights = pd.Series(1.0, index=selection.index)

    total = weights.sum()
    if total <= 0:
        return {}
    weights = weights / total
    if cash_buffer_pct:
        weights *= max(1.0 - cash_buffer_pct, 0.0)
    return {
        symbol: float(weight)
        for symbol, weight in zip(selection["symbol"], weights)
        if weight > 0
    }


def turnover(previous: Dict[str, float], new: Dict[str, float]) -> float:
    keys = set(previous) | set(new)
    return float(sum(abs(new.get(k, 0.0) - previous.get(k, 0.0)) for k in keys))


def portfolio_return(
    start_prices: pd.Series,
    end_prices: pd.Series,
    weights: Dict[str, float],
) -> float:
    if not weights:
        return 0.0
    total = 0.0
    for symbol, weight in weights.items():
        if symbol not in start_prices or symbol not in end_prices:
            continue
        start = start_prices[symbol]
        end = end_prices[symbol]
        if pd.isna(start) or pd.isna(end) or start <= 0:
            continue
        total += weight * (float(end) / float(start) - 1.0)
    return total
