"""Basic evaluation metrics for V0 backtests."""
from __future__ import annotations

import math
from typing import Dict

import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 52) -> float:
    if returns.empty:
        return 0.0
    growth = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return growth ** (1 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 52) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 52) -> float:
    excess = returns - rf / periods_per_year
    vol = annualized_volatility(excess, periods_per_year)
    if vol == 0:
        return 0.0
    return float(excess.mean() * periods_per_year / vol)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def summarize(timeline: pd.DataFrame, periods_per_year: int = 52) -> Dict[str, float]:
    returns = timeline["net_return"]
    equity = timeline["capital"]
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "n_periods": len(timeline),
    }
