"""Backtest engine for the V0 momentum baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..config import StrategyConfig
from ..data.universe import UniverseSelection, select_universe
from ..features import momentum, volatility
from ..utils import dates as date_utils
from . import portfolio


@dataclass
class BacktestResult:
    timeline: pd.DataFrame
    universe_history: List[UniverseSelection]


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.normalize()
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _trim_range(df: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    start = pd.to_datetime(strategy.backtest["start_date"])
    end = pd.to_datetime(strategy.backtest["end_date"])
    buffer = max(
        int(strategy.signal.get("lookback_days", 30)),
        int(strategy.portfolio.get("vol_lookback_days", 30)),
        int(strategy.universe.get("liquidity_lookback_days", 30)),
    )
    buffer_delta = pd.Timedelta(days=buffer * 2)
    mask = (df["date"] >= start - buffer_delta) & (df["date"] <= end)
    return df.loc[mask]


def _feature_columns(strategy: StrategyConfig) -> tuple[str, str]:
    momentum_window = int(strategy.signal.get("lookback_days", 30))
    vol_window = int(strategy.portfolio.get("vol_lookback_days", 30))
    return f"mom_{momentum_window}d", f"vol_{vol_window}d"


def _add_features(df: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    signal_col, vol_col = _feature_columns(strategy)
    momentum_window = int(strategy.signal.get("lookback_days", 30))
    vol_window = int(strategy.portfolio.get("vol_lookback_days", 30))
    df[signal_col] = momentum.log_momentum(df, window=momentum_window)
    df[vol_col] = volatility.realized_volatility(df, window=vol_window)
    return df


def _calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    return date_utils.normalize_calendar(df["date"].unique())


def _schedules(
    calendar: pd.DatetimeIndex,
    strategy: StrategyConfig,
) -> tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    backtest = strategy.backtest
    start = pd.to_datetime(backtest["start_date"])
    end = pd.to_datetime(backtest["end_date"])
    rebalance_freq = strategy.portfolio.get("rebalance_freq", "1W")
    universe_freq = strategy.universe.get("rebalance_freq", "1M")
    rebalance_dates = date_utils.generate_schedule(start, end, rebalance_freq, calendar)
    universe_dates = date_utils.generate_schedule(start, end, universe_freq, calendar)
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough rebalance dates in the requested interval.")
    if not universe_dates:
        universe_dates = [rebalance_dates[0]]
    return rebalance_dates, universe_dates


def run_backtest(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
    backtest_config: dict,
) -> BacktestResult:
    df = _prepare_data(prices)
    df = _trim_range(df, strategy)
    if df.empty:
        raise ValueError("No data available in the requested backtest window.")
    df = _add_features(df, strategy)
    calendar = _calendar(df)
    rebalance_dates, universe_dates = _schedules(calendar, strategy)

    signal_col, vol_col = _feature_columns(strategy)
    selection_pct = float(strategy.portfolio.get("selection_top_pct", 0.2))
    weighting = strategy.portfolio.get("weighting", "equal")
    cash_buffer = float(strategy.portfolio.get("cash_buffer_pct", 0.0))
    commission = float(strategy.execution.get("commission_pct", 0.0))

    capital = float(strategy.backtest.get("initial_capital", 10000.0))
    prev_weights: Dict[str, float] = {}
    universe_history: List[UniverseSelection] = []
    results: List[dict] = []

    universe_pointer = 0
    current_universe: List[str] = []

    for idx in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates[idx]
        next_date = rebalance_dates[idx + 1]

        while universe_pointer < len(universe_dates) and rebalance_date >= universe_dates[universe_pointer]:
            selection = select_universe(df, universe_dates[universe_pointer], strategy.universe)
            universe_history.append(selection)
            current_universe = selection.symbols
            universe_pointer += 1

        daily_slice = df[df["date"] == rebalance_date]
        eligible = daily_slice[daily_slice["symbol"].isin(current_universe)]
        signal_frame = eligible[["symbol", signal_col, vol_col, "close"]].rename(
            columns={signal_col: "signal", vol_col: "volatility"}
        )
        weights = portfolio.compute_weights(
            signal_frame,
            selection_top_pct=selection_pct,
            weighting=weighting,
            volatility_col="volatility",
            cash_buffer_pct=cash_buffer,
        )

        price_start = daily_slice.set_index("symbol")["close"]
        price_end = df[df["date"] == next_date].set_index("symbol")["close"]

        gross = portfolio.portfolio_return(price_start, price_end, weights)
        turn = portfolio.turnover(prev_weights, weights)
        cost = commission * turn
        net = gross - cost
        capital *= 1.0 + net

        results.append(
            {
                "date": rebalance_date,
                "capital": capital,
                "gross_return": gross,
                "net_return": net,
                "turnover": turn,
                "commission_cost": cost,
                "positions": len(weights),
                "universe_size": len(current_universe),
            }
        )
        prev_weights = weights

    timeline = pd.DataFrame(results)
    return BacktestResult(timeline=timeline, universe_history=universe_history)
