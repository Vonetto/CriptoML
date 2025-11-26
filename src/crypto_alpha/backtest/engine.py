"""Backtest engine for configurable strategies."""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..data.universe import UniverseSelection, select_universe
from ..features import momentum, volatility
from ..utils import dates as date_utils
from . import portfolio

logger = logging.getLogger(__name__)


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
    vol_window = int(strategy.portfolio.get("vol_lookback_days", 30))
    df[vol_col] = volatility.realized_volatility(df, window=vol_window)
    if strategy.signal.get("type", "momentum") == "momentum":
        momentum_window = int(strategy.signal.get("lookback_days", 30))
        df[signal_col] = momentum.log_momentum(df, window=momentum_window)
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


def _load_prediction_lookup(signal_config: dict) -> Dict[pd.Timestamp, pd.DataFrame]:
    path_value = signal_config.get("path")
    if not path_value:
        raise ValueError("Prediction signal requires a 'path'")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_parquet(path)
    if "prediction" not in df.columns or "symbol" not in df.columns:
        raise ValueError(f"Prediction file {path} missing required columns.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    grouped = {
        date: group[["symbol", "prediction"]].copy()
        for date, group in df.groupby("date")
    }
    return grouped


def _build_signal_frame(
    eligible: pd.DataFrame,
    signal_type: str,
    signal_col: str,
    vol_col: str,
    prediction_lookup: Optional[Dict[pd.Timestamp, pd.DataFrame]],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    frame = eligible[["symbol", vol_col]].rename(columns={vol_col: "volatility"})
    if signal_type == "momentum":
        frame["signal"] = eligible[signal_col]
        return frame
    if signal_type == "predictions":
        preds = prediction_lookup.get(as_of) if prediction_lookup else None
        if preds is None:
            frame["signal"] = pd.NA
            logger.debug("No predictions available for %s", as_of.date())
            return frame
        merged = frame.merge(preds, on="symbol", how="left")
        merged.rename(columns={"prediction": "signal"}, inplace=True)
        return merged
    raise ValueError(f"Unsupported signal type: {signal_type}")


def _risk_overlay_settings(strategy: StrategyConfig) -> Dict[str, float | bool | str]:
    cfg = strategy.risk_overlay or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "mode": "off"}
    return {
        "enabled": True,
        "mode": cfg.get("mode", "step"),
        "target_vol": float(cfg.get("target_vol_annual", 0.4)),
        "vol_window": int(cfg.get("vol_window_periods", 60)),
        "periods_per_year": float(cfg.get("periods_per_year", 52)),
        "max_leverage": float(cfg.get("max_gross_leverage", 1.5)),
        "dd_trigger": float(cfg.get("dd_trigger", 0.3)),
        "dd_hard": float(cfg.get("dd_hard", 0.5)),
        "dd_trigger_scale": float(cfg.get("dd_trigger_scale", 0.5)),
        "dd_proportionality": float(cfg.get("dd_proportionality", 1.5)),
        "min_scale": float(cfg.get("min_scale", 0.2)),
        "cooldown_rate": float(cfg.get("cooldown_rate", 0.0)),
    }


def _compute_vol_scale(
    returns_history: List[float],
    vol_window: int,
    target_vol: float,
    periods_per_year: float,
    max_leverage: float,
) -> tuple[float, float]:
    eps = 1e-6
    if vol_window <= 0 or len(returns_history) < vol_window:
        return max_leverage, 0.0
    window = np.array(returns_history[-vol_window:])
    realized = float(window.std(ddof=0))
    realized_annual = realized * np.sqrt(periods_per_year)
    if realized_annual <= 0:
        return max_leverage, realized_annual
    scale = min(max_leverage, target_vol / (realized_annual + eps))
    return scale, realized_annual


def _compute_drawdown_scale(
    capital: float,
    equity_peak: float,
    params: Dict[str, float | str],
) -> tuple[float, float, str]:
    if equity_peak <= 0:
        return 1.0, 0.0, "normal"
    drawdown = max(0.0, 1.0 - capital / equity_peak)
    mode = params.get("mode", "step")
    min_scale = float(params.get("min_scale", 0.2))
    if mode == "proportional":
        trigger = float(params.get("dd_trigger", 0.0))
        k = float(params.get("dd_proportionality", 1.5))
        if drawdown <= trigger:
            return 1.0, drawdown, "normal"
        scale = max(min_scale, 1.0 - k * drawdown)
        regime = "alert" if drawdown > trigger and scale > min_scale + 1e-6 else "panic"
        return scale, drawdown, regime
    else:
        dd_trigger = float(params.get("dd_trigger", 0.3))
        dd_hard = float(params.get("dd_hard", 0.5))
        dd_trigger_scale = float(params.get("dd_trigger_scale", 0.5))
        regime = "normal"
        scale = 1.0
        if drawdown >= dd_hard:
            scale = min_scale
            regime = "panic"
        elif drawdown >= dd_trigger:
            scale = max(dd_trigger_scale, min_scale)
            regime = "alert"
        return scale, drawdown, regime


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
    signal_type = strategy.signal.get("type", "momentum").lower()
    prediction_lookup: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    if signal_type == "predictions":
        prediction_lookup = _load_prediction_lookup(strategy.signal)

    risk_params = _risk_overlay_settings(strategy)

    selection_pct = float(strategy.portfolio.get("selection_top_pct", 0.2))
    weighting = strategy.portfolio.get("weighting", "equal")
    cash_buffer = float(strategy.portfolio.get("cash_buffer_pct", 0.0))
    commission = float(strategy.execution.get("commission_pct", 0.0))
    max_weight_pct = strategy.portfolio.get("max_weight_pct")
    long_short_cfg = strategy.portfolio.get("long_short", {})
    long_short_enabled = bool(long_short_cfg.get("enabled", False))

    capital = float(strategy.backtest.get("initial_capital", 10000.0))
    prev_weights: Dict[str, float] = {}
    universe_history: List[UniverseSelection] = []
    results: List[dict] = []
    returns_history: List[float] = []
    equity_peak = capital
    prev_drawdown = 0.0

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
        signal_frame = _build_signal_frame(
            eligible,
            signal_type,
            signal_col,
            vol_col,
            prediction_lookup,
            rebalance_date,
        )
        risk_scale = 1.0
        vol_scale = 1.0
        dd_scale = 1.0
        drawdown = 0.0
        realized_vol_annual = 0.0
        risk_regime = "off"
        if risk_params["enabled"]:
            vol_scale, realized_vol_annual = _compute_vol_scale(
                returns_history,
                risk_params["vol_window"],
                risk_params["target_vol"],
                risk_params["periods_per_year"],
                risk_params["max_leverage"],
            )
            dd_scale, drawdown, risk_regime = _compute_drawdown_scale(
                capital,
                equity_peak=max(equity_peak, capital),
                params=risk_params,
            )
            risk_scale = min(vol_scale, dd_scale)
            if risk_scale > risk_params["max_leverage"]:
                risk_scale = risk_params["max_leverage"]
            cooldown = float(risk_params.get("cooldown_rate", 0.0))
            if cooldown > 0:
                dd_trigger = float(risk_params.get("dd_trigger", 0.0))
                if drawdown < dd_trigger or drawdown < (prev_drawdown - 1e-6):
                    risk_scale = min(1.0, risk_scale + cooldown)
        else:
            risk_regime = "off"

        if long_short_enabled:
            weights = portfolio.compute_long_short_weights(
                signal_frame,
                long_pct=float(long_short_cfg.get("long_selection_pct", selection_pct)),
                short_pct=float(long_short_cfg.get("short_selection_pct", selection_pct)),
                weighting=long_short_cfg.get("weighting", weighting),
                volatility_col="volatility",
                gross_leverage=float(long_short_cfg.get("gross_leverage", 1.0)),
                max_weight_pct=long_short_cfg.get("max_weight_pct", max_weight_pct),
            )
        else:
            weights = portfolio.compute_weights(
                signal_frame,
                selection_top_pct=selection_pct,
                weighting=weighting,
                volatility_col="volatility",
                cash_buffer_pct=cash_buffer,
                max_weight_pct=max_weight_pct,
            )
        if risk_params["enabled"] and weights:
            weights = {symbol: weight * risk_scale for symbol, weight in weights.items()}

        price_start = daily_slice.set_index("symbol")["close"]
        price_end = df[df["date"] == next_date].set_index("symbol")["close"]

        gross = portfolio.portfolio_return(price_start, price_end, weights)
        turn = portfolio.turnover(prev_weights, weights)
        cost = commission * turn
        net = gross - cost
        capital *= 1.0 + net
        equity_peak = max(equity_peak, capital)
        returns_history.append(net)
        prev_drawdown = drawdown

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
                "risk_scale": risk_scale,
                "vol_scale": vol_scale,
                "dd_scale": dd_scale,
                "risk_regime": risk_regime,
                "drawdown": drawdown,
                "realized_vol_annual": realized_vol_annual,
            }
        )
        prev_weights = weights

    timeline = pd.DataFrame(results)
    return BacktestResult(timeline=timeline, universe_history=universe_history)
