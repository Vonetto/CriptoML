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
from bisect import bisect_right

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    timeline: pd.DataFrame
    universe_history: List[UniverseSelection]


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.normalize()
    if "funding_rate_1d" not in df.columns:
        df["funding_rate_1d"] = 0.0
    df["funding_rate_1d"] = df["funding_rate_1d"].fillna(0.0)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _load_regime_map(strategy: StrategyConfig, base: Path) -> dict[pd.Timestamp, str]:
    regime_cfg = strategy.regime or {}
    path = regime_cfg.get("file")
    if not path:
        return {}
    regime_path = base / path
    if not regime_path.exists():
        raise FileNotFoundError(f"Regime file not found: {regime_path}")
    df = pd.read_parquet(regime_path)
    if "date" not in df.columns or "regime" not in df.columns:
        raise ValueError(f"Regime file {regime_path} missing required columns (date, regime)")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return {d: r for d, r in zip(df["date"], df["regime"])}


def _regime_settings(strategy: StrategyConfig) -> dict[str, object]:
    cfg = strategy.regime or {}
    enabled = bool(cfg.get("enabled", False))
    return {
        "enabled": enabled,
        "file": cfg.get("file"),
        "lookback_days": int(cfg.get("lookback_days", 0)),
        "gross_normal": float(cfg.get("gross_normal", 1.0)),
        "gross_stress": float(cfg.get("gross_stress", 0.6)),
        "rebalance_normal": cfg.get("rebalance_normal", "2W"),
        "rebalance_stress": cfg.get("rebalance_stress", "1W"),
        "stress_label": cfg.get("stress_label", "STRESS"),
    }


def _hot_cold_settings(strategy: StrategyConfig) -> dict[str, object]:
    cfg = strategy.regime or {}
    hotcold = cfg.get("hot_cold", {}) if isinstance(cfg, dict) else {}
    enabled = bool(hotcold.get("enabled", False))
    return {
        "enabled": enabled,
        "window_periods": int(hotcold.get("window_periods", 4)),
        "vol_p_low": float(hotcold.get("vol_p_low", 0.2)),
        "vol_p_high": float(hotcold.get("vol_p_high", 0.8)),
        "gross_cold": float(hotcold.get("gross_cold", 0.4)),
        "gross_hot": float(hotcold.get("gross_hot", 1.2)),
    }

def _breadth_settings(strategy: StrategyConfig) -> dict[str, object]:
    cfg = strategy.portfolio or {}
    breadth = cfg.get("breadth_overlay", {}) if isinstance(cfg, dict) else {}
    enabled = bool(breadth.get("enabled", False))
    return {
        "enabled": enabled,
        "low_thresh": float(breadth.get("low_thresh", 0.30)),
        "high_thresh": float(breadth.get("high_thresh", 0.60)),
        "scale_low": float(breadth.get("scale_low", 0.30)),
        "scale_mid": float(breadth.get("scale_mid", 0.70)),
        "scale_high": float(breadth.get("scale_high", 1.0)),
        "signal_positive_col": breadth.get("signal_positive_col", None),  # default: use signal_col>0
        "long_only": bool(breadth.get("long_only", False)),
    }

def _mom7_filter_settings(strategy: StrategyConfig) -> dict[str, object]:
    cfg = strategy.portfolio or {}
    fcfg = cfg.get("filter_mom7", {}) if isinstance(cfg, dict) else {}
    enabled = bool(fcfg.get("enabled", False))
    return {
        "enabled": enabled,
        "scale_up": float(fcfg.get("scale_up", 1.2)),
        "scale_down": float(fcfg.get("scale_down", 0.5)),
    }

def _liq_exp(strategy: StrategyConfig) -> float:
    cfg = strategy.portfolio or {}
    return float(cfg.get("liq_exp", 1.0))

def _confidence_settings(strategy: StrategyConfig) -> dict[str, object]:
    cfg = strategy.portfolio or {}
    conf = cfg.get("confidence_scale", {}) if isinstance(cfg, dict) else {}
    enabled = bool(conf.get("enabled", False))
    return {
        "enabled": enabled,
        "min_scale": float(conf.get("min_scale", 0.5)),
        "max_scale": float(conf.get("max_scale", 1.2)),
        "clip_z": float(conf.get("clip_z", 3.0)),
    }

def _ic_weight_settings(strategy: StrategyConfig) -> dict[str, object]:
    cfg = strategy.portfolio or {}
    icw = cfg.get("ic_weighting", {}) if isinstance(cfg, dict) else {}
    enabled = bool(icw.get("enabled", False))
    return {
        "enabled": enabled,
        "lookback": int(icw.get("lookback", 10)),
        "ic_low": float(icw.get("ic_low", 0.0)),
        "ic_high": float(icw.get("ic_high", 0.05)),
        "scale_low": float(icw.get("scale_low", 0.5)),
        "scale_high": float(icw.get("scale_high", 1.0)),
    }



def _next_date(schedule: list[pd.Timestamp], current: pd.Timestamp) -> pd.Timestamp | None:
    idx = bisect_right(schedule, current)
    if idx >= len(schedule):
        return None
    return schedule[idx]


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
    # mom_7d for filters
    df["mom_7d"] = df.groupby("symbol") ["close"].transform(lambda s: np.log(s / s.shift(7)))
    # Liquidity proxy: rolling volume_quote and num_trades
    df["volq_30d"] = (
        df.sort_values(["symbol", "date"])
        .groupby("symbol")["volume_quote"]
        .transform(lambda s: s.rolling(vol_window, min_periods=vol_window//2).mean())
    )
    df["trades_30d"] = (
        df.sort_values(["symbol", "date"])
        .groupby("symbol")["num_trades"]
        .transform(lambda s: s.rolling(vol_window, min_periods=vol_window//2).mean())
    )
    df["liq_score"] = df["volq_30d"] * df["trades_30d"]
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
    meta_lookup: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    meta_threshold: float | None = None,
) -> pd.DataFrame:
    frame = eligible[["symbol", vol_col]].rename(columns={vol_col: "volatility"})
    if signal_type == "momentum":
        frame["signal"] = eligible[signal_col]
        # Optional meta-prob filter
        if meta_lookup is not None and meta_threshold is not None:
            probs = meta_lookup.get(as_of)
            if probs is not None:
                merged = frame.merge(probs, on="symbol", how="left")
                frame = merged[merged["meta_prob"] >= meta_threshold].copy()
                frame["signal"] = frame["signal"]
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
    meta_cfg = cfg.get("meta_label", {}) if isinstance(cfg, dict) else {}
    ic_cfg = cfg.get("ic_gating", {}) if isinstance(cfg, dict) else {}
    prop_cfg = cfg.get("prop_constraints", {}) if isinstance(cfg, dict) else {}
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
        "meta_label": {
            "enabled": bool(meta_cfg.get("enabled", False)),
            "lookback_ic": int(meta_cfg.get("lookback_ic", 10)),
            "min_scale": float(meta_cfg.get("min_scale", 0.2)),
            "max_scale": float(meta_cfg.get("max_scale", 1.0)),
            "ic_floor": float(meta_cfg.get("ic_floor", -0.05)),
            "ic_cap": float(meta_cfg.get("ic_cap", 0.15)),
        },
        "ic_gating": {
            "enabled": bool(ic_cfg.get("enabled", False)),
            "lookback_ic": int(ic_cfg.get("lookback_ic", 10)),
            "ic_low": float(ic_cfg.get("ic_low", 0.0)),
            "ic_high": float(ic_cfg.get("ic_high", 0.05)),
            "scale_low": float(ic_cfg.get("scale_low", 0.5)),
            "scale_high": float(ic_cfg.get("scale_high", 1.0)),
        },
        "prop_constraints": {
            "enabled": bool(prop_cfg.get("enabled", False)),
            # per-rebalance loss cap (relative, e.g., 0.05 = -5% max loss on a period)
            "loss_limit_pct": float(prop_cfg.get("loss_limit_pct", 0.0)),
            # cooldown periods to stay flat after a hit
            "cooloff_periods": int(prop_cfg.get("cooloff_periods", 0)),
            # equity stop: halt trading if capital falls below peak * (1 - equity_stop_pct)
            "equity_stop_pct": float(prop_cfg.get("equity_stop_pct", 0.0)),
        },
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
    base_root = Path(__file__).resolve().parents[3]
    df = _prepare_data(prices)
    df = _trim_range(df, strategy)
    if df.empty:
        raise ValueError("No data available in the requested backtest window.")
    df = _add_features(df, strategy)
    calendar = _calendar(df)
    start_bt = pd.to_datetime(strategy.backtest["start_date"])
    end_bt = pd.to_datetime(strategy.backtest["end_date"])
    rebalance_dates_base, universe_dates = _schedules(calendar, strategy)

    signal_col, vol_col = _feature_columns(strategy)
    signal_type = strategy.signal.get("type", "momentum").lower()
    prediction_lookup: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    if signal_type == "predictions":
        prediction_lookup = _load_prediction_lookup(strategy.signal)
    meta_lookup: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    meta_threshold = None
    if strategy.signal.get("meta_prob_path"):
        meta_path = base_root / strategy.signal["meta_prob_path"]
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta prob file not found: {meta_path}")
        meta_df = pd.read_parquet(meta_path)
        meta_df["date"] = pd.to_datetime(meta_df["date"]).dt.normalize()
        meta_df["symbol"] = meta_df["symbol"].astype(str)
        meta_lookup = {d: g[["symbol", "meta_prob"]].copy() for d, g in meta_df.groupby("date")}
        meta_threshold = float(strategy.signal.get("meta_threshold", 0.5))

    risk_params = _risk_overlay_settings(strategy)
    regime_params = _regime_settings(strategy)
    hotcold_params = _hot_cold_settings(strategy)
    mom7_filter = _mom7_filter_settings(strategy)
    conf_params = _confidence_settings(strategy)
    breadth_params = _breadth_settings(strategy)
    icw_params = _ic_weight_settings(strategy)
    regime_map: Dict[pd.Timestamp, str] = {}
    if regime_params["enabled"] and regime_params["file"]:
        regime_map = _load_regime_map(strategy, base_root)
    stress_label = regime_params.get("stress_label", "STRESS")

    if regime_params["enabled"]:
        rebalance_normal = regime_params["rebalance_normal"]
        rebalance_stress = regime_params["rebalance_stress"]
        sched_normal = sorted(date_utils.generate_schedule(start_bt, end_bt, rebalance_normal, calendar))
        sched_stress = sorted(date_utils.generate_schedule(start_bt, end_bt, rebalance_stress, calendar))
    else:
        sched_normal = rebalance_dates_base
        sched_stress = rebalance_dates_base

    selection_pct = float(strategy.portfolio.get("selection_top_pct", 0.2))
    weighting = strategy.portfolio.get("weighting", "equal")
    cash_buffer = float(strategy.portfolio.get("cash_buffer_pct", 0.0))
    commission = float(strategy.execution.get("commission_pct", 0.0))
    max_weight_pct = strategy.portfolio.get("max_weight_pct")
    cap_pct = strategy.portfolio.get("cap_pct")
    long_short_cfg = strategy.portfolio.get("long_short", {})
    long_short_enabled = bool(long_short_cfg.get("enabled", False))
    beta_neutral_cfg = long_short_cfg.get("beta_neutralize") if long_short_enabled else None

    capital = float(strategy.backtest.get("initial_capital", 10000.0))
    prev_weights: Dict[str, float] = {}
    universe_history: List[UniverseSelection] = []
    results: List[dict] = []
    returns_history: List[float] = []
    ret_history: List[float] = []
    vol_window_hist: List[float] = []
    equity_peak = capital
    prev_drawdown = 0.0
    prop_cooldown = 0
    prop_halted = False

    universe_pointer = 0
    current_universe: List[str] = []
    ic_history: List[float] = []

    if regime_params["enabled"]:
        initial_regime = regime_map.get(start_bt.normalize(), "NORMAL")
        initial_sched = sched_stress if initial_regime == stress_label else sched_normal
        rebalance_date = initial_sched[0]
    else:
        rebalance_date = rebalance_dates_base[0]

    # Slippage buckets (optional)
    exec_cfg = strategy.execution or {}
    sl_alpha_base = float(exec_cfg.get("slippage_alpha", 0.0))
    slip_cfg = exec_cfg.get("slippage_by_vol", {}) if isinstance(exec_cfg, dict) else {}
    slip_enabled = bool(slip_cfg.get("enabled", False))
    slip_bins = slip_cfg.get("pct_bins", [0.33, 0.66]) if slip_enabled else []
    slip_mults = slip_cfg.get("multipliers", [1.0, 1.0, 1.0]) if slip_enabled else []
    if slip_enabled:
        # ensure multipliers length = bins+1
        if len(slip_mults) != len(slip_bins) + 1:
            raise ValueError("slippage_by_vol.multipliers must have len = len(pct_bins)+1")
        # compute global quantile thresholds on vol_col
        try:
            slip_thresholds = [float(df[vol_col].quantile(q)) for q in slip_bins]
        except Exception as e:
            raise ValueError(f"Failed computing slippage quantiles on {vol_col}: {e}") from e
    else:
        slip_thresholds = []

    while True:
        regime_label = regime_map.get(rebalance_date, "NORMAL")
        is_stress = regime_params["enabled"] and regime_label == stress_label
        lb_days = int(regime_params.get("lookback_days", 0))
        if regime_params["enabled"] and lb_days > 0:
            window = pd.date_range(end=rebalance_date, periods=lb_days + 1, freq="D")
            if any(regime_map.get(d.normalize(), "NORMAL") == stress_label for d in window):
                is_stress = True
        target_sched = sched_stress if is_stress else sched_normal
        next_date = _next_date(target_sched, rebalance_date)
        if next_date is None:
            break

        while universe_pointer < len(universe_dates) and rebalance_date >= universe_dates[universe_pointer]:
            selection = select_universe(df, universe_dates[universe_pointer], strategy.universe)
            universe_history.append(selection)
            current_universe = selection.symbols
            universe_pointer += 1

        daily_slice = df[df["date"] == rebalance_date]
        eligible = daily_slice[daily_slice["symbol"].isin(current_universe)]
        eff_vol_col = vol_col
        if weighting == "inv_vol_liq" and "liq_score" in eligible.columns:
            liq = eligible["liq_score"]
            med = liq.median()
            exp = _liq_exp(strategy)
            adj = (liq / med) ** exp if med and pd.notna(med) else 1.0
            eligible = eligible.copy()
            eligible["vol_liq"] = eligible[vol_col] * adj
            eff_vol_col = "vol_liq"
        signal_frame = _build_signal_frame(
            eligible,
            signal_type,
            signal_col,
            vol_col,
            prediction_lookup,
            rebalance_date,
            meta_lookup=meta_lookup,
            meta_threshold=meta_threshold,
        )
        # Rolling returns window for beta-neutral (optional)
        returns_window = None
        if beta_neutral_cfg:
            lookback_beta = int(beta_neutral_cfg.get("lookback_days", 60))
            window_start = rebalance_date - pd.Timedelta(days=lookback_beta)
            sub = df[(df["date"] > window_start) & (df["date"] <= rebalance_date)]
            if not sub.empty:
                piv = sub.pivot_table(index="date", columns="symbol", values="close").pct_change().dropna(how="all")
                if not piv.empty:
                    returns_window = piv
        # Meta-label scaling based on recent IC of the signal
        meta_scale = 1.0
        meta_cfg = risk_params.get("meta_label", {}) if risk_params else {}
        if meta_cfg.get("enabled", False) and ic_history:
            lb = int(meta_cfg.get("lookback_ic", 10))
            window_ic = ic_history[-lb:] if lb > 0 else ic_history
            ic_mean = float(np.mean(window_ic)) if window_ic else 0.0
            ic_clipped = min(
                max(ic_mean, meta_cfg.get("ic_floor", -0.05)),
                meta_cfg.get("ic_cap", 0.15),
            )
            floor = meta_cfg.get("ic_floor", -0.05)
            cap = meta_cfg.get("ic_cap", 0.15)
            if cap - floor > 0:
                frac = (ic_clipped - floor) / (cap - floor)
            else:
                frac = 1.0
            meta_scale = meta_cfg.get("min_scale", 0.2) + frac * (
                meta_cfg.get("max_scale", 1.0) - meta_cfg.get("min_scale", 0.2)
            )
        # IC gating (step) based on recent IC mean
        ic_gate_scale = 1.0
        ic_gate_cfg = risk_params.get("ic_gating", {}) if risk_params else {}
        if ic_gate_cfg.get("enabled", False) and ic_history:
            lb_ic = int(ic_gate_cfg.get("lookback_ic", 10))
            window_ic = ic_history[-lb_ic:] if lb_ic > 0 else ic_history
            ic_mean = float(np.mean(window_ic)) if window_ic else 0.0
            ic_low = ic_gate_cfg.get("ic_low", 0.0)
            ic_high = ic_gate_cfg.get("ic_high", 0.05)
            scale_low = ic_gate_cfg.get("scale_low", 0.5)
            scale_high = ic_gate_cfg.get("scale_high", 1.0)
            if ic_mean < ic_low:
                ic_gate_scale = scale_low
            elif ic_mean > ic_high:
                ic_gate_scale = scale_high
            else:
                ic_gate_scale = 1.0

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

        prop_state = "off"
        prop_cfg = risk_params.get("prop_constraints", {}) if risk_params else {}
        prop_enabled = bool(prop_cfg.get("enabled", False))
        loss_limit = float(prop_cfg.get("loss_limit_pct", 0.0))
        cooloff_periods = int(prop_cfg.get("cooloff_periods", 0))
        equity_stop_pct = float(prop_cfg.get("equity_stop_pct", 0.0))

        # Apply prop halt/cooldown before sizing
        if prop_enabled and prop_halted:
            weights = {}
            prop_state = "halted"
        elif prop_enabled and prop_cooldown > 0:
            weights = {}
            prop_state = "cooldown"
        else:
            if long_short_enabled:
                dyn = long_short_cfg.get("dynamic_percentiles") or {}
                min_sym = int(dyn.get("min_symbols", 0))
                eff_top = float(dyn.get("top_pct", long_short_cfg.get("long_selection_pct", selection_pct)))
                eff_bot = float(dyn.get("bottom_pct", long_short_cfg.get("short_selection_pct", selection_pct)))
                if min_sym > 0 and eligible.shape[0] < min_sym:
                    eff_top = eff_bot = 0.0
                weights = portfolio.compute_long_short_weights(
                    signal_frame,
                    long_pct=eff_top,
                    short_pct=eff_bot,
                    weighting=long_short_cfg.get("weighting", weighting),
                    volatility_col="volatility" if eff_vol_col == vol_col else "vol_liq",
                    gross_leverage=float(long_short_cfg.get("gross_leverage", 1.0)),
                    max_weight_pct=long_short_cfg.get("max_weight_pct", max_weight_pct),
                    cap_pct=cap_pct,
                    beta_neutral_cfg=beta_neutral_cfg,
                    returns_window=returns_window,
                )
            else:
                weights = portfolio.compute_weights(
                    signal_frame,
                    selection_top_pct=selection_pct,
                    weighting=weighting,
                    volatility_col="volatility" if eff_vol_col == vol_col else "vol_liq",
                    cash_buffer_pct=cash_buffer,
                    max_weight_pct=max_weight_pct,
                    cap_pct=cap_pct,
                )
            if risk_params["enabled"] and weights:
                weights = {symbol: weight * risk_scale for symbol, weight in weights.items()}
            # Regime gross scaling
            if regime_params["enabled"] and weights:
                gross_scale = regime_params["gross_stress"] if is_stress else regime_params["gross_normal"]
                weights = {symbol: weight * gross_scale for symbol, weight in weights.items()}
            # Breadth overlay (simple 3-level scale)
            if breadth_params["enabled"] and not eligible.empty and weights:
                # determine positives using signal_col
                sig_col = breadth_params.get("signal_positive_col") or signal_col
                if sig_col in eligible.columns:
                    positives = (eligible[sig_col] > 0).sum()
                    breadth = positives / len(eligible)
                    if breadth < breadth_params["low_thresh"]:
                        b_scale = breadth_params["scale_low"]
                    elif breadth > breadth_params["high_thresh"]:
                        b_scale = breadth_params["scale_high"]
                    else:
                        b_scale = breadth_params["scale_mid"]
                    if b_scale != 1.0:
                        if breadth_params.get("long_only", False):
                            weights = {
                                symbol: (weight * b_scale if weight > 0 else weight)
                                for symbol, weight in weights.items()
                            }
                        else:
                            weights = {symbol: weight * b_scale for symbol, weight in weights.items()}
            # Hot/Cold overlay based on recent portfolio performance
            hotcold_scale = 1.0
            if hotcold_params["enabled"] and ret_history:
                w = hotcold_params["window_periods"]
                if len(ret_history) >= w:
                    window_rets = ret_history[-w:]
                    ret_w = float(np.prod([1 + r for r in window_rets]) - 1.0)
                    vol_w = float(np.std(window_rets, ddof=0))
                    vol_window_hist.append(vol_w)
                    lo_q = np.percentile(vol_window_hist, hotcold_params["vol_p_low"] * 100)
                    hi_q = np.percentile(vol_window_hist, hotcold_params["vol_p_high"] * 100)
                    if ret_w < 0 and vol_w > hi_q:
                        hotcold_scale = hotcold_params["gross_cold"]
                    elif ret_w > 0 and vol_w < lo_q:
                        hotcold_scale = hotcold_params["gross_hot"]
                if hotcold_scale != 1.0 and weights:
                    weights = {symbol: weight * hotcold_scale for symbol, weight in weights.items()}
            # Meta-label scaling
            if meta_scale != 1.0 and weights:
                weights = {symbol: weight * meta_scale for symbol, weight in weights.items()}
            # IC gating scaling
            if ic_gate_scale != 1.0 and weights:
                weights = {symbol: weight * ic_gate_scale for symbol, weight in weights.items()}
            # Mom7 filter scaling (alignment of short-term vs main signal)
            mom7_scale = 1.0
            if mom7_filter["enabled"] and "mom_7d" in eligible.columns and not signal_frame.empty and weights:
                mom7_map = eligible.set_index("symbol")["mom_7d"]
                sig_map = signal_frame.set_index("symbol")["signal"]
                new_weights = {}
                scales = []
                for sym, w in weights.items():
                    if sym in mom7_map and sym in sig_map:
                        sgn = sig_map[sym]
                        m7 = mom7_map[sym]
                        if pd.notna(sgn) and pd.notna(m7):
                            aligned = (sgn >= 0 and m7 >= 0) or (sgn <= 0 and m7 <= 0)
                            sc = mom7_filter["scale_up"] if aligned else mom7_filter["scale_down"]
                            scales.append(sc)
                            new_weights[sym] = w * sc
                            continue
                    new_weights[sym] = w
                if scales:
                    mom7_scale = float(np.mean(scales))
                weights = new_weights
            # IC-weighted scaling
            if icw_params["enabled"] and ic_history and weights:
                lw = icw_params["lookback"]
                if len(ic_history) >= lw and np.std(ic_history[-lw:]) > 0:
                    ic_roll = np.mean(ic_history[-lw:])
                    ic_clip = min(max(ic_roll, icw_params["ic_low"]), icw_params["ic_high"])
                    scale_ic = icw_params["scale_low"] + (ic_clip - icw_params["ic_low"]) * (
                        (icw_params["scale_high"] - icw_params["scale_low"]) / (icw_params["ic_high"] - icw_params["ic_low"] + 1e-9)
                    )
                    scale_ic = float(np.clip(scale_ic, icw_params["scale_low"], icw_params["scale_high"]))
                    if scale_ic != 1.0:
                        weights = {symbol: weight * scale_ic for symbol, weight in weights.items()}

        price_start = daily_slice.set_index("symbol")["close"]
        price_end = df[df["date"] == next_date].set_index("symbol")["close"]
        funding_series = daily_slice.set_index("symbol")["funding_rate_1d"]

        if weights:
            price_ret = portfolio.portfolio_return(price_start, price_end, weights)
            funding_ret = 0.0
            if not funding_series.empty:
                for sym, w in weights.items():
                    if sym in funding_series:
                        fr = funding_series[sym]
                        if pd.notna(fr):
                            funding_ret += w * (-float(fr))
            gross = price_ret + funding_ret
            turn = portfolio.turnover(prev_weights, weights)
            cost = commission * turn
            # slippage: base alpha optionally scaled by volatility buckets
            slippage_alpha = sl_alpha_base
            if slip_enabled and eligible.shape[0] > 0:
                vol_proxy = eligible[vol_col].median()
                bucket = 0
                for i, thr in enumerate(slip_thresholds):
                    if vol_proxy <= thr:
                        bucket = i
                        break
                else:
                    bucket = len(slip_mults) - 1
                slippage_alpha = sl_alpha_base * float(slip_mults[bucket])
            if slippage_alpha > 0.0:
                daily_vol_proxy = abs(price_ret)
                cost += slippage_alpha * daily_vol_proxy * turn
            # impact cost: k * sigma * sqrt(turnover)
            impact_cfg = exec_cfg.get("impact", {}) if isinstance(exec_cfg, dict) else {}
            if impact_cfg.get("enabled", False):
                k = float(impact_cfg.get("k", 0.0))
                vol_col_name = impact_cfg.get("vol_column", vol_col)
                # fallback: use daily volatility proxy if column missing
                if vol_col_name in eligible.columns:
                    sigma_val = float(eligible[vol_col_name].median())
                else:
                    sigma_val = abs(price_ret)
                impact_cost = k * sigma_val * (turn ** 0.5)
                cost += impact_cost
            net = gross - cost
        else:
            price_ret = 0.0
            funding_ret = 0.0
            gross = 0.0
            turn = 0.0
            cost = 0.0
            net = 0.0

        # Prop loss limit: cap loss and trigger cooldown
        if prop_enabled and loss_limit > 0.0 and net < -loss_limit:
            net = -loss_limit
            prop_cooldown = max(prop_cooldown, cooloff_periods)
            prop_state = "loss_limit"
        # Update capital
        capital *= 1.0 + net
        equity_peak = max(equity_peak, capital)
        returns_history.append(net)
        ret_history.append(net)
        prev_drawdown = drawdown
        # Equity stop
        if prop_enabled and equity_stop_pct > 0.0:
            if capital <= equity_peak * (1.0 - equity_stop_pct):
                prop_halted = True
                prop_state = "halted"

        if prop_enabled and prop_cooldown > 0 and prop_state != "halted":
            prop_cooldown -= 1

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
                "regime": regime_label,
                "gross_scale_regime": (regime_params["gross_stress"] if is_stress else regime_params["gross_normal"]),
                "hotcold_scale": locals().get("hotcold_scale", 1.0),
                "mom7_scale": mom7_scale if "mom7_scale" in locals() else 1.0,
                "breadth_scale": b_scale if "b_scale" in locals() else 1.0,
                "meta_scale": meta_scale,
                "ic_gate_scale": ic_gate_scale,
                "prop_state": prop_state,
                "prop_cooldown": prop_cooldown,
            }
        )
        prev_weights = weights
        rebalance_date = next_date

        # Compute IC for this period (signal vs forward return) and store for meta_label
        try:
            if not eligible.empty:
                fwd_ret = price_end.reindex(eligible["symbol"]).div(price_start.reindex(eligible["symbol"])) - 1.0
                sig = signal_frame.set_index("symbol")["signal"]
                aligned = pd.concat([sig, fwd_ret], axis=1, keys=["signal", "fwd"]).dropna()
                if (
                    len(aligned) >= 3
                    and aligned["signal"].std(ddof=0) > 0
                    and aligned["fwd"].std(ddof=0) > 0
                ):
                    ic = aligned["signal"].corr(aligned["fwd"], method="spearman")
                    if pd.notna(ic):
                        ic_history.append(float(ic))
        except Exception:
            logger.debug("Failed to compute IC at %s", rebalance_date, exc_info=True)

    timeline = pd.DataFrame(results)
    return BacktestResult(timeline=timeline, universe_history=universe_history)
