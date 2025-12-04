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


def _apply_weighting(
    selection: pd.DataFrame,
    weighting: str,
    volatility_col: str,
    returns_window: pd.DataFrame | None = None,
    cap_pct: float | None = None,
) -> pd.Series:
    if selection.empty:
        return pd.Series(dtype=float)
    if weighting == "inverse_vol" and volatility_col in selection:
        selection = selection.dropna(subset=[volatility_col])
        if selection.empty:
            return pd.Series(dtype=float)
        inv_vol = 1.0 / selection[volatility_col].replace(0, np.nan)
        weights = inv_vol.fillna(0.0)
    elif weighting == "max_div" and returns_window is not None:
        # Maximum diversification heuristic: w ∝ Σ^{-1} σ, w>=0
        symbols = selection["symbol"].tolist()
        sub = returns_window[symbols].dropna(how="all", axis=0)
        if sub.empty:
            return pd.Series(dtype=float)
        cov = sub.cov()
        # shrink towards diagonal for robustness
        diag = np.diag(np.diag(cov))
        cov = 0.8 * cov + 0.2 * diag
        try:
            inv_cov = np.linalg.pinv(cov.values)
        except np.linalg.LinAlgError:
            return pd.Series(dtype=float)
        vols = np.sqrt(np.diag(cov.values))
        vec = inv_cov @ vols
        vec = np.maximum(vec, 0)
        if vec.sum() == 0:
            return pd.Series(dtype=float)
        w = vec / vec.sum()
        if cap_pct is not None and cap_pct > 0:
            w = np.clip(w, 0, cap_pct)
            if w.sum() == 0:
                return pd.Series(dtype=float)
            w = w / w.sum()
        return pd.Series(w, index=selection.index)
    else:
        weights = pd.Series(1.0, index=selection.index)
    total = weights.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return weights / total


def compute_weights(
    df: pd.DataFrame,
    selection_top_pct: float,
    weighting: str = "equal",
    volatility_col: str = "volatility",
    returns_window: pd.DataFrame | None = None,
    cash_buffer_pct: float = 0.0,
    max_weight_pct: float | None = None,
    cap_pct: float | None = None,
) -> Dict[str, float]:
    """Return normalized weights according to the selected scheme."""

    keep = df.dropna(subset=["signal"])
    selection = _select_top(keep, selection_top_pct)
    if selection.empty:
        return {}

    weights = _apply_weighting(selection, weighting, volatility_col, returns_window, cap_pct)
    if weights.empty:
        return {}
    if max_weight_pct is not None and max_weight_pct > 0:
        weights = weights.clip(upper=max_weight_pct)
    if cap_pct is not None and cap_pct > 0:
        weights = weights.clip(upper=cap_pct)
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


def compute_long_short_weights(
    df: pd.DataFrame,
    long_pct: float,
    short_pct: float,
    weighting: str = "equal",
    volatility_col: str = "volatility",
    gross_leverage: float = 1.0,
    max_weight_pct: float | None = None,
    cap_pct: float | None = None,
    beta_neutral_cfg: dict | None = None,
    returns_window: pd.DataFrame | None = None,
) -> Dict[str, float]:
    keep = df.dropna(subset=["signal"])
    if keep.empty:
        return {}

    long_sel = keep.sort_values("signal", ascending=False).head(max(int(len(keep) * long_pct), 1))
    short_sel = keep.sort_values("signal", ascending=True).head(max(int(len(keep) * short_pct), 1))

    long_weights = _apply_weighting(long_sel, weighting, volatility_col, returns_window, cap_pct)
    short_weights = _apply_weighting(short_sel, weighting, volatility_col, returns_window, cap_pct)
    if long_weights.empty or short_weights.empty:
        return {}

    if max_weight_pct is not None and max_weight_pct > 0:
        long_weights = long_weights.clip(upper=max_weight_pct)
        short_weights = short_weights.clip(upper=max_weight_pct)
    if cap_pct is not None and cap_pct > 0:
        long_weights = long_weights.clip(upper=cap_pct)
        short_weights = short_weights.clip(upper=cap_pct)
    lw_sum = long_weights.sum()
    sw_sum = short_weights.sum()
    if lw_sum <= 0 or sw_sum <= 0:
        return {}
    long_weights = long_weights / lw_sum
    short_weights = short_weights / sw_sum

    half_gross = gross_leverage / 2.0
    long_weights *= half_gross
    short_weights *= half_gross

    weights: Dict[str, float] = {}
    for idx, symbol in long_sel["symbol"].items():
        weight = long_weights.loc[idx]
        weights[symbol] = weights.get(symbol, 0.0) + float(weight)
    for idx, symbol in short_sel["symbol"].items():
        weight = short_weights.loc[idx]
        weights[symbol] = weights.get(symbol, 0.0) - float(weight)

    # Optional beta-neutralization vs benchmark
    if beta_neutral_cfg and returns_window is not None:
        benchmark = beta_neutral_cfg.get("benchmark", "BTCUSDT")
        if benchmark in returns_window.columns and len(returns_window) > 10:
            var_b = returns_window[benchmark].var()
            if var_b and var_b > 0:
                betas = {}
                for sym in returns_window.columns:
                    if sym == benchmark:
                        continue
                    cov = returns_window[[sym, benchmark]].cov().iloc[0, 1]
                    if pd.notna(cov):
                        betas[sym] = cov / var_b
                beta_port = sum(weights.get(sym, 0.0) * betas.get(sym, 0.0) for sym in weights)
                if abs(beta_port) > 1e-6:
                    # hedge using benchmark if available
                    if benchmark in weights:
                        weights[benchmark] += -beta_port
                    else:
                        weights[benchmark] = -beta_port
    # Optional PCA neutralization of first component (crowding)
    if weighting == "pca_neutral" and returns_window is not None:
        syms = [sym for sym in weights.keys() if sym in returns_window.columns]
        if len(syms) >= 3:
            sub = returns_window[syms].dropna(how="any")
            if not sub.empty:
                corr = sub.corr()
                try:
                    eigvals, eigvecs = np.linalg.eigh(corr.values)
                    v1 = eigvecs[:, -1]  # leading eigenvector
                    w_vec = np.array([weights[s] for s in syms], dtype=float)
                    proj = np.dot(w_vec, v1) / (np.dot(v1, v1) + 1e-12)
                    w_adj = w_vec - proj * v1
                    # preserve gross (L1) to original
                    gross_old = np.sum(np.abs(w_vec)) + 1e-12
                    gross_new = np.sum(np.abs(w_adj)) + 1e-12
                    w_adj *= gross_old / gross_new
                    for s, w_new in zip(syms, w_adj):
                        weights[s] = float(w_new)
                except np.linalg.LinAlgError:
                    pass
    return {symbol: weight for symbol, weight in weights.items() if weight != 0}


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
        # In case of duplicate index entries, take the first
        if isinstance(start, pd.Series):
            start = start.iloc[0]
        if isinstance(end, pd.Series):
            end = end.iloc[0]
        if pd.isna(start) or pd.isna(end) or start <= 0:
            continue
        total += weight * (float(end) / float(start) - 1.0)
    return total
