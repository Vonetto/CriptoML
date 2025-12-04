"""
Interactive dashboard para correr backtests on‑demand y comparar estrategias/frecuencias.

Usage:
  streamlit run scripts/dashboard.py

Sidebar:
  - Estrategias (multi-select, top configs)
  - Frecuencias (multi-select: 1W, 2W)
  - Rango de fechas
  - Capital inicial

Output:
  - Curva(s) de equity con marcadores (pnl%)
  - Tabla resumida de métricas (Sharpe, AR, Vol, MDD, n_periods)
  - Tabla de timeline reescalado

Nota: la descomposición por activos no está disponible en modo on‑demand porque el motor
no devuelve órdenes. Para ver pesos por activo, usa los runs guardados en experiments/ que
incluyan orders.csv.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs/data/binance_daily.yaml"
BT_CONFIG = ROOT / "configs/backtest/daily_default.yaml"
STRATEGIES = {
    "Aggressive (crowd_penalty)": "v2_aggressive_crowdpen",
    "Defensive (crowd_penalty)": "v2_baseline_defensive",
}
FREQ_OPTIONS = ["1W", "2W"]
BUYHOLD_LABEL = "Buy & Hold BTC"

# --- helpers to run backtest on-demand -------------------------------------

# Ensure src in path
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crypto_alpha.config import load_yaml, load_strategy_config  # noqa: E402
from crypto_alpha.data.loaders import load_ohlcv  # noqa: E402
from crypto_alpha.backtest import engine_experimental as engine  # noqa: E402
from crypto_alpha.evaluation import metrics  # noqa: E402
import logging
logging.getLogger("py.warnings").setLevel(logging.ERROR)


@st.cache_resource
def load_prices():
    cfg = load_yaml(DATA_CONFIG)
    return load_ohlcv(cfg)


@st.cache_resource
def universe_min_date(universe_path: Path = ROOT / "data/processed/universe/v0b") -> pd.Timestamp:
    """Find earliest universe CSV date."""
    dates = []
    for csv in Path(universe_path).glob("universe_*.csv"):
        try:
            dstr = csv.stem.split("_")[-1]  # universe_YYYY-MM-DD
            dates.append(pd.to_datetime(dstr))
        except Exception:
            continue
    if not dates:
        return pd.Timestamp.min
    return min(dates)


def run_bt(strategy_name: str, start: str, end: str, rebalance_freq: str) -> Tuple[pd.DataFrame, Dict, Dict]:
    strat = load_strategy_config(strategy_name)
    strat.backtest["start_date"] = start
    strat.backtest["end_date"] = end
    strat.portfolio["rebalance_freq"] = rebalance_freq
    bt_cfg = load_yaml(BT_CONFIG)
    prices = load_prices()
    result = engine.run_backtest(prices, strat, bt_cfg)
    tl = result.timeline.copy()
    mt = metrics.summarize(tl)
    return tl, mt, (result.weights_history or {})


def main():
    st.set_page_config(layout="wide")
    st.title("On-demand Backtest Dashboard")
    st.sidebar.header("Parámetros")

    strat_labels = st.sidebar.multiselect(
        "Estrategias", list(STRATEGIES.keys()) + [BUYHOLD_LABEL], default=list(STRATEGIES.keys())
    )
    sel_strats = []
    include_bh = False
    for l in strat_labels:
        if l == BUYHOLD_LABEL:
            include_bh = True
        else:
            sel_strats.append(STRATEGIES[l])
    sel_freqs = st.sidebar.multiselect("Rebalance freq", FREQ_OPTIONS, default=["1W", "2W"])
    init_cap = st.sidebar.number_input("Capital inicial", value=100.0, min_value=1.0)

    # Date range
    prices = load_prices()
    min_price_d, max_price_d = prices["timestamp"].min().date(), prices["timestamp"].max().date()
    min_univ = universe_min_date().date()
    min_allowed = max(min_price_d, min_univ)
    date_input = st.sidebar.date_input(
        "Rango fechas (inicio / fin)",
        value=(min_allowed, max_price_d),
        min_value=min_allowed,
        max_value=max_price_d,
    )
    if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        start_d, end_d = date_input
    else:
        # fallback: single value => usa como fin, y start = min_allowed
        end_d = date_input if not isinstance(date_input, (list, tuple)) else date_input[0]
        start_d = min_allowed
    start_s, end_s = str(start_d), str(end_d)

    if st.sidebar.button("Correr backtest"):
        if start_d < min_allowed:
            st.error(f"La fecha inicial debe ser >= {min_allowed} (universo disponible desde ahí).")
            return

        fig_data = []
        metrics_rows = []
        timelines: List[Tuple[str, pd.DataFrame, Dict]] = []

        with st.spinner("Ejecutando backtests..."):
            # B&H if requested
            if include_bh:
                btc = prices[prices["symbol"] == "BTCUSDT"].copy()
                btc = btc[(btc["timestamp"] >= start_s) & (btc["timestamp"] <= end_s)].copy()
                if not btc.empty:
                    btc["date"] = btc["timestamp"].dt.normalize()
                    btc = btc.sort_values("date")
                    base_price = btc["close"].iloc[0]
                    btc["capital"] = init_cap * (btc["close"] / base_price)
                    btc["capital_scaled"] = btc["capital"]
                    btc["pnl_pct"] = btc["capital_scaled"] / init_cap - 1
                    btc["run"] = "B&H BTC"
                    fig_data.append(btc[["date", "pnl_pct", "run"]])
                    metrics_rows.append(
                        {
                            "run": "B&H BTC",
                            "sharpe": 0.0,
                            "ann_return": (btc["pnl_pct"].iloc[-1]) * 365 / max(1, len(btc)),
                            "ann_vol": 0.0,
                            "max_dd": (btc["pnl_pct"].cummax() - btc["pnl_pct"]).max(),
                            "n_periods": len(btc),
                        }
                    )
                    timelines.append(("B&H BTC", btc[["date", "capital_scaled", "pnl_pct"]], {}))

            for strat in sel_strats:
                for freq in sel_freqs:
                    tl, mt, whist = run_bt(strat, start_s, end_s, freq)
                    base_cap = tl["capital"].iloc[0]
                    tl["capital_scaled"] = tl["capital"] * (init_cap / base_cap)
                    tl["pnl_pct"] = tl["capital_scaled"] / init_cap - 1
                    run_label = f"{strat} {freq}"
                    tl["run"] = run_label
                    fig_data.append(tl[["date", "pnl_pct", "run"]])
                    metrics_rows.append(
                        {
                            "run": run_label,
                            "sharpe": mt.get("sharpe"),
                            "ann_return": mt.get("annualized_return"),
                            "ann_vol": mt.get("annualized_vol"),
                            "max_dd": mt.get("max_drawdown"),
                            "n_periods": mt.get("n_periods"),
                        }
                    )
                    timelines.append((run_label, tl, whist))

        if not fig_data:
            st.warning("No se generaron datos.")
            return

        plot_df = pd.concat(fig_data)
        st.subheader("Equity")
        with st.expander("Opciones de visualización", expanded=False):
            ds_every = st.slider("Mostrar cada N puntos (downsample)", 1, 10, 1, 1)
            use_log = st.checkbox("Escala logarítmica", value=False)

        plot_df_ds = plot_df.copy()
        if ds_every > 1:
            plot_df_ds = plot_df_ds.groupby("run").apply(
                lambda g: g.iloc[::ds_every]
            ).reset_index(drop=True)
        if not plot_df_ds.empty:
            fig = px.line(
                plot_df_ds,
                x="date",
                y="pnl_pct",
                color="run",
                markers=True,
                title="Equity (pnl%)",
            )
            if use_log:
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sin datos para graficar.")

        st.subheader("Métricas")
        st.dataframe(pd.DataFrame(metrics_rows))

        st.subheader("Timeline (reescalado)")
        tl_all = pd.concat([t for _, t, _ in timelines])
        st.dataframe(tl_all[["run", "date", "capital_scaled", "pnl_pct", "drawdown"]])

        st.subheader("Retornos por rebalance")
        ret_df_list = []
        for lbl, tl, _ in timelines:
            cols = [c for c in ["date", "gross_return", "pnl_pct"] if c in tl.columns]
            d = tl[cols].copy()
            d["run"] = lbl
            ret_df_list.append(d)
        ret_df = pd.concat(ret_df_list)
        st.dataframe(ret_df.sort_values(["run", "date"]))

        # detalle por fecha
        sel_date = st.date_input("Fecha a inspeccionar", value=plot_df["date"].max().date())
        sel_date_ts = pd.to_datetime(sel_date)
        st.markdown("**Detalle del día**")
        for lbl, tl, whist in timelines:
            row = tl[tl["date"] == sel_date_ts]
            st.markdown(f"- {lbl}")
            if row.empty:
                st.write("  Sin datos en esta fecha (no era día de rebalance).")
                continue
            out = {"capital_scaled": float(row["capital_scaled"].iloc[0])}
            if "gross_return" in row.columns:
                out["gross_return"] = float(row["gross_return"].iloc[0])
            if "drawdown" in row.columns:
                out["drawdown"] = float(row["drawdown"].iloc[0])
            st.write(out)
            # Weights breakdown if available
            if whist:
                w = whist.get(sel_date_ts.normalize())
                if w:
                    pos = pd.DataFrame(list(w.items()), columns=["symbol", "weight"]).sort_values(
                        "weight", ascending=False
                    )
                    st.dataframe(pos)
                else:
                    st.write("  Sin pesos registrados para esta fecha.")


if __name__ == "__main__":
    main()
