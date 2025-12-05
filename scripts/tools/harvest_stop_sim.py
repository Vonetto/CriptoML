#!/usr/bin/env python
"""
Offline simulator to apply a simple harvest + stop overlay on an existing
equity curve (e.g., output of run_backtest_experimental).

Logic
------
1) Stop-loss de portafolio:
   - Si el drawdown desde el máximo histórico supera `stop_dd`,
     se fija la exposición a 0 durante `stop_cooldown` días.
   - Tras el cooldown, la exposición vuelve a 1.0 (sin requisitos
     adicionales para simplificar; ajusta según tu criterio).

2) Harvest (retiro parcial):
   - Cada vez que el equity con overlay supera un nuevo máximo que esté
     al menos `harvest_step` por encima del último máximo “harvested”,
     se retira `harvest_pct` del equity a “cash”. El equity disminuye
     en esa cantidad; el cash acumulado sigue sumando al patrimonio
     total (wealth = equity + cash).

Entradas esperadas
------------------
- CSV de equity con columnas típicas de los backtests:
  * date (o datetime)
  * equity  (preferido)  o portfolio_value
  * gross_return o net_return (se usarán solo si no hay equity explícito)

Salida
------
- CSV con columnas: date, equity_base, equity_overlay, cash, wealth
- Métricas impresas por pantalla: final wealth, harvest_total, MDD, Sharpe.

Uso
---
python scripts/tools/harvest_stop_sim.py \
    --equity-file experiments/aggr_1w_20250601_20251203/20251203_155836/equity_curve.csv \
    --stop-dd 0.12 --stop-cooldown 20 \
    --harvest-step 0.08 --harvest-pct 0.30 \
    --output data/processed/overlays/harvest_stop_equity.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_equity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Infer date column
    date_col = None
    for c in ["date", "datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date/datetime column found in equity file.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if "equity" in df.columns:
        eq = df["equity"].astype(float).copy()
    elif "capital" in df.columns:
        eq = df["capital"].astype(float).copy()
    elif "portfolio_value" in df.columns:
        eq = df["portfolio_value"].astype(float).copy()
    elif "gross_return" in df.columns:
        # reconstruct from returns assuming initial 1.0
        eq = (1 + df["gross_return"].astype(float)).cumprod()
    else:
        raise ValueError("No equity/portfolio_value/capital/gross_return column found.")

    df_out = pd.DataFrame({"date": df[date_col], "equity_base": eq})
    return df_out


def apply_overlay(
    eq: pd.Series,
    dates: pd.Series,
    stop_dd: float,
    stop_cooldown: int,
    harvest_step: float,
    harvest_pct: float,
) -> pd.DataFrame:
    """Return equity overlay, cash, wealth."""
    eq = eq.to_numpy()
    n = len(eq)
    equity_overlay = np.zeros(n, dtype=float)
    cash = np.zeros(n, dtype=float)

    max_seen = eq[0]
    harvested_peak = eq[0]
    cash_val = 0.0
    exposure = 1.0
    cooldown = 0

    equity_overlay[0] = eq[0]
    cash[0] = 0.0

    for i in range(1, n):
        ret = eq[i] / eq[i - 1] - 1.0

        # Update drawdown on base equity
        max_seen = max(max_seen, eq[i - 1])
        dd = (eq[i - 1] / max_seen) - 1.0

        if cooldown > 0:
            cooldown -= 1
            if cooldown == 0:
                exposure = 1.0
        else:
            if dd <= -stop_dd:
                exposure = 0.0
                cooldown = stop_cooldown

        # Apply exposure
        equity_overlay[i] = equity_overlay[i - 1] * (1 + exposure * ret)

        # Harvest rule
        if equity_overlay[i] >= harvested_peak * (1 + harvest_step):
            harvest_amt = harvest_pct * equity_overlay[i]
            equity_overlay[i] -= harvest_amt
            cash_val += harvest_amt
            harvested_peak = equity_overlay[i]  # reset reference after harvest

        cash[i] = cash_val

    wealth = equity_overlay + cash
    return pd.DataFrame(
        {
            "date": dates,
            "equity_base": eq,
            "equity_overlay": equity_overlay,
            "cash": cash,
            "wealth": wealth,
        }
    )


def max_drawdown(series: np.ndarray) -> float:
    peak = series[0]
    mdd = 0.0
    for x in series:
        peak = max(peak, x)
        mdd = min(mdd, x / peak - 1.0)
    return mdd


def sharpe(series: np.ndarray, periods_per_year=252) -> float:
    if len(series) < 2:
        return 0.0
    r = series[1:] / series[:-1] - 1.0
    std = r.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return math.sqrt(periods_per_year) * r.mean() / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-file", required=True, type=Path)
    ap.add_argument("--stop-dd", type=float, default=0.12, help="Drawdown trigger, e.g. 0.12 means -12 pct")
    ap.add_argument("--stop-cooldown", type=int, default=20, help="Cooldown days after trigger")
    ap.add_argument("--harvest-step", type=float, default=0.08, help="Incremental peak step to harvest (fraction)")
    ap.add_argument("--harvest-pct", type=float, default=0.30, help="Fraction harvested when step hit (fraction)")
    ap.add_argument("--output", type=Path, default=Path("data/processed/overlays/harvest_stop_equity.csv"))
    args = ap.parse_args()

    df = load_equity(args.equity_file)
    out = apply_overlay(
        df["equity_base"],
        df["date"],
        stop_dd=args.stop_dd,
        stop_cooldown=args.stop_cooldown,
        harvest_step=args.harvest_step,
        harvest_pct=args.harvest_pct,
    )

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    wealth = out["wealth"].to_numpy()
    equity_overlay = out["equity_overlay"].to_numpy()

    print(f"Saved overlay curve to {out_path}")
    print(f"Final wealth: {wealth[-1]:.4f} | equity: {equity_overlay[-1]:.4f} | cash: {out['cash'].iloc[-1]:.4f}")
    print(f"Harvested total: {out['cash'].iloc[-1]:.4f}")
    print(f"MDD (wealth): {max_drawdown(wealth):.4f}")
    print(f"Sharpe (wealth): {sharpe(wealth):.3f}")


if __name__ == "__main__":
    main()
