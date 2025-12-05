#!/usr/bin/env bash
set -euo pipefail

# Dynamic date range: últimos 3 días hasta hoy (UTC)
START=$(/bin/date -u -v-5d +%Y-%m-%d)
END=$(/bin/date -u +%Y-%m-%d)
END_YEST=$(/bin/date -u -v-1d +%Y-%m-%d)

cd /Users/vicenteonetto/Desktop/Projects/CriptoML

VENVPY=/Users/vicenteonetto/Desktop/Projects/CriptoML/.venv/bin/python

$VENVPY scripts/etl/binance_futures.py ohlcv \
  --universe-dir data/processed/universe/v0b \
  --start "$START" --end "$END" \
  --interval 1d \
  --output-file data/processed/binance/ohlcv_1d.parquet \
  --append

$VENVPY scripts/live/generate_signals.py \
  --strategy v2_aggressive_crowdpen --capital 100 --end-date "$END_YEST"

$VENVPY scripts/live/paper_trade.py --init-cash-from-weights

# Harvest overlay (opcional, ya configurado aquí)
$VENVPY scripts/tools/harvest_stop_sim.py \
  --equity-file outputs/live/equity_curve_live.csv \
  --stop-dd 0.10 --stop-cooldown 10 \
  --harvest-step 0.30 --harvest-pct 0.05 \
  --output outputs/live/equity_curve_live_harvest.csv

# Registrar histórico de cash/wealth harvesteado (se acumula en harvest_cash_history.csv)
$VENVPY - <<'PY'
import pandas as pd, pathlib
harv = pathlib.Path("outputs/live/equity_curve_live_harvest.csv")
hist = pathlib.Path("outputs/live/harvest_cash_history.csv")
if harv.exists():
    df = pd.read_csv(harv)
    if len(df):
        row = df.iloc[-1]
        out = pd.DataFrame([{
            "run_date": pd.to_datetime(row["date"]).date(),
            "wealth": row.get("wealth", row.get("equity_overlay", 0) + row.get("cash", 0)),
            "equity_overlay": row.get("equity_overlay", 0),
            "cash": row.get("cash", 0),
        }])
        hist.parent.mkdir(parents=True, exist_ok=True)
        if hist.exists():
            out.to_csv(hist, mode="a", header=False, index=False)
        else:
            out.to_csv(hist, index=False)
PY
