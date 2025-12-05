"""
Generate end-of-day target weights for a given strategy using the experimental engine.
Outputs a CSV under outputs/live/weights_<YYYY-MM-DD>.csv with columns:
  date, symbol, weight, notional

Example:
  python scripts/live/generate_signals.py --strategy v2_aggressive_crowdpen --capital 10000
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys  # noqa: E402

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_alpha.config import load_strategy_config, load_yaml  # noqa: E402
from crypto_alpha.data.loaders import load_ohlcv  # noqa: E402
from crypto_alpha.backtest import engine_experimental as engine  # noqa: E402
from crypto_alpha.evaluation import metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, help="strategy name (configs/strategy/<name>.yaml)")
    p.add_argument("--capital", type=float, default=100.0, help="capital to scale weights")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (defaults to latest available)")
    p.add_argument("--output-dir", default="outputs/live", help="directory to save weights")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = load_strategy_config(args.strategy)
    data_cfg = load_yaml(ROOT / strat.references["data_config"])
    bt_cfg = load_yaml(ROOT / strat.references["backtest_config"])
    prices = load_ohlcv(data_cfg)

    # Determine end date
    max_dt = prices["timestamp"].max().normalize()
    # Por defecto usamos el penúltimo día disponible para evitar look‑ahead.
    if args.end_date:
        end_dt = pd.to_datetime(args.end_date)
    else:
        end_dt = max_dt - pd.Timedelta(days=1)
    end_dt = min(end_dt, max_dt)
    start_dt = pd.to_datetime(strat.backtest["start_date"])

    # Override dates and rebalance freq kept from strategy
    strat.backtest["start_date"] = start_dt.strftime("%Y-%m-%d")
    strat.backtest["end_date"] = end_dt.strftime("%Y-%m-%d")

    result = engine.run_backtest(prices, strat, bt_cfg)
    tl = result.timeline.copy()
    if tl.empty:
        raise SystemExit("Timeline empty; check data / dates.")
    weights_map = result.weights_history or {}
    if not weights_map:
        raise SystemExit("weights_history empty; engine did not produce weights.")

    # Elige la fecha de pesos más reciente <= end_dt (para evitar desalineación)
    dates_sorted = sorted(weights_map.keys())
    candidates = [d for d in dates_sorted if d <= end_dt]
    if candidates:
        last_date = candidates[-1]
    else:
        last_date = dates_sorted[-1]
    weights = weights_map.get(last_date)
    if not weights:
        raise SystemExit(f"No weights for last date {last_date}")

    # Capital base: si existe equity_curve_live, usa su último capital; de lo contrario usa args.capital
    capital_base = args.capital
    live_eq = Path("outputs/live/equity_curve_live.csv")
    if live_eq.exists():
        try:
            live_df = pd.read_csv(live_eq, parse_dates=["date"])
            if len(live_df):
                capital_base = float(live_df["capital"].iloc[-1])
        except Exception:
            pass

    capital_scaled = capital_base
    df = pd.DataFrame(list(weights.items()), columns=["symbol", "weight"])

    # Normalizar gross a 1.0 para evitar palanca accidental
    gross = df["weight"].abs().sum()
    if gross > 1.0:
        df["weight"] = df["weight"] / gross

    df["notional"] = df["weight"] * capital_scaled
    df.insert(0, "date", pd.to_datetime(last_date).date())

    out_path = out_dir / f"weights_{last_date.strftime('%Y-%m-%d')}.csv"
    df.to_csv(out_path, index=False)

    # Save summary
    mt = metrics.summarize(tl)
    summary = {
        "date": last_date.strftime("%Y-%m-%d"),
        "capital_scaled": capital_scaled,
        "weights_file": str(out_path),
        **{k: float(v) for k, v in mt.items()},
    }
    (out_dir / "weights_latest.json").write_text(pd.Series(summary).to_json())
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    main()
