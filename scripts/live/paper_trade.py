"""
Simple paper-trading simulator that consumes the latest weights CSV produced by generate_signals.py.

Assumptions:
 - Fills at the latest close price for each symbol on the weights date.
 - Costs: fixed rate (commission + slippage) applied on traded notional.
 - Marks equity at fill price (no intra-day PnL).

Outputs:
 - outputs/live/positions.json : current notional per symbol + cash
 - outputs/live/equity_curve_live.csv : date, capital, fees
 - outputs/live/trade_log.csv : date, symbol, notional_delta, fee
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys  # noqa: E402

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_alpha.config import load_yaml  # noqa: E402
from crypto_alpha.data.loaders import load_ohlcv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-file", default=None, help="weights csv; default: latest in outputs/live")
    p.add_argument("--cost-rate", type=float, default=0.0014, help="commission+slippage per notional")
    p.add_argument("--output-dir", default="outputs/live", help="directory for positions/equity/trades")
    p.add_argument(
        "--init-cash-from-weights",
        action="store_true",
        help="If positions.json missing, initialize cash from weights_latest.json capital_scaled",
    )
    return p.parse_args()


def latest_weights(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("weights_*.csv"))
    if not files:
        raise FileNotFoundError("No weights_*.csv found.")
    return files[-1]


def load_positions(path: Path) -> dict:
    if not path.exists():
        return {"cash": 0.0, "positions": {}}
    return json.loads(path.read_text())


def save_positions(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wfile = Path(args.weights_file) if args.weights_file else latest_weights(out_dir)
    weights = pd.read_csv(wfile)
    weights["date"] = pd.to_datetime(weights["date"])
    weights_date = weights["date"].iloc[0].normalize()

    # Load prices (usaremos las dos fechas más recientes para MTM y fill)
    data_cfg = load_yaml(ROOT / "configs/data/binance_daily.yaml")
    prices = load_ohlcv(data_cfg)
    prices["date"] = prices["timestamp"].dt.normalize()
    recent_dates = sorted(prices["date"].unique())
    if len(recent_dates) < 2:
        raise SystemExit("Not enough price history to mark-to-market.")
    trade_date = recent_dates[-1]          # última fecha disponible (fill/mark)
    prev_date = recent_dates[-2]           # fecha anterior (peso aplicado)
    if weights_date != prev_date:
        print(f"Warning: weights date {weights_date.date()} != prev_date {prev_date.date()}. Using latest weights anyway.")
    px_prev = prices[prices["date"] == prev_date].set_index("symbol")["close"]
    px = prices[prices["date"] == trade_date].set_index("symbol")["close"]

    # Current positions
    pos_path = out_dir / "positions.json"
    state = load_positions(pos_path)
    positions = state.get("positions", {})
    cash = float(state.get("cash", 0.0))

    # Initialize cash from weights_latest if requested and positions missing
    if args.init_cash_from_weights and cash == 0.0 and not positions:
        latest_json = out_dir / "weights_latest.json"
        if latest_json.exists():
            try:
                data = json.loads(latest_json.read_text())
                cash = float(data.get("capital_scaled", 0.0))
            except Exception:
                pass
    # If still zero cash and no positions, fall back to weights_latest capital_scaled
    if cash == 0.0 and not positions:
        latest_json = out_dir / "weights_latest.json"
        if latest_json.exists():
            try:
                data = json.loads(latest_json.read_text())
                cash = float(data.get("capital_scaled", 0.0))
            except Exception:
                cash = 0.0

    # Target notionals
    targets = dict(zip(weights["symbol"], weights["notional"]))
    symbols = set(targets.keys()) | set(positions.keys())

    trade_log = []
    fee_total = 0.0
    new_positions = {}

    # Mark-to-market existing positions from prev_date -> trade_date
    if positions:
        # compute returns
        common = set(px_prev.index) & set(px.index)
        rets = (px.loc[list(common)] / px_prev.loc[list(common)] - 1).fillna(0.0)
        for sym in list(positions.keys()):
            if sym in rets:
                positions[sym] = positions[sym] * (1 + rets.loc[sym])

    for sym in symbols:
        tgt = targets.get(sym, 0.0)
        cur = positions.get(sym, 0.0)
        delta = tgt - cur
        if delta == 0:
            new_positions[sym] = cur
            continue
        price = px.get(sym)
        if pd.isna(price):
            # skip if no price; keep old pos
            new_positions[sym] = cur
            continue
        fee = abs(delta) * args.cost_rate
        fee_total += fee
        cash -= fee  # pay fee
        new_positions[sym] = tgt
        trade_log.append({"date": trade_date, "symbol": sym, "notional_delta": delta, "fee": fee})

    # Mark equity (positions valued at fill price)
    equity = cash
    for sym, notional in new_positions.items():
        price = px.get(sym)
        if pd.isna(price):
            continue
        equity += notional  # notional already in USD terms

    # Persist
    save_positions(pos_path, {"cash": cash, "positions": new_positions, "last_date": trade_date.strftime("%Y-%m-%d")})

    # Append equity curve
    eq_path = out_dir / "equity_curve_live.csv"
    eq_df = pd.DataFrame([{"date": trade_date, "capital": equity, "fees": fee_total}])
    if eq_path.exists():
        prev = pd.read_csv(eq_path, parse_dates=["date"])
        eq_df = pd.concat([prev, eq_df]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    eq_df.to_csv(eq_path, index=False)

    # Append trade log
    tl_path = out_dir / "trade_log.csv"
    tl_df = pd.DataFrame(trade_log)
    if tl_path.exists():
        prev = pd.read_csv(tl_path, parse_dates=["date"])
        tl_df = pd.concat([prev, tl_df]).sort_values(["date", "symbol"])
    tl_df.to_csv(tl_path, index=False)

    print(f"Paper trade done for {trade_date.date()} | equity={equity:.2f} | fees={fee_total:.2f}")


if __name__ == "__main__":
    main()
