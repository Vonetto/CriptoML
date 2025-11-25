"""Plot equity curves from experiment outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px


def _resolve_experiment(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        csv_path = path / "equity_curve.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Directory {path} does not contain equity_curve.csv")
        return csv_path
    if path.suffix == ".csv":
        return path
    raise FileNotFoundError(f"Provided path {path} is neither a directory nor a CSV file")


def _btc_buy_hold_curve(
    parquet_path: Path,
    dates: pd.Series,
    base_capital: float,
    symbol: str = "BTCUSDT",
) -> pd.Series:
    target_index = pd.to_datetime(dates).dt.normalize()
    df = pd.read_parquet(parquet_path, columns=["timestamp", "symbol", "close"])
    btc = df[df["symbol"] == symbol].copy()
    if btc.empty:
        raise ValueError(f"{symbol} not found in {parquet_path}")
    btc["date"] = pd.to_datetime(btc["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    window = btc[(btc["date"] >= target_index.min()) & (btc["date"] <= target_index.max())]
    if window.empty:
        raise ValueError(
            f"No {symbol} data found between {target_index.min()} and {target_index.max()} in {parquet_path}"
        )
    window = window.sort_values("date").set_index("date")["close"]
    aligned = window.reindex(target_index, method="ffill")
    first_valid = aligned.first_valid_index()
    if first_valid is None:
        raise ValueError(f"{symbol} curve could not be aligned to equity dates.")
    normalized = aligned / aligned.loc[first_valid]
    normalized.index = target_index
    return normalized * base_capital


def plot_equity(
    csv_path: Path,
    output_path: Optional[Path] = None,
    btc_parquet: Optional[Path] = None,
    btc_symbol: str = "BTCUSDT",
) -> Path:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"Equity curve at {csv_path} is empty")
    plot_df = df[["date", "capital"]].copy()
    plot_df["series"] = "strategy"

    if btc_parquet:
        btc_curve = _btc_buy_hold_curve(btc_parquet, df["date"], float(df["capital"].iloc[0]), btc_symbol)
        btc_df = pd.DataFrame(
            {"date": btc_curve.index, "capital": btc_curve.values, "series": f"{btc_symbol} buy&hold"}
        )
        plot_df = pd.concat([plot_df, btc_df], ignore_index=True)

    fig = px.line(
        plot_df,
        x="date",
        y="capital",
        color="series",
        title=f"Equity Curve - {csv_path.parent.name}",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Capital",
        template="plotly_white",
    )
    target = output_path or csv_path.with_suffix(".html")
    fig.write_html(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot equity curve for a given experiment.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment directory or equity_curve.csv",
    )
    parser.add_argument(
        "--btc-parquet",
        default="",
        help="Optional path to OHLCV parquet with BTCUSDT prices to add buy&hold curve",
    )
    parser.add_argument(
        "--btc-symbol",
        default="BTCUSDT",
        help="Symbol to use for buy&hold comparison (default: BTCUSDT)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path for the HTML plot (defaults next to CSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = _resolve_experiment(args.experiment)
    output = Path(args.output).expanduser().resolve() if args.output else None
    btc_parquet = Path(args.btc_parquet).expanduser().resolve() if args.btc_parquet else None
    result = plot_equity(csv_path, output, btc_parquet=btc_parquet, btc_symbol=args.btc_symbol)
    print(f"Saved equity chart to: {result}")


if __name__ == "__main__":
    main()
