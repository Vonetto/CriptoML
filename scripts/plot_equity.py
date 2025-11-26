"""Plot equity curves from experiment outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px


def _resolve_experiment(entry: str) -> tuple[str, Path]:
    label = ""
    path_str = entry
    if "=" in entry:
        label, path_str = entry.split("=", 1)
        label = label.strip()
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        csv_path = path / "equity_curve.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Directory {path} does not contain equity_curve.csv")
        if not label:
            label = path.name
        return label or csv_path.parent.name, csv_path
    if path.suffix == ".csv":
        if not label:
            label = path.parent.name if path.parent.name else path.stem
        return label or path.stem, path
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
    experiments: List[Tuple[str, Path]],
    output_path: Optional[Path] = None,
    btc_parquet: Optional[Path] = None,
    btc_symbol: str = "BTCUSDT",
) -> Path:
    if not experiments:
        raise ValueError("At least one experiment must be provided.")
    frames: List[pd.DataFrame] = []
    base_dates: Optional[pd.Series] = None
    base_capital: Optional[float] = None

    for label, csv_path in experiments:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if df.empty:
            raise ValueError(f"Equity curve at {csv_path} is empty")
        tmp = df[["date", "capital"]].copy()
        tmp["series"] = label or csv_path.parent.name
        frames.append(tmp)
        if base_dates is None:
            base_dates = df["date"]
            base_capital = float(df["capital"].iloc[0])

    if btc_parquet and base_dates is not None and base_capital is not None:
        btc_curve = _btc_buy_hold_curve(btc_parquet, base_dates, base_capital, btc_symbol)
        btc_df = pd.DataFrame(
            {"date": btc_curve.index, "capital": btc_curve.values, "series": f"{btc_symbol} buy&hold"}
        )
        frames.append(btc_df)

    plot_df = pd.concat(frames, ignore_index=True)

    fig = px.line(
        plot_df,
        x="date",
        y="capital",
        color="series",
        title="Equity Curve Comparison",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Capital",
        template="plotly_white",
    )
    if output_path:
        target = output_path
    else:
        if len(experiments) == 1:
            target = experiments[0][1].with_suffix(".html")
        else:
            target = experiments[0][1].parent / "equity_compare.html"
    fig.write_html(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot equity curve for a given experiment.")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Path to experiment directory or equity_curve.csv. Optional label=path syntax.",
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
    experiments = [_resolve_experiment(entry) for entry in args.experiment]
    output = Path(args.output).expanduser().resolve() if args.output else None
    btc_parquet = Path(args.btc_parquet).expanduser().resolve() if args.btc_parquet else None
    result = plot_equity(experiments, output, btc_parquet=btc_parquet, btc_symbol=args.btc_symbol)
    print(f"Saved equity chart to: {result}")


if __name__ == "__main__":
    main()
