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


def plot_equity(csv_path: Path, output_path: Optional[Path] = None) -> Path:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"Equity curve at {csv_path} is empty")
    fig = px.line(df, x="date", y="capital", title=f"Equity Curve - {csv_path.parent.name}")
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
        "--output",
        default="",
        help="Optional path for the HTML plot (defaults next to CSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = _resolve_experiment(args.experiment)
    output = Path(args.output).expanduser().resolve() if args.output else None
    result = plot_equity(csv_path, output)
    print(f"Saved equity chart to: {result}")


if __name__ == "__main__":
    main()
