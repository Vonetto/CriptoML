"""Quick stats/plots for Ridge coefficient history."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Ridge coefficient history.")
    parser.add_argument(
        "--coefs-file",
        default="data/processed/predictions/v1/ridge_coefs.parquet",
        help="Parquet generated via train_v1.py --save-coefs ...",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/v1_eval/coefs",
        help="Directory to drop summary CSV/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coefs = pd.read_parquet(args.coefs_file)
    coefs["prediction_date"] = pd.to_datetime(coefs["prediction_date"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = coefs.groupby("feature")["coef"].agg(["mean", "std", "min", "max"])
    summary.to_csv(out_dir / "coef_summary.csv")
    print(summary)

    # Boxplot for coefficient distribution per feature
    fig = px.box(coefs, x="feature", y="coef", title="Ridge Coefficient Distribution")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Coefficient", template="plotly_white")
    fig.write_html(out_dir / "coef_boxplot.html")


if __name__ == "__main__":
    main()
