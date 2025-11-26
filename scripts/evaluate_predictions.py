"""Evaluate Ridge predictions vs baseline momentum signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IC/RankIC and top-k stats for predictions.")
    parser.add_argument(
        "--features",
        default="data/processed/features/v1/features.parquet",
        help="Path to features parquet (requires forward_return_5d, ret_7d, ret_30d).",
    )
    parser.add_argument(
        "--predictions",
        default="data/processed/predictions/v1/ridge_predictions.parquet",
        help="Path to Ridge predictions parquet.",
    )
    parser.add_argument(
        "--selection-pct",
        type=float,
        default=0.2,
        help="Top percentile for bucket comparison (default 0.2).",
    )
    parser.add_argument("--per-date-csv", default="", help="Optional CSV for per-date IC statistics.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary output.")
    return parser.parse_args()


def _corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    if len(series_a) < 3:
        return np.nan
    return series_a.corr(series_b, method=method)


def _rankic(group: pd.DataFrame, col: str) -> float:
    return _corr(group[col], group["forward_return_5d"], method="spearman")


def _ic(group: pd.DataFrame, col: str) -> float:
    return _corr(group[col], group["forward_return_5d"], method="pearson")


def _top_bucket_return(group: pd.DataFrame, col: str, pct: float) -> float:
    if group[col].isna().all():
        return np.nan
    n = max(int(len(group) * pct), 1)
    ranked = group.sort_values(col, ascending=False).head(n)
    return ranked["forward_return_5d"].mean()


def _aggregate_stats(series: pd.Series) -> Dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {"mean": np.nan, "std": np.nan, "t_stat": np.nan, "count": 0}
    mean = series.mean()
    std = series.std(ddof=1)
    t_stat = mean / (std / np.sqrt(len(series))) if std > 0 else np.nan
    return {"mean": mean, "std": std, "t_stat": t_stat, "count": int(len(series))}


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features)
    preds = pd.read_parquet(args.predictions)

    features["date"] = pd.to_datetime(features["date"])
    preds["date"] = pd.to_datetime(preds["date"])

    merged = features.merge(
        preds[["date", "symbol", "prediction"]],
        on=["date", "symbol"],
        how="inner",
    )
    merged = merged.dropna(subset=["forward_return_5d"])
    if merged.empty:
        raise ValueError("Merge of features and predictions is empty.")

    records: List[dict] = []
    for date, group in merged.groupby("date"):
        rec = {
            "date": date,
            "rankic_pred": _rankic(group, "prediction"),
            "ic_pred": _ic(group, "prediction"),
            "rankic_ret7d": _rankic(group, "ret_7d"),
            "rankic_ret30d": _rankic(group, "ret_30d"),
            "ic_ret7d": _ic(group, "ret_7d"),
            "ic_ret30d": _ic(group, "ret_30d"),
            "top_pred_return": _top_bucket_return(group, "prediction", args.selection_pct),
            "top_ret30d_return": _top_bucket_return(group, "ret_30d", args.selection_pct),
        }
        rec["top_return_diff"] = rec["top_pred_return"] - rec["top_ret30d_return"]
        records.append(rec)

    per_date = pd.DataFrame(records).sort_values("date")

    summary = {
        "selection_pct": args.selection_pct,
        "pred_rankic": _aggregate_stats(per_date["rankic_pred"]),
        "pred_ic": _aggregate_stats(per_date["ic_pred"]),
        "ret7d_rankic": _aggregate_stats(per_date["rankic_ret7d"]),
        "ret30d_rankic": _aggregate_stats(per_date["rankic_ret30d"]),
        "top_pred_return": _aggregate_stats(per_date["top_pred_return"]),
        "top_ret30d_return": _aggregate_stats(per_date["top_ret30d_return"]),
        "top_return_diff": _aggregate_stats(per_date["top_return_diff"]),
    }

    print("=== Prediction vs Baseline Evaluation ===")
    print(json.dumps(summary, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))

    if args.per_date_csv:
        out_path = Path(args.per_date_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_date.to_csv(out_path, index=False)
        print(f"Per-date stats saved to {out_path}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
