"""
Bootstrap and sub-period validation for crowd-penalty vs aggressive baseline.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_latest(exp_name: str) -> tuple[pd.Series, str]:
    runs = sorted(glob.glob(str(ROOT / "experiments" / exp_name / "*")))
    if not runs:
        raise FileNotFoundError(exp_name)
    run = runs[-1]
    tl = pd.read_csv(os.path.join(run, "timeline.csv"), parse_dates=["date"])
    daily = tl.groupby("date")["net_return"].sum()
    return daily, run


def block_bootstrap(returns: np.ndarray, block_len: int, n_samples: int) -> np.ndarray:
    n = len(returns)
    out = np.empty((n_samples, n))
    for i in range(n_samples):
        res = []
        while len(res) < n:
            start = np.random.randint(0, n - block_len + 1)
            res.extend(returns[start : start + block_len])
        out[i, :n] = res[:n]
    return out


def sharpe(arr: np.ndarray) -> float:
    std = arr.std()
    if std == 0:
        return 0.0
    return arr.mean() / std * np.sqrt(52)


def max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return dd.min()


def subperiod_stats(daily: pd.Series, splits: list[str]) -> dict:
    out = {}
    for start, end in zip(splits[:-1], splits[1:]):
        mask = (daily.index >= start) & (daily.index < end)
        r = daily.loc[mask].values
        if len(r) == 0:
            continue
        out[f"{start}_{end}"] = {
            "sharpe": sharpe(r),
            "mdd": max_drawdown(np.cumprod(1 + r)),
        }
    return out


def main():
    # use breadth run as baseline proxy (same metrics que agresivo original)
    baseline, run_base = load_latest("v2_baseline_aggressive_breadth")
    crowd, run_crowd = load_latest("v2_aggressive_crowdpen")

    # Subperiods
    splits = ["2019-11-01", "2021-01-01", "2022-07-01", "2023-12-01", "2025-07-02"]
    sub_base = subperiod_stats(baseline, splits)
    sub_crowd = subperiod_stats(crowd, splits)

    # Bootstrap
    block_len = 6
    n_samples = 300
    boot_base = block_bootstrap(baseline.values, block_len, n_samples)
    boot_crowd = block_bootstrap(crowd.values, block_len, n_samples)
    sr_base = np.array([sharpe(b) for b in boot_base])
    sr_crowd = np.array([sharpe(b) for b in boot_crowd])
    mdd_base = np.array([max_drawdown(np.cumprod(1 + b)) for b in boot_base])
    mdd_crowd = np.array([max_drawdown(np.cumprod(1 + b)) for b in boot_crowd])

    summary = {
        "baseline_run": run_base,
        "crowd_run": run_crowd,
        "subperiods": {
            "baseline": sub_base,
            "crowd": sub_crowd,
        },
        "bootstrap": {
            "sr_base_median": float(np.median(sr_base)),
            "sr_crowd_median": float(np.median(sr_crowd)),
            "sr_crowd_pctl_vs_base": float((sr_crowd > np.median(sr_base)).mean()),
            "mdd_base_median": float(np.median(mdd_base)),
            "mdd_crowd_median": float(np.median(mdd_crowd)),
            "mdd_crowd_pctl_vs_base": float((mdd_crowd > np.median(mdd_base)).mean()),
        },
    }
    out_path = ROOT / "experiments" / "v2_aggressive_crowdpen" / "validation_bootstrap.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
