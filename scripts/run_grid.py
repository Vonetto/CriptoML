"""Sweep risk_overlay parameters for a strategy."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Sequence

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crypto_alpha.backtest.engine import run_backtest  # noqa: E402
from crypto_alpha.config import load_strategy_config, load_yaml  # noqa: E402
from crypto_alpha.data.loaders import load_ohlcv  # noqa: E402
from crypto_alpha.evaluation import metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid search over risk overlay settings.")
    parser.add_argument("--strategy", required=True, help="Base strategy YAML (without .yaml).")
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.target_vol_annual (default: config value)",
    )
    parser.add_argument(
        "--dd-trigger",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.dd_trigger (default config)",
    )
    parser.add_argument(
        "--dd-scale",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.dd_trigger_scale (default config)",
    )
    parser.add_argument(
        "--min-scale",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.min_scale (default config)",
    )
    parser.add_argument(
        "--dd-proportionality",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.dd_proportionality (proportional mode)",
    )
    parser.add_argument(
        "--max-leverage",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.max_gross_leverage (default config)",
    )
    parser.add_argument(
        "--cooldown-rate",
        nargs="+",
        type=float,
        default=[],
        help="Values for risk_overlay.cooldown_rate (default config)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV path for the summary (defaults to experiments/grids/<strategy>_<ts>.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_strategy = args.strategy
    data = load_yaml(ROOT / "configs" / "strategy" / f"{base_strategy}.yaml")

    overrides_common = data.get("risk_overlay")
    if not overrides_common or not overrides_common.get("enabled", False):
        print("Warning: strategy does not have risk_overlay enabled; grid may have no effect.")

    data_config = load_yaml(ROOT / data["references"]["data_config"])
    backtest_config = load_yaml(ROOT / data["references"]["backtest_config"])
    prices = load_ohlcv(data_config, base_path=ROOT)

    def values_or_default(values: Sequence[float], default: float | None) -> List[float | None]:
        if values:
            return list(values)
        return [default]

    base_overlay = data.get("risk_overlay", {})
    target_vals = values_or_default(args.targets, base_overlay.get("target_vol_annual"))
    dd_trigger_vals = values_or_default(args.dd_trigger, base_overlay.get("dd_trigger"))
    dd_scale_vals = values_or_default(args.dd_scale, base_overlay.get("dd_trigger_scale"))
    min_scale_vals = values_or_default(args.min_scale, base_overlay.get("min_scale"))
    dd_prop_vals = values_or_default(args.dd_proportionality, base_overlay.get("dd_proportionality"))
    cooldown_vals = values_or_default(args.cooldown_rate, base_overlay.get("cooldown_rate", 0.0))
    max_lev_vals = values_or_default(args.max_leverage, base_overlay.get("max_gross_leverage"))

    combos = list(
        product(
            target_vals,
            dd_trigger_vals,
            dd_scale_vals,
            min_scale_vals,
            dd_prop_vals,
            cooldown_vals,
            max_lev_vals,
        )
    )

    records: List[dict] = []
    for target, dd_trigger, dd_scale, min_scale, dd_prop, cooldown, max_lev in combos:
        overrides = {}
        if target is not None:
            overrides["risk_overlay.target_vol_annual"] = target
        if dd_trigger is not None:
            overrides["risk_overlay.dd_trigger"] = dd_trigger
        if dd_scale is not None:
            overrides["risk_overlay.dd_trigger_scale"] = dd_scale
        if min_scale is not None:
            overrides["risk_overlay.min_scale"] = min_scale
        if dd_prop is not None:
            overrides["risk_overlay.dd_proportionality"] = dd_prop
        if cooldown is not None:
            overrides["risk_overlay.cooldown_rate"] = cooldown
        if max_lev is not None:
            overrides["risk_overlay.max_gross_leverage"] = max_lev
        strategy = load_strategy_config(base_strategy, overrides=overrides or None)
        result = run_backtest(prices, strategy, backtest_config)
        summary = metrics.summarize(result.timeline)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "experiments" / f"{strategy.name}_grid" / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        result.timeline.to_csv(out_dir / "equity_curve.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        summary_row = {
            "strategy": strategy.name,
            "base_strategy": base_strategy,
            "target_vol_annual": strategy.risk_overlay.get("target_vol_annual") if strategy.risk_overlay else None,
            "dd_trigger": strategy.risk_overlay.get("dd_trigger") if strategy.risk_overlay else None,
            "dd_trigger_scale": strategy.risk_overlay.get("dd_trigger_scale") if strategy.risk_overlay else None,
            "min_scale": strategy.risk_overlay.get("min_scale") if strategy.risk_overlay else None,
            "dd_proportionality": strategy.risk_overlay.get("dd_proportionality") if strategy.risk_overlay else None,
            "cooldown_rate": strategy.risk_overlay.get("cooldown_rate") if strategy.risk_overlay else None,
            "max_gross_leverage": strategy.risk_overlay.get("max_gross_leverage") if strategy.risk_overlay else None,
            "experiment_dir": str(out_dir),
        }
        summary_row.update(summary)
        records.append(summary_row)
        print(
            f"target_vol={summary_row['target_vol_annual']} "
            f"dd_trigger={summary_row['dd_trigger']} "
            f"dd_scale={summary_row['dd_trigger_scale']} "
            f"min_scale={summary_row['min_scale']} "
            f"dd_prop={summary_row['dd_proportionality']} "
            f"cooldown={summary_row['cooldown_rate']} "
            f"max_lev={summary_row['max_gross_leverage']} "
            f"→ sharpe={summary.get('sharpe', float('nan')):.3f}, "
            f"vol={summary.get('annualized_vol', float('nan')):.3f}, "
            f"max_dd={summary.get('max_drawdown', float('nan')):.3f}"
        )

    summary_df = pd.DataFrame(records)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = ROOT / "experiments" / "grids" / f"{base_strategy}_{timestamp}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Grid summary saved to {output_path}")


if __name__ == "__main__":
    main()
