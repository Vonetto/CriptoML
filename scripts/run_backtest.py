"""CLI to execute a strategy config (V0 baseline)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crypto_alpha.backtest.engine import run_backtest  # noqa: E402
from crypto_alpha.config import load_strategy_config, load_yaml  # noqa: E402
from crypto_alpha.data.loaders import load_ohlcv  # noqa: E402
from crypto_alpha.evaluation import metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strategy backtest.")
    parser.add_argument(
        "--strategy",
        default="v0_baseline",
        help="Name of the YAML file under configs/strategy (without extension)",
    )
    return parser.parse_args()


def _output_dir(backtest_config: dict, strategy_name: str) -> Path:
    base = Path(backtest_config.get("output", {}).get("root_dir", "experiments"))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / base / strategy_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_results(result, backtest_config: dict, strategy_name: str) -> Path:
    out_dir = _output_dir(backtest_config, strategy_name)
    result.timeline.to_csv(out_dir / "equity_curve.csv", index=False)
    summary = metrics.summarize(result.timeline)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return out_dir


def main() -> None:
    args = parse_args()
    strategy = load_strategy_config(args.strategy)
    data_config = load_yaml(ROOT / strategy.references["data_config"])
    backtest_config = load_yaml(ROOT / strategy.references["backtest_config"])

    prices = load_ohlcv(data_config, base_path=ROOT)
    result = run_backtest(prices, strategy, backtest_config)
    out_dir = _save_results(result, backtest_config, strategy.name)
    summary = metrics.summarize(result.timeline)

    print(f"Strategy: {strategy.name}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"Results saved under: {out_dir}")


if __name__ == "__main__":
    main()
