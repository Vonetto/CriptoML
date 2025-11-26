"""Train the V1 rolling Ridge model and emit PIT predictions."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_alpha.data.etl.features_v1 import FEATURE_COLUMNS  # noqa: E402
from crypto_alpha.models import RollingRidgeTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rolling Ridge model for V1.")
    parser.add_argument("--features-file", required=True, help="Parquet with engineered features.")
    parser.add_argument(
        "--output-file",
        default="data/processed/predictions/v1/ridge_predictions.parquet",
        help="Destination for the predictions parquet.",
    )
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge penalty alpha.")
    parser.add_argument(
        "--train-window",
        type=int,
        default=104,
        help="Rolling window size in rebalance periods (default: 104 ≈ 2 años).",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=2000,
        help="Minimum rows required to fit each window.",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated override for feature columns (default uses FEATURE_COLUMNS).",
    )
    parser.add_argument(
        "--save-coefs",
        default="",
        help="Optional parquet path to save rolling coefficients history.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    log = logging.getLogger("train_v1")

    feature_cols = (
        [col.strip() for col in args.features.split(",") if col.strip()]
        if args.features
        else FEATURE_COLUMNS
    )
    log.info("Loading features from %s", args.features_file)
    df = pd.read_parquet(args.features_file)
    trainer = RollingRidgeTrainer(
        feature_columns=feature_cols,
        alpha=args.alpha,
        train_window=args.train_window,
        min_train_rows=args.min_train_rows,
        collect_coefs=bool(args.save_coefs),
    )
    preds = trainer.run(df)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(output_path, index=False)
    log.info(
        "Saved predictions to %s (%d dates, %d rows)",
        output_path,
        preds['date'].nunique(),
        len(preds),
    )
    if args.save_coefs:
        coef_path = Path(args.save_coefs)
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        coef_df = pd.DataFrame(trainer.coef_history)
        coef_df.to_parquet(coef_path, index=False)
        log.info("Saved coefficient history to %s (%d rows)", coef_path, len(coef_df))


if __name__ == "__main__":
    main()
