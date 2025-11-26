"""Rolling Ridge Regression trainer for cross-sectional predictions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _cross_sectional_standardize(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby("date")
    for col in columns:
        df[col] = grouped[col].transform(_zscore)
    return df


@dataclass
class RollingRidgeTrainer:
    feature_columns: Sequence[str]
    alpha: float = 10.0
    train_window: int = 104  # number of rebalance periods (~weeks)
    min_train_rows: int = 2000
    dropna: bool = True
    collect_coefs: bool = False
    coef_history: List[dict] = field(default_factory=list, init=False)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        features = X.shape[1]
        penalty = self.alpha * np.eye(features)
        XtX = X.T @ X + penalty
        Xty = X.T @ y
        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(XtX) @ Xty
        return coef

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        required = set(self.feature_columns) | {"target_excess_return_5d", "date", "symbol"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Feature dataset missing columns: {missing}")

        working = df.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        working = working.sort_values(["date", "symbol"]).reset_index(drop=True)
        working = working.dropna(subset=self.feature_columns + ["target_excess_return_5d"])
        working = _cross_sectional_standardize(working, self.feature_columns)

        dates = working["date"].unique()
        if len(dates) <= self.train_window:
            raise ValueError("Not enough dates to run rolling training.")

        if self.collect_coefs:
            self.coef_history.clear()

        predictions = []
        for idx in range(self.train_window, len(dates)):
            test_date = dates[idx]
            train_dates = dates[idx - self.train_window : idx]
            mask = working["date"].isin(train_dates)
            train = working[mask]
            if len(train) < self.min_train_rows:
                logger.debug(
                    "Skipping %s: only %d training rows (<%d)",
                    test_date.date(),
                    len(train),
                    self.min_train_rows,
                )
                continue
            X_train = train[self.feature_columns].to_numpy()
            y_train = train["target_excess_return_5d"].to_numpy()
            coef = self._fit(X_train, y_train)

            test = working[working["date"] == test_date].copy()
            X_test = test[self.feature_columns].to_numpy()
            preds = X_test @ coef
            entry = pd.DataFrame(
                {
                    "date": test_date,
                    "symbol": test["symbol"].values,
                    "prediction": preds,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "n_train_rows": len(train),
                }
            )
            predictions.append(entry)
            if (idx - self.train_window) % 26 == 0:
                logger.info(
                    "Predicted %s (train %s → %s, samples=%d)",
                    test_date.date(),
                    train_dates[0].date(),
                    train_dates[-1].date(),
                    len(train),
                )
            if self.collect_coefs:
                for feature, weight in zip(self.feature_columns, coef):
                    self.coef_history.append(
                        {
                            "prediction_date": test_date,
                            "train_start": train_dates[0],
                            "train_end": train_dates[-1],
                            "feature": feature,
                            "coef": float(weight),
                            "n_train_rows": len(train),
                        }
                    )

        if not predictions:
            raise RuntimeError("No predictions generated. Check training window/min rows thresholds.")

        result = pd.concat(predictions, ignore_index=True)
        return result


__all__ = ["RollingRidgeTrainer"]
