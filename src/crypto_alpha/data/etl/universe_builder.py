"""Universe-building utilities for Binance Futures datasets."""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from crypto_alpha.data.exchanges import BinanceFuturesClient, CoinMarketCapClient


logger = logging.getLogger(__name__)


STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "TUSD"}
FIAT_BASES = {"BRL", "EUR", "TRY", "GBP", "AUD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def _is_leveraged_token(symbol: str) -> bool:
    base = symbol.replace("USDT", "")
    return any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES)


def _should_skip(base_asset: str, symbol: str) -> bool:
    upper = base_asset.upper()
    if upper in STABLECOINS:
        return True
    if upper in FIAT_BASES:
        return True
    if _is_leveraged_token(symbol.upper()):
        return True
    return False


def build_universe_v0a(
    client: BinanceFuturesClient,
    min_volume_usd: float = 10_000_000,
    top_n: int = 40,
    output_path: Path | str = "data/processed/universe/binance_v0a_latest.csv",
) -> pd.DataFrame:
    """Build the provisional (V0a) universe using 24h volumes."""

    contracts = pd.DataFrame(client.list_perpetual_contracts())
    if contracts.empty:
        raise RuntimeError("Unable to list Binance contracts.")
    contracts = contracts[contracts["status"] == "TRADING"]
    contracts = contracts[["symbol", "baseAsset", "quoteAsset"]]

    tickers = client.fetch_24h_tickers()
    tickers = tickers[tickers["symbol"].isin(contracts["symbol"])]
    tickers = tickers.rename(columns={"quote_volume": "volume_24h_usd"})

    merged = contracts.merge(tickers[["symbol", "volume_24h_usd"]], on="symbol", how="left")
    merged["volume_24h_usd"] = merged["volume_24h_usd"].fillna(0)
    merged = merged[merged["volume_24h_usd"] >= min_volume_usd]
    merged = merged[~merged.apply(lambda row: _should_skip(row.baseAsset, row.symbol), axis=1)]
    merged.sort_values("volume_24h_usd", ascending=False, inplace=True)
    merged["rank"] = range(1, len(merged) + 1)
    selection = merged.head(top_n).reset_index(drop=True)
    selection["as_of"] = pd.Timestamp.utcnow().normalize()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_path, index=False)
    return selection


def _candidate_contracts(
    contracts: pd.DataFrame,
    allowed_bases: Iterable[str] | None = None,
) -> List[str]:
    df = contracts.copy()
    if allowed_bases is not None:
        allowed = {base.upper() for base in allowed_bases}
        df = df[df["baseAsset"].str.upper().isin(allowed)]
    df = df[~df.apply(lambda row: _should_skip(row.baseAsset, row.symbol), axis=1)]
    return df["symbol"].tolist()


def _rolling_metrics(
    client: BinanceFuturesClient,
    symbol: str,
    window_start: datetime,
    window_end: datetime,
    min_points: int,
) -> dict | None:
    ohlcv = client.fetch_klines(symbol, start=window_start, end=window_end, interval="1d")
    if ohlcv.empty or len(ohlcv) < min_points:
        logger.debug("Skipping %s: insufficient OHLCV history", symbol)
        return None
    volume = ohlcv["volume_quote"].mean()
    trades = ohlcv.get("trade_count")
    avg_trades = trades.mean() if trades is not None else None
    return {
        "symbol": symbol,
        "avg_volume_30d": volume,
        "avg_trades_30d": avg_trades,
    }


def _resolve_allowed_bases(
    cmc_client: CoinMarketCapClient | None,
    as_of: datetime,
    pool_size: int,
) -> List[str] | None:
    if cmc_client is None:
        return None
    try:
        symbols = cmc_client.top_market_cap_symbols(as_of, limit=pool_size)
    except RuntimeError as exc:
        logger.warning("CoinMarketCap request failed (%s). Proceeding without pre-filter.", exc)
        return None
    return symbols or None


def build_universe_v0b(
    client: BinanceFuturesClient,
    start: datetime,
    end: datetime,
    lookback_days: int = 30,
    min_volume_usd: float = 10_000_000,
    top_n: int = 40,
    output_dir: Path | str = "data/processed/universe/v0b",
    cmc_client: CoinMarketCapClient | None = None,
    pool_size: int = 120,
    cache_metrics: bool = True,
    cache_dir: Path | str | None = None,
    resume: bool = True,
    status_log: Path | str | None = None,
) -> List[Path]:
    """Build monthly point-in-time universes ranked by liquidity."""

    contracts = pd.DataFrame(client.list_perpetual_contracts())
    contracts = contracts[contracts["status"] == "TRADING"]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path: Path | None = None
    if cache_metrics:
        cache_path = Path(cache_dir) if cache_dir else output_root / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
    status_path = Path(status_log) if status_log else output_root / "universe_status.csv"
    if not status_path.exists():
        _init_status_log(status_path)

    anchors = pd.date_range(start=start, end=end, freq="MS")
    saved_paths: List[Path] = []

    for as_of in tqdm(anchors, desc="Universe months", unit="month"):
        outfile = output_root / f"universe_{as_of:%Y-%m-%d}.csv"
        if resume and outfile.exists():
            logger.info("Skipping %s (already exists)", as_of.strftime("%Y-%m"))
            saved_paths.append(outfile)
            _append_status_row(
                status_path,
                as_of,
                pool_size=0,
                metrics_count=0,
                selected_count=0,
                status="skipped",
                note="file_exists",
            )
            continue
        window_end = as_of - pd.Timedelta(days=1)
        window_start = window_end - pd.Timedelta(days=lookback_days - 1)
        allowed_bases = _resolve_allowed_bases(cmc_client, as_of.to_pydatetime(), pool_size)
        symbols = _candidate_contracts(contracts, allowed_bases)
        if not symbols:
            logger.warning(
                "No contracts available for %s (allowed=%s)",
                as_of.strftime("%Y-%m"),
                "all" if allowed_bases is None else len(allowed_bases),
            )
            _append_status_row(
                status_path,
                as_of,
                pool_size=0,
                metrics_count=0,
                selected_count=0,
                status="no_symbols",
                note="filters_removed_all",
            )
            continue
        df = _load_cached_metrics(cache_path, as_of, lookback_days)
        if df is None:
            measurements: List[dict] = []
            for symbol in tqdm(
                symbols,
                desc=f"Universe {as_of:%Y-%m}",
                leave=False,
            ):
                metrics = _rolling_metrics(
                    client,
                    symbol,
                    window_start.to_pydatetime(),
                    (window_end + pd.Timedelta(days=1)).to_pydatetime(),
                    min_points=lookback_days,
                )
                if metrics is None:
                    continue
                measurements.append(metrics)

            if not measurements:
                logger.warning(
                    "No liquidity metrics for %s (pool=%d). Check data availability or thresholds.",
                    as_of.strftime("%Y-%m"),
                    len(symbols),
                )
                _append_status_row(
                    status_path,
                    as_of,
                    pool_size=len(symbols),
                    metrics_count=0,
                    selected_count=0,
                    status="no_metrics",
                    note="insufficient_history",
                )
                continue
            df = pd.DataFrame(measurements)
            df["as_of"] = as_of
            df["lookback_days"] = lookback_days
            if cache_path is not None:
                _save_cached_metrics(cache_path, as_of, df)
        else:
            logger.info("Using cached metrics for %s", as_of.strftime("%Y-%m"))

        logger.info(
            "%s: metrics computed for %d/%d symbols",
            as_of.strftime("%Y-%m"),
            len(df),
            len(symbols),
        )
        filtered = df[df["avg_volume_30d"] >= min_volume_usd].copy()
        if filtered.empty:
            logger.warning(
                "%s: all %d symbols fell below threshold (vol>=%.0f)",
                as_of.strftime("%Y-%m"),
                len(df),
                min_volume_usd,
            )
            _append_status_row(
                status_path,
                as_of,
                pool_size=len(symbols),
                metrics_count=len(df),
                selected_count=0,
                status="filtered_out",
                note="volume_threshold",
            )
            continue
        filtered["liquidity_score"] = filtered["avg_volume_30d"]
        filtered.sort_values(["avg_volume_30d"], ascending=False, inplace=True)
        filtered["rank"] = range(1, len(filtered) + 1)
        selection = filtered.head(top_n).copy()
        selection["as_of"] = as_of
        selection.to_csv(outfile, index=False)
        logger.info(
            "Universe %s saved with %d symbols (pool=%d)",
            as_of.strftime("%Y-%m"),
            len(selection),
            len(symbols),
        )
        _append_status_row(
            status_path,
            as_of,
            pool_size=len(symbols),
            metrics_count=len(df),
            selected_count=len(selection),
            status="saved",
            note="",
        )
        saved_paths.append(outfile)
    return saved_paths


def _init_status_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "as_of",
                "pool_size",
                "metrics_count",
                "selected_count",
                "status",
                "note",
            ],
        )
        writer.writeheader()


def _append_status_row(
    path: Path,
    as_of: pd.Timestamp,
    pool_size: int,
    metrics_count: int,
    selected_count: int,
    status: str,
    note: str,
) -> None:
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "as_of",
                "pool_size",
                "metrics_count",
                "selected_count",
                "status",
                "note",
            ],
        )
        writer.writerow(
            {
                "as_of": as_of.strftime("%Y-%m-%d"),
                "pool_size": pool_size,
                "metrics_count": metrics_count,
                "selected_count": selected_count,
                "status": status,
                "note": note,
            }
        )


def _metrics_cache_path(cache_dir: Path, as_of: pd.Timestamp) -> Path:
    return cache_dir / f"metrics_{as_of:%Y-%m-%d}.parquet"


def _load_cached_metrics(
    cache_dir: Path | None,
    as_of: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame | None:
    if cache_dir is None:
        return None
    path = _metrics_cache_path(cache_dir, as_of)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not df.empty and "lookback_days" in df.columns:
        cached = int(df["lookback_days"].iloc[0])
        if cached != lookback_days:
            logger.info(
                "Ignoring cache for %s due to lookback mismatch (%s vs %s)",
                as_of.strftime("%Y-%m"),
                cached,
                lookback_days,
            )
            return None
    return df


def _save_cached_metrics(cache_dir: Path, as_of: pd.Timestamp, df: pd.DataFrame) -> None:
    path = _metrics_cache_path(cache_dir, as_of)
    df.to_parquet(path, index=False)


__all__ = ["build_universe_v0a", "build_universe_v0b"]
