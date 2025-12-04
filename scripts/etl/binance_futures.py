"""CLI utilities to run Binance Futures ETL pipelines."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crypto_alpha.data.etl import (
    build_features_v1,
    build_features_v2,
    build_universe_v0a,
    build_universe_v0b,
    build_btc_regime,
    build_macro_regime,
    download_ohlcv,
    download_open_interest,
    download_funding,
)
from crypto_alpha.data.exchanges import BinanceFuturesClient, CoinMarketCapClient


def _parse_date(value: str) -> datetime:
    return pd.Timestamp(value, tz="UTC").to_pydatetime()


def _load_symbols_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"File {path} missing 'symbol' column")
    return df["symbol"].astype(str).str.upper().tolist()


def _load_symbols_from_dir(directory: str) -> List[str]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Universe directory not found: {directory}")
    symbols: set[str] = set()
    for csv_path in sorted(root.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=["symbol"])
        except ValueError:
            # Skip auxiliary files like universe_status.csv
            continue
        symbols.update(df["symbol"].astype(str).str.upper().tolist())
    if not symbols:
        raise ValueError(f"No symbols found under {directory}")
    return sorted(symbols)


def _resolve_symbols(args, client: BinanceFuturesClient) -> List[str]:
    if args.symbols:
        return [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    if args.universe_csv:
        return _load_symbols_from_csv(args.universe_csv)
    if args.universe_dir:
        return _load_symbols_from_dir(args.universe_dir)
    if args.all_contracts:
        contracts = client.list_perpetual_contracts()
        return [c["symbol"].upper() for c in contracts]
    raise ValueError("Provide --symbols, --universe-csv, --universe-dir or --all-contracts")


def ohlcv_command(args) -> None:
    client = BinanceFuturesClient()
    symbols = _resolve_symbols(args, client)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    download_ohlcv(
        symbols,
        start=start,
        end=end,
        interval=args.interval,
        output_dir=args.output_dir,
        output_file=args.output_file or None,
        client=client,
        append=args.append,
    )
    client.close()


def open_interest_command(args) -> None:
    client = BinanceFuturesClient()
    symbols = _resolve_symbols(args, client)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    download_open_interest(
        symbols,
        start=start,
        end=end,
        period=args.period,
        output_dir=args.output_dir,
        output_file=args.output_file or None,
        client=client,
    )
    client.close()

def funding_command(args) -> None:
    client = BinanceFuturesClient()
    symbols = _resolve_symbols(args, client)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    download_funding(
        symbols,
        start=start,
        end=end,
        output_dir=args.output_dir,
        output_file=args.output_file or None,
        client=client,
    )
    client.close()

def btc_regime_command(args) -> None:
    build_btc_regime(
        ohlcv_path=args.ohlcv_file,
        funding_path=args.funding_file,
        output_path=args.output_file,
        symbol=args.symbol,
        vol_threshold=args.vol_threshold,
        dd_threshold=args.dd_threshold,
        funding_threshold=args.funding_threshold,
        entry_days=args.entry_days,
        exit_days=args.exit_days,
        hl_threshold=args.hl_threshold,
        ret_30d_threshold=args.ret_30d_threshold,
    )


def macro_regime_command(args) -> None:
    build_macro_regime(
        output_path=args.output_file,
        start=args.start,
        end=args.end,
        api_key=args.api_key,
        vol_threshold=args.vol_threshold,
        dxy_threshold=args.dxy_threshold,
        ret_30d_threshold=args.ret_30d_threshold,
        entry_days=args.entry_days,
        exit_days=args.exit_days,
    )


def universe_v0a_command(args) -> None:
    client = BinanceFuturesClient()
    build_universe_v0a(
        client,
        min_volume_usd=args.min_volume,
        top_n=args.top_n,
        output_path=args.output_path,
    )
    client.close()


def universe_v0b_command(args) -> None:
    client = BinanceFuturesClient()
    cmc = None
    if args.use_cmc:
        cmc = CoinMarketCapClient(api_key=args.cmc_key or None)
    cache_dir = None if args.no_cache else (args.cache_dir or None)
    build_universe_v0b(
        client,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        lookback_days=args.lookback,
        min_volume_usd=args.min_volume,
        active_vol_threshold=args.active_vol_threshold,
        min_active_days=args.min_active_days,
        min_trades=args.min_trades if args.min_trades > 0 else None,
        filter_fake_volume=args.filter_fake_volume,
        min_trades_filter=args.min_trades_filter,
        max_vol_per_trade=args.max_vol_per_trade,
        top_n=args.top_n,
        output_dir=args.output_dir,
        cmc_client=cmc,
        pool_size=args.pool_size,
        cache_metrics=not args.no_cache,
        cache_dir=cache_dir,
        resume=not args.no_resume,
        status_log=args.status_log or None,
    )
    client.close()
    if cmc:
        cmc.close()


def features_v1_command(args) -> None:
    build_features_v1(
        ohlcv_path=args.ohlcv_file,
        universe_dir=args.universe_dir,
        output_path=args.output_file,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        rebalance_freq=args.rebalance_freq,
        active_vol_threshold=args.active_vol_threshold,
    )

def features_v2_command(args) -> None:
    build_features_v2(
        ohlcv_path=args.ohlcv_file,
        funding_path=args.funding_file,
        universe_dir=args.universe_dir,
        output_path=args.output_file,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        rebalance_freq=args.rebalance_freq,
        active_vol_threshold=args.active_vol_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance Futures ETL helper")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for ETL diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbols", help="Comma-separated list of symbols", default="")
    common.add_argument("--universe-csv", help="CSV containing a symbol column", default="")
    common.add_argument(
        "--universe-dir",
        help="Directory with PIT CSVs (union of symbols will be used)",
        default="",
    )
    common.add_argument(
        "--all-contracts", action="store_true", help="Use every USDT perp currently trading"
    )

    ohlcv = subparsers.add_parser(
        "ohlcv", parents=[common], help="Download OHLCV data from Binance Futures"
    )
    ohlcv.add_argument("--start", required=True)
    ohlcv.add_argument("--end", required=True)
    ohlcv.add_argument("--interval", default="1d")
    ohlcv.add_argument("--output-dir", default="data/raw/binance_futures/ohlcv")
    ohlcv.add_argument("--output-file", default="")
    ohlcv.add_argument(
        "--append",
        action="store_true",
        help="Si existe output-file, concatena y deduplica por timestamp+symbol en lugar de sobrescribir",
    )
    ohlcv.set_defaults(func=ohlcv_command)

    oi = subparsers.add_parser(
        "open-interest", parents=[common], help="Download open interest history"
    )
    oi.add_argument("--start", required=True)
    oi.add_argument("--end", required=True)
    oi.add_argument("--period", default="1d")
    oi.add_argument("--output-dir", default="data/raw/binance_futures/open_interest")
    oi.add_argument("--output-file", default="")
    oi.set_defaults(func=open_interest_command)

    funding = subparsers.add_parser(
        "funding", parents=[common], help="Download funding rates (8h aggregated to 1d)"
    )
    funding.add_argument("--start", required=True)
    funding.add_argument("--end", required=True)
    funding.add_argument("--output-dir", default="data/raw/binance_futures/funding")
    funding.add_argument("--output-file", default="")
    funding.set_defaults(func=funding_command)

    v0a = subparsers.add_parser("universe-v0a", help="Build provisional liquidity universe")
    v0a.add_argument("--min-volume", type=float, default=15_000_000)
    v0a.add_argument("--top-n", type=int, default=40)
    v0a.add_argument(
        "--output-path",
        default="data/processed/universe/binance_v0a_latest.csv",
    )
    v0a.set_defaults(func=universe_v0a_command)

    v0b = subparsers.add_parser("universe-v0b", help="Build monthly point-in-time universes")
    v0b.add_argument("--start", required=True)
    v0b.add_argument("--end", required=True)
    v0b.add_argument("--lookback", type=int, default=30)
    v0b.add_argument("--min-volume", type=float, default=15_000_000)
    v0b.add_argument(
        "--active-vol-threshold",
        type=float,
        default=5_000_000,
        help="USD volume threshold per day to count towards persistence (default: 5M)",
    )
    v0b.add_argument(
        "--min-active-days",
        type=int,
        default=20,
        help="Minimum # of days within lookback where volume >= active-vol-threshold",
    )
    v0b.add_argument(
        "--min-trades",
        type=float,
        default=0.0,
        help="Optional minimum avg trade count over lookback (0 disables filter)",
    )
    v0b.add_argument(
        "--filter-fake-volume",
        action="store_true",
        help="Apply extra filters: min avg trades and max vol_per_trade",
    )
    v0b.add_argument(
        "--min-trades-filter",
        type=float,
        default=0.0,
        help="Drop symbols with avg_trades_30d below this (used when --filter-fake-volume)",
    )
    v0b.add_argument(
        "--max-vol-per-trade",
        type=float,
        default=None,
        help="Drop symbols with vol_per_trade above this (used when --filter-fake-volume)",
    )
    v0b.add_argument(
        "--vol-per-trade-z",
        type=float,
        default=None,
        help="If set, drop symbols with vol_per_trade z-score above this (upper tail).",
    )
    v0b.add_argument("--top-n", type=int, default=40)
    v0b.add_argument("--output-dir", default="data/processed/universe/v0b")
    v0b.add_argument("--pool-size", type=int, default=120)
    v0b.add_argument("--use-cmc", action="store_true", help="Enable CoinMarketCap pre-filter")
    v0b.add_argument("--cmc-key", default="", help="Override CoinMarketCap API key or fall back to env var")
    v0b.add_argument("--no-cache", action="store_true", help="Disable caching of monthly metrics")
    v0b.add_argument(
        "--cache-dir",
        default="",
        help="Directory to store cached liquidity metrics (default: <output>/cache)",
    )
    v0b.add_argument("--no-resume", action="store_true", help="Recompute even if monthly CSV exists")
    v0b.add_argument(
        "--status-log",
        default="",
        help="Optional path for the month-by-month status CSV",
    )
    v0b.set_defaults(func=universe_v0b_command)

    features = subparsers.add_parser("features-v1", help="Build V1 technical feature dataset")
    features.add_argument("--ohlcv-file", required=True, help="Path to OHLCV parquet")
    features.add_argument("--universe-dir", required=True, help="Directory with PIT CSVs")
    features.add_argument("--output-file", default="data/processed/features/v1/features.parquet")
    features.add_argument("--start", required=True)
    features.add_argument("--end", required=True)
    features.add_argument("--rebalance-freq", default="1W", help="Rebalance frequency (default 1W)")
    features.add_argument(
        "--active-vol-threshold",
        type=float,
        default=5_000_000,
        help="Daily USD volume threshold for persistence calculations",
    )
    features.set_defaults(func=features_v1_command)

    features_v2 = subparsers.add_parser("features-v2", help="Build V2 feature set (mom + carry)")
    features_v2.add_argument("--ohlcv-file", required=True, help="Path to OHLCV parquet")
    features_v2.add_argument("--funding-file", required=True, help="Path to daily funding parquet")
    features_v2.add_argument("--universe-dir", required=True, help="Directory with PIT CSVs")
    features_v2.add_argument(
        "--output-file", default="data/processed/features/v2/features.parquet"
    )
    features_v2.add_argument("--start", required=True)
    features_v2.add_argument("--end", required=True)
    features_v2.add_argument(
        "--rebalance-freq", default="1W", help="Rebalance frequency (default 1W)"
    )
    features_v2.add_argument(
        "--active-vol-threshold",
        type=float,
        default=5_000_000,
        help="Daily USD volume threshold for persistence calculations",
    )
    features_v2.set_defaults(func=features_v2_command)

    regime = subparsers.add_parser("btc-regime", help="Build BTC regime file")
    regime.add_argument("--ohlcv-file", required=True, help="Path to OHLCV parquet")
    regime.add_argument("--funding-file", required=True, help="Path to daily funding parquet")
    regime.add_argument("--symbol", default="BTCUSDT", help="BTC symbol to use (default BTCUSDT)")
    regime.add_argument("--output-file", default="data/processed/regime/btc_regime.parquet")
    regime.add_argument("--vol-threshold", default="p80", help="Vol threshold (pXX or numeric)")
    regime.add_argument("--dd-threshold", type=float, default=-0.50, help="DD threshold (e.g. -0.5)")
    regime.add_argument(
        "--funding-threshold",
        type=float,
        default=0.0005,
        help="Abs funding mean 30d threshold (fraction, default 0.0005 ~= 0.05 pct per 8h)",
    )
    regime.add_argument(
        "--hl-threshold",
        default=None,
        help="High/low range 30d threshold (pXX or numeric); if None, disabled",
    )
    regime.add_argument(
        "--ret-30d-threshold",
        type=float,
        default=None,
        help="ret_30d threshold to flag stress if return <= threshold (e.g. -0.10).",
    )
    regime.add_argument(
        "--entry-days",
        type=int,
        default=1,
        help="Consecutive stress_raw days to enter STRESS (hysteresis)",
    )
    regime.add_argument(
        "--exit-days",
        type=int,
        default=1,
        help="Consecutive normal days to exit STRESS (hysteresis)",
    )
    regime.set_defaults(func=btc_regime_command)

    macro = subparsers.add_parser("macro-regime", help="Build macro (FRED) regime file")
    macro.add_argument("--output-file", default="data/processed/regime/macro_regime.parquet")
    macro.add_argument("--start", default="2015-01-01")
    macro.add_argument("--end", default="2025-12-31")
    macro.add_argument("--api-key", default="", help="FRED API key (or set FRED_API_KEY env)")
    macro.add_argument("--vol-threshold", default="p80", help="VIX threshold (pXX or numeric)")
    macro.add_argument("--dxy-threshold", default="p80", help="DXY threshold (pXX or numeric)")
    macro.add_argument("--ret-30d-threshold", type=float, default=-0.05, help="SPX 30d return threshold")
    macro.add_argument("--entry-days", type=int, default=3, help="Consecutive stress days to enter")
    macro.add_argument("--exit-days", type=int, default=10, help="Consecutive normal days to exit")
    macro.set_defaults(func=macro_regime_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
