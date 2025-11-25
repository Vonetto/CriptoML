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
    build_universe_v0a,
    build_universe_v0b,
    download_ohlcv,
    download_open_interest,
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
