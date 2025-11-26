"""Lightweight Binance Futures client tailored for ETL jobs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests import Response, Session

DEFAULT_INTERVAL = "1d"


def _to_millis(value: datetime | pd.Timestamp | int | float | str | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        timestamp = pd.Timestamp(value, tz="UTC")
    elif isinstance(value, pd.Timestamp):
        timestamp = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    elif isinstance(value, datetime):
        timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        timestamp = pd.Timestamp(timestamp)
    else:  # pragma: no cover - defensive branch
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")
    return int(timestamp.timestamp() * 1000)


def _interval_to_millis(interval: str) -> int:
    delta = pd.to_timedelta(interval)
    return int(delta.total_seconds() * 1000)


@dataclass
class BinanceFuturesClient:
    """Minimal HTTP client for Binance USDⓈ-M Futures endpoints."""

    base_url: str = "https://fapi.binance.com"
    timeout: int = 10
    rate_limit: float = 0.2
    max_retries: int = 5
    backoff_base: float = 0.5
    backoff_factor: float = 1.5
    session: Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self._exchange_info: Dict[str, object] | None = None

    # -- Generic helpers -----------------------------------------------------------------
    def _request(self, path: str, params: Optional[Dict[str, object]] = None) -> Response:
        last_error: requests.HTTPError | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            if self.rate_limit:
                time.sleep(self.rate_limit)
            try:
                response = self.session.get(
                    f"{self.base_url}{path}", params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429 and attempt < attempts - 1:
                    delay = self.backoff_base * (self.backoff_factor ** attempt)
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Binance request failed without HTTP response.")

    def _paginate_klines(
        self,
        symbol: str,
        interval: str,
        start: Optional[int],
        end: Optional[int],
        limit: int,
    ) -> List[List[object]]:
        interval_ms = _interval_to_millis(interval)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end

        data: List[List[object]] = []
        while True:
            chunk = self._request("/fapi/v1/klines", params=params).json()
            if not chunk:
                break
            data.extend(chunk)
            last_open_time = chunk[-1][0]
            next_start = last_open_time + interval_ms
            if end is not None and next_start >= end:
                break
            params["startTime"] = next_start
        return data

    # -- Public endpoints ----------------------------------------------------------------
    def exchange_info(self) -> Dict[str, object]:
        if self._exchange_info is None:
            self._exchange_info = self._request("/fapi/v1/exchangeInfo").json()
        return self._exchange_info

    def list_perpetual_contracts(self, quote_asset: str = "USDT") -> List[Dict[str, object]]:
        info = self.exchange_info()
        symbols = info.get("symbols", [])
        result = []
        for symbol in symbols:
            if symbol.get("contractType") != "PERPETUAL":
                continue
            if quote_asset and symbol.get("quoteAsset") != quote_asset:
                continue
            result.append(symbol)
        return result

    def fetch_24h_tickers(self) -> pd.DataFrame:
        payload = self._request("/fapi/v1/ticker/24hr").json()
        df = pd.DataFrame(payload)
        if df.empty:
            return df
        numeric_cols = [
            "volume",
            "quoteVolume",
            "countTrades",
            "openPrice",
            "highPrice",
            "lowPrice",
            "lastPrice",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.rename(columns={"symbol": "symbol", "quoteVolume": "quote_volume"}, inplace=True)
        return df

    def fetch_klines(
        self,
        symbol: str,
        interval: str = DEFAULT_INTERVAL,
        start: datetime | pd.Timestamp | int | float | str | None = None,
        end: datetime | pd.Timestamp | int | float | str | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)
        rows = self._paginate_klines(symbol, interval, start_ms, end_ms, limit)
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_base_volume",
            "taker_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return df
        df["symbol"] = symbol
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_base_volume",
            "taker_quote_volume",
            "trade_count",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
        df.rename(columns={"quote_volume": "volume_quote"}, inplace=True)
        df.rename(columns={"trade_count": "num_trades"}, inplace=True)
        keep = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_quote",
            "num_trades",
            "taker_quote_volume",
        ]
        return df[keep]

    def fetch_open_interest_history(
        self,
        symbol: str,
        start: datetime | pd.Timestamp | int | float | str | None = None,
        end: datetime | pd.Timestamp | int | float | str | None = None,
        period: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        params: Dict[str, object] = {"symbol": symbol, "period": period, "limit": limit}
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        try:
            payload = self._request("/futures/data/openInterestHist", params=params).json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in {400, 429}:
                return pd.DataFrame()
            raise
        df = pd.DataFrame(payload)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df["open_interest"] = pd.to_numeric(df.get("sumOpenInterest"), errors="coerce")
        df["open_interest_usd"] = pd.to_numeric(
            df.get("sumOpenInterestValue"), errors="coerce"
        )
        keep = ["timestamp", "symbol", "open_interest", "open_interest_usd"]
        return df[keep]

    def fetch_funding_history(
        self,
        symbol: str,
        start: datetime | pd.Timestamp | int | float | str | None = None,
        end: datetime | pd.Timestamp | int | float | str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch funding rates (8h) for a symbol, paginated, returned as a DataFrame."""

        start_ms = _to_millis(start)
        end_ms = _to_millis(end)
        params: Dict[str, object] = {"symbol": symbol, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        data: List[Dict[str, object]] = []
        while True:
            payload = self._request("/fapi/v1/fundingRate", params=params).json()
            if not payload:
                break
            data.extend(payload)
            last_time = payload[-1]["fundingTime"]
            next_start = int(last_time) + 1
            if end_ms is not None and next_start >= end_ms:
                break
            params["startTime"] = next_start

        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert(None)
        df["funding_rate"] = pd.to_numeric(df.get("fundingRate"), errors="coerce")
        df["symbol"] = symbol
        keep = ["timestamp", "symbol", "funding_rate"]
        return df[keep]

    def close(self) -> None:
        self.session.close()


__all__ = ["BinanceFuturesClient"]
