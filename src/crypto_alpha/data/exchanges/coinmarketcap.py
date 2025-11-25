"""CoinMarketCap helper client for market-cap based filters."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests import Session


@dataclass
class CoinMarketCapClient:
    """Thin wrapper around CoinMarketCap's REST API."""

    base_url: str = "https://pro-api.coinmarketcap.com"
    timeout: int = 15
    rate_limit: float = 1.2
    session: Session = field(default_factory=requests.Session)
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = (
                os.getenv("CMC_KEY")
                or os.getenv("CMC_PRO_API_KEY")
                or os.getenv("COINMARKETCAP_API_KEY")
            )
        if not self.api_key:
            raise RuntimeError(
                "CoinMarketCap API key missing. Set CMC_KEY or CMC_PRO_API_KEY in the environment."
            )

    def _request(self, path: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if self.rate_limit:
            time.sleep(self.rate_limit)
        headers = {"X-CMC_PRO_API_KEY": self.api_key}
        response = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout, headers=headers
        )
        if response.status_code == 401:
            raise RuntimeError("CoinMarketCap API rejected the key (401). Check CMC_KEY/plan permissions.")
        if response.status_code == 403:
            raise RuntimeError("CoinMarketCap plan does not permit this endpoint (403).")
        if response.status_code == 429:
            time.sleep(2 * (self.rate_limit or 1.0))
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status", {})
        error_code = status.get("error_code")
        if error_code:
            raise RuntimeError(
                f"CMC error {error_code}: {status.get('error_message', 'unknown error')}"
            )
        return payload

    def listings_latest(self, limit: int = 200, convert: str = "USD") -> pd.DataFrame:
        params = {
            "start": 1,
            "limit": limit,
            "convert": convert,
            "sort": "market_cap",
            "sort_dir": "desc",
        }
        payload = self._request("/v1/cryptocurrency/listings/latest", params=params)
        return pd.DataFrame(payload.get("data", []))

    def listings_historical(
        self,
        date: datetime,
        limit: int = 200,
        convert: str = "USD",
    ) -> pd.DataFrame:
        params = {
            "date": date.strftime("%Y-%m-%d"),
            "start": 1,
            "limit": limit,
            "convert": convert,
            "sort": "market_cap",
            "sort_dir": "desc",
        }
        payload = self._request("/v1/cryptocurrency/listings/historical", params=params)
        return pd.DataFrame(payload.get("data", []))

    def top_market_cap_symbols(
        self,
        as_of: datetime,
        limit: int = 120,
        allow_latest_fallback: bool = True,
    ) -> List[str]:
        try:
            df = self.listings_historical(as_of, limit=limit)
        except RuntimeError as exc:
            if not allow_latest_fallback:
                raise
            df = self.listings_latest(limit=limit)
            if df.empty:
                raise RuntimeError(
                    f"CoinMarketCap historical listings unavailable ({exc}). Latest listings also empty."
                )
        if df.empty and allow_latest_fallback:
            df = self.listings_latest(limit=limit)
        if df.empty:
            return []
        return df["symbol"].str.upper().tolist()

    def close(self) -> None:
        self.session.close()


__all__ = ["CoinMarketCapClient"]
