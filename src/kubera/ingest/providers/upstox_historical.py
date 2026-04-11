"""Upstox v2 historical daily OHLCV.

Requires ``KUBERA_UPSTOX_ACCESS_TOKEN`` (Bearer). Obtain it by completing the OAuth flow
and exchanging the authorization ``code`` via ``kubera.ingest.providers.upstox_token``
or Upstox's documented POST to ``/v2/login/authorization/token``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from kubera.ingest.market_data import (
    HistoricalFetchRequest,
    HistoricalMarketDataProvider,
    HistoricalMarketDataProviderError,
)


UPSTOX_HISTORICAL_URL = "https://api.upstox.com/v2/historical-candle"


class UpstoxHistoricalDataProvider(HistoricalMarketDataProvider):
    """Daily OHLCV via Upstox Open API (Bearer token)."""

    provider_name = "upstox"

    def __init__(self, access_token: str) -> None:
        self._access_token = access_token.strip()

    def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
        if not self._access_token:
            raise HistoricalMarketDataProviderError(
                "Upstox access token is empty. Set KUBERA_UPSTOX_ACCESS_TOKEN."
            )
        instrument_key = request.provider_symbol.strip()
        if not instrument_key:
            raise HistoricalMarketDataProviderError(
                "Upstox requires provider_symbol (instrument_key) in the ticker catalog "
                "under provider_symbol_map['upstox']."
            )

        rows: list[dict[str, Any]] = []
        chunk_end = request.end_date
        start = request.start_date
        while chunk_end >= start:
            chunk_start = max(start, chunk_end - timedelta(days=364))
            encoded_key = quote(instrument_key, safe="")
            url = (
                f"{UPSTOX_HISTORICAL_URL}/{encoded_key}/day/"
                f"{chunk_end.isoformat()}/{chunk_start.isoformat()}"
            )
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json",
                },
                timeout=120,
            )
            if response.status_code != 200:
                raise HistoricalMarketDataProviderError(
                    f"Upstox historical request failed ({response.status_code}): {response.text[:500]}"
                )
            payload = response.json()
            if payload.get("status") != "success":
                raise HistoricalMarketDataProviderError(
                    f"Upstox historical response not successful: {payload!r}"
                )
            candles = (payload.get("data") or {}).get("candles") or []
            for candle in candles:
                if not candle or len(candle) < 6:
                    continue
                ts, open_v, high_v, low_v, close_v, volume_v = candle[:6]
                rows.append(
                    {
                        "date": ts,
                        "Open": float(open_v),
                        "High": float(high_v),
                        "Low": float(low_v),
                        "Close": float(close_v),
                        "Adj Close": float(close_v),
                        "Volume": float(volume_v),
                    }
                )
            chunk_end = chunk_start - timedelta(days=1)

        if not rows:
            raise HistoricalMarketDataProviderError("Upstox returned no candle rows.")

        frame = pd.DataFrame(rows)
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        frame = frame.sort_values("date").reset_index(drop=True)
        return frame
