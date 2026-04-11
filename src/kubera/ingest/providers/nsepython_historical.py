"""NSE equity history via nsepython (NSE only)."""

from __future__ import annotations

import pandas as pd

from kubera.ingest.market_data import (
    HistoricalFetchRequest,
    HistoricalMarketDataProvider,
    HistoricalMarketDataProviderError,
)


class NsePythonHistoricalDataProvider(HistoricalMarketDataProvider):
    """Daily OHLCV via nsepython equity_history (NSE EQ series)."""

    provider_name = "nsepython"

    def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
        if request.exchange.strip().upper() != "NSE":
            raise HistoricalMarketDataProviderError(
                "nsepython historical provider only supports NSE. "
                f"Got exchange={request.exchange}."
            )
        try:
            from nsepython import equity_history
        except ImportError as exc:
            raise HistoricalMarketDataProviderError(
                'nsepython is not installed. Install optional deps: pip install -e ".[nsepython]"'
            ) from exc

        symbol = request.provider_symbol.strip() or request.ticker.strip()
        start_s = request.start_date.strftime("%d-%m-%Y")
        end_s = request.end_date.strftime("%d-%m-%Y")
        try:
            raw_frame = equity_history(symbol, "EQ", start_s, end_s)
        except KeyError as exc:
            raise HistoricalMarketDataProviderError(
                "nsepython equity_history failed: historical API response missing expected keys "
                "(not live-quote/nse_eq). Upstream may raise KeyError('data') on empty or 404 "
                "responses; see nsepython#74."
            ) from exc
        if raw_frame is None or raw_frame.empty:
            raise HistoricalMarketDataProviderError("nsepython equity_history returned no rows.")

        column_map = {
            "CH_TIMESTAMP": "date",
            "CH_OPENING_PRICE": "Open",
            "CH_TRADE_HIGH_PRICE": "High",
            "CH_TRADE_LOW_PRICE": "Low",
            "CH_CLOSING_PRICE": "Close",
            "CH_TOT_TRADED_QTY": "Volume",
        }
        missing = [c for c in column_map if c not in raw_frame.columns]
        if missing:
            raise HistoricalMarketDataProviderError(
                f"nsepython frame missing expected columns {missing}. Got: {list(raw_frame.columns)}"
            )

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(raw_frame["CH_TIMESTAMP"], errors="coerce")
        out["Open"] = pd.to_numeric(raw_frame["CH_OPENING_PRICE"], errors="coerce")
        out["High"] = pd.to_numeric(raw_frame["CH_TRADE_HIGH_PRICE"], errors="coerce")
        out["Low"] = pd.to_numeric(raw_frame["CH_TRADE_LOW_PRICE"], errors="coerce")
        out["Close"] = pd.to_numeric(raw_frame["CH_CLOSING_PRICE"], errors="coerce")
        out["Adj Close"] = out["Close"]
        out["Volume"] = pd.to_numeric(raw_frame["CH_TOT_TRADED_QTY"], errors="coerce")
        out = out.dropna(subset=["date"])
        out = out.sort_values("date").reset_index(drop=True)
        return out
