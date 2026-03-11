"""Data ingestion modules live here."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CompanyNewsProvider",
    "HistoricalFetchResult",
    "HistoricalMarketDataProvider",
    "HistoricalMarketDataProviderError",
    "MarketauxNewsProvider",
    "NewsDiscoveryRequest",
    "NewsIngestionError",
    "NewsIngestionResult",
    "YFinanceHistoricalDataProvider",
    "fetch_company_news",
    "fetch_historical_market_data",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name in {
            "CompanyNewsProvider",
            "MarketauxNewsProvider",
            "NewsDiscoveryRequest",
            "NewsIngestionError",
            "NewsIngestionResult",
            "fetch_company_news",
        }:
            from kubera.ingest import news_data

            return getattr(news_data, name)

        from kubera.ingest import market_data

        return getattr(market_data, name)
    raise AttributeError(f"module 'kubera.ingest' has no attribute {name!r}")
