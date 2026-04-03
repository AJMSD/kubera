"""Company news ingestion for Kubera."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pandas as pd
import requests

from kubera.config import (
    AppSettings,
    NewsIngestionSettings,
    load_settings,
    resolve_runtime_settings,
)
from kubera.utils.hashing import compute_file_sha256
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot
from kubera.utils.time_utils import utc_to_market_time


MARKETAUX_ENTITY_SEARCH_URL = "https://api.marketaux.com/v1/entity/search"
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"
ALPHAVANTAGE_NEWS_URL = "https://www.alphavantage.co/query"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
NSE_HOME_URL = "https://www.nseindia.com"
NSE_CORP_INFO_URL = "https://www.nseindia.com/api/corp-info"
ALPHAVANTAGE_PROVIDER_NAME = "alphavantage"
GOOGLE_NEWS_PROVIDER_NAME = "google_news_rss"
NSE_ANNOUNCEMENTS_PROVIDER_NAME = "nse_announcements"
ECONOMIC_TIMES_PROVIDER_NAME = "economic_times"
TRACKING_QUERY_PREFIXES = ("utm_", "ga_", "fbclid", "gclid", "mc_", "ref")
HOSTNAME_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,63}$")
ARTICLE_STRIP_TAGS = (
    "aside",
    "footer",
    "form",
    "header",
    "iframe",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
)
PROCESSED_NEWS_COLUMNS = (
    "article_id",
    "ticker",
    "exchange",
    "provider",
    "discovery_mode",
    "provider_uuid",
    "article_title",
    "article_url",
    "canonical_url",
    "source_domain",
    "provider_source",
    "source_name",
    "published_at_raw",
    "published_at_utc",
    "published_at_ist",
    "published_date_ist",
    "summary_snippet",
    "full_text",
    "content_origin",
    "text_acquisition_mode",
    "text_acquisition_reason",
    "fetch_warning_flag",
    "fetch_error",
    "http_status",
    "provider_entity_payload",
    "raw_snapshot_path",
    "fetched_at_utc",
)


class NewsIngestionError(RuntimeError):
    """Raised when Stage 5 news ingestion cannot continue."""


@dataclass(frozen=True)
class NewsDiscoveryRequest:
    ticker: str
    exchange: str
    provider: str
    company_name: str
    search_aliases: tuple[str, ...]
    sector_name: str | None
    industry_name: str | None
    sector_query_terms: tuple[str, ...]
    macro_query_terms: tuple[str, ...]
    lookback_days: int
    published_after: datetime
    published_before: datetime
    language: str
    country: str
    marketaux_limit_per_request: int
    max_articles_per_run: int


@dataclass(frozen=True)
class ArticleFetchResult:
    full_text: str | None
    text_acquisition_mode: str
    text_acquisition_reason: str
    fetch_warning_flag: bool
    fetch_error: str | None
    http_status: int | None
    resolved_article_url: str | None = None
    attempt_count: int = 1
    retry_count: int = 0


@dataclass(frozen=True)
class NewsIngestionResult:
    raw_snapshot_path: Path
    cleaned_table_path: Path
    metadata_path: Path
    row_count: int
    duplicate_count: int
    dropped_row_count: int
    coverage_start: date | None
    coverage_end: date | None


@dataclass(frozen=True)
class NewsNormalizationResult:
    final_articles: list[dict[str, Any]]
    dropped_rows: list[dict[str, Any]]
    duplicate_count: int
    acquisition_diagnostics: list[dict[str, Any]]
    updated_article_fetch_cache: dict[str, dict[str, Any]]
    cache_hit_count: int
    fresh_fetch_count: int
    expired_cache_count: int
    article_fetch_attempt_count: int
    article_fetch_retry_count: int


@dataclass(frozen=True)
class CollectedNewsSource:
    provider_name: str
    normalized_articles: list[dict[str, Any]]
    dropped_rows: list[dict[str, Any]]
    warnings: list[str]
    provider_request_count: int
    provider_request_retry_count: int
    raw_payload: dict[str, Any]
    discovery_mode: str | None = None
    search_query: str | None = None
    resolved_symbols: tuple[str, ...] = ()
    entity_matches: list[dict[str, Any]] = field(default_factory=list)
    entity_payload_count: int = 0
    news_payload_count: int = 0


ArticleFetcher = Callable[[dict[str, Any], NewsIngestionSettings], ArticleFetchResult]


class CompanyNewsProvider(ABC):
    """Boundary for company-news discovery providers."""

    provider_name: str

    @abstractmethod
    def search_entities(
        self,
        request: NewsDiscoveryRequest,
        query: str,
    ) -> dict[str, Any]:
        """Search provider entities using one company alias."""

    @abstractmethod
    def fetch_news_page(
        self,
        request: NewsDiscoveryRequest,
        *,
        page: int,
        symbols: tuple[str, ...] | None = None,
        search_query: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one provider page of news articles."""

    def get_retry_summary(self) -> dict[str, int]:
        """Return retry counters collected during this run when available."""

        return {
            "provider_request_count": 0,
            "provider_request_retry_count": 0,
        }


class MarketauxNewsProvider(CompanyNewsProvider):
    """Marketaux-backed provider for company news discovery."""

    provider_name = "marketaux"

    def __init__(
        self,
        api_token: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 15,
        retry_attempts: int = 3,
    ) -> None:
        self._api_token = api_token
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._provider_request_count = 0
        self._provider_request_retry_count = 0

    def search_entities(
        self,
        request: NewsDiscoveryRequest,
        query: str,
    ) -> dict[str, Any]:
        params = {
            "api_token": self._api_token,
            "search": query,
            "countries": request.country,
            "exchanges": request.exchange,
            "types": "equity",
        }
        return self._get_json(MARKETAUX_ENTITY_SEARCH_URL, params=params)

    def fetch_news_page(
        self,
        request: NewsDiscoveryRequest,
        *,
        page: int,
        symbols: tuple[str, ...] | None = None,
        search_query: str | None = None,
    ) -> dict[str, Any]:
        params = {
            "api_token": self._api_token,
            "countries": request.country,
            "language": request.language,
            "published_after": format_marketaux_datetime(request.published_after),
            "published_before": format_marketaux_datetime(request.published_before),
            "limit": request.marketaux_limit_per_request,
            "page": page,
            "group_similar": "true",
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
            params["filter_entities"] = "true"
            params["must_have_entities"] = "true"
        elif search_query:
            params["search"] = search_query
            params["sort"] = "relevance_score"
            params["must_have_entities"] = "false"
        else:
            raise NewsIngestionError("Expected symbols or search query for Marketaux news fetch.")
        return self._get_json(MARKETAUX_NEWS_URL, params=params)

    def _get_json(self, url: str, *, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        self._provider_request_count += 1
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise NewsIngestionError(
                        f"News provider returned an unexpected payload type: {type(payload)!r}"
                    )
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self._retry_attempts:
                    break
                self._provider_request_retry_count += 1
                time.sleep(0.5 * attempt)
        raise NewsIngestionError(f"News provider request failed: {last_error}") from last_error

    def get_retry_summary(self) -> dict[str, int]:
        return {
            "provider_request_count": int(self._provider_request_count),
            "provider_request_retry_count": int(self._provider_request_retry_count),
        }


class AlphaVantageNewsProvider:
    """Alpha Vantage NEWS_SENTIMENT provider for ticker-scoped article discovery."""

    provider_name = ALPHAVANTAGE_PROVIDER_NAME
    minimum_pause_seconds = 1.0

    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 15,
        retry_attempts: int = 3,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._provider_request_count = 0
        self._provider_request_retry_count = 0

    def fetch_feed(self, request: NewsDiscoveryRequest) -> dict[str, Any]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": build_alphavantage_ticker(request),
            "time_from": format_alphavantage_datetime(request.published_after),
            "time_to": format_alphavantage_datetime(request.published_before),
            "sort": "RELEVANCE",
            "limit": max(request.max_articles_per_run * 3, 50),
            "apikey": self._api_key,
        }
        return self._get_json(ALPHAVANTAGE_NEWS_URL, params=params)

    def _get_json(self, url: str, *, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        self._provider_request_count += 1
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise NewsIngestionError(
                        f"Alpha Vantage returned an unexpected payload type: {type(payload)!r}"
                    )
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self._retry_attempts:
                    break
                self._provider_request_retry_count += 1
                time.sleep(self.minimum_pause_seconds * attempt)
        raise NewsIngestionError(
            f"Alpha Vantage request failed: {last_error}"
        ) from last_error

    def get_retry_summary(self) -> dict[str, int]:
        return {
            "provider_request_count": int(self._provider_request_count),
            "provider_request_retry_count": int(self._provider_request_retry_count),
        }


class GoogleNewsRssProvider:
    """Google News RSS search provider for free supplemental coverage."""

    provider_name = GOOGLE_NEWS_PROVIDER_NAME
    minimum_pause_seconds = 0.5

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 15,
        retry_attempts: int = 3,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._provider_request_count = 0
        self._provider_request_retry_count = 0

    def fetch_feed(
        self,
        request: NewsDiscoveryRequest,
    ) -> str:
        params = {
            "q": build_google_news_query(request),
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en",
        }
        return self._get_text(GOOGLE_NEWS_RSS_URL, params=params)

    def _get_text(self, url: str, *, params: dict[str, Any]) -> str:
        last_error: Exception | None = None
        self._provider_request_count += 1
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self._retry_attempts:
                    break
                self._provider_request_retry_count += 1
                time.sleep(0.5 * attempt)
        raise NewsIngestionError(f"Google News RSS request failed: {last_error}") from last_error

    def get_retry_summary(self) -> dict[str, int]:
        return {
            "provider_request_count": int(self._provider_request_count),
            "provider_request_retry_count": int(self._provider_request_retry_count),
        }


class EconomicTimesProvider:
    """Economic Times news search provider for Indian market coverage."""

    provider_name = ECONOMIC_TIMES_PROVIDER_NAME
    minimum_pause_seconds = 1.0

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 15,
        retry_attempts: int = 3,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._provider_request_count = 0
        self._provider_request_retry_count = 0

    def fetch_feed(
        self,
        request: NewsDiscoveryRequest,
    ) -> str:
        # Search using Google News specifically restricted to Economic Times
        params = {
            "q": f"{build_google_news_query(request)} site:economictimes.indiatimes.com",
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en",
        }
        return self._get_text(GOOGLE_NEWS_RSS_URL, params=params)

    def _get_text(self, url: str, *, params: dict[str, Any]) -> str:
        last_error: Exception | None = None
        self._provider_request_count += 1
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self._retry_attempts:
                    break
                self._provider_request_retry_count += 1
                time.sleep(0.5 * attempt)
        raise NewsIngestionError(f"Economic Times RSS request failed: {last_error}") from last_error

    def get_retry_summary(self) -> dict[str, int]:
        return {
            "provider_request_count": int(self._provider_request_count),
            "provider_request_retry_count": int(self._provider_request_retry_count),
        }


class NseAnnouncementsProvider:
    """NSE corporate announcements provider with cookie priming."""

    provider_name = NSE_ANNOUNCEMENTS_PROVIDER_NAME
    minimum_pause_seconds = 1.0

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 15,
        retry_attempts: int = 3,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._provider_request_count = 0
        self._provider_request_retry_count = 0
        self._primed_session = False

    def fetch_announcements(
        self,
        request: NewsDiscoveryRequest,
        *,
        pause_seconds: float,
    ) -> list[dict[str, Any]]:
        self._prime_session(pause_seconds=pause_seconds)
        pause_before_provider_request(pause_seconds)
        payload = self._get_json(
            NSE_CORP_INFO_URL,
            params={
                "symbol": request.ticker,
                "corpType": "announcements",
                "market": "equities",
            },
        )
        return extract_nse_announcements_payload(payload)

    def _prime_session(self, *, pause_seconds: float) -> None:
        if self._primed_session:
            return
        pause_before_provider_request(pause_seconds)
        self._get_text(NSE_HOME_URL)
        self._primed_session = True

    def _build_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0",
            "Referer": NSE_HOME_URL,
            "Accept": "application/json, text/plain, */*",
        }

    def _get_text(self, url: str) -> str:
        last_error: Exception | None = None
        self._provider_request_count += 1
        headers = self._build_headers()
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self._retry_attempts:
                    break
                self._provider_request_retry_count += 1
                time.sleep(0.5 * attempt)
        raise NewsIngestionError(f"NSE home-page request failed: {last_error}") from last_error

    def _get_json(self, url: str, *, params: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        self._provider_request_count += 1
        headers = self._build_headers()
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self._retry_attempts:
                    break
                self._provider_request_retry_count += 1
                time.sleep(0.5 * attempt)
        raise NewsIngestionError(f"NSE announcements request failed: {last_error}") from last_error

    def get_retry_summary(self) -> dict[str, int]:
        return {
            "provider_request_count": int(self._provider_request_count),
            "provider_request_retry_count": int(self._provider_request_retry_count),
        }


def build_provider_retry_summary(
    news_provider: CompanyNewsProvider,
    *,
    entity_payloads: list[dict[str, Any]],
    news_payloads: list[dict[str, Any]],
) -> dict[str, int]:
    """Resolve deterministic provider retry counters for this Stage 5 run."""

    summary = news_provider.get_retry_summary()
    provider_request_count = int(summary.get("provider_request_count", 0) or 0)
    provider_request_retry_count = int(summary.get("provider_request_retry_count", 0) or 0)
    if provider_request_count <= 0:
        provider_request_count = len(entity_payloads) + len(news_payloads)
    return {
        "provider_request_count": provider_request_count,
        "provider_request_retry_count": provider_request_retry_count,
    }


def extract_nse_announcements_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize the live NSE response into one list of announcement rows."""

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        announcements = payload.get("data")
        if isinstance(announcements, list):
            return [row for row in announcements if isinstance(row, dict)]

    raise NewsIngestionError("NSE announcements endpoint returned an unexpected payload.")


def resolve_configured_news_sources(
    settings: AppSettings,
) -> list[
    CompanyNewsProvider
    | AlphaVantageNewsProvider
    | GoogleNewsRssProvider
    | NseAnnouncementsProvider
    | EconomicTimesProvider
]:
    """Resolve the configured Stage 5 provider set for one run."""

    providers: list[
        CompanyNewsProvider
        | AlphaVantageNewsProvider
        | GoogleNewsRssProvider
        | NseAnnouncementsProvider
        | EconomicTimesProvider
    ] = []
    configured_provider = resolve_news_provider(settings)
    if configured_provider is not None:
        providers.append(configured_provider)
    if settings.news_ingestion.enable_google_news_rss:
        providers.append(
            GoogleNewsRssProvider(
                timeout_seconds=settings.news_ingestion.request_timeout_seconds,
                retry_attempts=settings.news_ingestion.article_retry_attempts,
            )
        )
    if getattr(settings.news_ingestion, "enable_economic_times", False):
        providers.append(
            EconomicTimesProvider(
                timeout_seconds=settings.news_ingestion.request_timeout_seconds,
                retry_attempts=settings.news_ingestion.article_retry_attempts,
            )
        )
    if settings.news_ingestion.enable_nse_announcements:
        providers.append(
            NseAnnouncementsProvider(
                timeout_seconds=settings.news_ingestion.request_timeout_seconds,
                retry_attempts=settings.news_ingestion.article_retry_attempts,
            )
        )
    return providers


def collect_news_source_results(
    *,
    settings: AppSettings,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    provider: CompanyNewsProvider | AlphaVantageNewsProvider | None = None,
) -> tuple[list[CollectedNewsSource], list[str]]:
    """Collect normalized candidates from the configured source set."""

    if provider is not None:
        if isinstance(provider, AlphaVantageNewsProvider):
            return (
                [
                    collect_alphavantage_source(
                        provider=provider,
                        request=request,
                        raw_snapshot_path=raw_snapshot_path,
                        fetched_at_utc=fetched_at_utc,
                        market_settings=settings.market,
                        news_settings=settings.news_ingestion,
                    )
                ],
                [],
            )
        return (
            [
                collect_marketaux_source(
                    provider=provider,
                    request=request,
                    raw_snapshot_path=raw_snapshot_path,
                    fetched_at_utc=fetched_at_utc,
                    market_settings=settings.market,
                    news_settings=settings.news_ingestion,
                )
            ],
            [],
        )

    source_results: list[CollectedNewsSource] = []
    source_warnings: list[str] = []
    for active_provider in resolve_configured_news_sources(settings):
        try:
            if isinstance(active_provider, CompanyNewsProvider):
                source_results.append(
                    collect_marketaux_source(
                        provider=active_provider,
                        request=request,
                        raw_snapshot_path=raw_snapshot_path,
                        fetched_at_utc=fetched_at_utc,
                        market_settings=settings.market,
                        news_settings=settings.news_ingestion,
                    )
                )
            elif isinstance(active_provider, AlphaVantageNewsProvider):
                source_results.append(
                    collect_alphavantage_source(
                        provider=active_provider,
                        request=request,
                        raw_snapshot_path=raw_snapshot_path,
                        fetched_at_utc=fetched_at_utc,
                        market_settings=settings.market,
                        news_settings=settings.news_ingestion,
                    )
                )
            elif isinstance(active_provider, GoogleNewsRssProvider):
                source_results.append(
                    collect_google_news_source(
                        provider=active_provider,
                        request=request,
                        raw_snapshot_path=raw_snapshot_path,
                        fetched_at_utc=fetched_at_utc,
                        market_settings=settings.market,
                        news_settings=settings.news_ingestion,
                    )
                )
            elif isinstance(active_provider, EconomicTimesProvider):
                source_results.append(
                    collect_economic_times_source(
                        provider=active_provider,
                        request=request,
                        raw_snapshot_path=raw_snapshot_path,
                        fetched_at_utc=fetched_at_utc,
                        market_settings=settings.market,
                        news_settings=settings.news_ingestion,
                    )
                )
            else:
                source_results.append(
                    collect_nse_announcements_source(
                        provider=active_provider,
                        request=request,
                        raw_snapshot_path=raw_snapshot_path,
                        fetched_at_utc=fetched_at_utc,
                        market_settings=settings.market,
                        news_settings=settings.news_ingestion,
                    )
                )
        except Exception as exc:
            provider_name = getattr(active_provider, "provider_name", "unknown_provider")
            source_warnings.append(f"{provider_name}_failed")
            source_results.append(
                CollectedNewsSource(
                    provider_name=provider_name,
                    normalized_articles=[],
                    dropped_rows=[],
                    warnings=[sanitize_provider_warning(str(exc))],
                    provider_request_count=int(
                        getattr(active_provider, "get_retry_summary", lambda: {})().get(
                            "provider_request_count",
                            0,
                        )
                    ),
                    provider_request_retry_count=int(
                        getattr(active_provider, "get_retry_summary", lambda: {})().get(
                            "provider_request_retry_count",
                            0,
                        )
                    ),
                    raw_payload={
                        "provider": provider_name,
                        "status": "failed",
                        "error": sanitize_provider_warning(str(exc)),
                    },
                )
            )
    return source_results, source_warnings


def collect_marketaux_source(
    *,
    provider: CompanyNewsProvider,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    news_settings: NewsIngestionSettings,
) -> CollectedNewsSource:
    """Collect normalized candidates from one Marketaux-style provider."""

    normalized_request = (
        replace(request, provider=provider.provider_name)
        if request.provider != provider.provider_name
        else request
    )
    entity_payloads, resolved_symbols, entity_matches = resolve_provider_entities(
        provider,
        normalized_request,
        provider_request_pause_seconds=news_settings.provider_request_pause_seconds,
    )
    discovery_mode, search_query, news_payloads = discover_company_news(
        provider,
        normalized_request,
        resolved_symbols=resolved_symbols,
        provider_request_pause_seconds=news_settings.provider_request_pause_seconds,
    )
    provider_retry_summary = build_provider_retry_summary(
        provider,
        entity_payloads=entity_payloads,
        news_payloads=news_payloads,
    )
    normalized_articles, dropped_rows = normalize_marketaux_payload_articles(
        news_payloads=news_payloads,
        request=normalized_request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=fetched_at_utc,
        market_settings=market_settings,
        discovery_mode=discovery_mode,
        resolved_symbols=resolved_symbols,
    )
    warnings: list[str] = []
    if discovery_mode == "search_fallback":
        warnings.append("search_fallback_used")
    return CollectedNewsSource(
        provider_name=provider.provider_name,
        normalized_articles=normalized_articles,
        dropped_rows=dropped_rows,
        warnings=warnings,
        provider_request_count=provider_retry_summary["provider_request_count"],
        provider_request_retry_count=provider_retry_summary["provider_request_retry_count"],
        raw_payload={
            "provider": provider.provider_name,
            "status": "ok",
            "discovery_mode": discovery_mode,
            "search_query": search_query,
            "resolved_symbols": list(resolved_symbols),
            "entity_matches": entity_matches,
            "entity_search_payloads": entity_payloads,
            "news_payloads": news_payloads,
            "warnings": warnings,
        },
        discovery_mode=discovery_mode,
        search_query=search_query,
        resolved_symbols=resolved_symbols,
        entity_matches=entity_matches,
        entity_payload_count=len(entity_payloads),
        news_payload_count=len(news_payloads),
    )


def collect_alphavantage_source(
    *,
    provider: AlphaVantageNewsProvider,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    news_settings: NewsIngestionSettings,
) -> CollectedNewsSource:
    """Collect normalized candidates from Alpha Vantage NEWS_SENTIMENT."""

    pause_before_provider_request(
        max(news_settings.provider_request_pause_seconds, provider.minimum_pause_seconds)
    )
    payload = provider.fetch_feed(request)
    feed_items = extract_alphavantage_feed(payload)
    normalized_articles, dropped_rows = normalize_alphavantage_feed(
        feed_items=feed_items,
        request=request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=fetched_at_utc,
        market_settings=market_settings,
    )
    retry_summary = provider.get_retry_summary()
    return CollectedNewsSource(
        provider_name=provider.provider_name,
        normalized_articles=normalized_articles,
        dropped_rows=dropped_rows,
        warnings=[],
        provider_request_count=int(retry_summary["provider_request_count"]),
        provider_request_retry_count=int(retry_summary["provider_request_retry_count"]),
        raw_payload={
            "provider": provider.provider_name,
            "status": "ok",
            "ticker": build_alphavantage_ticker(request),
            "feed_count": len(feed_items),
            "payload": payload,
        },
        discovery_mode="ticker_sentiment",
        search_query=build_alphavantage_ticker(request),
        news_payload_count=1,
    )


def collect_economic_times_source(
    *,
    provider: EconomicTimesProvider,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    news_settings: NewsIngestionSettings,
) -> CollectedNewsSource:
    """Collect normalized candidates from Economic Times via Google News RSS wrapper."""

    pause_before_provider_request(
        max(news_settings.provider_request_pause_seconds, provider.minimum_pause_seconds)
    )
    rss_text = provider.fetch_feed(request)
    normalized_articles, dropped_rows, item_count = normalize_google_news_rss_payload(
        rss_text=rss_text,
        request=request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=fetched_at_utc,
        market_settings=market_settings,
    )
    # Ensure provider remains tagged as ET not Google News
    for article in normalized_articles:
        article["provider"] = provider.provider_name
    for row in dropped_rows:
        row["provider"] = provider.provider_name

    retry_summary = provider.get_retry_summary()
    return CollectedNewsSource(
        provider_name=provider.provider_name,
        normalized_articles=normalized_articles,
        dropped_rows=dropped_rows,
        warnings=[],
        provider_request_count=int(retry_summary["provider_request_count"]),
        provider_request_retry_count=int(retry_summary["provider_request_retry_count"]),
        raw_payload={
            "provider": provider.provider_name,
            "status": "ok",
            "query": build_google_news_query(request) + " site:economictimes.indiatimes.com",
            "rss_item_count": item_count,
            "rss_text": rss_text,
        },
        discovery_mode="rss_search",
        search_query=build_google_news_query(request) + " site:economictimes.indiatimes.com",
        news_payload_count=1,
    )


def collect_google_news_source(
    *,
    provider: GoogleNewsRssProvider,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    news_settings: NewsIngestionSettings,
) -> CollectedNewsSource:
    """Collect normalized candidates from Google News RSS."""

    pause_before_provider_request(
        max(news_settings.provider_request_pause_seconds, provider.minimum_pause_seconds)
    )
    rss_text = provider.fetch_feed(request)
    normalized_articles, dropped_rows, item_count = normalize_google_news_rss_payload(
        rss_text=rss_text,
        request=request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=fetched_at_utc,
        market_settings=market_settings,
    )
    retry_summary = provider.get_retry_summary()
    return CollectedNewsSource(
        provider_name=provider.provider_name,
        normalized_articles=normalized_articles,
        dropped_rows=dropped_rows,
        warnings=[],
        provider_request_count=int(retry_summary["provider_request_count"]),
        provider_request_retry_count=int(retry_summary["provider_request_retry_count"]),
        raw_payload={
            "provider": provider.provider_name,
            "status": "ok",
            "query": build_google_news_query(request),
            "rss_item_count": item_count,
            "rss_text": rss_text,
        },
        discovery_mode="rss_search",
        search_query=build_google_news_query(request),
        news_payload_count=1,
    )


def collect_nse_announcements_source(
    *,
    provider: NseAnnouncementsProvider,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    news_settings: NewsIngestionSettings,
) -> CollectedNewsSource:
    """Collect normalized candidates from NSE corporate announcements."""

    pause_seconds = max(
        news_settings.provider_request_pause_seconds,
        provider.minimum_pause_seconds,
    )
    announcements = provider.fetch_announcements(request, pause_seconds=pause_seconds)
    normalized_articles, dropped_rows = normalize_nse_announcement_rows(
        announcements=announcements,
        request=request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=fetched_at_utc,
        market_settings=market_settings,
    )
    retry_summary = provider.get_retry_summary()
    return CollectedNewsSource(
        provider_name=provider.provider_name,
        normalized_articles=normalized_articles,
        dropped_rows=dropped_rows,
        warnings=[],
        provider_request_count=int(retry_summary["provider_request_count"]),
        provider_request_retry_count=int(retry_summary["provider_request_retry_count"]),
        raw_payload={
            "provider": provider.provider_name,
            "status": "ok",
            "announcement_count": len(announcements),
            "announcements_payload": announcements,
        },
        discovery_mode="corp_announcements",
        news_payload_count=1,
    )


def normalize_marketaux_payload_articles(
    *,
    news_payloads: list[dict[str, Any]],
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    discovery_mode: str,
    resolved_symbols: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize Marketaux payload pages into article candidates."""

    normalized_candidates: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for payload in news_payloads:
        articles = payload.get("data", [])
        if not isinstance(articles, list):
            raise NewsIngestionError("News provider returned an unexpected data payload.")
        for article in articles:
            if not isinstance(article, dict):
                continue
            normalized_article, reasons = normalize_marketaux_article(
                article,
                request=request,
                raw_snapshot_path=raw_snapshot_path,
                fetched_at_utc=fetched_at_utc,
                market_settings=market_settings,
                discovery_mode=discovery_mode,
                resolved_symbols=resolved_symbols,
            )
            if reasons:
                dropped_rows.append(
                    {
                        "provider_uuid": article.get("uuid"),
                        "url": article.get("url"),
                        "reasons": reasons,
                    }
                )
                continue
            normalized_candidates.append(normalized_article)
    return normalized_candidates, dropped_rows


def acquire_normalized_articles(
    *,
    normalized_candidates: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    request: NewsDiscoveryRequest,
    fetched_at_utc: datetime,
    news_settings: NewsIngestionSettings,
    article_fetcher: ArticleFetcher,
    article_fetch_cache: dict[str, dict[str, Any]],
) -> NewsNormalizationResult:
    """Deduplicate, prioritize, and enrich normalized article candidates."""

    deduped_articles, duplicate_count = dedupe_normalized_articles(normalized_candidates)
    prioritized_articles = prioritize_normalized_articles(deduped_articles, request=request)
    acquisition_diagnostics: list[dict[str, Any]] = []
    final_articles: list[dict[str, Any]] = []
    updated_article_fetch_cache = dict(article_fetch_cache)
    cache_hit_count = 0
    fresh_fetch_count = 0
    expired_cache_count = 0
    article_fetch_attempt_count = 0
    article_fetch_retry_count = 0
    for article in prioritized_articles[: request.max_articles_per_run]:
        cached_fetch_result, cache_age_hours = resolve_cached_article_fetch_result(
            article,
            cache_entries=updated_article_fetch_cache,
            fetched_at_utc=fetched_at_utc,
            ttl_hours=news_settings.article_cache_ttl_hours,
        )
        cache_hit = cached_fetch_result is not None
        if cache_hit:
            cache_hit_count += 1
            fetch_result = cached_fetch_result
        else:
            if cache_age_hours is not None:
                expired_cache_count += 1
            pause_before_article_request(news_settings.article_request_pause_seconds)
            fetch_result = article_fetcher(article, news_settings)
            fresh_fetch_count += 1
            article_fetch_attempt_count += int(fetch_result.attempt_count)
            article_fetch_retry_count += int(fetch_result.retry_count)
            update_article_fetch_cache(
                article,
                cache_entries=updated_article_fetch_cache,
                fetch_result=fetch_result,
                fetched_at_utc=fetched_at_utc,
            )

        final_article = article.copy()
        final_article["full_text"] = fetch_result.full_text
        final_article["content_origin"] = determine_content_origin(
            text_acquisition_mode=fetch_result.text_acquisition_mode,
        )
        final_article["text_acquisition_mode"] = fetch_result.text_acquisition_mode
        final_article["text_acquisition_reason"] = fetch_result.text_acquisition_reason
        final_article["fetch_warning_flag"] = fetch_result.fetch_warning_flag
        final_article["fetch_error"] = fetch_result.fetch_error
        final_article["http_status"] = fetch_result.http_status
        if fetch_result.resolved_article_url is not None:
            final_article["canonical_url"] = fetch_result.resolved_article_url
            resolved_source_domain = extract_source_domain(fetch_result.resolved_article_url)
            if resolved_source_domain is not None:
                final_article["source_domain"] = resolved_source_domain
                final_article["source_name"] = canonicalize_source_name(
                    provider_source=clean_text(final_article.get("provider_source")),
                    source_domain=resolved_source_domain,
                ) or final_article.get("source_name")
        final_articles.append(final_article)
        acquisition_diagnostics.append(
            {
                "article_id": article["article_id"],
                "article_url": article["article_url"],
                "source_name": article.get("source_name"),
                "provider": article.get("provider"),
                "company_specificity_score": score_company_specificity(
                    article,
                    request=request,
                ),
                "text_acquisition_mode": fetch_result.text_acquisition_mode,
                "text_acquisition_reason": fetch_result.text_acquisition_reason,
                "fetch_warning_flag": fetch_result.fetch_warning_flag,
                "fetch_error": fetch_result.fetch_error,
                "http_status": fetch_result.http_status,
                "resolved_article_url": fetch_result.resolved_article_url,
                "content_origin": final_article["content_origin"],
                "cache_hit": cache_hit,
                "cache_age_hours": cache_age_hours,
                "attempt_count": int(fetch_result.attempt_count) if not cache_hit else 0,
                "retry_count": int(fetch_result.retry_count) if not cache_hit else 0,
            }
        )

    return NewsNormalizationResult(
        final_articles=final_articles,
        dropped_rows=dropped_rows,
        duplicate_count=duplicate_count,
        acquisition_diagnostics=acquisition_diagnostics,
        updated_article_fetch_cache=updated_article_fetch_cache,
        cache_hit_count=cache_hit_count,
        fresh_fetch_count=fresh_fetch_count,
        expired_cache_count=expired_cache_count,
        article_fetch_attempt_count=article_fetch_attempt_count,
        article_fetch_retry_count=article_fetch_retry_count,
    )


def fetch_company_news(
    settings: AppSettings,
    *,
    published_before: datetime | None = None,
    lookback_days: int | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
    provider: CompanyNewsProvider | AlphaVantageNewsProvider | None = None,
    article_fetcher: ArticleFetcher | None = None,
    ensure_fresh_until: datetime | None = None,
) -> NewsIngestionResult:
    """
    Fetch, normalize, deduplicate, and persist company news.

    If ensure_fresh_until is set, computes adaptive lookback to fill gaps
    between existing news coverage and the target datetime.
    """

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(runtime_settings, path_manager)
    write_settings_snapshot(runtime_settings, run_context.config_snapshot_path)
    logger = configure_logging(run_context, runtime_settings.run.log_level)
    stage_start = time.perf_counter()

    # If ensure_fresh_until is set, compute adaptive lookback
    effective_lookback_days = lookback_days
    if ensure_fresh_until is not None:
        configured_lookback = lookback_days or runtime_settings.news_ingestion.lookback_days
        adaptive_lookback = compute_adaptive_lookback(
            runtime_settings,
            ticker=ticker,
            exchange=exchange,
            target_end_datetime=ensure_fresh_until,
            configured_lookback_days=configured_lookback,
        )
        effective_lookback_days = adaptive_lookback
        # Also set published_before to ensure_fresh_until if not explicitly provided
        if published_before is None:
            published_before = ensure_fresh_until

    request = build_news_discovery_request(
        runtime_settings,
        published_before=published_before,
        lookback_days=effective_lookback_days,
    )
    acquisition_handler = article_fetcher or acquire_article_text_fallback

    raw_snapshot_path = path_manager.build_raw_news_data_path(
        request.ticker,
        run_context.run_id,
    )
    source_results, source_warnings = collect_news_source_results(
        settings=runtime_settings,
        request=request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=run_context.started_at_utc,
        provider=provider,
    )
    normalized_candidates = [
        article
        for source_result in source_results
        for article in source_result.normalized_articles
    ]
    dropped_rows = [
        dropped_row
        for source_result in source_results
        for dropped_row in source_result.dropped_rows
    ]
    providers_used = [
        source_result.provider_name
        for source_result in source_results
        if source_result.raw_payload.get("status") == "ok"
    ]
    aggregated_provider_request_count = sum(
        source_result.provider_request_count for source_result in source_results
    )
    aggregated_provider_request_retry_count = sum(
        source_result.provider_request_retry_count for source_result in source_results
    )
    aggregated_entity_payload_count = sum(
        source_result.entity_payload_count for source_result in source_results
    )
    aggregated_news_payload_count = sum(
        source_result.news_payload_count for source_result in source_results
    )

    article_fetch_cache_path = path_manager.build_article_fetch_cache_path(
        request.ticker,
        request.exchange,
    )
    article_fetch_cache = load_article_fetch_cache(article_fetch_cache_path)
    normalization_result = acquire_normalized_articles(
        normalized_candidates=normalized_candidates,
        dropped_rows=dropped_rows,
        request=request,
        fetched_at_utc=run_context.started_at_utc,
        news_settings=runtime_settings.news_ingestion,
        article_fetcher=acquisition_handler,
        article_fetch_cache=article_fetch_cache,
    )
    write_json_file(
        article_fetch_cache_path,
        build_article_fetch_cache_payload(
            ticker=request.ticker,
            exchange=request.exchange,
            generated_at_utc=run_context.started_at_utc,
            cache_entries=normalization_result.updated_article_fetch_cache,
        ),
    )
    warnings: list[str] = []
    warnings.extend(sorted(set(source_warnings)))
    for source_result in source_results:
        warnings.extend(source_result.warnings)
    warnings.extend(
        build_degraded_source_warnings(
            request=request,
            final_articles=normalization_result.final_articles,
            providers_used=providers_used,
        )
    )
    if not normalization_result.final_articles:
        warnings.append("no_articles_found")
    warnings = dedupe_preserve_order(warnings)

    cleaned_table_path = path_manager.build_processed_news_data_path(
        request.ticker,
        request.exchange,
    )
    metadata_path = path_manager.build_processed_news_metadata_path(
        request.ticker,
        request.exchange,
    )
    cleaned_table_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame = pd.DataFrame(
        normalization_result.final_articles,
        columns=PROCESSED_NEWS_COLUMNS,
    )
    final_frame.to_csv(cleaned_table_path, index=False)

    raw_snapshot_payload = build_raw_news_snapshot_payload(
        request=request,
        run_id=run_context.run_id,
        fetched_at_utc=run_context.started_at_utc,
        source_results=source_results,
        acquisition_diagnostics=normalization_result.acquisition_diagnostics,
        providers_used=providers_used,
        article_fetch_cache_path=article_fetch_cache_path,
        cache_hit_count=normalization_result.cache_hit_count,
        fresh_fetch_count=normalization_result.fresh_fetch_count,
        expired_cache_count=normalization_result.expired_cache_count,
        provider_request_count=aggregated_provider_request_count,
        provider_request_retry_count=aggregated_provider_request_retry_count,
        article_fetch_attempt_count=normalization_result.article_fetch_attempt_count,
        article_fetch_retry_count=normalization_result.article_fetch_retry_count,
        fetch_policy=build_fetch_policy_metadata(runtime_settings.news_ingestion),
        timing=build_stage_timing_payload(
            started_at_utc=run_context.started_at_utc,
            elapsed_seconds=round(time.perf_counter() - stage_start, 6),
        ),
        workload=build_stage5_workload_payload(
            entity_payload_count=aggregated_entity_payload_count,
            news_payload_count=aggregated_news_payload_count,
            output_row_count=int(len(final_frame)),
            dropped_row_count=int(len(normalization_result.dropped_rows)),
        ),
    )
    write_json_file(raw_snapshot_path, raw_snapshot_payload)

    elapsed_seconds = round(time.perf_counter() - stage_start, 6)
    finished_at_utc = datetime.now(timezone.utc)
    metadata = build_news_metadata(
        request=request,
        provider_label=resolve_provider_label(providers_used),
        cleaned_table_path=cleaned_table_path,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=run_context.started_at_utc,
        source_results=source_results,
        providers_used=providers_used,
        final_frame=final_frame,
        duplicate_count=normalization_result.duplicate_count,
        dropped_rows=normalization_result.dropped_rows,
        warnings=warnings,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
        article_fetch_cache_path=article_fetch_cache_path,
        cache_hit_count=normalization_result.cache_hit_count,
        fresh_fetch_count=normalization_result.fresh_fetch_count,
        expired_cache_count=normalization_result.expired_cache_count,
        provider_request_count=aggregated_provider_request_count,
        provider_request_retry_count=aggregated_provider_request_retry_count,
        article_fetch_attempt_count=normalization_result.article_fetch_attempt_count,
        article_fetch_retry_count=normalization_result.article_fetch_retry_count,
        news_settings=runtime_settings.news_ingestion,
        started_at_utc=run_context.started_at_utc,
        finished_at_utc=finished_at_utc,
        elapsed_seconds=elapsed_seconds,
        entity_payload_count=aggregated_entity_payload_count,
        news_payload_count=aggregated_news_payload_count,
    )
    write_json_file(metadata_path, metadata)

    logger.info(
        "Company news ready | ticker=%s | exchange=%s | providers=%s | rows=%s | dropped_rows=%s | duplicates=%s | cache_hits=%s | acquisition_modes=%s | elapsed=%.3fs | processed_csv=%s",
        request.ticker,
        request.exchange,
        providers_used,
        len(final_frame),
        len(normalization_result.dropped_rows),
        normalization_result.duplicate_count,
        normalization_result.cache_hit_count,
        metadata["text_acquisition_mode_counts"],
        elapsed_seconds,
        cleaned_table_path,
    )

    return NewsIngestionResult(
        raw_snapshot_path=raw_snapshot_path,
        cleaned_table_path=cleaned_table_path,
        metadata_path=metadata_path,
        row_count=len(final_frame),
        duplicate_count=normalization_result.duplicate_count,
        dropped_row_count=len(normalization_result.dropped_rows),
        coverage_start=date.fromisoformat(metadata["coverage_start"]) if metadata["coverage_start"] else None,
        coverage_end=date.fromisoformat(metadata["coverage_end"]) if metadata["coverage_end"] else None,
    )


def compute_adaptive_lookback(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    target_end_datetime: datetime,
    configured_lookback_days: int,
    buffer_days: int = 2,
) -> int:
    """
    Compute adaptive lookback days based on existing news coverage gap.

    If existing coverage ends on March 1 and target is April 2:
    - Gap: 32 days
    - Return: max(gap + buffer, configured_lookback_days)

    This ensures we fill gaps while respecting minimum lookback configuration.
    """
    from pathlib import Path
    import json

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)

    metadata_path = path_manager.build_processed_news_metadata_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )

    # If no existing metadata, use configured lookback
    if not metadata_path.exists():
        return configured_lookback_days

    # Try to read existing coverage
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        coverage_end_str = metadata.get("published_before")
        if not coverage_end_str:
            return configured_lookback_days

        # Parse existing coverage end datetime
        existing_end_datetime = datetime.fromisoformat(coverage_end_str)
        if existing_end_datetime.tzinfo is None:
            existing_end_datetime = existing_end_datetime.replace(tzinfo=timezone.utc)
        else:
            existing_end_datetime = existing_end_datetime.astimezone(timezone.utc)

        # Ensure target is timezone-aware
        if target_end_datetime.tzinfo is None:
            target_end_datetime = target_end_datetime.replace(tzinfo=timezone.utc)
        else:
            target_end_datetime = target_end_datetime.astimezone(timezone.utc)

        # Calculate gap in days
        gap_seconds = (target_end_datetime - existing_end_datetime).total_seconds()
        gap_days = max(0, int(gap_seconds / 86400))  # 86400 seconds per day

        # Return max of (gap + buffer, configured_lookback)
        adaptive_lookback = gap_days + buffer_days
        return max(adaptive_lookback, configured_lookback_days)

    except (json.JSONDecodeError, ValueError, KeyError, OSError):
        # On any error reading metadata, fall back to configured lookback
        return configured_lookback_days


def build_news_discovery_request(
    settings: AppSettings,
    *,
    published_before: datetime | None = None,
    lookback_days: int | None = None,
) -> NewsDiscoveryRequest:
    """Build the normalized news discovery request from settings."""

    resolved_published_before = (
        published_before.astimezone(timezone.utc)
        if published_before is not None
        else datetime.now(timezone.utc)
    )
    resolved_lookback_days = lookback_days or settings.news_ingestion.lookback_days
    if resolved_lookback_days < 1:
        raise NewsIngestionError("News lookback days must be at least 1.")

    return NewsDiscoveryRequest(
        ticker=settings.ticker.symbol,
        exchange=settings.ticker.exchange,
        provider=settings.providers.news_provider,
        company_name=settings.ticker.company_name,
        search_aliases=settings.ticker.search_aliases,
        sector_name=settings.ticker.sector_name,
        industry_name=settings.ticker.industry_name,
        sector_query_terms=settings.ticker.sector_query_terms,
        macro_query_terms=settings.ticker.macro_query_terms,
        lookback_days=resolved_lookback_days,
        published_after=resolved_published_before - timedelta(days=resolved_lookback_days),
        published_before=resolved_published_before,
        language=settings.news_ingestion.language,
        country=settings.news_ingestion.country,
        marketaux_limit_per_request=settings.news_ingestion.marketaux_limit_per_request,
        max_articles_per_run=settings.news_ingestion.max_articles_per_run,
    )


def resolve_news_provider(
    settings: AppSettings,
) -> CompanyNewsProvider | AlphaVantageNewsProvider | None:
    """Resolve the configured paid provider when one is enabled."""

    provider_name = settings.providers.news_provider.strip().lower()
    if provider_name in {"", "not_configured", "none", "disabled", "public"}:
        return None
    if provider_name == "marketaux":
        if not settings.providers.news_api_key:
            raise NewsIngestionError("Marketaux news ingestion requires KUBERA_NEWS_API_KEY.")
        return MarketauxNewsProvider(
            settings.providers.news_api_key,
            timeout_seconds=settings.news_ingestion.request_timeout_seconds,
            retry_attempts=settings.news_ingestion.article_retry_attempts,
        )
    if provider_name == ALPHAVANTAGE_PROVIDER_NAME:
        if not settings.providers.alphavantage_api_key:
            raise NewsIngestionError(
                "Alpha Vantage news ingestion requires KUBERA_ALPHAVANTAGE_API_KEY."
            )
        return AlphaVantageNewsProvider(
            settings.providers.alphavantage_api_key,
            timeout_seconds=settings.news_ingestion.request_timeout_seconds,
            retry_attempts=settings.news_ingestion.article_retry_attempts,
        )
    raise NewsIngestionError(
        f"Unsupported news provider: {settings.providers.news_provider}"
    )


def resolve_provider_entities(
    provider: CompanyNewsProvider,
    request: NewsDiscoveryRequest,
    *,
    provider_request_pause_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], tuple[str, ...], list[dict[str, Any]]]:
    """Resolve the requested ticker through provider entity search."""

    payloads: list[dict[str, Any]] = []
    candidates: list[tuple[int, dict[str, Any]]] = []
    for alias in request.search_aliases:
        pause_before_provider_request(provider_request_pause_seconds)
        payload = provider.search_entities(request, alias)
        payloads.append({"query": alias, "response": payload})
        for entity in payload.get("data", []):
            if not isinstance(entity, dict):
                continue
            score = score_entity_candidate(entity, request, alias)
            if score > 0:
                candidates.append((score, entity))

    candidates.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("symbol", "")).upper(),
            str(item[1].get("name", "")).lower(),
        )
    )
    resolved_symbols: list[str] = []
    entity_matches: list[dict[str, Any]] = []
    for score, entity in candidates:
        symbol = str(entity.get("symbol", "")).strip().upper()
        if not symbol or symbol in resolved_symbols:
            continue
        resolved_symbols.append(symbol)
        entity_matches.append({"score": score, **entity})
        if symbol == request.ticker.upper():
            return payloads, (symbol,), entity_matches[:1]
        if len(resolved_symbols) >= 3:
            break

    return payloads, tuple(resolved_symbols), entity_matches


def score_entity_candidate(
    entity: dict[str, Any],
    request: NewsDiscoveryRequest,
    alias: str,
) -> int:
    """Score one entity result against the configured ticker."""

    score = 0
    symbol = str(entity.get("symbol", "")).strip().upper()
    name = normalize_text_for_matching(entity.get("name"))
    alias_normalized = normalize_text_for_matching(alias)
    company_name = normalize_text_for_matching(request.company_name)
    if symbol == request.ticker.upper():
        score += 100
    if name == company_name:
        score += 60
    if name == alias_normalized:
        score += 30
    if str(entity.get("exchange", "")).strip().upper() == request.exchange.upper():
        score += 20
    if str(entity.get("country", "")).strip().lower() == request.country.lower():
        score += 10
    if str(entity.get("type", "")).strip().lower() == "equity":
        score += 5
    return score


def discover_company_news(
    provider: CompanyNewsProvider,
    request: NewsDiscoveryRequest,
    *,
    resolved_symbols: tuple[str, ...],
    provider_request_pause_seconds: float = 0.0,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Discover provider news payloads for the configured company."""

    discovery_mode = "entity_symbols"
    search_query: str | None = None
    if not resolved_symbols:
        discovery_mode = "search_fallback"
        search_query = build_alias_search_query(request.search_aliases)

    payloads: list[dict[str, Any]] = []
    page = 1
    article_count = 0
    while article_count < request.max_articles_per_run:
        pause_before_provider_request(provider_request_pause_seconds)
        payload = provider.fetch_news_page(
            request,
            page=page,
            symbols=resolved_symbols if resolved_symbols else None,
            search_query=search_query,
        )
        payloads.append(payload)
        page_articles = payload.get("data", [])
        if not isinstance(page_articles, list):
            raise NewsIngestionError("News provider returned an unexpected data payload.")
        article_count += len(page_articles)
        if len(page_articles) < request.marketaux_limit_per_request:
            break
        page += 1

    return discovery_mode, search_query, payloads


def normalize_news_articles(
    *,
    news_payloads: list[dict[str, Any]],
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    news_settings: NewsIngestionSettings,
    article_fetcher: ArticleFetcher,
    discovery_mode: str,
    resolved_symbols: tuple[str, ...],
    article_fetch_cache: dict[str, dict[str, Any]],
) -> NewsNormalizationResult:
    """Normalize provider articles, drop invalid rows, dedupe, and acquire text."""

    normalized_candidates: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for payload in news_payloads:
        articles = payload.get("data", [])
        if not isinstance(articles, list):
            raise NewsIngestionError("News provider returned an unexpected data payload.")
        for article in articles:
            if not isinstance(article, dict):
                continue
            normalized_article, reasons = normalize_marketaux_article(
                article,
                request=request,
                raw_snapshot_path=raw_snapshot_path,
                fetched_at_utc=fetched_at_utc,
                market_settings=market_settings,
                discovery_mode=discovery_mode,
                resolved_symbols=resolved_symbols,
            )
            if reasons:
                dropped_rows.append(
                    {
                        "provider_uuid": article.get("uuid"),
                        "url": article.get("url"),
                        "reasons": reasons,
                    }
                )
                continue
            normalized_candidates.append(normalized_article)

    deduped_articles, duplicate_count = dedupe_normalized_articles(normalized_candidates)
    acquisition_diagnostics: list[dict[str, Any]] = []
    final_articles: list[dict[str, Any]] = []
    updated_article_fetch_cache = dict(article_fetch_cache)
    cache_hit_count = 0
    fresh_fetch_count = 0
    expired_cache_count = 0
    article_fetch_attempt_count = 0
    article_fetch_retry_count = 0
    for article in deduped_articles[: request.max_articles_per_run]:
        cached_fetch_result, cache_age_hours = resolve_cached_article_fetch_result(
            article,
            cache_entries=updated_article_fetch_cache,
            fetched_at_utc=fetched_at_utc,
            ttl_hours=news_settings.article_cache_ttl_hours,
        )
        cache_hit = cached_fetch_result is not None
        if cache_hit:
            cache_hit_count += 1
            fetch_result = cached_fetch_result
        else:
            if cache_age_hours is not None:
                expired_cache_count += 1
            pause_before_article_request(news_settings.article_request_pause_seconds)
            fetch_result = article_fetcher(article, news_settings)
            fresh_fetch_count += 1
            article_fetch_attempt_count += int(fetch_result.attempt_count)
            article_fetch_retry_count += int(fetch_result.retry_count)
            update_article_fetch_cache(
                article,
                cache_entries=updated_article_fetch_cache,
                fetch_result=fetch_result,
                fetched_at_utc=fetched_at_utc,
            )

        final_article = article.copy()
        final_article["full_text"] = fetch_result.full_text
        final_article["content_origin"] = determine_content_origin(
            text_acquisition_mode=fetch_result.text_acquisition_mode,
        )
        final_article["text_acquisition_mode"] = fetch_result.text_acquisition_mode
        final_article["text_acquisition_reason"] = fetch_result.text_acquisition_reason
        final_article["fetch_warning_flag"] = fetch_result.fetch_warning_flag
        final_article["fetch_error"] = fetch_result.fetch_error
        final_article["http_status"] = fetch_result.http_status
        final_articles.append(final_article)
        acquisition_diagnostics.append(
            {
                "article_id": article["article_id"],
                "article_url": article["article_url"],
                "source_name": article.get("source_name"),
                "text_acquisition_mode": fetch_result.text_acquisition_mode,
                "text_acquisition_reason": fetch_result.text_acquisition_reason,
                "fetch_warning_flag": fetch_result.fetch_warning_flag,
                "fetch_error": fetch_result.fetch_error,
                "http_status": fetch_result.http_status,
                "content_origin": final_article["content_origin"],
                "cache_hit": cache_hit,
                "cache_age_hours": cache_age_hours,
                "attempt_count": int(fetch_result.attempt_count) if not cache_hit else 0,
                "retry_count": int(fetch_result.retry_count) if not cache_hit else 0,
            }
        )

    return NewsNormalizationResult(
        final_articles=final_articles,
        dropped_rows=dropped_rows,
        duplicate_count=duplicate_count,
        acquisition_diagnostics=acquisition_diagnostics,
        updated_article_fetch_cache=updated_article_fetch_cache,
        cache_hit_count=cache_hit_count,
        fresh_fetch_count=fresh_fetch_count,
        expired_cache_count=expired_cache_count,
        article_fetch_attempt_count=article_fetch_attempt_count,
        article_fetch_retry_count=article_fetch_retry_count,
    )


def load_article_fetch_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load the persisted article fetch cache if it exists."""

    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        return {}
    return {
        str(cache_key): entry
        for cache_key, entry in raw_entries.items()
        if isinstance(entry, dict)
    }


def build_article_fetch_cache_payload(
    *,
    ticker: str,
    exchange: str,
    generated_at_utc: datetime,
    cache_entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the persisted Stage 5 article fetch cache payload."""

    return {
        "ticker": ticker,
        "exchange": exchange,
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "entry_count": len(cache_entries),
        "entries": cache_entries,
    }


def build_article_fetch_cache_key(article: dict[str, Any]) -> str | None:
    """Build the stable cache key used for article text reuse."""

    article_url = clean_text(article.get("canonical_url")) or clean_text(article.get("article_url"))
    return canonicalize_article_url(article_url)


def resolve_cached_article_fetch_result(
    article: dict[str, Any],
    *,
    cache_entries: dict[str, dict[str, Any]],
    fetched_at_utc: datetime,
    ttl_hours: int,
) -> tuple[ArticleFetchResult | None, float | None]:
    """Return a fresh cached article fetch result when the TTL still allows reuse."""

    if ttl_hours <= 0:
        return None, None

    cache_key = build_article_fetch_cache_key(article)
    if cache_key is None:
        return None, None

    entry = cache_entries.get(cache_key)
    if entry is None:
        return None, None

    cached_at_raw = clean_text(entry.get("cached_at_utc"))
    if not cached_at_raw:
        return None, None

    try:
        cached_at = pd.Timestamp(cached_at_raw)
    except (TypeError, ValueError):
        return None, None
    if cached_at.tzinfo is None:
        cached_at = cached_at.tz_localize("UTC")
    cached_at_utc = cached_at.tz_convert("UTC").to_pydatetime()
    cache_age_hours = (
        fetched_at_utc.astimezone(timezone.utc) - cached_at_utc
    ).total_seconds() / 3600.0
    if cache_age_hours < 0 or cache_age_hours > ttl_hours:
        return None, cache_age_hours

    return (
        ArticleFetchResult(
            full_text=entry.get("full_text"),
            text_acquisition_mode=str(entry.get("text_acquisition_mode") or "headline_only"),
            text_acquisition_reason=str(
                entry.get("text_acquisition_reason") or "article_fetch_cache"
            ),
            fetch_warning_flag=bool(entry.get("fetch_warning_flag")),
            fetch_error=clean_text(entry.get("fetch_error")),
            http_status=(
                int(entry["http_status"])
                if entry.get("http_status") is not None
                else None
            ),
            resolved_article_url=clean_text(entry.get("resolved_article_url")),
            attempt_count=0,
            retry_count=0,
        ),
        cache_age_hours,
    )


def update_article_fetch_cache(
    article: dict[str, Any],
    *,
    cache_entries: dict[str, dict[str, Any]],
    fetch_result: ArticleFetchResult,
    fetched_at_utc: datetime,
) -> None:
    """Persist one article fetch result into the reusable cache."""

    cache_key = build_article_fetch_cache_key(article)
    if cache_key is None:
        return

    cache_entries[cache_key] = {
        "cached_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "article_id": clean_text(article.get("article_id")),
        "canonical_url": clean_text(article.get("canonical_url")),
        "article_url": clean_text(article.get("article_url")),
        "source_name": clean_text(article.get("source_name")),
        "full_text": fetch_result.full_text,
        "text_acquisition_mode": fetch_result.text_acquisition_mode,
        "text_acquisition_reason": fetch_result.text_acquisition_reason,
        "fetch_warning_flag": fetch_result.fetch_warning_flag,
        "fetch_error": fetch_result.fetch_error,
        "http_status": fetch_result.http_status,
        "resolved_article_url": fetch_result.resolved_article_url,
        "attempt_count": int(fetch_result.attempt_count),
        "retry_count": int(fetch_result.retry_count),
    }


def pause_before_article_request(pause_seconds: float) -> None:
    """Apply the configured pacing between uncached article fetches."""

    if pause_seconds > 0:
        time.sleep(pause_seconds)


def pause_before_provider_request(pause_seconds: float) -> None:
    """Apply the configured pacing between provider discovery requests."""

    if pause_seconds > 0:
        time.sleep(pause_seconds)


def determine_content_origin(*, text_acquisition_mode: str) -> str:
    """Classify where the stored article text came from."""

    if text_acquisition_mode == "full_article":
        return "direct_publisher_text"
    if text_acquisition_mode == "headline_plus_snippet":
        return "aggregator_text"
    return "snippet_only"


def normalize_marketaux_article(
    article: dict[str, Any],
    *,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
    discovery_mode: str,
    resolved_symbols: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize one Marketaux article into the processed schema."""

    reasons: list[str] = []
    article_title = clean_text(article.get("title"))
    if not article_title:
        reasons.append("missing_title")

    published_at_raw = clean_text(article.get("published_at"))
    published_at_utc = parse_provider_timestamp(published_at_raw)
    if published_at_utc is None:
        reasons.append("invalid_published_at")

    article_url = clean_text(article.get("url"))
    canonical_url, article_url_validation_reason = validate_article_url(article_url)
    if article_url and canonical_url is None:
        reasons.append(article_url_validation_reason or "invalid_article_url")

    if reasons:
        return {}, reasons

    assert published_at_utc is not None
    published_at_ist = utc_to_market_time(published_at_utc, market_settings)
    provider_uuid = clean_text(article.get("uuid"))
    summary_snippet = clean_text(article.get("description")) or clean_text(article.get("snippet"))
    provider_source = clean_text(article.get("source"))
    source_domain = extract_source_domain(canonical_url or article_url or provider_source)
    source_name = canonicalize_source_name(
        provider_source=provider_source,
        source_domain=source_domain,
    )
    provider_entities = filter_provider_entities(
        article.get("entities"),
        resolved_symbols=resolved_symbols,
    )
    article_id = build_article_id(
        provider_uuid=provider_uuid,
        canonical_url=canonical_url,
        article_url=article_url,
        article_title=article_title,
        published_at_utc=published_at_utc,
    )

    return (
        {
            "article_id": article_id,
            "ticker": request.ticker,
            "exchange": request.exchange,
            "provider": request.provider,
            "discovery_mode": discovery_mode,
            "provider_uuid": provider_uuid,
            "article_title": article_title,
            "article_url": article_url,
            "canonical_url": canonical_url,
            "source_domain": source_domain,
            "provider_source": provider_source,
            "source_name": source_name,
            "published_at_raw": published_at_raw,
            "published_at_utc": published_at_utc.isoformat(),
            "published_at_ist": published_at_ist.isoformat(),
            "published_date_ist": published_at_ist.date().isoformat(),
            "summary_snippet": summary_snippet,
            "full_text": None,
            "content_origin": None,
            "text_acquisition_mode": None,
            "text_acquisition_reason": None,
            "fetch_warning_flag": None,
            "fetch_error": None,
            "http_status": None,
            "provider_entity_payload": json.dumps(provider_entities, sort_keys=True),
            "raw_snapshot_path": str(raw_snapshot_path),
            "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        },
        [],
    )


def build_alphavantage_ticker(request: NewsDiscoveryRequest) -> str:
    """Build the Alpha Vantage ticker argument for the configured company."""

    return request.ticker


def extract_alphavantage_feed(payload: Any) -> list[dict[str, Any]]:
    """Normalize the Alpha Vantage payload into one list of feed rows."""

    if not isinstance(payload, dict):
        raise NewsIngestionError("Alpha Vantage payload was not a JSON object.")
    feed_items = payload.get("feed")
    if isinstance(feed_items, list):
        return [row for row in feed_items if isinstance(row, dict)]
    if payload.get("Information") or payload.get("Error Message") or payload.get("Note"):
        message = (
            clean_text(payload.get("Error Message"))
            or clean_text(payload.get("Information"))
            or clean_text(payload.get("Note"))
        )
        raise NewsIngestionError(message or "Alpha Vantage returned an unusable payload.")
    raise NewsIngestionError("Alpha Vantage payload did not contain a feed list.")


def normalize_alphavantage_feed(
    *,
    feed_items: list[dict[str, Any]],
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize Alpha Vantage feed rows into the processed schema."""

    normalized_articles: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for article in feed_items:
        normalized_article, reasons = normalize_alphavantage_article(
            article,
            request=request,
            raw_snapshot_path=raw_snapshot_path,
            fetched_at_utc=fetched_at_utc,
            market_settings=market_settings,
        )
        if reasons:
            dropped_rows.append(
                {
                    "provider": ALPHAVANTAGE_PROVIDER_NAME,
                    "provider_uuid": clean_text(article.get("url")),
                    "url": article.get("url"),
                    "reasons": reasons,
                }
            )
            continue
        normalized_articles.append(normalized_article)
    return normalized_articles, dropped_rows


def normalize_alphavantage_article(
    article: dict[str, Any],
    *,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize one Alpha Vantage article into the processed schema."""

    reasons: list[str] = []
    article_title = clean_text(article.get("title"))
    if not article_title:
        reasons.append("missing_title")

    published_at_raw = clean_text(article.get("time_published"))
    published_at_utc = parse_provider_timestamp(published_at_raw)
    if published_at_utc is None:
        reasons.append("invalid_published_at")

    article_url = clean_text(article.get("url"))
    canonical_url, article_url_validation_reason = validate_article_url(article_url)
    if article_url and canonical_url is None:
        reasons.append(article_url_validation_reason or "invalid_article_url")

    source_name = clean_text(article.get("source"))
    if not source_name:
        reasons.append("missing_source_name")

    ticker_sentiment = filter_alphavantage_ticker_sentiment(
        article.get("ticker_sentiment"),
        ticker=request.ticker,
    )
    if not ticker_sentiment:
        reasons.append("missing_requested_ticker_sentiment")

    if reasons:
        return {}, reasons

    assert published_at_utc is not None
    published_at_ist = utc_to_market_time(published_at_utc, market_settings)
    source_domain = extract_source_domain(canonical_url or article_url or source_name)
    canonical_source_name = canonicalize_source_name(
        provider_source=source_name,
        source_domain=source_domain,
    )
    summary_snippet = clean_text(article.get("summary"))
    article_id = build_article_id(
        provider_uuid=None,
        canonical_url=canonical_url,
        article_url=article_url,
        article_title=article_title,
        published_at_utc=published_at_utc,
    )
    provider_payload = {
        "ticker_sentiment": ticker_sentiment,
        "overall_sentiment_score": clean_text(article.get("overall_sentiment_score")),
        "overall_sentiment_label": clean_text(article.get("overall_sentiment_label")),
        "topics": article.get("topics"),
    }
    return (
        {
            "article_id": article_id,
            "ticker": request.ticker,
            "exchange": request.exchange,
            "provider": ALPHAVANTAGE_PROVIDER_NAME,
            "discovery_mode": "ticker_sentiment",
            "provider_uuid": None,
            "article_title": article_title,
            "article_url": article_url,
            "canonical_url": canonical_url,
            "source_domain": source_domain,
            "provider_source": source_name,
            "source_name": canonical_source_name,
            "published_at_raw": published_at_raw,
            "published_at_utc": published_at_utc.isoformat(),
            "published_at_ist": published_at_ist.isoformat(),
            "published_date_ist": published_at_ist.date().isoformat(),
            "summary_snippet": summary_snippet,
            "full_text": None,
            "content_origin": None,
            "text_acquisition_mode": None,
            "text_acquisition_reason": None,
            "fetch_warning_flag": None,
            "fetch_error": None,
            "http_status": None,
            "provider_entity_payload": json.dumps(provider_payload, sort_keys=True),
            "raw_snapshot_path": str(raw_snapshot_path),
            "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        },
        [],
    )


def filter_alphavantage_ticker_sentiment(
    payload: Any,
    *,
    ticker: str,
) -> list[dict[str, Any]]:
    """Keep only the Alpha Vantage ticker-sentiment rows that match the active ticker."""

    if not isinstance(payload, list):
        return []
    normalized_ticker = ticker.strip().upper()
    filtered_rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if clean_text(entry.get("ticker")) != normalized_ticker:
            continue
        filtered_rows.append(entry)
    return filtered_rows


def build_google_news_query(request: NewsDiscoveryRequest) -> str:
    """Build the Google News RSS search query for one company."""

    quoted_aliases = tuple(
        f"\"{alias}\"" if " " in alias else alias
        for alias in dict.fromkeys((request.company_name, request.ticker, *request.search_aliases))
        if alias.strip()
    )
    company_clause = " OR ".join(quoted_aliases)

    context_terms: list[str] = []
    if request.industry_name:
        context_terms.append(f"\"{request.industry_name}\"")
    if request.sector_name:
        context_terms.append(f"\"{request.sector_name}\"")
    context_terms.extend(f"\"{term}\"" for term in request.sector_query_terms)
    context_terms.extend(f"\"{term}\"" for term in request.macro_query_terms)
    context_clause = " OR ".join(dict.fromkeys(context_terms))

    if context_clause:
        return f"({company_clause}) ({request.exchange}) ({context_clause})"
    return f"({company_clause}) {request.exchange}"


def normalize_google_news_rss_payload(
    *,
    rss_text: str,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Normalize one Google News RSS response into the processed schema."""

    try:
        root = ET.fromstring(rss_text)
    except ET.ParseError as exc:
        raise NewsIngestionError(f"Google News RSS XML was invalid: {exc}") from exc

    normalized_articles: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    items = root.findall(".//item")
    for item in items:
        normalized_article, reasons = normalize_google_news_rss_item(
            item,
            request=request,
            raw_snapshot_path=raw_snapshot_path,
            fetched_at_utc=fetched_at_utc,
            market_settings=market_settings,
        )
        if reasons:
            dropped_rows.append(
                {
                    "provider": GOOGLE_NEWS_PROVIDER_NAME,
                    "provider_uuid": None,
                    "url": item.findtext("link"),
                    "reasons": reasons,
                }
            )
            continue
        normalized_articles.append(normalized_article)
    return normalized_articles, dropped_rows, len(items)


def normalize_google_news_rss_item(
    item: ET.Element,
    *,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize one Google News RSS item into the processed schema."""

    reasons: list[str] = []
    article_title = clean_text(item.findtext("title"))
    if not article_title:
        reasons.append("missing_title")

    published_at_raw = clean_text(item.findtext("pubDate"))
    published_at_utc = parse_provider_timestamp(published_at_raw)
    if published_at_utc is None:
        reasons.append("invalid_published_at")

    article_url = clean_text(item.findtext("link"))
    validated_article_url, article_url_validation_reason = validate_article_url(article_url)
    if article_url and validated_article_url is None:
        reasons.append(article_url_validation_reason or "invalid_article_url")
    article_url = validated_article_url or article_url

    source_element = item.find("source")
    source_name = clean_text(source_element.text if source_element is not None else None)
    source_url = clean_text(
        source_element.attrib.get("url") if source_element is not None else None
    )
    if not source_name:
        reasons.append("missing_source_name")

    if reasons:
        return {}, reasons

    assert published_at_utc is not None
    published_at_ist = utc_to_market_time(published_at_utc, market_settings)
    source_domain = extract_source_domain(source_url or source_name)
    canonical_source_name = canonicalize_source_name(
        provider_source=source_name,
        source_domain=source_domain,
    )
    article_id = build_article_id(
        provider_uuid=None,
        canonical_url=None,
        article_url=article_url,
        article_title=article_title,
        published_at_utc=published_at_utc,
    )
    provider_payload = {
        "query": build_google_news_query(request),
        "source_url": source_url,
    }
    return (
        {
            "article_id": article_id,
            "ticker": request.ticker,
            "exchange": request.exchange,
            "provider": GOOGLE_NEWS_PROVIDER_NAME,
            "discovery_mode": "rss_search",
            "provider_uuid": None,
            "article_title": article_title,
            "article_url": article_url,
            "canonical_url": None,
            "source_domain": source_domain,
            "provider_source": source_name,
            "source_name": canonical_source_name,
            "published_at_raw": published_at_raw,
            "published_at_utc": published_at_utc.isoformat(),
            "published_at_ist": published_at_ist.isoformat(),
            "published_date_ist": published_at_ist.date().isoformat(),
            "summary_snippet": None,
            "full_text": None,
            "content_origin": None,
            "text_acquisition_mode": None,
            "text_acquisition_reason": None,
            "fetch_warning_flag": None,
            "fetch_error": None,
            "http_status": None,
            "provider_entity_payload": json.dumps(provider_payload, sort_keys=True),
            "raw_snapshot_path": str(raw_snapshot_path),
            "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        },
        [],
    )


def normalize_nse_announcement_rows(
    *,
    announcements: list[dict[str, Any]],
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize NSE announcement rows into the processed schema."""

    normalized_articles: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    for announcement in announcements:
        normalized_article, reasons = normalize_nse_announcement_row(
            announcement,
            request=request,
            raw_snapshot_path=raw_snapshot_path,
            fetched_at_utc=fetched_at_utc,
            market_settings=market_settings,
        )
        if reasons:
            dropped_rows.append(
                {
                    "provider": NSE_ANNOUNCEMENTS_PROVIDER_NAME,
                    "provider_uuid": clean_text(announcement.get("attchmntFile")),
                    "url": announcement.get("attchmntFile"),
                    "reasons": reasons,
                }
            )
            continue
        normalized_articles.append(normalized_article)
    return normalized_articles, dropped_rows


def normalize_nse_announcement_row(
    announcement: dict[str, Any],
    *,
    request: NewsDiscoveryRequest,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    market_settings: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize one NSE corporate announcement into the processed schema."""

    reasons: list[str] = []
    article_title = clean_text(announcement.get("subject"))
    if not article_title:
        reasons.append("missing_title")

    published_at_raw = clean_text(announcement.get("exchdisstime"))
    published_at_utc = parse_market_timestamp(
        published_at_raw,
        market_settings=market_settings,
    )
    if published_at_utc is None:
        reasons.append("invalid_published_at")

    attachment_path = clean_text(announcement.get("attchmntFile"))
    article_url = None
    canonical_url = None
    if attachment_path:
        candidate_url = attachment_path
        if candidate_url.startswith("/"):
            candidate_url = f"{NSE_HOME_URL}{candidate_url}"
        validated_url, article_url_validation_reason = validate_article_url(candidate_url)
        if validated_url is None:
            reasons.append(article_url_validation_reason or "invalid_article_url")
        else:
            article_url = validated_url
            canonical_url = validated_url

    if reasons:
        return {}, reasons

    assert published_at_utc is not None
    published_at_ist = utc_to_market_time(published_at_utc, market_settings)
    source_domain = extract_source_domain(NSE_HOME_URL)
    article_id = build_article_id(
        provider_uuid=attachment_path,
        canonical_url=canonical_url,
        article_url=article_url,
        article_title=article_title,
        published_at_utc=published_at_utc,
    )
    provider_payload = {
        "bflag": clean_text(announcement.get("bflag")),
        "attachment_path": attachment_path,
    }
    return (
        {
            "article_id": article_id,
            "ticker": request.ticker,
            "exchange": request.exchange,
            "provider": NSE_ANNOUNCEMENTS_PROVIDER_NAME,
            "discovery_mode": "corp_announcements",
            "provider_uuid": attachment_path,
            "article_title": article_title,
            "article_url": article_url,
            "canonical_url": canonical_url,
            "source_domain": source_domain,
            "provider_source": "NSE Corporate Announcements",
            "source_name": "NSE Corporate Announcements",
            "published_at_raw": published_at_raw,
            "published_at_utc": published_at_utc.isoformat(),
            "published_at_ist": published_at_ist.isoformat(),
            "published_date_ist": published_at_ist.date().isoformat(),
            "summary_snippet": clean_text(announcement.get("desc")),
            "full_text": None,
            "content_origin": None,
            "text_acquisition_mode": None,
            "text_acquisition_reason": None,
            "fetch_warning_flag": None,
            "fetch_error": None,
            "http_status": None,
            "provider_entity_payload": json.dumps(provider_payload, sort_keys=True),
            "raw_snapshot_path": str(raw_snapshot_path),
            "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        },
        [],
    )

def dedupe_normalized_articles(
    articles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate normalized articles using conservative Stage 5 rules."""

    if not articles:
        return [], 0

    working_frame = pd.DataFrame(articles)
    working_frame["summary_length"] = working_frame["summary_snippet"].fillna("").str.len()
    working_frame["title_normalized"] = working_frame["article_title"].fillna("").map(normalize_text_for_matching)
    working_frame["published_day_utc"] = pd.to_datetime(working_frame["published_at_utc"]).dt.strftime(
        "%Y-%m-%d"
    )
    working_frame = working_frame.sort_values(
        by=[
            "summary_length",
            "published_at_utc",
            "provider_uuid",
            "article_id",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    kept_rows: list[dict[str, Any]] = []
    seen_canonical_urls: set[str] = set()
    seen_source_title_days: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for row in working_frame.to_dict(orient="records"):
        canonical_url = clean_text(row.get("canonical_url"))
        if canonical_url:
            if canonical_url in seen_canonical_urls:
                duplicate_count += 1
                continue
            seen_canonical_urls.add(canonical_url)
        else:
            same_source_key = (
                clean_text(row.get("source_domain")) or "",
                clean_text(row.get("title_normalized")) or "",
                clean_text(row.get("published_day_utc")) or "",
            )
            if all(same_source_key):
                if same_source_key in seen_source_title_days:
                    duplicate_count += 1
                    continue
                seen_source_title_days.add(same_source_key)

        kept_rows.append({column: row.get(column) for column in PROCESSED_NEWS_COLUMNS})

    kept_rows.sort(
        key=lambda row: (
            row["published_at_utc"] or "",
            row["article_id"] or "",
        ),
        reverse=True,
    )
    return kept_rows, duplicate_count


def prioritize_normalized_articles(
    articles: list[dict[str, Any]],
    *,
    request: NewsDiscoveryRequest,
) -> list[dict[str, Any]]:
    """Order deduplicated articles so high-relevance company coverage survives the cap."""

    return sorted(
        articles,
        key=lambda row: (
            provider_priority_rank(
                clean_text(row.get("provider")),
                configured_provider_name=request.provider,
            ),
            -score_company_specificity(row, request=request),
            -resolve_article_sort_timestamp(row),
            clean_text(row.get("source_name")) or "",
            clean_text(row.get("article_id")) or "",
        ),
    )


def provider_priority_rank(
    provider_name: str | None,
    *,
    configured_provider_name: str | None = None,
) -> int:
    """Return the deterministic source priority used before article capping."""

    normalized_provider = clean_text(provider_name) or ""
    normalized_configured_provider = (clean_text(configured_provider_name) or "").lower()
    if normalized_provider == NSE_ANNOUNCEMENTS_PROVIDER_NAME:
        return 0
    if normalized_provider == ALPHAVANTAGE_PROVIDER_NAME:
        return 1
    if normalized_provider == GOOGLE_NEWS_PROVIDER_NAME:
        return 3
    if normalized_provider and normalized_provider == normalized_configured_provider:
        return 2
    if normalized_provider:
        return 2
    return 9


def resolve_article_sort_timestamp(article: dict[str, Any]) -> float:
    """Return a stable UTC timestamp for article ordering."""

    published_at_raw = clean_text(article.get("published_at_utc"))
    if not published_at_raw:
        return 0.0
    try:
        return pd.Timestamp(published_at_raw).timestamp()
    except (TypeError, ValueError):
        return 0.0


def score_company_specificity(
    article: dict[str, Any],
    *,
    request: NewsDiscoveryRequest,
) -> int:
    """Score how explicitly one article appears to target the configured company."""

    title_text = normalize_text_for_matching(article.get("article_title"))
    summary_text = normalize_text_for_matching(article.get("summary_snippet"))
    aliases = build_company_specificity_aliases(request)
    title_match_count = count_alias_matches(title_text, aliases=aliases)
    summary_match_count = count_alias_matches(summary_text, aliases=aliases)
    payload_bonus = score_provider_payload_specificity(article, request=request)
    discovery_bonus = {
        "corp_announcements": 4,
        "ticker_sentiment": 3,
        "entity_symbols": 2,
        "search_fallback": 1,
    }.get(clean_text(article.get("discovery_mode")) or "", 0)
    return min(title_match_count, 3) * 4 + min(summary_match_count, 2) * 2 + payload_bonus + discovery_bonus


def build_company_specificity_aliases(request: NewsDiscoveryRequest) -> tuple[str, ...]:
    """Build the normalized company-alias list used for relevance scoring."""

    aliases: list[str] = []
    seen_aliases: set[str] = set()
    for candidate in (request.ticker, request.company_name, *request.search_aliases):
        normalized_candidate = normalize_text_for_matching(candidate)
        if not normalized_candidate or normalized_candidate in seen_aliases:
            continue
        seen_aliases.add(normalized_candidate)
        aliases.append(normalized_candidate)
    aliases.sort(key=lambda value: (-len(value), value))
    return tuple(aliases)


def count_alias_matches(text: str, *, aliases: tuple[str, ...]) -> int:
    """Count distinct alias matches within normalized free text."""

    if not text:
        return 0
    return sum(1 for alias in aliases if text_contains_alias(text, alias))


def text_contains_alias(text: str, alias: str) -> bool:
    """Return True when the normalized alias appears as its own token span."""

    if not text or not alias:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def score_provider_payload_specificity(
    article: dict[str, Any],
    *,
    request: NewsDiscoveryRequest,
) -> int:
    """Score specificity signals carried inside the provider payload."""

    payload = parse_provider_entity_payload(article.get("provider_entity_payload"))
    if isinstance(payload, dict):
        ticker_sentiment = payload.get("ticker_sentiment")
        if isinstance(ticker_sentiment, list) and ticker_sentiment:
            return 3
        payload_name = normalize_text_for_matching(payload.get("name"))
        if payload_name and payload_name == normalize_text_for_matching(request.company_name):
            return 2
        return 0
    if not isinstance(payload, list):
        return 0

    normalized_company_name = normalize_text_for_matching(request.company_name)
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if clean_text(entry.get("symbol")) == request.ticker:
            return 3
        if normalize_text_for_matching(entry.get("name")) == normalized_company_name:
            return 2
    return 0


def parse_provider_entity_payload(payload: Any) -> Any:
    """Parse the persisted provider entity payload when it is saved as JSON text."""

    if isinstance(payload, (list, dict)):
        return payload
    if not isinstance(payload, str):
        return None
    try:
        return json.loads(payload)
    except ValueError:
        return None


def build_degraded_source_warnings(
    *,
    request: NewsDiscoveryRequest,
    final_articles: list[dict[str, Any]],
    providers_used: list[str],
) -> list[str]:
    """Derive run-level warnings when Stage 5 coverage quality looks degraded."""

    if not final_articles:
        return []

    warnings: list[str] = []
    normalized_providers = {
        clean_text(provider_name) or ""
        for provider_name in providers_used
        if clean_text(provider_name)
    }
    if normalized_providers == {GOOGLE_NEWS_PROVIDER_NAME}:
        warnings.append("degraded_source_google_only")

    fallback_count = sum(
        1
        for article in final_articles
        if clean_text(article.get("content_origin")) != "direct_publisher_text"
    )
    if fallback_count / len(final_articles) > 0.5:
        warnings.append("degraded_source_fallback_majority")

    low_specificity_count = sum(
        1
        for article in final_articles
        if score_company_specificity(article, request=request) < 4
    )
    if low_specificity_count / len(final_articles) > 0.5:
        warnings.append("degraded_source_low_specificity")

    return warnings


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Collapse duplicate warning codes while preserving their first-seen order."""

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned_value = clean_text(value)
        if not cleaned_value or cleaned_value in seen:
            continue
        seen.add(cleaned_value)
        deduped.append(cleaned_value)
    return deduped


def acquire_article_text_fallback(
    article: dict[str, Any],
    settings: NewsIngestionSettings,
) -> ArticleFetchResult:
    """Fetch article HTML and degrade to provider text when full extraction is weak."""

    article_url = clean_text(article.get("article_url")) or clean_text(article.get("canonical_url"))
    validated_article_url, article_url_validation_reason = validate_article_url(article_url)
    if validated_article_url is None:
        return build_article_fallback_result(
            article,
            reason=article_url_validation_reason or "invalid_article_url",
            attempt_count=0,
            retry_count=0,
        )
    article_url = validated_article_url

    last_error: str | None = None
    last_status: int | None = None
    attempt_count = 0
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml",
    }
    for attempt in range(1, settings.article_retry_attempts + 1):
        attempt_count = attempt
        try:
            response = requests.get(
                article_url,
                headers=headers,
                timeout=settings.article_fetch_timeout_seconds,
            )
            last_status = response.status_code
            response.raise_for_status()
            resolved_article_url, _ = validate_article_url(
                clean_text(getattr(response, "url", None)) or article_url
            )
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if content_type and "html" not in content_type:
                return build_article_fallback_result(
                    article,
                    reason="non_html_response",
                    http_status=last_status,
                    resolved_article_url=resolved_article_url,
                    attempt_count=attempt_count,
                    retry_count=max(attempt_count - 1, 0),
                )

            extracted_text, extraction_reason = extract_article_text_from_html(response.text)
            is_usable, usability_reason = is_materially_richer_article_text(
                extracted_text,
                article_title=clean_text(article.get("article_title")),
                summary_snippet=clean_text(article.get("summary_snippet")),
                min_chars=settings.full_text_min_chars,
            )
            if is_usable and extracted_text is not None:
                return ArticleFetchResult(
                    full_text=extracted_text,
                    text_acquisition_mode="full_article",
                    text_acquisition_reason=extraction_reason,
                    fetch_warning_flag=False,
                    fetch_error=None,
                    http_status=last_status,
                    resolved_article_url=resolved_article_url,
                    attempt_count=attempt_count,
                    retry_count=max(attempt_count - 1, 0),
                )

            return build_article_fallback_result(
                article,
                reason=usability_reason or extraction_reason,
                http_status=last_status,
                resolved_article_url=resolved_article_url,
                attempt_count=attempt_count,
                retry_count=max(attempt_count - 1, 0),
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == settings.article_retry_attempts:
                break
            time.sleep(0.5 * attempt)

    return build_article_fallback_result(
        article,
        reason="page_fetch_failed",
        fetch_error=last_error,
        http_status=last_status,
        attempt_count=attempt_count,
        retry_count=max(attempt_count - 1, 0),
    )


def build_article_fallback_result(
    article: dict[str, Any],
    *,
    reason: str,
    fetch_error: str | None = None,
    http_status: int | None = None,
    resolved_article_url: str | None = None,
    attempt_count: int = 1,
    retry_count: int = 0,
) -> ArticleFetchResult:
    """Build the degraded text result from provider headline and snippet fields."""

    article_title = clean_text(article.get("article_title")) or ""
    summary_snippet = clean_text(article.get("summary_snippet"))
    if summary_snippet:
        return ArticleFetchResult(
            full_text=build_provider_fallback_text(article_title, summary_snippet),
            text_acquisition_mode="headline_plus_snippet",
            text_acquisition_reason=reason,
            fetch_warning_flag=True,
            fetch_error=fetch_error,
            http_status=http_status,
            resolved_article_url=resolved_article_url,
            attempt_count=attempt_count,
            retry_count=retry_count,
        )
    return ArticleFetchResult(
        full_text=article_title,
        text_acquisition_mode="headline_only",
        text_acquisition_reason=reason,
        fetch_warning_flag=True,
        fetch_error=fetch_error,
        http_status=http_status,
        resolved_article_url=resolved_article_url,
        attempt_count=attempt_count,
        retry_count=retry_count,
    )


def build_provider_fallback_text(article_title: str, summary_snippet: str | None) -> str:
    """Compose the fallback text payload for later extraction stages."""

    if summary_snippet:
        return f"{article_title}\n\n{summary_snippet}"
    return article_title


def extract_article_text_from_html(html: str) -> tuple[str | None, str]:
    """Extract article body text using article, main, then paragraph fallback."""

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ARTICLE_STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    article_text = extract_paragraph_text(soup.find("article"))
    if article_text:
        return article_text, "article_tag"

    main_text = extract_paragraph_text(soup.find("main"))
    if main_text:
        return main_text, "main_tag"

    paragraph_text = extract_paragraph_text(soup)
    if paragraph_text:
        return paragraph_text, "paragraph_fallback"

    return None, "no_article_text_found"


def extract_paragraph_text(container: Any) -> str | None:
    """Extract deduplicated paragraph text from one HTML container."""

    if container is None:
        return None

    paragraphs: list[str] = []
    seen_paragraphs: set[str] = set()
    for paragraph in container.find_all("p"):
        text = collapse_whitespace(paragraph.get_text(" ", strip=True))
        if not text:
            continue
        normalized_text = normalize_text_for_matching(text)
        if normalized_text in seen_paragraphs:
            continue
        seen_paragraphs.add(normalized_text)
        paragraphs.append(text)

    if not paragraphs:
        return None
    return "\n\n".join(paragraphs)


def is_materially_richer_article_text(
    extracted_text: str | None,
    *,
    article_title: str | None,
    summary_snippet: str | None,
    min_chars: int,
) -> tuple[bool, str | None]:
    """Validate that fetched article text is richer than provider headline/snippet text."""

    cleaned_text = clean_text(extracted_text)
    if cleaned_text is None:
        return False, "no_article_text_found"
    if len(cleaned_text) < min_chars:
        return False, "article_text_below_min_chars"

    baseline_text = build_provider_fallback_text(article_title or "", summary_snippet)
    normalized_candidate = normalize_text_for_matching(cleaned_text)
    normalized_baseline = normalize_text_for_matching(baseline_text)
    if normalized_baseline and normalized_candidate == normalized_baseline:
        return False, "article_text_matches_provider_text"
    if normalized_baseline and len(normalized_candidate) < len(normalized_baseline) + 80:
        return False, "article_text_not_materially_richer"
    return True, None


def build_raw_news_snapshot_payload(
    *,
    request: NewsDiscoveryRequest,
    run_id: str,
    fetched_at_utc: datetime,
    source_results: list[CollectedNewsSource],
    acquisition_diagnostics: list[dict[str, Any]],
    providers_used: list[str],
    article_fetch_cache_path: Path,
    cache_hit_count: int,
    fresh_fetch_count: int,
    expired_cache_count: int,
    provider_request_count: int,
    provider_request_retry_count: int,
    article_fetch_attempt_count: int,
    article_fetch_retry_count: int,
    fetch_policy: dict[str, Any],
    timing: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    """Build the raw run snapshot payload for Stage 5."""

    primary_result = next(
        (result for result in source_results if result.discovery_mode or result.resolved_symbols),
        source_results[0] if source_results else None,
    )
    return {
        "provider": resolve_provider_label(providers_used),
        "providers_used": providers_used,
        "ticker": request.ticker,
        "exchange": request.exchange,
        "company_name": request.company_name,
        "search_aliases": list(request.search_aliases),
        "lookback_days": request.lookback_days,
        "published_after": request.published_after.isoformat(),
        "published_before": request.published_before.isoformat(),
        "language": request.language,
        "country": request.country,
        "run_id": run_id,
        "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "discovery_mode": primary_result.discovery_mode if primary_result else None,
        "search_query": primary_result.search_query if primary_result else None,
        "resolved_symbols": list(primary_result.resolved_symbols) if primary_result else [],
        "entity_matches": primary_result.entity_matches if primary_result else [],
        "article_fetch_cache_path": str(article_fetch_cache_path),
        "cache_hit_count": cache_hit_count,
        "fresh_fetch_count": fresh_fetch_count,
        "expired_cache_count": expired_cache_count,
        "provider_request_count": int(provider_request_count),
        "provider_request_retry_count": int(provider_request_retry_count),
        "article_fetch_attempt_count": int(article_fetch_attempt_count),
        "article_fetch_retry_count": int(article_fetch_retry_count),
        "fetch_policy": fetch_policy,
        "timing": timing,
        "workload": workload,
        "provider_summaries": [build_provider_summary_payload(result) for result in source_results],
        "article_fetch_diagnostics": acquisition_diagnostics,
    }


def build_news_metadata(
    *,
    request: NewsDiscoveryRequest,
    provider_label: str,
    cleaned_table_path: Path,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    source_results: list[CollectedNewsSource],
    providers_used: list[str],
    final_frame: pd.DataFrame,
    duplicate_count: int,
    dropped_rows: list[dict[str, Any]],
    warnings: list[str],
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
    article_fetch_cache_path: Path,
    cache_hit_count: int,
    fresh_fetch_count: int,
    expired_cache_count: int,
    provider_request_count: int,
    provider_request_retry_count: int,
    article_fetch_attempt_count: int,
    article_fetch_retry_count: int,
    news_settings: NewsIngestionSettings,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    elapsed_seconds: float,
    entity_payload_count: int,
    news_payload_count: int,
) -> dict[str, Any]:
    """Build the metadata payload for the persisted Stage 5 outputs."""

    coverage_start = None
    coverage_end = None
    if not final_frame.empty:
        coverage_start = str(final_frame["published_date_ist"].min())
        coverage_end = str(final_frame["published_date_ist"].max())
    primary_result = next(
        (result for result in source_results if result.discovery_mode or result.resolved_symbols),
        source_results[0] if source_results else None,
    )
    coverage_notes = build_news_coverage_notes(
        request=request,
        source_results=source_results,
        final_frame=final_frame,
    )

    return {
        "provider": provider_label,
        "providers_used": providers_used,
        "ticker": request.ticker,
        "exchange": request.exchange,
        "company_name": request.company_name,
        "search_aliases": list(request.search_aliases),
        "lookback_days": request.lookback_days,
        "published_after": request.published_after.isoformat(),
        "published_before": request.published_before.isoformat(),
        "language": request.language,
        "country": request.country,
        "discovery_mode": primary_result.discovery_mode if primary_result else None,
        "search_query": primary_result.search_query if primary_result else None,
        "resolved_symbols": list(primary_result.resolved_symbols) if primary_result else [],
        "entity_matches": primary_result.entity_matches if primary_result else [],
        "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "processed_news_path": str(cleaned_table_path),
        "processed_news_hash": compute_file_sha256(cleaned_table_path),
        "raw_snapshot_path": str(raw_snapshot_path),
        "raw_snapshot_hash": compute_file_sha256(raw_snapshot_path),
        "article_fetch_cache_path": str(article_fetch_cache_path),
        "article_fetch_cache_hash": compute_file_sha256(article_fetch_cache_path),
        "row_count": int(len(final_frame)),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "duplicate_count": duplicate_count,
        "dropped_row_count": len(dropped_rows),
        "dropped_rows": dropped_rows,
        "cache_hit_count": cache_hit_count,
        "fresh_fetch_count": fresh_fetch_count,
        "expired_cache_count": expired_cache_count,
        "provider_request_count": int(provider_request_count),
        "provider_request_retry_count": int(provider_request_retry_count),
        "article_fetch_attempt_count": int(article_fetch_attempt_count),
        "article_fetch_retry_count": int(article_fetch_retry_count),
        "fetch_policy": build_fetch_policy_metadata(news_settings),
        "timing": build_stage_timing_payload(
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            elapsed_seconds=elapsed_seconds,
        ),
        "workload": build_stage5_workload_payload(
            entity_payload_count=entity_payload_count,
            news_payload_count=news_payload_count,
            output_row_count=int(len(final_frame)),
            dropped_row_count=int(len(dropped_rows)),
        ),
        "provider_summaries": [build_provider_summary_payload(result) for result in source_results],
        "provider_limitations": describe_provider_limitations(providers_used),
        "source_terms_review_required": True,
        "coverage_notes": coverage_notes,
        "source_name_counts": count_series_values(final_frame, "source_name"),
        "text_acquisition_mode_counts": count_series_values(final_frame, "text_acquisition_mode"),
        "content_origin_counts": count_series_values(final_frame, "content_origin"),
        "source_domain_counts": count_series_values(final_frame, "source_domain"),
        "warnings": warnings,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def build_stage_timing_payload(
    *,
    started_at_utc: datetime,
    elapsed_seconds: float,
    finished_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build one shared stage-timing payload."""

    return {
        "started_at_utc": started_at_utc.astimezone(timezone.utc).isoformat(),
        "finished_at_utc": (
            finished_at_utc.astimezone(timezone.utc).isoformat()
            if finished_at_utc is not None
            else None
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def build_stage5_workload_payload(
    *,
    entity_payload_count: int,
    news_payload_count: int,
    output_row_count: int,
    dropped_row_count: int,
) -> dict[str, int]:
    """Build the shared Stage 5 workload summary."""

    return {
        "entity_payload_count": int(entity_payload_count),
        "news_payload_count": int(news_payload_count),
        "output_row_count": int(output_row_count),
        "dropped_row_count": int(dropped_row_count),
    }


def build_fetch_policy_metadata(settings: NewsIngestionSettings) -> dict[str, Any]:
    """Describe the active Stage 5 fetch policy in metadata."""

    return {
        "request_timeout_seconds": settings.request_timeout_seconds,
        "article_fetch_timeout_seconds": settings.article_fetch_timeout_seconds,
        "article_retry_attempts": settings.article_retry_attempts,
        "article_cache_ttl_hours": settings.article_cache_ttl_hours,
        "provider_request_pause_seconds": settings.provider_request_pause_seconds,
        "article_request_pause_seconds": settings.article_request_pause_seconds,
        "full_text_min_chars": settings.full_text_min_chars,
        "enable_google_news_rss": settings.enable_google_news_rss,
        "enable_nse_announcements": settings.enable_nse_announcements,
    }


def describe_provider_limitations(provider_names: list[str]) -> list[str]:
    """Record known Stage 5 discovery and fetch limits for the active provider set."""

    limitations: list[str] = []
    normalized_providers = sorted({clean_text(name) or "unknown_provider" for name in provider_names})
    if "marketaux" in normalized_providers:
        limitations.extend(
            [
                "Entity resolution depends on provider search coverage and symbol mapping quality.",
                "Indian equity coverage can be sparse outside larger companies and major events.",
            ]
        )
    if ALPHAVANTAGE_PROVIDER_NAME in normalized_providers:
        limitations.extend(
            [
                "Alpha Vantage NEWS_SENTIMENT coverage is ticker-scoped and can be throttled on free-tier access.",
                "Provider summaries can still require direct article fetches to recover richer publisher text.",
            ]
        )
    if GOOGLE_NEWS_PROVIDER_NAME in normalized_providers:
        limitations.append(
            "Google News RSS feed links can be aggregator redirects and may fall back to headline-level text."
        )
    if NSE_ANNOUNCEMENTS_PROVIDER_NAME in normalized_providers:
        limitations.append(
            "NSE announcement coverage is highly relevant but is limited to corporate filings rather than broader news."
        )
    limitations.append("Public article URLs can disappear, block scraping, or return aggregator snippets only.")
    limitations.append("Provider and publisher terms should be reviewed before wider automation.")
    return limitations


def build_news_coverage_notes(
    *,
    request: NewsDiscoveryRequest,
    source_results: list[CollectedNewsSource],
    final_frame: pd.DataFrame,
) -> list[str]:
    """Summarize the practical Stage 5 coverage conditions for one run."""

    notes = [
        f"Discovery used aliases {list(request.search_aliases)} across {len(source_results)} source adapters.",
    ]
    for source_result in source_results:
        if source_result.resolved_symbols:
            notes.append(
                f"{source_result.provider_name} resolved symbols {list(source_result.resolved_symbols)}."
            )
        elif source_result.discovery_mode:
            notes.append(
                f"{source_result.provider_name} used discovery mode {source_result.discovery_mode}."
            )
    if final_frame.empty:
        notes.append("No normalized articles were available in the requested lookback window.")
        return notes

    unique_sources = int(final_frame["source_name"].nunique(dropna=True))
    notes.append(f"Usable articles came from {unique_sources} canonical sources.")
    if (final_frame["content_origin"] == "direct_publisher_text").any():
        notes.append("At least one article used direct publisher page text.")
    if (final_frame["content_origin"] != "direct_publisher_text").any():
        notes.append("Some articles relied on aggregator or snippet fallback text.")
    return notes


def resolve_provider_label(providers_used: list[str]) -> str:
    """Collapse the active provider set into the persisted top-level label."""

    if not providers_used:
        return "multi_source"
    unique_providers = sorted(set(providers_used))
    if len(unique_providers) == 1:
        return unique_providers[0]
    return "multi_source"


def build_provider_summary_payload(source_result: CollectedNewsSource) -> dict[str, Any]:
    """Build one provider-level summary payload for raw snapshots and metadata."""

    return {
        "provider": source_result.provider_name,
        "status": source_result.raw_payload.get("status", "ok"),
        "warning_count": len(source_result.warnings),
        "warnings": source_result.warnings,
        "normalized_article_count": len(source_result.normalized_articles),
        "dropped_row_count": len(source_result.dropped_rows),
        "provider_request_count": int(source_result.provider_request_count),
        "provider_request_retry_count": int(source_result.provider_request_retry_count),
        "discovery_mode": source_result.discovery_mode,
        "search_query": source_result.search_query,
        "resolved_symbols": list(source_result.resolved_symbols),
        "entity_matches": source_result.entity_matches,
        "raw_payload": source_result.raw_payload,
    }


def count_series_values(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    """Count string values in one DataFrame column for metadata."""

    if frame.empty:
        return {}
    counts = frame[column_name].fillna("null").value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def sanitize_provider_warning(value: str) -> str:
    """Collapse one provider warning into a log-safe short string."""

    return " ".join(str(value).split())


def parse_provider_timestamp(raw_value: str | None) -> datetime | None:
    """Parse a provider timestamp into a UTC datetime."""

    if raw_value is None or not raw_value.strip():
        return None
    compact_value = raw_value.strip()
    for format_string, expected_length in (
        ("%Y%m%dT%H%M%S", 15),
        ("%Y%m%dT%H%M", 13),
    ):
        if len(compact_value) != expected_length:
            continue
        try:
            return datetime.strptime(compact_value, format_string).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    try:
        timestamp = pd.Timestamp(compact_value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def parse_market_timestamp(
    raw_value: str | None,
    *,
    market_settings: Any,
) -> datetime | None:
    """Parse a provider timestamp and treat naive values as market-local time."""

    if raw_value is None or not raw_value.strip():
        return None
    try:
        timestamp = pd.Timestamp(raw_value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(market_settings.timezone_name)
    return timestamp.tz_convert("UTC").to_pydatetime()


def format_marketaux_datetime(value: datetime) -> str:
    """Format a UTC datetime for Marketaux query parameters."""

    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def format_alphavantage_datetime(value: datetime) -> str:
    """Format a UTC datetime for Alpha Vantage NEWS_SENTIMENT filters."""

    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def build_alias_search_query(search_aliases: tuple[str, ...]) -> str:
    """Build a conservative OR query from configured aliases."""

    return " | ".join(f"\"{escape_search_term(alias)}\"" for alias in search_aliases)


def escape_search_term(value: str) -> str:
    """Escape special Marketaux search characters."""

    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
    return escaped


def filter_provider_entities(
    raw_entities: Any,
    *,
    resolved_symbols: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Keep the provider entity payload focused on the resolved ticker symbols."""

    if not isinstance(raw_entities, list):
        return []
    if not resolved_symbols:
        return [entity for entity in raw_entities if isinstance(entity, dict)]
    resolved_symbol_set = {symbol.upper() for symbol in resolved_symbols}
    filtered = [
        entity
        for entity in raw_entities
        if isinstance(entity, dict)
        and str(entity.get("symbol", "")).strip().upper() in resolved_symbol_set
    ]
    return filtered


def build_article_id(
    *,
    provider_uuid: str | None,
    canonical_url: str | None,
    article_url: str | None,
    article_title: str,
    published_at_utc: datetime,
) -> str:
    """Build a stable article identifier for persisted rows."""

    if provider_uuid:
        return provider_uuid
    digest = hashlib.sha256()
    for part in (
        canonical_url or "",
        article_url or "",
        article_title,
        published_at_utc.isoformat(),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:24]


def canonicalize_article_url(raw_url: str | None) -> str | None:
    """Canonicalize article URLs for dedupe and traceability."""

    canonical_url, _ = validate_article_url(raw_url)
    return canonical_url


def validate_article_url(raw_url: str | None) -> tuple[str | None, str | None]:
    """Validate and canonicalize an article URL for safe fetching."""

    cleaned_url = clean_text(raw_url)
    if not cleaned_url:
        return None, None

    try:
        parsed = urlparse(cleaned_url)
    except ValueError:
        return None, "malformed_article_url"

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None, "invalid_article_url"
    if parsed.username or parsed.password:
        return None, "article_url_contains_credentials"

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return None, "invalid_article_url"
    if not is_safe_public_article_host(hostname):
        return None, "disallowed_article_url_host"

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.strip().lower()
        if any(
            normalized_key == prefix or normalized_key.startswith(prefix)
            for prefix in TRACKING_QUERY_PREFIXES
        ):
            continue
        query_items.append((key, value))
    query_items.sort()

    try:
        port = parsed.port
    except ValueError:
        return None, "malformed_article_url"
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    return (
        urlunparse(
            (
                parsed.scheme.lower(),
                netloc,
                path,
                "",
                urlencode(query_items, doseq=True),
                "",
            )
        ),
        None,
    )


def is_safe_public_article_host(hostname: str) -> bool:
    """Return True when the resolved article host looks public and fetch-safe."""

    normalized_host = hostname.strip().strip(".").lower()
    if not normalized_host or normalized_host == "localhost" or "." not in normalized_host:
        return False

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        address = None
    if address is not None:
        return bool(address.is_global)

    if len(normalized_host) > 253:
        return False

    labels = normalized_host.split(".")
    if not labels or labels[-1].isdigit():
        return False
    for label in labels:
        if not label or label.startswith("-") or label.endswith("-"):
            return False
        if not HOSTNAME_LABEL_PATTERN.fullmatch(label):
            return False
    return True


def extract_source_domain(raw_value: str | None) -> str | None:
    """Extract a normalized source domain from a URL or domain string."""

    cleaned_value = clean_text(raw_value)
    if not cleaned_value:
        return None
    parsed = urlparse(cleaned_value if "://" in cleaned_value else f"https://{cleaned_value}")
    hostname = (parsed.hostname or "").lower()
    return hostname or None


def canonicalize_source_name(
    *,
    provider_source: str | None,
    source_domain: str | None,
) -> str | None:
    """Build a stable publisher name while preserving the raw provider label separately."""

    cleaned_source = clean_text(provider_source)
    domain_label = humanize_domain_label(source_domain)
    if cleaned_source is None:
        return domain_label

    normalized_source = normalize_text_for_matching(cleaned_source)
    normalized_domain = normalize_text_for_matching(source_domain)
    normalized_domain_label = normalize_text_for_matching(domain_label)
    if normalized_source in {normalized_domain, normalized_domain_label}:
        return domain_label
    return collapse_whitespace(cleaned_source)


def humanize_domain_label(source_domain: str | None) -> str | None:
    """Convert a hostname into a readable canonical source label."""

    cleaned_domain = clean_text(source_domain)
    if not cleaned_domain:
        return None

    domain = cleaned_domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    domain_parts = [part for part in domain.split(".") if part]
    if not domain_parts:
        return None

    root = domain_parts[0]
    if len(domain_parts) >= 3 and root in {"co", "com", "net", "org"}:
        root = domain_parts[-3]
    words = [word for word in root.replace("-", " ").replace("_", " ").split() if word]
    if not words:
        return None
    return " ".join(word.capitalize() for word in words)


def normalize_text_for_matching(value: Any) -> str:
    """Normalize free text for stable matching and dedupe keys."""

    cleaned = clean_text(value)
    return " ".join(cleaned.lower().split()) if cleaned else ""


def collapse_whitespace(value: str) -> str:
    """Collapse repeated internal whitespace while preserving original casing."""

    return " ".join(value.split())


def clean_text(value: Any) -> str | None:
    """Trim text values and collapse empty strings to None."""

    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def parse_published_before(raw_value: str) -> datetime:
    """Parse an ISO-like published-before CLI value into UTC."""

    timestamp = pd.Timestamp(raw_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 5 news-ingestion command arguments."""

    parser = argparse.ArgumentParser(description="Fetch Kubera company news.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Override the configured news lookback window in days.",
    )
    parser.add_argument(
        "--published-before",
        help="Use a specific UTC cutoff in an ISO-style format supported by pandas Timestamp.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 5 news-ingestion command."""

    args = parse_args(argv)
    settings = load_settings()
    published_before = (
        parse_published_before(args.published_before)
        if args.published_before
        else None
    )
    fetch_company_news(
        settings,
        published_before=published_before,
        lookback_days=args.lookback_days,
        ticker=args.ticker,
        exchange=args.exchange,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
