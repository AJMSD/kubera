"""Company news ingestion for Kubera."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

from kubera.config import AppSettings, NewsIngestionSettings, load_settings
from kubera.utils.hashing import compute_file_sha256
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot
from kubera.utils.time_utils import utc_to_market_time


MARKETAUX_ENTITY_SEARCH_URL = "https://api.marketaux.com/v1/entity/search"
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"
TRACKING_QUERY_PREFIXES = ("utm_", "ga_", "fbclid", "gclid", "mc_", "ref")
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
    "published_at_raw",
    "published_at_utc",
    "published_at_ist",
    "published_date_ist",
    "summary_snippet",
    "full_text",
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
                time.sleep(0.5 * attempt)
        raise NewsIngestionError(f"News provider request failed: {last_error}") from last_error


def fetch_company_news(
    settings: AppSettings,
    *,
    published_before: datetime | None = None,
    lookback_days: int | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
    provider: CompanyNewsProvider | None = None,
    article_fetcher: ArticleFetcher | None = None,
) -> NewsIngestionResult:
    """Fetch, normalize, deduplicate, and persist company news."""

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

    request = build_news_discovery_request(
        runtime_settings,
        published_before=published_before,
        lookback_days=lookback_days,
    )
    news_provider = provider or resolve_news_provider(runtime_settings)
    if request.provider != news_provider.provider_name:
        request = replace(request, provider=news_provider.provider_name)
    acquisition_handler = article_fetcher or acquire_article_text_fallback

    entity_payloads, resolved_symbols, entity_matches = resolve_provider_entities(
        news_provider,
        request,
    )
    discovery_mode, search_query, news_payloads = discover_company_news(
        news_provider,
        request,
        resolved_symbols=resolved_symbols,
    )

    raw_snapshot_path = path_manager.build_raw_news_data_path(
        request.ticker,
        run_context.run_id,
    )
    normalized_articles, dropped_rows, duplicate_count, acquisition_diagnostics = normalize_news_articles(
        news_payloads=news_payloads,
        request=request,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=run_context.started_at_utc,
        market_settings=runtime_settings.market,
        news_settings=runtime_settings.news_ingestion,
        article_fetcher=acquisition_handler,
        discovery_mode=discovery_mode,
        resolved_symbols=resolved_symbols,
    )
    warnings: list[str] = []
    if discovery_mode == "search_fallback":
        warnings.append("search_fallback_used")
    if not normalized_articles:
        warnings.append("no_articles_found")

    cleaned_table_path = path_manager.build_processed_news_data_path(
        request.ticker,
        request.exchange,
    )
    metadata_path = path_manager.build_processed_news_metadata_path(
        request.ticker,
        request.exchange,
    )
    cleaned_table_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame = pd.DataFrame(normalized_articles, columns=PROCESSED_NEWS_COLUMNS)
    final_frame.to_csv(cleaned_table_path, index=False)

    raw_snapshot_payload = build_raw_news_snapshot_payload(
        request=request,
        run_id=run_context.run_id,
        fetched_at_utc=run_context.started_at_utc,
        entity_payloads=entity_payloads,
        news_payloads=news_payloads,
        acquisition_diagnostics=acquisition_diagnostics,
        entity_matches=entity_matches,
        resolved_symbols=resolved_symbols,
        discovery_mode=discovery_mode,
        search_query=search_query,
    )
    write_json_file(raw_snapshot_path, raw_snapshot_payload)

    metadata = build_news_metadata(
        request=request,
        news_provider=news_provider,
        cleaned_table_path=cleaned_table_path,
        raw_snapshot_path=raw_snapshot_path,
        fetched_at_utc=run_context.started_at_utc,
        entity_matches=entity_matches,
        resolved_symbols=resolved_symbols,
        discovery_mode=discovery_mode,
        search_query=search_query,
        final_frame=final_frame,
        duplicate_count=duplicate_count,
        dropped_rows=dropped_rows,
        warnings=warnings,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
    )
    write_json_file(metadata_path, metadata)

    logger.info(
        "Company news ready | ticker=%s | exchange=%s | provider=%s | rows=%s | dropped_rows=%s | duplicates=%s | acquisition_modes=%s | processed_csv=%s",
        request.ticker,
        request.exchange,
        news_provider.provider_name,
        len(final_frame),
        len(dropped_rows),
        duplicate_count,
        metadata["text_acquisition_mode_counts"],
        cleaned_table_path,
    )

    return NewsIngestionResult(
        raw_snapshot_path=raw_snapshot_path,
        cleaned_table_path=cleaned_table_path,
        metadata_path=metadata_path,
        row_count=len(final_frame),
        duplicate_count=duplicate_count,
        dropped_row_count=len(dropped_rows),
        coverage_start=date.fromisoformat(metadata["coverage_start"]) if metadata["coverage_start"] else None,
        coverage_end=date.fromisoformat(metadata["coverage_end"]) if metadata["coverage_end"] else None,
    )


def resolve_runtime_settings(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
) -> AppSettings:
    """Apply lightweight runtime ticker overrides for the news command."""

    if ticker is None and exchange is None:
        return settings

    resolved_symbol = (ticker or settings.ticker.symbol).strip().upper()
    resolved_exchange = (exchange or settings.ticker.exchange).strip().upper()
    updated_ticker = replace(
        settings.ticker,
        symbol=resolved_symbol,
        exchange=resolved_exchange,
    )
    return replace(settings, ticker=updated_ticker)


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
        lookback_days=resolved_lookback_days,
        published_after=resolved_published_before - timedelta(days=resolved_lookback_days),
        published_before=resolved_published_before,
        language=settings.news_ingestion.language,
        country=settings.news_ingestion.country,
        marketaux_limit_per_request=settings.news_ingestion.marketaux_limit_per_request,
        max_articles_per_run=settings.news_ingestion.max_articles_per_run,
    )


def resolve_news_provider(settings: AppSettings) -> CompanyNewsProvider:
    """Resolve the active news provider from settings."""

    provider_name = settings.providers.news_provider.strip().lower()
    if provider_name == "marketaux":
        if not settings.providers.news_api_key:
            raise NewsIngestionError("Marketaux news ingestion requires KUBERA_NEWS_API_KEY.")
        return MarketauxNewsProvider(
            settings.providers.news_api_key,
            timeout_seconds=settings.news_ingestion.request_timeout_seconds,
            retry_attempts=settings.news_ingestion.article_retry_attempts,
        )
    raise NewsIngestionError(
        f"Unsupported news provider: {settings.providers.news_provider}"
    )


def resolve_provider_entities(
    provider: CompanyNewsProvider,
    request: NewsDiscoveryRequest,
) -> tuple[list[dict[str, Any]], tuple[str, ...], list[dict[str, Any]]]:
    """Resolve the requested ticker through provider entity search."""

    payloads: list[dict[str, Any]] = []
    candidates: list[tuple[int, dict[str, Any]]] = []
    for alias in request.search_aliases:
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
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
    for article in deduped_articles[: request.max_articles_per_run]:
        fetch_result = article_fetcher(article, news_settings)
        final_article = article.copy()
        final_article["full_text"] = fetch_result.full_text
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
                "text_acquisition_mode": fetch_result.text_acquisition_mode,
                "text_acquisition_reason": fetch_result.text_acquisition_reason,
                "fetch_warning_flag": fetch_result.fetch_warning_flag,
                "fetch_error": fetch_result.fetch_error,
                "http_status": fetch_result.http_status,
            }
        )

    return final_articles, dropped_rows, duplicate_count, acquisition_diagnostics


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
    canonical_url = canonicalize_article_url(article_url)
    if article_url and canonical_url is None:
        reasons.append("invalid_article_url")

    if reasons:
        return {}, reasons

    assert published_at_utc is not None
    published_at_ist = utc_to_market_time(published_at_utc, market_settings)
    provider_uuid = clean_text(article.get("uuid"))
    summary_snippet = clean_text(article.get("description")) or clean_text(article.get("snippet"))
    provider_source = clean_text(article.get("source"))
    source_domain = extract_source_domain(canonical_url or article_url or provider_source)
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
            "published_at_raw": published_at_raw,
            "published_at_utc": published_at_utc.isoformat(),
            "published_at_ist": published_at_ist.isoformat(),
            "published_date_ist": published_at_ist.date().isoformat(),
            "summary_snippet": summary_snippet,
            "full_text": None,
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


def acquire_article_text_fallback(
    article: dict[str, Any],
    settings: NewsIngestionSettings,
) -> ArticleFetchResult:
    """Fetch article HTML and degrade to provider text when full extraction is weak."""

    article_url = clean_text(article.get("article_url")) or clean_text(article.get("canonical_url"))
    if canonicalize_article_url(article_url) is None:
        return build_article_fallback_result(
            article,
            reason="invalid_article_url",
        )

    last_error: str | None = None
    last_status: int | None = None
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml",
    }
    for attempt in range(1, settings.article_retry_attempts + 1):
        try:
            response = requests.get(
                article_url,
                headers=headers,
                timeout=settings.article_fetch_timeout_seconds,
            )
            last_status = response.status_code
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if content_type and "html" not in content_type:
                return build_article_fallback_result(
                    article,
                    reason="non_html_response",
                    http_status=last_status,
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
                )

            return build_article_fallback_result(
                article,
                reason=usability_reason or extraction_reason,
                http_status=last_status,
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
    )


def build_article_fallback_result(
    article: dict[str, Any],
    *,
    reason: str,
    fetch_error: str | None = None,
    http_status: int | None = None,
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
        )
    return ArticleFetchResult(
        full_text=article_title,
        text_acquisition_mode="headline_only",
        text_acquisition_reason=reason,
        fetch_warning_flag=True,
        fetch_error=fetch_error,
        http_status=http_status,
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
    entity_payloads: list[dict[str, Any]],
    news_payloads: list[dict[str, Any]],
    acquisition_diagnostics: list[dict[str, Any]],
    entity_matches: list[dict[str, Any]],
    resolved_symbols: tuple[str, ...],
    discovery_mode: str,
    search_query: str | None,
) -> dict[str, Any]:
    """Build the raw run snapshot payload for Stage 5."""

    return {
        "provider": request.provider,
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
        "discovery_mode": discovery_mode,
        "search_query": search_query,
        "resolved_symbols": list(resolved_symbols),
        "entity_matches": entity_matches,
        "entity_search_payloads": entity_payloads,
        "news_payloads": news_payloads,
        "article_fetch_diagnostics": acquisition_diagnostics,
    }


def build_news_metadata(
    *,
    request: NewsDiscoveryRequest,
    news_provider: CompanyNewsProvider,
    cleaned_table_path: Path,
    raw_snapshot_path: Path,
    fetched_at_utc: datetime,
    entity_matches: list[dict[str, Any]],
    resolved_symbols: tuple[str, ...],
    discovery_mode: str,
    search_query: str | None,
    final_frame: pd.DataFrame,
    duplicate_count: int,
    dropped_rows: list[dict[str, Any]],
    warnings: list[str],
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
) -> dict[str, Any]:
    """Build the metadata payload for the persisted Stage 5 outputs."""

    coverage_start = None
    coverage_end = None
    if not final_frame.empty:
        coverage_start = str(final_frame["published_date_ist"].min())
        coverage_end = str(final_frame["published_date_ist"].max())

    return {
        "provider": news_provider.provider_name,
        "ticker": request.ticker,
        "exchange": request.exchange,
        "company_name": request.company_name,
        "search_aliases": list(request.search_aliases),
        "lookback_days": request.lookback_days,
        "published_after": request.published_after.isoformat(),
        "published_before": request.published_before.isoformat(),
        "language": request.language,
        "country": request.country,
        "discovery_mode": discovery_mode,
        "search_query": search_query,
        "resolved_symbols": list(resolved_symbols),
        "entity_matches": entity_matches,
        "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "processed_news_path": str(cleaned_table_path),
        "processed_news_hash": compute_file_sha256(cleaned_table_path),
        "raw_snapshot_path": str(raw_snapshot_path),
        "raw_snapshot_hash": compute_file_sha256(raw_snapshot_path),
        "row_count": int(len(final_frame)),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "duplicate_count": duplicate_count,
        "dropped_row_count": len(dropped_rows),
        "dropped_rows": dropped_rows,
        "text_acquisition_mode_counts": count_series_values(final_frame, "text_acquisition_mode"),
        "source_domain_counts": count_series_values(final_frame, "source_domain"),
        "warnings": warnings,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def count_series_values(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    """Count string values in one DataFrame column for metadata."""

    if frame.empty:
        return {}
    counts = frame[column_name].fillna("null").value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def parse_provider_timestamp(raw_value: str | None) -> datetime | None:
    """Parse a provider timestamp into a UTC datetime."""

    if raw_value is None or not raw_value.strip():
        return None
    try:
        timestamp = pd.Timestamp(raw_value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def format_marketaux_datetime(value: datetime) -> str:
    """Format a UTC datetime for Marketaux query parameters."""

    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


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

    cleaned_url = clean_text(raw_url)
    if not cleaned_url:
        return None
    parsed = urlparse(cleaned_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None

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

    hostname = (parsed.hostname or "").lower()
    port = parsed.port
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

    return urlunparse(
        (
            parsed.scheme.lower(),
            netloc,
            path,
            "",
            urlencode(query_items, doseq=True),
            "",
        )
    )


def extract_source_domain(raw_value: str | None) -> str | None:
    """Extract a normalized source domain from a URL or domain string."""

    cleaned_value = clean_text(raw_value)
    if not cleaned_value:
        return None
    parsed = urlparse(cleaned_value if "://" in cleaned_value else f"https://{cleaned_value}")
    hostname = (parsed.hostname or "").lower()
    return hostname or None


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
