from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Callable

import pandas as pd
import pytest
import requests

from kubera.config import load_settings
from kubera.ingest.news_data import (
    ALPHAVANTAGE_PROVIDER_NAME,
    ArticleFetchResult,
    AlphaVantageNewsProvider,
    CollectedNewsSource,
    CompanyNewsProvider,
    MarketauxNewsProvider,
    build_fetch_policy_metadata,
    collect_news_source_results,
    compute_adaptive_lookback,
    ECONOMIC_TIMES_PROVIDER_NAME,
    GOOGLE_NEWS_PROVIDER_NAME,
    GoogleNewsRssProvider,
    NewsDiscoveryRequest,
    NewsIngestionError,
    NSE_ANNOUNCEMENTS_PROVIDER_NAME,
    NseAnnouncementsProvider,
    PROCESSED_NEWS_COLUMNS,
    acquire_article_text_fallback,
    build_google_news_query,
    build_news_discovery_request,
    canonicalize_article_url,
    dedupe_normalized_articles,
    fetch_company_news,
    normalize_alphavantage_feed,
    normalize_google_news_rss_payload,
    normalize_nse_announcement_rows,
    prioritize_normalized_articles,
    main,
    parse_provider_timestamp,
    resolve_configured_news_sources,
    resolve_news_provider,
    resolve_provider_entities,
    validate_article_url,
)
import kubera.ingest.news_data as news_data_module
from kubera.utils.hashing import compute_file_sha256


class FakeNewsProvider(CompanyNewsProvider):
    provider_name = "fake_news"

    def __init__(
        self,
        *,
        entity_search_payloads: dict[str, dict[str, object]] | None = None,
        news_pages: list[dict[str, object]] | None = None,
    ) -> None:
        self._entity_search_payloads = entity_search_payloads or {}
        self._news_pages = news_pages or []
        self.entity_queries: list[str] = []
        self.news_calls: list[dict[str, object]] = []

    def search_entities(
        self,
        request: NewsDiscoveryRequest,
        query: str,
    ) -> dict[str, object]:
        del request
        self.entity_queries.append(query)
        return self._entity_search_payloads.get(query, {"data": []})

    def fetch_news_page(
        self,
        request: NewsDiscoveryRequest,
        *,
        page: int,
        symbols: tuple[str, ...] | None = None,
        search_query: str | None = None,
    ) -> dict[str, object]:
        self.news_calls.append(
            {
                "ticker": request.ticker,
                "exchange": request.exchange,
                "page": page,
                "symbols": symbols,
                "search_query": search_query,
            }
        )
        if 1 <= page <= len(self._news_pages):
            return self._news_pages[page - 1]
        return {"data": []}


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        url: str = "https://example.com/article",
        headers: dict[str, str] | None = None,
        raise_error: requests.RequestException | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._raise_error = raise_error

    def raise_for_status(self) -> None:
        if self._raise_error is not None:
            raise self._raise_error

    def json(self) -> object:
        raise ValueError("No JSON payload configured for FakeResponse.")


class FakeJsonResponse(FakeResponse):
    def __init__(
        self,
        *,
        json_payload: object,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        raise_error: requests.RequestException | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            text="",
            headers=headers or {"Content-Type": "application/json; charset=utf-8"},
            raise_error=raise_error,
        )
        self._json_payload = json_payload

    def json(self) -> object:
        return self._json_payload


class FakeNseSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"url": url, **kwargs})
        if not self._responses:
            raise AssertionError("No fake NSE responses remaining.")
        return self._responses.pop(0)


def make_provider_article(
    *,
    uuid: str,
    title: str,
    url: str,
    published_at: str = "2026-03-10T06:30:00Z",
    description: str = "Provider summary snippet.",
    source: str = "Example News",
    entities: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "uuid": uuid,
        "title": title,
        "url": url,
        "published_at": published_at,
        "description": description,
        "source": source,
        "entities": entities
        or [
            {
                "symbol": "INFY",
                "name": "Infosys Limited",
                "exchange": "NSE",
                "country": "in",
                "type": "equity",
            }
        ],
    }


def make_alphavantage_article(
    *,
    title: str = "Infosys wins a banking modernization contract",
    url: str = "https://example.com/alphavantage-article",
    time_published: str = "20260310T063000",
    source: str = "Example News",
    summary: str = "Infosys secured a multi-year deal with a global banking client.",
    ticker: str = "INFY",
) -> dict[str, object]:
    return {
        "title": title,
        "url": url,
        "time_published": time_published,
        "source": source,
        "summary": summary,
        "overall_sentiment_score": "0.18",
        "overall_sentiment_label": "Somewhat-Bullish",
        "topics": [{"topic": "Technology", "relevance_score": "0.61"}],
        "ticker_sentiment": [
            {
                "ticker": ticker,
                "relevance_score": "0.87",
                "ticker_sentiment_score": "0.29",
                "ticker_sentiment_label": "Bullish",
            }
        ],
    }


def make_article_fetcher(
    *,
    mode: str = "full_article",
    reason: str = "test_fetcher",
    warning: bool = False,
    status_code: int | None = 200,
    attempt_count: int = 1,
    retry_count: int = 0,
) -> Callable[[dict[str, object], object], ArticleFetchResult]:
    def _fetcher(article: dict[str, object], settings) -> ArticleFetchResult:
        del settings
        return ArticleFetchResult(
            full_text=f"{article['article_title']} body text",
            text_acquisition_mode=mode,
            text_acquisition_reason=reason,
            fetch_warning_flag=warning,
            fetch_error=None,
            http_status=status_code,
            attempt_count=attempt_count,
            retry_count=retry_count,
        )

    return _fetcher


def make_normalized_article(
    article_id: str,
    *,
    provider: str = "marketaux",
    article_title: str = "Infosys wins a contract",
    article_url: str = "https://example.com/article",
    canonical_url: str | None = "https://example.com/article",
    source_domain: str = "example.com",
    provider_source: str = "Example News",
    source_name: str = "Example News",
    published_at_utc: str = "2026-03-10T06:30:00+00:00",
    published_date_ist: str = "2026-03-10",
    summary_snippet: str = "Provider summary",
) -> dict[str, object]:
    row = {column: None for column in PROCESSED_NEWS_COLUMNS}
    row.update(
        {
            "article_id": article_id,
            "ticker": "INFY",
            "exchange": "NSE",
            "provider": provider,
            "discovery_mode": "entity_symbols",
            "provider_uuid": article_id,
            "article_title": article_title,
            "article_url": article_url,
            "canonical_url": canonical_url,
            "source_domain": source_domain,
            "provider_source": provider_source,
            "source_name": source_name,
            "published_at_raw": published_at_utc,
            "published_at_utc": published_at_utc,
            "published_at_ist": "2026-03-10T12:00:00+05:30",
            "published_date_ist": published_date_ist,
            "summary_snippet": summary_snippet,
            "content_origin": None,
            "provider_entity_payload": "[]",
            "raw_snapshot_path": "data/raw/news/INFY/run.json",
            "fetched_at_utc": "2026-03-11T00:00:00+00:00",
        }
    )
    return row


def make_source_result(
    provider_name: str,
    *,
    articles: list[dict[str, object]] | None = None,
    warnings: list[str] | None = None,
    discovery_mode: str | None = None,
) -> CollectedNewsSource:
    return CollectedNewsSource(
        provider_name=provider_name,
        normalized_articles=[dict(article) for article in (articles or [])],
        dropped_rows=[],
        warnings=warnings or [],
        provider_request_count=1,
        provider_request_retry_count=0,
        raw_payload={"provider": provider_name, "status": "ok"},
        discovery_mode=discovery_mode,
    )


def test_resolve_provider_entities_prefers_exact_symbol_exchange_country(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)
    provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NYSE",
                        "country": "us",
                        "type": "equity",
                    },
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    },
                ]
            }
        }
    )

    payloads, resolved_symbols, entity_matches = resolve_provider_entities(provider, request)

    assert len(payloads) == len(request.search_aliases)
    assert resolved_symbols == ("INFY",)
    assert entity_matches[0]["exchange"] == "NSE"
    assert entity_matches[0]["country"] == "in"


def test_parse_provider_timestamp_normalizes_mixed_offsets() -> None:
    utc_timestamp = parse_provider_timestamp("2026-03-10T06:30:00Z")
    ist_timestamp = parse_provider_timestamp("2026-03-10 12:00:00+05:30")

    assert utc_timestamp is not None
    assert ist_timestamp is not None
    assert utc_timestamp.isoformat() == "2026-03-10T06:30:00+00:00"
    assert ist_timestamp.isoformat() == "2026-03-10T06:30:00+00:00"


def test_parse_provider_timestamp_supports_alphavantage_compact_formats() -> None:
    timestamp_with_seconds = parse_provider_timestamp("20260310T063000")
    timestamp_without_seconds = parse_provider_timestamp("20260310T0630")

    assert timestamp_with_seconds is not None
    assert timestamp_without_seconds is not None
    assert timestamp_with_seconds.isoformat() == "2026-03-10T06:30:00+00:00"
    assert timestamp_without_seconds.isoformat() == "2026-03-10T06:30:00+00:00"


def test_canonicalize_article_url_and_dedupe_rules() -> None:
    canonical_url = canonicalize_article_url(
        "HTTPS://Example.com:443/path/?utm_source=feed&b=2&a=1#fragment"
    )
    deduped_articles, duplicate_count = dedupe_normalized_articles(
        [
            make_normalized_article(
                "article-1",
                canonical_url="https://example.com/article",
                article_url="https://example.com/article?utm_source=feed",
            ),
            make_normalized_article(
                "article-2",
                canonical_url="https://example.com/article",
                article_url="https://example.com/article?utm_medium=email",
            ),
            make_normalized_article(
                "article-3",
                canonical_url=None,
                article_url="https://example.com/other",
                source_domain="example.com",
            ),
            make_normalized_article(
                "article-4",
                canonical_url=None,
                article_url="https://example.com/third",
                source_domain="example.com",
            ),
            make_normalized_article(
                "article-5",
                canonical_url=None,
                article_url="https://another.example.com/other",
                source_domain="another.example.com",
            ),
        ]
    )

    assert canonical_url == "https://example.com/path?a=1&b=2"
    assert duplicate_count == 2
    assert len(deduped_articles) == 3
    assert {article["article_id"] for article in deduped_articles} == {
        "article-1",
        "article-3",
        "article-5",
    }


def test_fetch_company_news_persists_outputs_and_traceability(
    isolated_repo,
) -> None:
    settings = load_settings()
    provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Infosys wins a large contract",
                        url="https://example.com/article?utm_source=feed",
                    ),
                    make_provider_article(
                        uuid="news-2",
                        title="Infosys wins a large contract duplicate",
                        url="https://example.com/article?utm_medium=email",
                    ),
                ]
            }
        ],
    )

    result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=make_article_fetcher(),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    raw_snapshot = json.loads(result.raw_snapshot_path.read_text(encoding="utf-8"))

    assert result.row_count == 1
    assert result.duplicate_count == 1
    assert cleaned_frame["raw_snapshot_path"].tolist() == [str(result.raw_snapshot_path)]
    assert cleaned_frame["provider"].tolist() == ["fake_news"]
    assert cleaned_frame["source_name"].tolist() == ["Example News"]
    assert cleaned_frame["content_origin"].tolist() == ["direct_publisher_text"]
    assert metadata["run_id"] == raw_snapshot["run_id"]
    assert metadata["processed_news_path"] == str(result.cleaned_table_path)
    assert metadata["raw_snapshot_path"] == str(result.raw_snapshot_path)
    assert metadata["processed_news_hash"] == compute_file_sha256(result.cleaned_table_path)
    assert metadata["raw_snapshot_hash"] == compute_file_sha256(result.raw_snapshot_path)
    assert metadata["source_name_counts"] == {"Example News": 1}
    assert metadata["content_origin_counts"] == {"direct_publisher_text": 1}
    assert metadata["fetch_policy"]["article_cache_ttl_hours"] == 24
    assert metadata["fetch_policy"]["provider_request_pause_seconds"] == pytest.approx(0.5)
    assert metadata["timing"]["elapsed_seconds"] >= 0.0
    assert metadata["workload"]["entity_payload_count"] == 3
    assert metadata["workload"]["news_payload_count"] == 1
    assert metadata["source_terms_review_required"] is True
    assert any("terms" in note.lower() for note in metadata["provider_limitations"])
    assert raw_snapshot["article_fetch_diagnostics"][0]["text_acquisition_mode"] == "full_article"
    assert raw_snapshot["article_fetch_diagnostics"][0]["cache_hit"] is False
    assert raw_snapshot["resolved_symbols"] == ["INFY"]
    assert raw_snapshot["timing"]["elapsed_seconds"] >= 0.0


def test_fetch_company_news_persists_empty_outputs_for_quiet_window(
    isolated_repo,
) -> None:
    settings = load_settings()
    provider = FakeNewsProvider(news_pages=[{"data": []}])

    result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=make_article_fetcher(),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    raw_snapshot = json.loads(result.raw_snapshot_path.read_text(encoding="utf-8"))

    assert result.row_count == 0
    assert list(cleaned_frame.columns) == list(PROCESSED_NEWS_COLUMNS)
    assert metadata["warnings"] == ["search_fallback_used", "no_articles_found"]
    assert metadata["row_count"] == 0
    assert raw_snapshot["discovery_mode"] == "search_fallback"
    assert raw_snapshot["resolved_symbols"] == []


def test_fetch_company_news_reuses_recent_article_cache(
    isolated_repo,
) -> None:
    settings = load_settings()
    provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Infosys signs a services deal",
                        url="https://example.com/article?utm_source=feed",
                    )
                ]
            }
        ],
    )
    fetch_call_count = 0

    def counting_fetcher(article: dict[str, object], settings) -> ArticleFetchResult:
        nonlocal fetch_call_count
        fetch_call_count += 1
        return make_article_fetcher()(article, settings)

    first_result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=counting_fetcher,
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )
    second_result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=counting_fetcher,
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    second_metadata = json.loads(second_result.metadata_path.read_text(encoding="utf-8"))
    second_raw_snapshot = json.loads(second_result.raw_snapshot_path.read_text(encoding="utf-8"))

    assert first_result.row_count == 1
    assert second_result.row_count == 1
    assert fetch_call_count == 1
    assert second_metadata["cache_hit_count"] == 1
    assert second_metadata["fresh_fetch_count"] == 0
    assert second_raw_snapshot["article_fetch_diagnostics"][0]["cache_hit"] is True


def test_normalize_alphavantage_feed_maps_items_into_processed_schema(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)

    normalized_articles, dropped_rows = normalize_alphavantage_feed(
        feed_items=[make_alphavantage_article()],
        request=request,
        raw_snapshot_path=isolated_repo / "data" / "raw" / "news" / "alphavantage.json",
        fetched_at_utc=datetime(2026, 3, 11, tzinfo=timezone.utc),
        market_settings=settings.market,
    )

    assert dropped_rows == []
    assert len(normalized_articles) == 1
    assert normalized_articles[0]["provider"] == ALPHAVANTAGE_PROVIDER_NAME
    assert normalized_articles[0]["discovery_mode"] == "ticker_sentiment"
    assert normalized_articles[0]["published_at_utc"] == "2026-03-10T06:30:00+00:00"
    assert normalized_articles[0]["source_name"] == "Example News"
    provider_payload = json.loads(str(normalized_articles[0]["provider_entity_payload"]))
    assert provider_payload["ticker_sentiment"][0]["ticker"] == "INFY"


def test_build_news_discovery_request_carries_ticker_query_metadata(isolated_repo) -> None:
    settings = load_settings()

    request = build_news_discovery_request(settings)

    assert request.sector_name == "Information Technology"
    assert request.industry_name == "IT Services and Consulting"
    assert request.sector_query_terms == (
        "information technology",
        "IT services",
        "digital transformation",
    )
    assert request.macro_query_terms == (
        "NSE IT index",
        "India technology exports",
    )


def test_build_google_news_query_uses_company_and_context_terms(isolated_repo) -> None:
    query = build_google_news_query(build_news_discovery_request(load_settings()))

    assert "Infosys Limited" in query
    assert "INFY" in query
    assert "Information Technology" in query
    assert "IT Services and Consulting" in query
    assert "NSE IT index" in query


def test_normalize_google_news_rss_payload_maps_items_into_processed_schema(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)
    rss_text = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Infosys wins a banking technology contract</title>
          <link>https://news.google.com/rss/articles/CBMiZGh0dHBzOi8vZXhhbXBsZS5jb20vaW5mb3N5cy1kZWFs0gEA</link>
          <pubDate>Tue, 10 Mar 2026 06:30:00 GMT</pubDate>
          <source url="https://example.com">Example News</source>
        </item>
      </channel>
    </rss>
    """

    normalized_articles, dropped_rows, item_count = normalize_google_news_rss_payload(
        rss_text=rss_text,
        request=request,
        raw_snapshot_path=isolated_repo / "data" / "raw" / "rss.json",
        fetched_at_utc=datetime(2026, 3, 11, tzinfo=timezone.utc),
        market_settings=settings.market,
    )

    assert item_count == 1
    assert dropped_rows == []
    assert normalized_articles[0]["provider"] == GOOGLE_NEWS_PROVIDER_NAME
    assert normalized_articles[0]["source_name"] == "Example News"
    assert normalized_articles[0]["source_domain"] == "example.com"
    assert normalized_articles[0]["published_at_utc"] == "2026-03-10T06:30:00+00:00"


def test_normalize_nse_announcement_rows_maps_primary_source_fields(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)
    announcements = [
        {
            "subject": "Board Meeting Outcome",
            "desc": "The board approved an interim dividend.",
            "attchmntFile": "/content/example.pdf",
            "bflag": "false",
            "exchdisstime": "2026-03-10 17:05:00",
        }
    ]

    normalized_articles, dropped_rows = normalize_nse_announcement_rows(
        announcements=announcements,
        request=request,
        raw_snapshot_path=isolated_repo / "data" / "raw" / "nse.json",
        fetched_at_utc=datetime(2026, 3, 11, tzinfo=timezone.utc),
        market_settings=settings.market,
    )

    assert dropped_rows == []
    assert normalized_articles[0]["provider"] == NSE_ANNOUNCEMENTS_PROVIDER_NAME
    assert normalized_articles[0]["source_name"] == "NSE Corporate Announcements"
    assert normalized_articles[0]["summary_snippet"] == "The board approved an interim dividend."
    assert normalized_articles[0]["article_url"] == "https://www.nseindia.com/content/example.pdf"
    assert normalized_articles[0]["published_at_utc"] == "2026-03-10T11:35:00+00:00"


def test_nse_announcements_provider_accepts_enveloped_data_payload(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)
    session = FakeNseSession(
        [
            FakeResponse(text="<html></html>"),
            FakeJsonResponse(
                json_payload={
                    "data": [
                        {
                            "subject": "Board Meeting Outcome",
                            "desc": "The board approved an interim dividend.",
                            "attchmntFile": "/content/example.pdf",
                            "exchdisstime": "2026-03-10 17:05:00",
                        }
                    ],
                    "msg": "success",
                }
            ),
        ]
    )
    provider = NseAnnouncementsProvider(session=session)

    announcements = provider.fetch_announcements(request, pause_seconds=0.0)

    assert len(announcements) == 1
    assert announcements[0]["subject"] == "Board Meeting Outcome"
    assert session.calls[0]["url"] == "https://www.nseindia.com"
    assert session.calls[1]["url"] == "https://www.nseindia.com/api/corp-info"


def test_nse_announcements_provider_accepts_empty_enveloped_payload_without_failure(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)
    session = FakeNseSession(
        [
            FakeResponse(text="<html></html>"),
            FakeJsonResponse(json_payload={"data": [], "msg": "no data found"}),
        ]
    )
    provider = NseAnnouncementsProvider(session=session)

    announcements = provider.fetch_announcements(request, pause_seconds=0.0)

    assert announcements == []


def test_nse_announcements_provider_rejects_unexpected_payload_shape(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_news_discovery_request(settings)
    session = FakeNseSession(
        [
            FakeResponse(text="<html></html>"),
            FakeJsonResponse(json_payload={"msg": "unexpected"}),
        ]
    )
    provider = NseAnnouncementsProvider(session=session)

    with pytest.raises(NewsIngestionError, match="unexpected payload"):
        provider.fetch_announcements(request, pause_seconds=0.0)


def test_resolve_configured_news_sources_uses_free_sources_without_marketaux_key(
    isolated_repo,
) -> None:
    provider_names = [provider.provider_name for provider in resolve_configured_news_sources(load_settings())]

    assert provider_names == [
        GOOGLE_NEWS_PROVIDER_NAME,
        ECONOMIC_TIMES_PROVIDER_NAME,
        NSE_ANNOUNCEMENTS_PROVIDER_NAME,
    ]


def test_resolve_news_provider_supports_alphavantage(
    isolated_repo,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", ALPHAVANTAGE_PROVIDER_NAME)
    monkeypatch.setenv("KUBERA_ALPHAVANTAGE_API_KEY", "alphavantage-test-key")

    provider = resolve_news_provider(load_settings())

    assert isinstance(provider, AlphaVantageNewsProvider)


def test_resolve_configured_news_sources_includes_alphavantage_when_enabled(
    isolated_repo,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", ALPHAVANTAGE_PROVIDER_NAME)
    monkeypatch.setenv("KUBERA_ALPHAVANTAGE_API_KEY", "alphavantage-test-key")

    provider_names = [provider.provider_name for provider in resolve_configured_news_sources(load_settings())]

    assert provider_names == [
        ALPHAVANTAGE_PROVIDER_NAME,
        GOOGLE_NEWS_PROVIDER_NAME,
        ECONOMIC_TIMES_PROVIDER_NAME,
        NSE_ANNOUNCEMENTS_PROVIDER_NAME,
    ]


def test_fetch_company_news_merges_multi_source_articles_and_prioritizes_nse(
    isolated_repo,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_MAX_ARTICLES_PER_RUN", "2")
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.ingest.news_data.collect_news_source_results",
        lambda **kwargs: (
            [
                make_source_result(
                    "marketaux",
                    articles=[
                        make_normalized_article(
                            "marketaux-1",
                            published_at_utc="2026-03-10T06:30:00+00:00",
                        )
                    ],
                    discovery_mode="entity_symbols",
                ),
                make_source_result(
                    GOOGLE_NEWS_PROVIDER_NAME,
                    articles=[
                        make_normalized_article(
                            "google-1",
                            provider=GOOGLE_NEWS_PROVIDER_NAME,
                            published_at_utc="2026-03-10T06:30:00+00:00",
                            article_url="https://news.google.com/rss/articles/google-1",
                            canonical_url=None,
                            source_domain="example.net",
                            source_name="Example Net",
                            article_title="Infosys featured in Google News",
                        )
                    ],
                    discovery_mode="rss_search",
                ),
                make_source_result(
                    NSE_ANNOUNCEMENTS_PROVIDER_NAME,
                    articles=[
                        make_normalized_article(
                            "nse-1",
                            provider=NSE_ANNOUNCEMENTS_PROVIDER_NAME,
                            published_at_utc="2026-03-10T06:30:00+00:00",
                            article_url="https://www.nseindia.com/content/example.pdf",
                            canonical_url="https://www.nseindia.com/content/example.pdf",
                            source_domain="www.nseindia.com",
                            article_title="Board Meeting Outcome",
                            summary_snippet="Dividend approved",
                        )
                    ],
                    discovery_mode="corp_announcements",
                ),
            ],
            [],
        ),
    )

    result = fetch_company_news(
        settings,
        article_fetcher=make_article_fetcher(),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert cleaned_frame["provider"].tolist() == [
        NSE_ANNOUNCEMENTS_PROVIDER_NAME,
        "marketaux",
    ]
    assert metadata["provider"] == "multi_source"
    assert metadata["providers_used"] == [
        "marketaux",
        GOOGLE_NEWS_PROVIDER_NAME,
        NSE_ANNOUNCEMENTS_PROVIDER_NAME,
    ]


def test_prioritize_normalized_articles_prefers_specific_company_matches_within_provider(
    isolated_repo,
) -> None:
    request = build_news_discovery_request(load_settings())
    prioritized = prioritize_normalized_articles(
        [
            make_normalized_article(
                "generic-marketaux",
                provider="marketaux",
                published_at_utc="2026-03-10T08:00:00+00:00",
                article_title="IT services stocks broadly gain",
                summary_snippet="Large-cap technology names traded higher.",
            ),
            make_normalized_article(
                "specific-marketaux",
                provider="marketaux",
                published_at_utc="2026-03-10T07:00:00+00:00",
                article_title="Infosys wins a strategic banking modernization deal",
                summary_snippet="Infosys said the multi-year program expands its existing client work.",
            ),
            make_normalized_article(
                "google-row",
                provider=GOOGLE_NEWS_PROVIDER_NAME,
                published_at_utc="2026-03-10T09:00:00+00:00",
                article_title="Infosys mentioned in Google News roundup",
                article_url="https://news.google.com/rss/articles/google-row",
                canonical_url=None,
            ),
        ],
        request=request,
    )

    assert [row["article_id"] for row in prioritized] == [
        "specific-marketaux",
        "generic-marketaux",
        "google-row",
    ]


def test_fetch_company_news_records_provider_failure_warnings_without_aborting(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.ingest.news_data.resolve_configured_news_sources",
        lambda _settings: [GoogleNewsRssProvider()],
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.collect_google_news_source",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("google throttled")),
    )

    result = fetch_company_news(
        settings,
        article_fetcher=make_article_fetcher(),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.row_count == 0
    assert "google_news_rss_failed" in metadata["warnings"]
    assert metadata["provider_summaries"][0]["status"] == "failed"
    assert "google throttled" in metadata["provider_summaries"][0]["warnings"][0]


def test_fetch_company_news_emits_degraded_source_warnings_for_google_only_generic_fallback(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.ingest.news_data.collect_news_source_results",
        lambda **kwargs: (
            [
                make_source_result(
                    GOOGLE_NEWS_PROVIDER_NAME,
                    articles=[
                        make_normalized_article(
                            "google-generic-1",
                            provider=GOOGLE_NEWS_PROVIDER_NAME,
                            article_title="IT services shares move higher",
                            summary_snippet="Technology stocks broadly gained in Mumbai trading.",
                            article_url="https://news.google.com/rss/articles/google-generic-1",
                            canonical_url=None,
                            source_domain="news.google.com",
                            source_name="Google News",
                        ),
                        make_normalized_article(
                            "google-generic-2",
                            provider=GOOGLE_NEWS_PROVIDER_NAME,
                            article_title="Indian large-cap tech stocks rise",
                            summary_snippet="Broker commentary highlighted sector-level momentum.",
                            article_url="https://news.google.com/rss/articles/google-generic-2",
                            canonical_url=None,
                            source_domain="news.google.com",
                            source_name="Google News",
                        ),
                    ],
                    discovery_mode="rss_search",
                )
            ],
            [],
        ),
    )

    result = fetch_company_news(
        settings,
        article_fetcher=make_article_fetcher(mode="headline_only", warning=True),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert "degraded_source_google_only" in metadata["warnings"]
    assert "degraded_source_fallback_majority" in metadata["warnings"]
    assert "degraded_source_low_specificity" in metadata["warnings"]


def test_fetch_company_news_supports_catalog_backed_alternate_ticker(
    isolated_repo,
) -> None:
    settings = load_settings()
    provider = FakeNewsProvider(
        entity_search_payloads={
            "TCS": {
                "data": [
                    {
                        "symbol": "TCS",
                        "name": "Tata Consultancy Services",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="tcs-news-1",
                        title="TCS expands a banking modernization program",
                        url="https://example.com/tcs-article",
                        entities=[
                            {
                                "symbol": "TCS",
                                "name": "Tata Consultancy Services",
                                "exchange": "NSE",
                                "country": "in",
                                "type": "equity",
                            }
                        ],
                    )
                ]
            }
        ],
    )

    result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=make_article_fetcher(
            mode="headline_plus_snippet",
            warning=True,
            attempt_count=3,
            retry_count=2,
        ),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
        ticker="TCS",
        exchange="NSE",
    )

    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    raw_snapshot = json.loads(result.raw_snapshot_path.read_text(encoding="utf-8"))

    assert result.cleaned_table_path.name == "TCS_NSE_news.csv"
    assert cleaned_frame["ticker"].tolist() == ["TCS"]
    assert metadata["ticker"] == "TCS"
    assert metadata["company_name"] == "Tata Consultancy Services"
    assert metadata["article_fetch_attempt_count"] == 3
    assert metadata["article_fetch_retry_count"] == 2
    assert metadata["provider_request_count"] > 0
    assert metadata["provider_request_retry_count"] == 0
    assert raw_snapshot["resolved_symbols"] == ["TCS"]
    assert raw_snapshot["article_fetch_diagnostics"][0]["attempt_count"] == 3
    assert raw_snapshot["article_fetch_diagnostics"][0]["retry_count"] == 2


def test_fetch_company_news_applies_request_pacing_between_uncached_fetches(
    isolated_repo,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS", "0")
    monkeypatch.setenv("KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS", "0.25")
    settings = load_settings()
    provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Infosys signs a services deal",
                        url="https://example.com/article-1",
                    ),
                    make_provider_article(
                        uuid="news-2",
                        title="Infosys updates a client relationship",
                        url="https://example.com/article-2",
                    ),
                ]
            }
        ],
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr("kubera.ingest.news_data.time.sleep", sleep_calls.append)

    result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=make_article_fetcher(mode="headline_plus_snippet", warning=True),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.row_count == 2
    assert sleep_calls == [0.25, 0.25]
    assert metadata["fetch_policy"]["article_request_pause_seconds"] == 0.25
    assert metadata["content_origin_counts"] == {"aggregator_text": 2}


def test_fetch_company_news_applies_provider_request_pacing(
    isolated_repo,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_ALIASES", "INFY")
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS", "0.25")
    settings = load_settings()
    provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Infosys signs a services deal",
                        url="https://example.com/article-1",
                    ),
                    make_provider_article(
                        uuid="news-2",
                        title="Infosys updates a client relationship",
                        url="https://example.com/article-2",
                    ),
                ]
            }
        ],
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr("kubera.ingest.news_data.time.sleep", sleep_calls.append)

    result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=make_article_fetcher(),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.row_count == 2
    assert sleep_calls[:2] == [0.25, 0.25]
    assert metadata["fetch_policy"]["provider_request_pause_seconds"] == 0.25


def test_validate_article_url_rejects_suspicious_targets() -> None:
    assert validate_article_url("javascript:alert(1)") == (None, "invalid_article_url")
    assert validate_article_url("https://user:pass@example.com/article") == (
        None,
        "article_url_contains_credentials",
    )
    assert validate_article_url("https://127.0.0.1/internal") == (
        None,
        "disallowed_article_url_host",
    )
    assert validate_article_url("https://localhost/admin") == (
        None,
        "disallowed_article_url_host",
    )


def test_acquire_article_text_fallback_rejects_private_hosts_without_fetch() -> None:
    result = acquire_article_text_fallback(
        {
            "article_title": "Loopback target",
            "summary_snippet": "Provider summary",
            "article_url": "https://127.0.0.1/internal",
        },
        load_settings().news_ingestion,
    )

    assert result.text_acquisition_mode == "headline_plus_snippet"
    assert result.text_acquisition_reason == "disallowed_article_url_host"
    assert result.attempt_count == 0


def test_fetch_company_news_drops_rows_with_missing_title_and_invalid_timestamp(
    isolated_repo,
) -> None:
    settings = load_settings()
    provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    {
                        **make_provider_article(
                            uuid="bad-title",
                            title="Infosys headline placeholder",
                            url="https://example.com/bad-title",
                        ),
                        "title": "",
                    },
                    {
                        **make_provider_article(
                            uuid="bad-time",
                            title="Infosys bad timestamp",
                            url="https://example.com/bad-time",
                        ),
                        "published_at": "not-a-timestamp",
                    },
                    make_provider_article(
                        uuid="good-row",
                        title="Infosys lands a new banking contract",
                        url="https://example.com/good-row",
                    ),
                ]
            }
        ],
    )

    result = fetch_company_news(
        settings,
        provider=provider,
        article_fetcher=make_article_fetcher(),
        published_before=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.row_count == 1
    assert metadata["dropped_row_count"] == 2
    assert metadata["dropped_rows"] == [
        {
            "provider_uuid": "bad-title",
            "reasons": ["missing_title"],
            "url": "https://example.com/bad-title",
        },
        {
            "provider_uuid": "bad-time",
            "reasons": ["invalid_published_at"],
            "url": "https://example.com/bad-time",
        },
    ]


def test_acquire_article_text_fallback_uses_full_article_when_page_is_richer(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings().news_ingestion
    article = make_normalized_article(
        "article-1",
        article_title="Infosys signs a strategic deal",
        summary_snippet="Short provider summary.",
    )
    html = """
    <html>
      <body>
        <article>
          <p>Infosys signed a multi-year agreement with a major banking client in Europe.</p>
          <p>The deal expands digital transformation work across cloud, cyber security, and core platform modernization.</p>
          <p>Management said the contract should support medium-term revenue visibility and hiring in delivery teams.</p>
        </article>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "kubera.ingest.news_data.requests.get",
        lambda *args, **kwargs: FakeResponse(text=html),
    )

    result = acquire_article_text_fallback(article, settings)

    assert result.text_acquisition_mode == "full_article"
    assert result.fetch_warning_flag is False
    assert result.http_status == 200
    assert result.full_text is not None
    assert "multi-year agreement" in result.full_text


def test_acquire_article_text_fallback_records_resolved_article_url(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings().news_ingestion
    article = make_normalized_article(
        "article-resolved-url",
        article_url="https://news.google.com/rss/articles/google-redirect",
        canonical_url=None,
        article_title="Infosys signs a strategic deal",
        summary_snippet="Short provider summary.",
    )
    html = """
    <html>
      <body>
        <article>
          <p>Infosys signed a multi-year agreement with a major banking client in Europe.</p>
          <p>The deal expands digital transformation work across cloud, cyber security, and core platform modernization.</p>
          <p>Management said the contract should support medium-term revenue visibility and hiring in delivery teams.</p>
        </article>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "kubera.ingest.news_data.requests.get",
        lambda *args, **kwargs: FakeResponse(
            text=html,
            url="https://publisher.example.com/articles/infosys-deal?utm_source=google",
        ),
    )

    result = acquire_article_text_fallback(article, settings)

    assert result.text_acquisition_mode == "full_article"
    assert result.resolved_article_url == "https://publisher.example.com/articles/infosys-deal"


def test_acquire_article_text_fallback_uses_headline_plus_snippet_for_short_pages(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings().news_ingestion
    article = make_normalized_article(
        "article-2",
        article_title="Infosys announces an update",
        summary_snippet="Provider summary remains available.",
    )
    html = """
    <html>
      <body>
        <article>
          <p>Infosys announces an update.</p>
        </article>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "kubera.ingest.news_data.requests.get",
        lambda *args, **kwargs: FakeResponse(text=html),
    )

    result = acquire_article_text_fallback(article, settings)

    assert result.text_acquisition_mode == "headline_plus_snippet"
    assert result.fetch_warning_flag is True
    assert result.fetch_error is None
    assert result.full_text == "Infosys announces an update\n\nProvider summary remains available."


def test_acquire_article_text_fallback_uses_headline_only_without_snippet(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings().news_ingestion
    article = make_normalized_article(
        "article-3",
        article_title="Infosys updates guidance",
        summary_snippet=None,
    )
    html = """
    <html>
      <body>
        <main>
          <p>Infosys updates guidance.</p>
        </main>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "kubera.ingest.news_data.requests.get",
        lambda *args, **kwargs: FakeResponse(text=html),
    )

    result = acquire_article_text_fallback(article, settings)

    assert result.text_acquisition_mode == "headline_only"
    assert result.fetch_warning_flag is True
    assert result.full_text == "Infosys updates guidance"


def test_acquire_article_text_fallback_handles_dead_urls_with_degraded_success(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings().news_ingestion
    article = make_normalized_article(
        "article-4",
        article_title="Infosys faces an outage",
        summary_snippet="Provider summary is still available after fetch failure.",
    )
    error = requests.HTTPError("404 Client Error")
    monkeypatch.setattr("kubera.ingest.news_data.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "kubera.ingest.news_data.requests.get",
        lambda *args, **kwargs: FakeResponse(
            status_code=404,
            raise_error=error,
        ),
    )

    result = acquire_article_text_fallback(article, settings)

    assert result.text_acquisition_mode == "headline_plus_snippet"
    assert result.fetch_warning_flag is True
    assert result.http_status == 404
    assert result.fetch_error == "HTTPError: 404 Client Error"


def test_acquire_article_text_fallback_records_retry_counts(
    isolated_repo,
    monkeypatch,
) -> None:
    settings = load_settings().news_ingestion
    article = make_normalized_article(
        "article-5",
        article_title="TCS signs a transformation deal",
        summary_snippet="Provider summary remains available.",
    )
    html = """
    <html>
      <body>
        <article>
          <p>TCS signed a multi-year transformation agreement with a global banking client.</p>
          <p>The engagement covers cloud migration, application modernization, and operating-model changes.</p>
          <p>The company expects the deal to support medium-term delivery utilization.</p>
          <p>Executives said the program expands existing managed-services work into data, cyber security, and core-platform engineering.</p>
          <p>The mandate is expected to ramp over several quarters and includes multi-region delivery teams and new platform migration milestones.</p>
        </article>
      </body>
    </html>
    """
    call_count = {"value": 0}

    def flaky_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise requests.Timeout("temporary timeout")
        return FakeResponse(text=html)

    monkeypatch.setattr("kubera.ingest.news_data.time.sleep", lambda seconds: None)
    monkeypatch.setattr("kubera.ingest.news_data.requests.get", flaky_get)

    result = acquire_article_text_fallback(article, settings)

    assert result.text_acquisition_mode == "full_article"
    assert result.attempt_count == 2
    assert result.retry_count == 1


def test_news_command_smoke_writes_outputs(
    isolated_repo,
    monkeypatch,
) -> None:
    fake_provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Infosys signs a strategic deal",
                        url="https://example.com/strategic-deal",
                    )
                ]
            }
        ],
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.resolve_news_provider",
        lambda settings: fake_provider,
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.acquire_article_text_fallback",
        make_article_fetcher(mode="headline_plus_snippet", warning=True),
    )

    exit_code = main(
        [
            "--ticker",
            "INFY",
            "--exchange",
            "NSE",
            "--lookback-days",
            "7",
            "--published-before",
            "2026-03-11T00:00:00Z",
        ]
    )

    assert exit_code == 0
    assert (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_news.csv"
    ).exists()
    assert (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_news.metadata.json"
    ).exists()


def test_compute_adaptive_lookback_returns_configured_when_no_existing_data(
    isolated_repo,
) -> None:
    settings = load_settings()

    adaptive_lookback = compute_adaptive_lookback(
        settings,
        target_end_datetime=datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc),
        configured_lookback_days=30,
    )

    assert adaptive_lookback == 30


def test_compute_adaptive_lookback_calculates_gap_correctly(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()

    # First, fetch some news to create existing metadata
    fake_provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Test Article",
                        url="https://example.com/article1",
                        published_at="2026-03-01T12:00:00Z",
                    )
                ]
            }
        ],
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.resolve_news_provider",
        lambda settings: fake_provider,
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.acquire_article_text_fallback",
        make_article_fetcher(mode="headline_only"),
    )

    # Fetch news with published_before = March 1
    fetch_company_news(
        settings,
        published_before=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
        lookback_days=30,
    )

    # Now compute adaptive lookback for April 2 (32 days gap)
    adaptive_lookback = compute_adaptive_lookback(
        settings,
        target_end_datetime=datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc),
        configured_lookback_days=30,
        buffer_days=2,
    )

    # Gap is 32 days, buffer is 2, so adaptive = 34
    # Should return max(34, 30) = 34
    assert adaptive_lookback == 34


def test_compute_adaptive_lookback_respects_minimum_configured_lookback(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()

    # First, fetch some news
    fake_provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Test Article",
                        url="https://example.com/article1",
                        published_at="2026-03-10T12:00:00Z",
                    )
                ]
            }
        ],
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.resolve_news_provider",
        lambda settings: fake_provider,
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.acquire_article_text_fallback",
        make_article_fetcher(mode="headline_only"),
    )

    # Fetch news with published_before = March 10
    fetch_company_news(
        settings,
        published_before=datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc),
        lookback_days=30,
    )

    # Now compute adaptive lookback for March 13 (3 days gap, 2 buffer = 5 total)
    adaptive_lookback = compute_adaptive_lookback(
        settings,
        target_end_datetime=datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc),
        configured_lookback_days=30,
        buffer_days=2,
    )

    # Gap is 3 days, buffer is 2, so adaptive = 5
    # Should return max(5, 30) = 30 (respects minimum)
    assert adaptive_lookback == 30


def test_ensure_fresh_until_uses_adaptive_lookback(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()

    # First fetch with published_before = March 1
    first_provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-1",
                        title="Test Article 1",
                        url="https://example.com/article1",
                        published_at="2026-02-28T12:00:00Z",
                    )
                ]
            }
        ],
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.resolve_news_provider",
        lambda settings: first_provider,
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.acquire_article_text_fallback",
        make_article_fetcher(mode="headline_only"),
    )

    fetch_company_news(
        settings,
        published_before=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
        lookback_days=7,
    )

    # Second fetch with ensure_fresh_until = April 2 (32 day gap)
    second_provider = FakeNewsProvider(
        entity_search_payloads={
            "INFY": {
                "data": [
                    {
                        "symbol": "INFY",
                        "name": "Infosys Limited",
                        "exchange": "NSE",
                        "country": "in",
                        "type": "equity",
                    }
                ]
            }
        },
        news_pages=[
            {
                "data": [
                    make_provider_article(
                        uuid="news-2",
                        title="Test Article 2",
                        url="https://example.com/article2",
                        published_at="2026-04-01T12:00:00Z",
                    )
                ]
            }
        ],
    )
    monkeypatch.setattr(
        "kubera.ingest.news_data.resolve_news_provider",
        lambda settings: second_provider,
    )

    result = fetch_company_news(
        settings,
        lookback_days=7,
        ensure_fresh_until=datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc),
    )

    # Verify the adaptive lookback was used (gap + buffer > configured)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    # The request should have used adaptive lookback (32 + 2 = 34 days)
    # Check that published_before was set to ensure_fresh_until
    assert metadata["published_before"] == "2026-04-02T12:00:00+00:00"
    # Lookback should be >= 34 (gap + buffer)
    assert metadata["lookback_days"] >= 34


def test_marketaux_provider_passes_tuple_timeout_to_requests(isolated_repo) -> None:
    captured: dict[str, object] = {}

    class RecordingSession:
        def get(self, url: str, **kwargs: object) -> object:
            captured["timeout"] = kwargs.get("timeout")

            class Ok:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    return {"data": []}

            return Ok()

    settings = load_settings()
    request = build_news_discovery_request(settings)
    provider = MarketauxNewsProvider(
        "token",
        session=RecordingSession(),
        connect_timeout_seconds=2.5,
        read_timeout_seconds=18,
    )
    provider.search_entities(request, "INFY")
    assert captured["timeout"] == (2.5, 18.0)


def test_build_fetch_policy_metadata_includes_marketaux_timeouts(isolated_repo) -> None:
    settings = load_settings().news_ingestion
    meta = build_fetch_policy_metadata(settings)
    assert meta["marketaux_connect_timeout_seconds"] == pytest.approx(10.0)
    assert meta["marketaux_read_timeout_seconds"] == 15


def test_collect_news_source_results_runs_google_after_marketaux_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", "marketaux")
    monkeypatch.setenv("KUBERA_NEWS_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("KUBERA_NEWS_ENABLE_NSE_ANNOUNCEMENTS", "false")
    monkeypatch.setenv("KUBERA_NEWS_ENABLE_ECONOMIC_TIMES", "false")

    settings = load_settings()
    request = build_news_discovery_request(settings)
    raw_path = isolated_repo / "raw_snapshot.json"

    def fail_marketaux(**kwargs: object) -> None:
        raise NewsIngestionError("simulated marketaux failure")

    def stub_google(**kwargs: object) -> CollectedNewsSource:
        return make_source_result("google_news_rss", articles=[])

    monkeypatch.setattr(news_data_module, "collect_marketaux_source", fail_marketaux)
    monkeypatch.setattr(news_data_module, "collect_google_news_source", stub_google)

    results, warnings = collect_news_source_results(
        settings=settings,
        request=request,
        raw_snapshot_path=raw_path,
        fetched_at_utc=datetime.now(timezone.utc),
    )
    assert "marketaux_failed" in warnings
    assert len(results) == 2
    assert results[0].provider_name == "marketaux"
    assert results[0].raw_payload.get("status") == "failed"
    assert results[1].provider_name == "google_news_rss"
