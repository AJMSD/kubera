from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Callable

import pandas as pd
import pytest
import requests

from kubera.config import load_settings
from kubera.ingest.news_data import (
    ArticleFetchResult,
    CollectedNewsSource,
    CompanyNewsProvider,
    GOOGLE_NEWS_PROVIDER_NAME,
    GoogleNewsRssProvider,
    NewsDiscoveryRequest,
    NewsIngestionError,
    NSE_ANNOUNCEMENTS_PROVIDER_NAME,
    NseAnnouncementsProvider,
    PROCESSED_NEWS_COLUMNS,
    acquire_article_text_fallback,
    build_news_discovery_request,
    canonicalize_article_url,
    dedupe_normalized_articles,
    fetch_company_news,
    normalize_google_news_rss_payload,
    normalize_nse_announcement_rows,
    prioritize_normalized_articles,
    main,
    parse_provider_timestamp,
    resolve_configured_news_sources,
    resolve_provider_entities,
    validate_article_url,
)
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
        headers: dict[str, str] | None = None,
        raise_error: requests.RequestException | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
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

    assert provider_names == [GOOGLE_NEWS_PROVIDER_NAME, NSE_ANNOUNCEMENTS_PROVIDER_NAME]


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
