from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Callable

import pandas as pd

from kubera.config import load_settings
from kubera.ingest.news_data import (
    ArticleFetchResult,
    CompanyNewsProvider,
    NewsDiscoveryRequest,
    PROCESSED_NEWS_COLUMNS,
    build_news_discovery_request,
    canonicalize_article_url,
    dedupe_normalized_articles,
    fetch_company_news,
    main,
    parse_provider_timestamp,
    resolve_provider_entities,
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
        )

    return _fetcher


def make_normalized_article(
    article_id: str,
    *,
    article_title: str = "Infosys wins a contract",
    article_url: str = "https://example.com/article",
    canonical_url: str | None = "https://example.com/article",
    source_domain: str = "example.com",
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
            "provider": "marketaux",
            "discovery_mode": "entity_symbols",
            "provider_uuid": article_id,
            "article_title": article_title,
            "article_url": article_url,
            "canonical_url": canonical_url,
            "source_domain": source_domain,
            "provider_source": "Example News",
            "published_at_raw": published_at_utc,
            "published_at_utc": published_at_utc,
            "published_at_ist": "2026-03-10T12:00:00+05:30",
            "published_date_ist": published_date_ist,
            "summary_snippet": summary_snippet,
            "provider_entity_payload": "[]",
            "raw_snapshot_path": "data/raw/news/INFY/run.json",
            "fetched_at_utc": "2026-03-11T00:00:00+00:00",
        }
    )
    return row


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
    assert metadata["run_id"] == raw_snapshot["run_id"]
    assert metadata["processed_news_path"] == str(result.cleaned_table_path)
    assert metadata["raw_snapshot_path"] == str(result.raw_snapshot_path)
    assert metadata["processed_news_hash"] == compute_file_sha256(result.cleaned_table_path)
    assert metadata["raw_snapshot_hash"] == compute_file_sha256(result.raw_snapshot_path)
    assert raw_snapshot["article_fetch_diagnostics"][0]["text_acquisition_mode"] == "full_article"
    assert raw_snapshot["resolved_symbols"] == ["INFY"]


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
