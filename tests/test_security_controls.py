from __future__ import annotations

from pathlib import Path

from kubera.ingest.news_data import build_article_fallback_result, sanitize_provider_warning
from kubera.pilot.live_pilot import prefix_metadata_warnings
from kubera.utils.logging import sanitize_log_text


def test_sanitize_log_text_redacts_tokens_and_query_secrets() -> None:
    message = (
        "Request failed for https://example.com/feed?api_key=secret-key-123&token=abc123 "
        "Authorization: Bearer top-secret-token "
        "x-goog-api-key=another-secret"
    )

    sanitized = sanitize_log_text(message)

    assert "secret-key-123" not in sanitized
    assert "abc123" not in sanitized
    assert "top-secret-token" not in sanitized
    assert "another-secret" not in sanitized
    assert sanitized.count("[redacted]") >= 3
    assert "https://example.com/feed?" in sanitized


def test_sanitize_log_text_redacts_sensitive_url_params_but_keeps_safe_ones() -> None:
    message = "Failure: https://example.com/feed?token=my-token&cursor=abc123"

    sanitized = sanitize_log_text(message)

    assert "my-token" not in sanitized
    assert "token=[redacted]" in sanitized
    assert "cursor=abc123" in sanitized


def test_prefix_metadata_warnings_redacts_sensitive_fragments() -> None:
    warnings = prefix_metadata_warnings(
        {"warnings": ["fetch failed token=abc123", "Authorization: Bearer super-secret"]},
        "stage5",
    )

    assert len(warnings) == 2
    assert all("[redacted]" in warning for warning in warnings)
    assert all("abc123" not in warning for warning in warnings)
    assert all("super-secret" not in warning for warning in warnings)


def test_prefix_metadata_warnings_compacts_stage5_provider_failure_details() -> None:
    warnings = prefix_metadata_warnings(
        {
            "warnings": [
                "nse_rss request failed: HTTPSConnectionPool(host='www.nseindia.com', port=443): Read timed out.",
                "bse_rss_failed",
                "marketaux request failed: 503 Server Error: Service Unavailable",
            ]
        },
        "stage5",
    )

    assert warnings == [
        "stage5:nse_rss_failed",
        "stage5:bse_rss_failed",
        "stage5:marketaux_failed",
    ]


def test_sanitize_provider_warning_redacts_query_secrets() -> None:
    warning = (
        "nse_rss request failed: "
        "https://example.com/feed?api_key=super-secret&token=abc123&cursor=safe"
    )

    sanitized = sanitize_provider_warning(warning)

    assert "super-secret" not in sanitized
    assert "abc123" not in sanitized
    assert "api_key=[redacted]" in sanitized
    assert "token=abc123" not in sanitized


def test_article_fallback_result_sanitizes_persisted_fetch_error() -> None:
    result = build_article_fallback_result(
        {"article_title": "Headline", "summary_snippet": "Snippet"},
        reason="page_fetch_failed",
        fetch_error=(
            "RequestException: https://news.example.com/item?access_token=top-secret&cursor=keep"
        ),
    )

    assert result.fetch_error is not None
    assert "top-secret" not in result.fetch_error
    assert "access_token=[redacted]" in result.fetch_error
    assert "cursor=keep" not in result.fetch_error


def test_gitignore_covers_local_artifacts_and_sensitive_files() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for expected_entry in (
        ".env",
        "PRD.md",
        "checklist.md",
        "codebase_analysis_report.md",
        "data/",
        "artifacts/",
        "story/",
    ):
        assert expected_entry in gitignore
