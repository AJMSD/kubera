from __future__ import annotations

from pathlib import Path

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
