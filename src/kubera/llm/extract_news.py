"""Stage 6 LLM extraction helpers for Kubera."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd
import requests

from kubera.config import AppSettings, load_settings, resolve_runtime_settings
from kubera.utils.hashing import compute_file_sha256
from kubera.utils.logging import configure_logging, sanitize_log_text
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot


SCHEMA_VERSION = "1"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# For large offline backfills, consider the Gemini Batch API (separate rate limits; not for pilot latency).
ARTICLE_TEXT_START_MARKER = "<article_text>"
ARTICLE_TEXT_END_MARKER = "</article_text>"
PROMPT_JSON_FIELDS = (
    "ticker",
    "company_name",
    "relevance_score",
    "sentiment_label",
    "sentiment_score",
    "event_type",
    "event_severity",
    "expected_horizon",
    "directional_bias",
    "confidence_score",
    "rationale_short",
    "extraction_mode",
    "content_quality_score",
    "warning_flag",
)
SOURCE_NEWS_REQUIRED_COLUMNS = (
    "article_id",
    "ticker",
    "exchange",
    "provider",
    "article_title",
    "article_url",
    "canonical_url",
    "source_domain",
    "provider_source",
    "published_at_utc",
    "published_at_ist",
    "published_date_ist",
    "summary_snippet",
    "full_text",
    "text_acquisition_mode",
    "text_acquisition_reason",
    "fetch_warning_flag",
)
EXTRACTED_NEWS_COLUMNS = (
    "article_id",
    "ticker",
    "exchange",
    "company_name",
    "article_title",
    "article_url",
    "canonical_url",
    "source_domain",
    "provider",
    "provider_source",
    "published_at_utc",
    "published_at_ist",
    "published_date_ist",
    "extraction_mode",
    "content_quality_score",
    "warning_flag",
    "source_fetch_warning_flag",
    "prompt_truncated",
    "article_input_hash",
    "request_mode",
    "recovery_reason",
    "recovery_status",
    "llm_provider",
    "llm_model",
    "prompt_version",
    "schema_version",
    "relevance_score",
    "sentiment_label",
    "sentiment_score",
    "event_type",
    "event_severity",
    "expected_horizon",
    "directional_bias",
    "confidence_score",
    "rationale_short",
)
ALLOWED_SENTIMENT_LABELS = frozenset({"positive", "neutral", "negative"})
ALLOWED_DIRECTIONAL_BIAS = frozenset({"bullish", "neutral", "bearish"})
ALLOWED_HORIZONS = frozenset({"intraday", "short_term", "medium_term", "long_term"})
ALLOWED_EVENT_TYPES = frozenset(
    {
        "earnings",
        "guidance",
        "deal_win",
        "product_launch",
        "partnership",
        "acquisition",
        "lawsuit",
        "regulation",
        "analyst_commentary",
        "market_reaction",
        "management_change",
        "supply_chain",
        "sector_development",
        "macro_spillover",
        "government_policy_impact",
        "rupee_commodity_sensitivity",
        "other",
    }
)
ALLOWED_EXTRACTION_MODES = frozenset(
    {"full_article", "headline_plus_snippet", "headline_only"}
)
CONTENT_QUALITY_BY_MODE = {
    "full_article": 1.0,
    "headline_plus_snippet": 0.75,
    "headline_only": 0.5,
}
REQUEST_MODE_PLAIN_TEXT = "plain_text"
REQUEST_MODE_URL_CONTEXT = "url_context"
REQUEST_MODE_GOOGLE_SEARCH = "google_search"
ALLOWED_REQUEST_MODES = frozenset(
    {
        REQUEST_MODE_PLAIN_TEXT,
        REQUEST_MODE_URL_CONTEXT,
        REQUEST_MODE_GOOGLE_SEARCH,
    }
)
ALLOWED_RECOVERY_STATUSES = frozenset(
    {"not_needed", "skipped", "exhausted", "succeeded"}
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ENUM_NORMALIZER = re.compile(r"[^a-z0-9]+")


class LlmExtractionError(RuntimeError):
    """Raised when Stage 6 extraction cannot continue."""


class SchemaValidationError(LlmExtractionError):
    """Raised when a model payload fails schema validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class RetryableProviderError(LlmExtractionError):
    """Raised when the provider failed in a retryable way."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        raw_payload: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.raw_payload = raw_payload
        super().__init__(message)


class NonRetryableProviderError(LlmExtractionError):
    """Raised when the provider failed in a non-retryable way."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        raw_payload: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.raw_payload = raw_payload
        super().__init__(message)


@dataclass(frozen=True)
class PreparedArticleInput:
    """Normalized article payload sent to the LLM."""

    article_id: str
    source_row: dict[str, Any]
    prompt_article_text: str
    prompt_truncated: bool
    extraction_mode: str
    content_quality_score: float
    warning_flag: bool
    article_input_hash: str


@dataclass(frozen=True)
class ExtractionRequestOptions:
    """One configured model request, optionally with Gemini tools enabled."""

    model: str
    request_mode: str = REQUEST_MODE_PLAIN_TEXT
    url_context_urls: tuple[str, ...] = ()
    enable_google_search: bool = False

    def __post_init__(self) -> None:
        if self.request_mode not in ALLOWED_REQUEST_MODES:
            raise LlmExtractionError(f"Unsupported request mode: {self.request_mode}")
        if not self.model.strip():
            raise LlmExtractionError("Request model must not be empty.")


@dataclass(frozen=True)
class ProviderTextResponse:
    """Structured provider response text plus transport metadata."""

    response_text: str
    raw_payload: dict[str, Any]
    status_code: int
    finish_reason: str | None
    retrieved_urls: tuple[str, ...] = ()
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LlmExtractionRunResult:
    """Persisted Stage 6 extraction artifact summary."""

    extraction_table_path: Path
    metadata_path: Path
    failure_log_path: Path
    raw_snapshot_path: Path
    source_row_count: int
    success_count: int
    failure_count: int
    cache_hit_count: int
    fresh_call_count: int


@dataclass(frozen=True)
class SourceExtractionOutcome:
    """One source-row extraction outcome, ready for deterministic aggregation."""

    success_row: dict[str, Any] | None
    failure: dict[str, Any] | None
    article_run: dict[str, Any]
    cache_hit_count: int
    fresh_call_count: int
    provider_request_count: int
    retry_count: int
    llm_provider: str


class StructuredNewsExtractionClient(ABC):
    """Boundary for structured article extraction providers."""

    provider_name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        options: ExtractionRequestOptions | None = None,
    ) -> ProviderTextResponse:
        """Return one provider text response for the given prompt."""


class GeminiApiExtractionClient(StructuredNewsExtractionClient):
    """Gemma client backed by the Gemini API generateContent endpoint."""

    provider_name = "gemini_api"

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        timeout_seconds: int,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def generate(
        self,
        prompt: str,
        *,
        options: ExtractionRequestOptions | None = None,
    ) -> ProviderTextResponse:
        request_options = options or ExtractionRequestOptions(model=self._model)
        request_body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "candidateCount": 1,
            },
        }
        tools: list[dict[str, Any]] = []
        if request_options.request_mode == REQUEST_MODE_URL_CONTEXT:
            tools.append({"urlContext": {}})
        if request_options.enable_google_search:
            tools.append({"googleSearch": {}})
        if tools:
            request_body["tools"] = tools

        response = self._session.post(
            f"{GEMINI_API_BASE_URL}/{request_options.model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            json=request_body,
            timeout=self._timeout_seconds,
        )

        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise RetryableProviderError(
                f"Gemini API request failed with status {response.status_code}.",
                status_code=response.status_code,
                raw_payload=response.text,
            )
        if 400 <= response.status_code < 500:
            raise NonRetryableProviderError(
                f"Gemini API request failed with status {response.status_code}.",
                status_code=response.status_code,
                raw_payload=response.text,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RetryableProviderError(
                "Gemini API response was not valid JSON.",
                status_code=response.status_code,
                raw_payload=response.text,
            ) from exc

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            prompt_feedback = payload.get("promptFeedback", {})
            block_reason = sanitize_prompt_text(prompt_feedback.get("blockReason"))
            raise NonRetryableProviderError(
                block_reason or "Gemini API response did not include a candidate.",
                status_code=response.status_code,
                raw_payload=json.dumps(payload, sort_keys=True),
            )

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise RetryableProviderError(
                "Gemini API candidate payload was malformed.",
                status_code=response.status_code,
                raw_payload=json.dumps(payload, sort_keys=True),
            )

        content = candidate.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        response_text = "\n".join(
            str(part.get("text"))
            for part in parts
            if isinstance(part, dict) and part.get("text") is not None
        ).strip()
        if not response_text:
            raise RetryableProviderError(
                "Gemini API candidate did not include response text.",
                status_code=response.status_code,
                raw_payload=json.dumps(payload, sort_keys=True),
            )

        return ProviderTextResponse(
            response_text=response_text,
            raw_payload=payload,
            status_code=response.status_code,
            finish_reason=sanitize_prompt_text(candidate.get("finishReason")) or None,
            retrieved_urls=extract_retrieved_urls(payload),
            citations=extract_response_citations(payload),
        )

def sanitize_prompt_text(value: Any) -> str:
    """Remove control characters and collapse internal whitespace."""

    if value is None:
        return ""
    cleaned = _CONTROL_CHARACTERS.sub(" ", str(value))
    return " ".join(cleaned.split())


def extract_retrieved_urls(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect retrieved URLs from a Gemini response payload."""

    deduped_urls: list[str] = []
    seen_urls: set[str] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).strip().lower()
                if normalized_key in {
                    "url",
                    "uri",
                    "retrievedurl",
                    "retrieved_url",
                    "sourceurl",
                    "source_url",
                }:
                    candidate_url = sanitize_prompt_text(item)
                    if candidate_url.startswith(("http://", "https://")) and candidate_url not in seen_urls:
                        seen_urls.add(candidate_url)
                        deduped_urls.append(candidate_url)
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(payload)
    return tuple(deduped_urls)


def extract_response_citations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract citation-like URL/title pairs from a Gemini response payload."""

    citations: list[dict[str, Any]] = []
    candidate_payloads = payload.get("candidates")
    if not isinstance(candidate_payloads, list):
        return citations

    for candidate in candidate_payloads:
        if not isinstance(candidate, dict):
            continue
        grounding_metadata = candidate.get("groundingMetadata")
        if not isinstance(grounding_metadata, dict):
            continue
        grounding_chunks = grounding_metadata.get("groundingChunks")
        if not isinstance(grounding_chunks, list):
            continue
        for chunk in grounding_chunks:
            if not isinstance(chunk, dict):
                continue
            web_chunk = chunk.get("web")
            if not isinstance(web_chunk, dict):
                continue
            url = sanitize_prompt_text(web_chunk.get("uri"))
            title = sanitize_prompt_text(web_chunk.get("title"))
            if url.startswith(("http://", "https://")):
                citations.append({"title": title or None, "url": url})
    return citations


def sanitize_article_prompt_text(value: Any) -> str:
    """Neutralize article markers so source text cannot escape the bounded prompt section."""

    cleaned = sanitize_prompt_text(value)
    return (
        cleaned.replace(ARTICLE_TEXT_START_MARKER, "[article_text]")
        .replace(ARTICLE_TEXT_END_MARKER, "[/article_text]")
    )


def normalize_enum(value: Any) -> str:
    """Convert a user or model enum value into normalized snake_case."""

    cleaned = sanitize_prompt_text(value).lower()
    if not cleaned:
        return ""
    normalized = _ENUM_NORMALIZER.sub("_", cleaned).strip("_")
    return normalized


def normalize_free_text(value: Any) -> str:
    """Normalize free text for tolerant equality checks."""

    return sanitize_prompt_text(value).casefold()


def resolve_source_name(source_row: Mapping[str, Any]) -> str:
    """Resolve the preferred display name for the article source."""

    for field_name in ("provider_source", "source_domain", "provider"):
        resolved = sanitize_prompt_text(source_row.get(field_name))
        if resolved:
            return resolved
    return ""


def normalize_timestamp(raw_value: Any) -> str:
    """Normalize a timestamp-like value into a UTC ISO string."""

    value = sanitize_prompt_text(raw_value)
    if not value:
        raise ValueError("Expected a non-empty timestamp value.")

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def coerce_float(value: Any, *, field_name: str, minimum: float, maximum: float) -> float:
    """Coerce one numeric field and validate its closed range."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError([f"{field_name} must be numeric."]) from exc

    if not math.isfinite(parsed):
        raise SchemaValidationError([f"{field_name} must be finite."])
    if parsed < minimum or parsed > maximum:
        raise SchemaValidationError(
            [f"{field_name} must be between {minimum} and {maximum}."]
        )
    return parsed


def coerce_bool(value: Any, *, field_name: str) -> bool:
    """Coerce a model boolean field from common JSON-like spellings."""

    if isinstance(value, bool):
        return value

    normalized = sanitize_prompt_text(value).lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise SchemaValidationError([f"{field_name} must be a boolean."])


def compute_content_quality_score(extraction_mode: str) -> float:
    """Return the deterministic content quality score for one extraction mode."""

    normalized_mode = normalize_enum(extraction_mode)
    if normalized_mode not in CONTENT_QUALITY_BY_MODE:
        raise LlmExtractionError(f"Unsupported extraction mode: {extraction_mode}")
    return CONTENT_QUALITY_BY_MODE[normalized_mode]


def build_article_fallback_text(article_title: str, summary_snippet: str) -> str:
    """Compose a headline/snippet fallback string for prompt inputs."""

    title = sanitize_prompt_text(article_title)
    snippet = sanitize_prompt_text(summary_snippet)
    if title and snippet:
        return f"{title}\n\n{snippet}"
    return title or snippet


def prepare_article_input(
    source_row: Mapping[str, Any],
    *,
    company_name: str,
    max_input_chars: int,
) -> PreparedArticleInput:
    """Normalize one Stage 5 row into the bounded Stage 6 prompt input."""

    normalized_row = {key: source_row.get(key) for key in SOURCE_NEWS_REQUIRED_COLUMNS}
    article_id = sanitize_prompt_text(normalized_row.get("article_id"))
    extraction_mode = normalize_enum(normalized_row.get("text_acquisition_mode"))
    if not article_id:
        raise LlmExtractionError("Stage 5 row is missing article_id.")
    if extraction_mode not in ALLOWED_EXTRACTION_MODES:
        raise LlmExtractionError(
            f"Stage 5 row {article_id} has unsupported text_acquisition_mode: {normalized_row.get('text_acquisition_mode')}"
        )

    published_at_utc = normalize_timestamp(normalized_row.get("published_at_utc"))
    prompt_text = sanitize_article_prompt_text(normalized_row.get("full_text"))
    if not prompt_text:
        prompt_text = sanitize_article_prompt_text(
            build_article_fallback_text(
            sanitize_prompt_text(normalized_row.get("article_title")),
            sanitize_prompt_text(normalized_row.get("summary_snippet")),
            )
        )
    if not prompt_text:
        raise LlmExtractionError(f"Stage 5 row {article_id} does not contain usable article text.")

    prompt_truncated = len(prompt_text) > max_input_chars
    prompt_article_text = prompt_text[:max_input_chars]
    content_quality_score = compute_content_quality_score(extraction_mode)
    source_warning_flag = coerce_bool(
        normalized_row.get("fetch_warning_flag"),
        field_name="fetch_warning_flag",
    )
    warning_flag = source_warning_flag or prompt_truncated

    input_payload = {
        "article_id": article_id,
        "ticker": sanitize_prompt_text(normalized_row.get("ticker")).upper(),
        "exchange": sanitize_prompt_text(normalized_row.get("exchange")).upper(),
        "company_name": sanitize_prompt_text(company_name),
        "article_title": sanitize_prompt_text(normalized_row.get("article_title")),
        "provider": sanitize_prompt_text(normalized_row.get("provider")),
        "provider_source": resolve_source_name(normalized_row),
        "source_domain": sanitize_prompt_text(normalized_row.get("source_domain")),
        "article_url": sanitize_prompt_text(normalized_row.get("article_url")),
        "canonical_url": sanitize_prompt_text(normalized_row.get("canonical_url")),
        "published_at_utc": published_at_utc,
        "published_at_ist": sanitize_prompt_text(normalized_row.get("published_at_ist")),
        "published_date_ist": sanitize_prompt_text(normalized_row.get("published_date_ist")),
        "summary_snippet": sanitize_prompt_text(normalized_row.get("summary_snippet")),
        "text_acquisition_reason": sanitize_prompt_text(
            normalized_row.get("text_acquisition_reason")
        ),
        "extraction_mode": extraction_mode,
        "content_quality_score": content_quality_score,
        "warning_flag": warning_flag,
        "prompt_article_text": prompt_article_text,
    }
    article_input_hash = build_article_input_hash(input_payload)

    normalized_row["published_at_utc"] = published_at_utc
    normalized_row["ticker"] = input_payload["ticker"]
    normalized_row["exchange"] = input_payload["exchange"]
    return PreparedArticleInput(
        article_id=article_id,
        source_row=normalized_row,
        prompt_article_text=prompt_article_text,
        prompt_truncated=prompt_truncated,
        extraction_mode=extraction_mode,
        content_quality_score=content_quality_score,
        warning_flag=warning_flag,
        article_input_hash=article_input_hash,
    )


def build_article_input_hash(input_payload: Mapping[str, Any]) -> str:
    """Hash the bounded prompt input so repeated runs can reuse cached outputs."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(dict(input_payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    )
    return digest.hexdigest()


def build_extraction_prompt(
    prepared_article: PreparedArticleInput,
    *,
    company_name: str,
    prompt_version: str,
    request_mode: str = REQUEST_MODE_PLAIN_TEXT,
    recovery_reason: str | None = None,
    url_context_urls: tuple[str, ...] = (),
    retry_reason: str | None = None,
) -> str:
    """Build the single-user prompt sent to Gemma."""

    source_row = prepared_article.source_row
    retry_suffix = ""
    if retry_reason:
        retry_suffix = (
            "\nPrevious response failed validation. Correct the issue and return one JSON object only.\n"
            f"Failure reason: {sanitize_prompt_text(retry_reason)}\n"
        )
    recovery_suffix = ""
    if request_mode == REQUEST_MODE_URL_CONTEXT and url_context_urls:
        recovery_suffix = (
            "\nRecovery mode: url_context.\n"
            "Use the Gemini URL context tool to read the linked publisher URL before extracting fields.\n"
            f"Preferred URL: {sanitize_prompt_text(url_context_urls[0])}\n"
        )
    elif request_mode == REQUEST_MODE_GOOGLE_SEARCH:
        recovery_suffix = (
            "\nRecovery mode: google_search.\n"
            "Use Google Search only to recover missing context for this company and article topic.\n"
        )
    if recovery_reason:
        recovery_suffix += f"Recovery reason: {sanitize_prompt_text(recovery_reason)}\n"

    return (
        "You extract structured stock-news signals for a machine learning pipeline.\n"
        "Return exactly one JSON object and no prose, markdown, or code fences.\n"
        "Do not invent missing facts. If the article does not support a stronger claim, choose neutral or other values.\n"
        "Use the provided deterministic metadata as fixed context; do not infer different values.\n"
        f"Prompt version: {sanitize_prompt_text(prompt_version)}\n"
        f"{retry_suffix}"
        f"{recovery_suffix}"
        "Allowed sentiment_label: positive, neutral, negative.\n"
        "Allowed directional_bias: bullish, neutral, bearish.\n"
        "Allowed expected_horizon: intraday, short_term, medium_term, long_term.\n"
        "Allowed event_type: earnings, guidance, deal_win, product_launch, partnership, acquisition, lawsuit, regulation, analyst_commentary, market_reaction, management_change, supply_chain, sector_development, macro_spillover, government_policy_impact, rupee_commodity_sensitivity, other.\n"
        "Fields and ranges:\n"
        "- relevance_score: 0 to 1\n"
        "- sentiment_score: -1 to 1\n"
        "- event_severity: 0 to 1\n"
        "- confidence_score: 0 to 1\n"
        "- content_quality_score: 0 to 1\n"
        "Required JSON keys in this exact naming:\n"
        f"{', '.join(PROMPT_JSON_FIELDS)}\n"
        "Deterministic metadata (source-of-truth context, persisted from the source row):\n"
        f"- ticker: {sanitize_prompt_text(source_row.get('ticker')).upper()}\n"
        f"- company_name: {sanitize_prompt_text(company_name)}\n"
        f"- article_title: {sanitize_prompt_text(source_row.get('article_title'))}\n"
        f"- published_at: {sanitize_prompt_text(source_row.get('published_at_utc'))}\n"
        f"- source: {resolve_source_name(source_row)}\n"
        f"- extraction_mode: {prepared_article.extraction_mode}\n"
        f"- content_quality_score: {prepared_article.content_quality_score:.2f}\n"
        f"- warning_flag: {'true' if prepared_article.warning_flag else 'false'}\n"
        "Article metadata:\n"
        f"- exchange: {sanitize_prompt_text(source_row.get('exchange')).upper()}\n"
        f"- source_domain: {sanitize_prompt_text(source_row.get('source_domain'))}\n"
        f"- article_url: {sanitize_prompt_text(source_row.get('article_url'))}\n"
        f"- canonical_url: {sanitize_prompt_text(source_row.get('canonical_url'))}\n"
        f"- published_at_ist: {sanitize_prompt_text(source_row.get('published_at_ist'))}\n"
        f"- published_date_ist: {sanitize_prompt_text(source_row.get('published_date_ist'))}\n"
        f"- text_acquisition_reason: {sanitize_prompt_text(source_row.get('text_acquisition_reason'))}\n"
        f"{ARTICLE_TEXT_START_MARKER}\n"
        f"{prepared_article.prompt_article_text}\n"
        f"{ARTICLE_TEXT_END_MARKER}\n"
    )


def parse_first_json_object(raw_text: str) -> dict[str, Any]:
    """Parse the first top-level JSON object from a model text response."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text):
        try:
            parsed, _ = decoder.raw_decode(raw_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LlmExtractionError("Model response did not contain a valid JSON object.")


def validate_extraction_payload(
    payload: Mapping[str, Any],
    *,
    prepared_article: PreparedArticleInput,
    company_name: str,
    llm_provider: str,
    llm_model: str,
    prompt_version: str,
    request_mode: str,
) -> dict[str, Any]:
    """Validate one model payload and convert it into a persisted row."""

    missing_fields = [
        field_name for field_name in PROMPT_JSON_FIELDS if field_name not in payload
    ]
    if missing_fields:
        raise SchemaValidationError(
            [f"Missing required fields: {', '.join(sorted(missing_fields))}"]
        )

    source_row = prepared_article.source_row
    errors: list[str] = []

    ticker = sanitize_prompt_text(payload.get("ticker")).upper()
    if ticker != sanitize_prompt_text(source_row.get("ticker")).upper():
        errors.append("ticker does not match the source row")

    model_company_name = normalize_free_text(payload.get("company_name"))
    if model_company_name != normalize_free_text(company_name):
        errors.append("company_name does not match the configured company")

    extraction_mode = normalize_enum(payload.get("extraction_mode"))
    if extraction_mode != prepared_article.extraction_mode:
        errors.append("extraction_mode does not match the source row")

    try:
        content_quality_score = coerce_float(
            payload.get("content_quality_score"),
            field_name="content_quality_score",
            minimum=0.0,
            maximum=1.0,
        )
    except SchemaValidationError as exc:
        errors.extend(exc.errors)
        content_quality_score = -1.0
    if (
        content_quality_score >= 0
        and not math.isclose(
            content_quality_score,
            prepared_article.content_quality_score,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        errors.append("content_quality_score does not match the deterministic score")

    try:
        warning_flag = coerce_bool(payload.get("warning_flag"), field_name="warning_flag")
    except SchemaValidationError as exc:
        errors.extend(exc.errors)
        warning_flag = not prepared_article.warning_flag
    if warning_flag != prepared_article.warning_flag:
        errors.append("warning_flag does not match the deterministic flag")

    sentiment_label = normalize_enum(payload.get("sentiment_label"))
    if sentiment_label not in ALLOWED_SENTIMENT_LABELS:
        errors.append(
            f"sentiment_label must be one of {sorted(ALLOWED_SENTIMENT_LABELS)}"
        )

    directional_bias = normalize_enum(payload.get("directional_bias"))
    if directional_bias not in ALLOWED_DIRECTIONAL_BIAS:
        errors.append(
            f"directional_bias must be one of {sorted(ALLOWED_DIRECTIONAL_BIAS)}"
        )

    expected_horizon = normalize_enum(payload.get("expected_horizon"))
    if expected_horizon not in ALLOWED_HORIZONS:
        errors.append(
            f"expected_horizon must be one of {sorted(ALLOWED_HORIZONS)}"
        )

    event_type = normalize_enum(payload.get("event_type"))
    if event_type not in ALLOWED_EVENT_TYPES:
        errors.append(f"event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}")

    numeric_fields: dict[str, float] = {}
    for field_name, minimum, maximum in (
        ("relevance_score", 0.0, 1.0),
        ("sentiment_score", -1.0, 1.0),
        ("event_severity", 0.0, 1.0),
        ("confidence_score", 0.0, 1.0),
    ):
        try:
            numeric_fields[field_name] = coerce_float(
                payload.get(field_name),
                field_name=field_name,
                minimum=minimum,
                maximum=maximum,
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)

    rationale_short = sanitize_prompt_text(payload.get("rationale_short"))
    if not rationale_short:
        errors.append("rationale_short must not be empty")

    if errors:
        raise SchemaValidationError(errors)

    return {
        "article_id": prepared_article.article_id,
        "ticker": sanitize_prompt_text(source_row.get("ticker")).upper(),
        "exchange": sanitize_prompt_text(source_row.get("exchange")).upper(),
        "company_name": sanitize_prompt_text(company_name),
        "article_title": sanitize_prompt_text(source_row.get("article_title")),
        "article_url": sanitize_prompt_text(source_row.get("article_url")),
        "canonical_url": sanitize_prompt_text(source_row.get("canonical_url")),
        "source_domain": sanitize_prompt_text(source_row.get("source_domain")),
        "provider": sanitize_prompt_text(source_row.get("provider")),
        "provider_source": resolve_source_name(source_row),
        "published_at_utc": sanitize_prompt_text(source_row.get("published_at_utc")),
        "published_at_ist": sanitize_prompt_text(source_row.get("published_at_ist")),
        "published_date_ist": sanitize_prompt_text(source_row.get("published_date_ist")),
        "extraction_mode": prepared_article.extraction_mode,
        "content_quality_score": prepared_article.content_quality_score,
        "warning_flag": prepared_article.warning_flag,
        "source_fetch_warning_flag": coerce_bool(
            source_row.get("fetch_warning_flag"),
            field_name="fetch_warning_flag",
        ),
        "prompt_truncated": prepared_article.prompt_truncated,
        "article_input_hash": prepared_article.article_input_hash,
        "request_mode": sanitize_prompt_text(request_mode),
        "recovery_reason": None,
        "recovery_status": "not_needed",
        "llm_provider": sanitize_prompt_text(llm_provider),
        "llm_model": sanitize_prompt_text(llm_model),
        "prompt_version": sanitize_prompt_text(prompt_version),
        "schema_version": SCHEMA_VERSION,
        "relevance_score": numeric_fields["relevance_score"],
        "sentiment_label": sentiment_label,
        "sentiment_score": numeric_fields["sentiment_score"],
        "event_type": event_type,
        "event_severity": numeric_fields["event_severity"],
        "expected_horizon": expected_horizon,
        "directional_bias": directional_bias,
        "confidence_score": numeric_fields["confidence_score"],
        "rationale_short": rationale_short,
    }


def resolve_news_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    news_table_path: str | Path | None,
) -> Path:
    """Resolve the Stage 5 processed news table path."""

    if news_table_path is not None:
        return Path(news_table_path).expanduser().resolve()

    return path_manager.build_processed_news_data_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def infer_news_metadata_path(news_table_path: Path) -> Path | None:
    """Infer the Stage 5 processed news metadata path when it follows the standard naming rule."""

    if news_table_path.suffix.lower() != ".csv":
        return None

    candidate = news_table_path.with_name(f"{news_table_path.stem}.metadata.json")
    if candidate.exists():
        return candidate
    return None


def read_processed_news(news_table_path: Path) -> pd.DataFrame:
    """Read the persisted Stage 5 processed news table."""

    try:
        return pd.read_csv(news_table_path)
    except FileNotFoundError as exc:
        raise LlmExtractionError(
            f"Processed news file does not exist: {news_table_path}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise LlmExtractionError(
            f"Processed news file is empty: {news_table_path}"
        ) from exc


def validate_processed_news_frame(
    news_frame: pd.DataFrame,
    *,
    ticker: str,
    exchange: str,
) -> pd.DataFrame:
    """Validate the Stage 5 processed news table before extraction."""

    missing_columns = [
        column for column in SOURCE_NEWS_REQUIRED_COLUMNS if column not in news_frame.columns
    ]
    if missing_columns:
        raise LlmExtractionError(
            f"Processed news table is missing required columns: {missing_columns}"
        )

    working_frame = news_frame.copy()
    if working_frame.empty:
        return working_frame

    source_tickers = {
        sanitize_prompt_text(value).upper()
        for value in working_frame["ticker"].dropna().unique().tolist()
    }
    source_exchanges = {
        sanitize_prompt_text(value).upper()
        for value in working_frame["exchange"].dropna().unique().tolist()
    }
    if source_tickers != {ticker.upper()}:
        raise LlmExtractionError(
            f"Processed news ticker values do not match the requested ticker: {sorted(source_tickers)}"
        )
    if source_exchanges != {exchange.upper()}:
        raise LlmExtractionError(
            f"Processed news exchange values do not match the requested exchange: {sorted(source_exchanges)}"
        )

    article_ids = working_frame["article_id"].map(sanitize_prompt_text)
    if any(not article_id for article_id in article_ids.tolist()):
        raise LlmExtractionError("Processed news table contains empty article_id values.")
    if article_ids.duplicated().any():
        raise LlmExtractionError("Processed news table contains duplicate article_id values.")

    return working_frame.reset_index(drop=True)


def build_cache_key(
    *,
    article_input_hash: Any,
    request_mode: Any,
    llm_provider: Any,
    llm_model: Any,
    prompt_version: Any,
    schema_version: Any,
) -> tuple[str, str, str, str, str, str]:
    """Build the success-cache key for one extracted article."""

    return (
        sanitize_prompt_text(article_input_hash),
        sanitize_prompt_text(request_mode) or REQUEST_MODE_PLAIN_TEXT,
        sanitize_prompt_text(llm_provider),
        sanitize_prompt_text(llm_model),
        sanitize_prompt_text(prompt_version),
        sanitize_prompt_text(schema_version),
    )


def load_extraction_cache(
    cache_path: Path,
    *,
    force: bool,
) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    """Load previously persisted successful extractions for cache reuse."""

    if force or not cache_path.exists():
        return {}

    try:
        cached_frame = pd.read_csv(cache_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return {}

    required_cache_columns = {
        "article_input_hash",
        "llm_provider",
        "llm_model",
        "prompt_version",
        "schema_version",
    }
    if not required_cache_columns.issubset(cached_frame.columns):
        return {}

    cache: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in cached_frame.to_dict(orient="records"):
        key = build_cache_key(
            article_input_hash=row.get("article_input_hash"),
            request_mode=row.get("request_mode"),
            llm_provider=row.get("llm_provider"),
            llm_model=row.get("llm_model"),
            prompt_version=row.get("prompt_version"),
            schema_version=row.get("schema_version"),
        )
        if not all(key):
            continue
        cache[key] = row
    return cache


def resolve_extraction_client(
    settings: AppSettings,
    *,
    client: StructuredNewsExtractionClient | None = None,
) -> StructuredNewsExtractionClient:
    """Resolve the active Stage 6 extraction client."""

    if client is not None:
        return client

    provider_name = settings.providers.llm_provider.strip().lower()
    if provider_name != "gemini_api":
        raise LlmExtractionError(
            f"Unsupported LLM provider for Stage 6: {settings.providers.llm_provider}"
        )
    if not settings.providers.llm_api_key:
        raise LlmExtractionError("Stage 6 extraction requires KUBERA_LLM_API_KEY.")

    return GeminiApiExtractionClient(
        settings.providers.llm_api_key,
        model=settings.llm_extraction.model,
        timeout_seconds=settings.llm_extraction.request_timeout_seconds,
    )


def count_series_values(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    """Count string values in one DataFrame column for metadata."""

    if frame.empty or column_name not in frame.columns:
        return {}
    counts = frame[column_name].fillna("null").value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def build_failure_entry(
    *,
    prepared_article: PreparedArticleInput,
    llm_provider: str,
    llm_model: str,
    prompt_version: str,
    failure_category: str,
    error_message: str,
    attempt_logs: list[dict[str, Any]],
    schema_errors: list[str] | None = None,
    raw_response_text: str | None = None,
    provider_status_code: int | None = None,
) -> dict[str, Any]:
    """Build one persisted failure entry."""

    source_row = prepared_article.source_row
    sanitized_error_message = sanitize_log_text(error_message)
    return {
        "article_id": prepared_article.article_id,
        "article_input_hash": prepared_article.article_input_hash,
        "ticker": sanitize_prompt_text(source_row.get("ticker")).upper(),
        "exchange": sanitize_prompt_text(source_row.get("exchange")).upper(),
        "article_title": sanitize_prompt_text(source_row.get("article_title")),
        "provider": sanitize_prompt_text(source_row.get("provider")),
        "source_domain": sanitize_prompt_text(source_row.get("source_domain")),
        "source_name": resolve_source_name(source_row),
        "published_at_utc": sanitize_prompt_text(source_row.get("published_at_utc")),
        "extraction_mode": prepared_article.extraction_mode,
        "llm_provider": sanitize_prompt_text(llm_provider),
        "llm_model": sanitize_prompt_text(llm_model),
        "prompt_version": sanitize_prompt_text(prompt_version),
        "schema_version": SCHEMA_VERSION,
        "failure_category": sanitize_prompt_text(failure_category),
        "error_message": sanitize_prompt_text(sanitized_error_message),
        "attempt_count": len(attempt_logs),
        "schema_errors": schema_errors or [],
        "provider_status_code": provider_status_code,
        "raw_response_text": raw_response_text,
    }


def attempt_backoff(seconds: float) -> None:
    """Sleep between retries so tests can patch the delay hook cleanly."""

    time.sleep(seconds)


def build_article_run_trace(
    prepared_article: PreparedArticleInput,
    attempt_logs: list[dict[str, Any]],
    *,
    request_mode: str,
    llm_model: str,
    recovery_reason: str | None = None,
    tool_urls: tuple[str, ...] = (),
    cache_hit: bool = False,
) -> dict[str, Any]:
    """Build the raw per-article trace stored in the run snapshot."""

    return {
        "article_id": prepared_article.article_id,
        "article_input_hash": prepared_article.article_input_hash,
        "extraction_mode": prepared_article.extraction_mode,
        "request_mode": request_mode,
        "llm_model": llm_model,
        "recovery_reason": sanitize_prompt_text(recovery_reason) or None,
        "tool_urls": list(tool_urls),
        "cache_hit": cache_hit,
        "attempt_count": len(attempt_logs),
        "attempts": attempt_logs,
    }


@dataclass
class RecoveryBudgetTracker:
    """Track per-run recovery usage and per-model request budgets."""

    max_articles_per_run: int
    attempted_articles: int = 0
    model_request_counts: dict[str, int] = field(default_factory=dict)

    def can_attempt_article(self) -> bool:
        return self.max_articles_per_run > 0 and self.attempted_articles < self.max_articles_per_run

    def mark_article_attempt(self) -> None:
        if self.max_articles_per_run > 0:
            self.attempted_articles += 1

    def can_use_model(
        self,
        model_settings: Any,
    ) -> bool:
        used_requests = int(self.model_request_counts.get(model_settings.model, 0))
        if (
            model_settings.requests_per_minute_limit > 0
            and used_requests >= model_settings.requests_per_minute_limit
        ):
            return False
        if (
            model_settings.requests_per_day_limit > 0
            and used_requests >= model_settings.requests_per_day_limit
        ):
            return False
        return True

    def note_model_requests(self, model: str, request_count: int) -> None:
        if request_count <= 0:
            return
        self.model_request_counts[model] = int(self.model_request_counts.get(model, 0)) + int(
            request_count
        )


def generate_plain_text_with_tiered_models(
    *,
    settings: AppSettings,
    api_key: str,
    prompt: str,
    client: StructuredNewsExtractionClient | None = None,
) -> tuple[str, str]:
    """
    Try the primary LLM model, then recovery_model_pool entries (deduped by model id),
    with per-tier retries and pool budget tracking. Plain text only (no tools).

    Returns (response_text, model_id). Raises LlmExtractionError if every tier fails or is
    skipped by budget.
    """

    llm = settings.llm_extraction
    primary = llm.model.strip()
    retry_attempts = llm.retry_attempts
    retry_base = llm.retry_base_delay_seconds

    budget = RecoveryBudgetTracker(max_articles_per_run=1_000_000)
    tiers: list[tuple[str, Any]] = [(primary, None)]
    seen: set[str] = {primary}
    for entry in llm.recovery_model_pool:
        mid = str(entry.model).strip()
        if mid in seen:
            continue
        seen.add(mid)
        tiers.append((mid, entry))

    active = client or GeminiApiExtractionClient(
        str(api_key),
        model=primary,
        timeout_seconds=llm.request_timeout_seconds,
    )

    last_error: str | None = None
    for model_id, pool_entry in tiers:
        if pool_entry is not None and not budget.can_use_model(pool_entry):
            continue
        provider_request_count = 0
        for attempt_number in range(1, retry_attempts + 1):
            provider_request_count += 1
            try:
                response = active.generate(
                    prompt,
                    options=ExtractionRequestOptions(
                        model=model_id,
                        request_mode=REQUEST_MODE_PLAIN_TEXT,
                    ),
                )
            except RetryableProviderError as exc:
                last_error = sanitize_log_text(str(exc))
                if attempt_number < retry_attempts:
                    attempt_backoff(retry_base * attempt_number)
                continue
            except NonRetryableProviderError as exc:
                last_error = sanitize_log_text(str(exc))
                break
            except Exception as exc:
                last_error = sanitize_log_text(str(exc))
                break

            text = sanitize_log_text(response.response_text).strip()
            if not text:
                last_error = "Gemini returned an empty response."
                if attempt_number < retry_attempts:
                    attempt_backoff(retry_base * attempt_number)
                continue

            if pool_entry is not None:
                budget.note_model_requests(pool_entry.model, provider_request_count)
            return text, model_id

        if pool_entry is not None:
            budget.note_model_requests(pool_entry.model, provider_request_count)

    raise LlmExtractionError(
        last_error or "All explanation models failed or were skipped by budget."
    )


def is_weak_source_article(prepared_article: PreparedArticleInput) -> bool:
    """Return True when the Stage 5 text looks weak enough to justify recovery."""

    return (
        prepared_article.extraction_mode != "full_article"
        or prepared_article.warning_flag
    )


def determine_recovery_reason(prepared_article: PreparedArticleInput) -> str | None:
    """Explain why a row qualifies for the Stage 6 recovery router."""

    if prepared_article.extraction_mode == "headline_only":
        return "headline_only_source_text"
    if prepared_article.extraction_mode == "headline_plus_snippet":
        return "headline_plus_snippet_source_text"
    if prepared_article.warning_flag:
        return "source_warning_flag"
    return None


def resolve_url_context_urls(source_row: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve one deterministic URL set for Gemini URL context recovery."""

    urls: list[str] = []
    seen: set[str] = set()
    for field_name in ("canonical_url", "article_url"):
        candidate_url = sanitize_prompt_text(source_row.get(field_name))
        if not candidate_url or candidate_url in seen:
            continue
        if not candidate_url.startswith(("http://", "https://")):
            continue
        seen.add(candidate_url)
        urls.append(candidate_url)
    return tuple(urls)


def clone_cached_success_row(
    cached_row: Mapping[str, Any],
    *,
    request_mode: str,
) -> dict[str, Any]:
    """Normalize one cached success row into the current extracted schema."""

    normalized_row = {
        column: cached_row.get(column)
        for column in EXTRACTED_NEWS_COLUMNS
    }
    normalized_row["request_mode"] = sanitize_prompt_text(
        cached_row.get("request_mode")
    ) or request_mode
    normalized_row["recovery_reason"] = cached_row.get("recovery_reason")
    normalized_row["recovery_status"] = sanitize_prompt_text(
        cached_row.get("recovery_status")
    ) or "not_needed"
    return normalized_row


def apply_recovery_metadata(
    success_row: Mapping[str, Any],
    *,
    recovery_reason: str | None,
    recovery_status: str,
) -> dict[str, Any]:
    """Overlay final recovery-routing metadata onto one success row."""

    if recovery_status not in ALLOWED_RECOVERY_STATUSES:
        raise LlmExtractionError(f"Unsupported recovery status: {recovery_status}")

    output_row = {
        column: success_row.get(column)
        for column in EXTRACTED_NEWS_COLUMNS
    }
    output_row["recovery_reason"] = sanitize_prompt_text(recovery_reason) or None
    output_row["recovery_status"] = recovery_status
    return output_row


def build_recovery_request_options(
    *,
    settings: AppSettings,
    prepared_article: PreparedArticleInput,
    budget_tracker: RecoveryBudgetTracker,
) -> tuple[list[ExtractionRequestOptions], str | None]:
    """Build the ordered weak-row recovery request plan for one article."""

    if settings.llm_extraction.recovery_max_articles_per_run <= 0:
        return [], "skipped"
    if not budget_tracker.can_attempt_article():
        return [], "exhausted"

    request_options: list[ExtractionRequestOptions] = []
    url_context_urls = resolve_url_context_urls(prepared_article.source_row)
    for model_settings in settings.llm_extraction.recovery_model_pool:
        if not budget_tracker.can_use_model(model_settings):
            continue
        if (
            settings.llm_extraction.recovery_url_context_enabled
            and model_settings.supports_url_context
            and url_context_urls
        ):
            request_options.append(
                ExtractionRequestOptions(
                    model=model_settings.model,
                    request_mode=REQUEST_MODE_URL_CONTEXT,
                    url_context_urls=url_context_urls,
                )
            )
        if (
            settings.llm_extraction.recovery_google_search_enabled
            and model_settings.supports_google_search
        ):
            request_options.append(
                ExtractionRequestOptions(
                    model=model_settings.model,
                    request_mode=REQUEST_MODE_GOOGLE_SEARCH,
                    url_context_urls=(),
                    enable_google_search=True,
                )
            )

    if request_options:
        return request_options, None
    if settings.llm_extraction.recovery_google_search_enabled:
        return [], "exhausted"
    if settings.llm_extraction.recovery_url_context_enabled and not url_context_urls:
        return [], "skipped"
    return [], "skipped"


def execute_extraction_request(
    *,
    prepared_article: PreparedArticleInput,
    company_name: str,
    llm_provider: str,
    llm_model: str,
    prompt_version: str,
    request_mode: str,
    recovery_reason: str | None,
    url_context_urls: tuple[str, ...],
    retry_attempts: int,
    retry_base_delay_seconds: float,
    client: StructuredNewsExtractionClient | None,
    cache: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any], int, int, bool]:
    """Execute one plain-text or recovery request with cache reuse."""

    cache_key = build_cache_key(
        article_input_hash=prepared_article.article_input_hash,
        request_mode=request_mode,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_version=prompt_version,
        schema_version=SCHEMA_VERSION,
    )
    cached_row = cache.get(cache_key)
    if cached_row is not None:
        article_run = build_article_run_trace(
            prepared_article,
            [],
            request_mode=request_mode,
            llm_model=llm_model,
            recovery_reason=recovery_reason,
            tool_urls=url_context_urls,
            cache_hit=True,
        )
        article_run["final_status"] = "success"
        return (
            clone_cached_success_row(cached_row, request_mode=request_mode),
            None,
            article_run,
            0,
            0,
            True,
        )

    if client is None:
        raise LlmExtractionError("Stage 6 extraction requires an active client for uncached requests.")

    success_row, failure, article_run, provider_request_count, retry_count = extract_one_article(
        prepared_article=prepared_article,
        company_name=company_name,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_version=prompt_version,
        request_mode=request_mode,
        recovery_reason=recovery_reason,
        url_context_urls=url_context_urls,
        retry_attempts=retry_attempts,
        retry_base_delay_seconds=retry_base_delay_seconds,
        client=client,
    )
    article_run["final_status"] = "success" if success_row is not None else "failure"
    if failure is not None:
        article_run["failure_category"] = failure["failure_category"]
    return (
        success_row,
        failure,
        article_run,
        provider_request_count,
        retry_count,
        False,
    )


def extract_one_article(
    *,
    prepared_article: PreparedArticleInput,
    company_name: str,
    llm_provider: str,
    llm_model: str,
    prompt_version: str,
    request_mode: str,
    recovery_reason: str | None,
    url_context_urls: tuple[str, ...],
    retry_attempts: int,
    retry_base_delay_seconds: float,
    client: StructuredNewsExtractionClient,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any], int, int]:
    """Extract one article with retries and rich traceability."""

    attempt_logs: list[dict[str, Any]] = []
    provider_request_count = 0
    retry_count = 0
    retry_reason: str | None = None

    for attempt_number in range(1, retry_attempts + 1):
        prompt = build_extraction_prompt(
            prepared_article,
            company_name=company_name,
            prompt_version=prompt_version,
            request_mode=request_mode,
            recovery_reason=recovery_reason,
            url_context_urls=url_context_urls,
            retry_reason=retry_reason,
        )
        provider_request_count += 1

        try:
            response = client.generate(
                prompt,
                options=ExtractionRequestOptions(
                    model=llm_model,
                    request_mode=request_mode,
                    url_context_urls=url_context_urls,
                    enable_google_search=request_mode == REQUEST_MODE_GOOGLE_SEARCH,
                ),
            )
        except requests.RequestException as exc:
            error_message = sanitize_log_text(f"{type(exc).__name__}: {exc}")
            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_char_count": len(prompt),
                    "outcome": "retryable_network_error",
                    "error_message": error_message,
                }
            )
            if attempt_number < retry_attempts:
                retry_count += 1
                retry_reason = error_message
                attempt_backoff(retry_base_delay_seconds * attempt_number)
                continue
            failure = build_failure_entry(
                prepared_article=prepared_article,
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_version=prompt_version,
                failure_category="retryable_network_error",
                error_message=error_message,
                attempt_logs=attempt_logs,
            )
            return (
                None,
                failure,
                build_article_run_trace(
                    prepared_article,
                    attempt_logs,
                    request_mode=request_mode,
                    llm_model=llm_model,
                    recovery_reason=recovery_reason,
                    tool_urls=url_context_urls,
                ),
                provider_request_count,
                retry_count,
            )
        except RetryableProviderError as exc:
            sanitized_error_message = sanitize_log_text(str(exc))
            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_char_count": len(prompt),
                    "outcome": "retryable_provider_error",
                    "error_message": sanitized_error_message,
                    "provider_status_code": exc.status_code,
                    "raw_response_text": exc.raw_payload,
                }
            )
            if attempt_number < retry_attempts:
                retry_count += 1
                retry_reason = sanitized_error_message
                attempt_backoff(retry_base_delay_seconds * attempt_number)
                continue
            failure = build_failure_entry(
                prepared_article=prepared_article,
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_version=prompt_version,
                failure_category="retryable_provider_error",
                error_message=sanitized_error_message,
                attempt_logs=attempt_logs,
                raw_response_text=exc.raw_payload,
                provider_status_code=exc.status_code,
            )
            return (
                None,
                failure,
                build_article_run_trace(
                    prepared_article,
                    attempt_logs,
                    request_mode=request_mode,
                    llm_model=llm_model,
                    recovery_reason=recovery_reason,
                    tool_urls=url_context_urls,
                ),
                provider_request_count,
                retry_count,
            )
        except NonRetryableProviderError as exc:
            sanitized_error_message = sanitize_log_text(str(exc))
            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_char_count": len(prompt),
                    "outcome": "non_retryable_provider_error",
                    "error_message": sanitized_error_message,
                    "provider_status_code": exc.status_code,
                    "raw_response_text": exc.raw_payload,
                }
            )
            failure = build_failure_entry(
                prepared_article=prepared_article,
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_version=prompt_version,
                failure_category="non_retryable_provider_error",
                error_message=sanitized_error_message,
                attempt_logs=attempt_logs,
                raw_response_text=exc.raw_payload,
                provider_status_code=exc.status_code,
            )
            return (
                None,
                failure,
                build_article_run_trace(
                    prepared_article,
                    attempt_logs,
                    request_mode=request_mode,
                    llm_model=llm_model,
                    recovery_reason=recovery_reason,
                    tool_urls=url_context_urls,
                ),
                provider_request_count,
                retry_count,
            )

        try:
            parsed_payload = parse_first_json_object(response.response_text)
        except LlmExtractionError as exc:
            sanitized_error_message = sanitize_log_text(str(exc))
            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_char_count": len(prompt),
                    "outcome": "malformed_model_output",
                    "error_message": sanitized_error_message,
                    "provider_status_code": response.status_code,
                    "model_finish_reason": response.finish_reason,
                    "raw_response_text": response.response_text,
                    "retrieved_urls": list(response.retrieved_urls),
                    "citations": response.citations,
                }
            )
            if attempt_number < retry_attempts:
                retry_count += 1
                retry_reason = sanitized_error_message
                attempt_backoff(retry_base_delay_seconds * attempt_number)
                continue
            failure = build_failure_entry(
                prepared_article=prepared_article,
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_version=prompt_version,
                failure_category="malformed_model_output",
                error_message=sanitized_error_message,
                attempt_logs=attempt_logs,
                raw_response_text=response.response_text,
                provider_status_code=response.status_code,
            )
            return (
                None,
                failure,
                build_article_run_trace(
                    prepared_article,
                    attempt_logs,
                    request_mode=request_mode,
                    llm_model=llm_model,
                    recovery_reason=recovery_reason,
                    tool_urls=url_context_urls,
                ),
                provider_request_count,
                retry_count,
            )

        try:
            success_row = validate_extraction_payload(
                parsed_payload,
                prepared_article=prepared_article,
                company_name=company_name,
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_version=prompt_version,
                request_mode=request_mode,
            )
        except SchemaValidationError as exc:
            sanitized_error_message = sanitize_log_text(str(exc))
            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_char_count": len(prompt),
                    "outcome": "schema_validation_error",
                    "error_message": sanitized_error_message,
                    "schema_errors": exc.errors,
                    "provider_status_code": response.status_code,
                    "model_finish_reason": response.finish_reason,
                    "raw_response_text": response.response_text,
                    "retrieved_urls": list(response.retrieved_urls),
                    "citations": response.citations,
                }
            )
            if attempt_number < retry_attempts:
                retry_count += 1
                retry_reason = sanitize_log_text("; ".join(exc.errors))
                attempt_backoff(retry_base_delay_seconds * attempt_number)
                continue
            failure = build_failure_entry(
                prepared_article=prepared_article,
                llm_provider=llm_provider,
                llm_model=llm_model,
                prompt_version=prompt_version,
                failure_category="schema_validation_error",
                error_message=sanitized_error_message,
                attempt_logs=attempt_logs,
                schema_errors=exc.errors,
                raw_response_text=response.response_text,
                provider_status_code=response.status_code,
            )
            return (
                None,
                failure,
                build_article_run_trace(
                    prepared_article,
                    attempt_logs,
                    request_mode=request_mode,
                    llm_model=llm_model,
                    recovery_reason=recovery_reason,
                    tool_urls=url_context_urls,
                ),
                provider_request_count,
                retry_count,
            )

        attempt_logs.append(
            {
                "attempt_number": attempt_number,
                "prompt_char_count": len(prompt),
                "outcome": "success",
                "provider_status_code": response.status_code,
                "model_finish_reason": response.finish_reason,
                "raw_response_text": response.response_text,
                "retrieved_urls": list(response.retrieved_urls),
                "citations": response.citations,
            }
        )
        return (
            success_row,
            None,
            build_article_run_trace(
                prepared_article,
                attempt_logs,
                request_mode=request_mode,
                llm_model=llm_model,
                recovery_reason=recovery_reason,
                tool_urls=url_context_urls,
            ),
            provider_request_count,
            retry_count,
        )

    raise AssertionError("Expected the extraction loop to return before exhaustion.")


def build_failure_log_payload(
    *,
    settings: AppSettings,
    source_news_path: Path,
    failures: list[dict[str, Any]],
    generated_at_utc: datetime,
) -> dict[str, Any]:
    """Build the persisted Stage 6 failure log payload."""

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "source_news_path": str(source_news_path),
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "failure_count": len(failures),
        "failures": failures,
    }


def build_raw_snapshot_payload(
    *,
    settings: AppSettings,
    source_news_path: Path,
    source_news_hash: str,
    source_news_metadata_path: Path | None,
    source_news_metadata_hash: str | None,
    run_id: str,
    generated_at_utc: datetime,
    llm_provider: str,
    llm_model: str,
    prompt_version: str,
    article_runs: list[dict[str, Any]],
    cache_hit_count: int,
    fresh_call_count: int,
    provider_request_count: int,
    retry_count: int,
    timing: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    """Build the raw Stage 6 run snapshot payload."""

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "company_name": settings.ticker.company_name,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "source_news_path": str(source_news_path),
        "source_news_hash": source_news_hash,
        "source_news_metadata_path": str(source_news_metadata_path) if source_news_metadata_path else None,
        "source_news_metadata_hash": source_news_metadata_hash,
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "run_id": run_id,
        "cache_hit_count": cache_hit_count,
        "fresh_call_count": fresh_call_count,
        "provider_request_count": provider_request_count,
        "retry_count": retry_count,
        "timing": timing,
        "workload": workload,
        "article_runs": article_runs,
    }


def build_extraction_metadata(
    *,
    settings: AppSettings,
    extraction_table_path: Path,
    failure_log_path: Path,
    raw_snapshot_path: Path,
    source_news_path: Path,
    source_news_hash: str,
    source_news_metadata_path: Path | None,
    source_news_metadata_hash: str | None,
    extracted_frame: pd.DataFrame,
    source_row_count: int,
    cache_hit_count: int,
    fresh_call_count: int,
    provider_request_count: int,
    retry_count: int,
    failure_count: int,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the persisted Stage 6 metadata payload."""

    coverage_start = None
    coverage_end = None
    warnings: list[str] = []
    if not extracted_frame.empty:
        coverage_start = str(extracted_frame["published_date_ist"].min())
        coverage_end = str(extracted_frame["published_date_ist"].max())
    else:
        warnings.append("no_successful_extractions")
    if source_row_count == 0:
        warnings.append("no_source_articles")
    if failure_count > 0:
        warnings.append("extraction_failures_present")
    if (
        not extracted_frame.empty
        and "recovery_status" in extracted_frame.columns
        and (extracted_frame["recovery_status"] == "exhausted").any()
    ):
        warnings.append("recovery_exhausted_present")
    if (
        not extracted_frame.empty
        and "recovery_status" in extracted_frame.columns
        and (extracted_frame["recovery_status"] == "skipped").any()
    ):
        warnings.append("recovery_skipped_present")

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "company_name": settings.ticker.company_name,
        "llm_provider": settings.providers.llm_provider,
        "llm_model": settings.llm_extraction.model,
        "prompt_version": settings.llm_extraction.prompt_version,
        "schema_version": SCHEMA_VERSION,
        "extraction_table_path": str(extraction_table_path),
        "extraction_table_hash": compute_file_sha256(extraction_table_path),
        "failure_log_path": str(failure_log_path),
        "failure_log_hash": compute_file_sha256(failure_log_path),
        "raw_snapshot_path": str(raw_snapshot_path),
        "raw_snapshot_hash": compute_file_sha256(raw_snapshot_path),
        "source_news_path": str(source_news_path),
        "source_news_hash": source_news_hash,
        "source_news_metadata_path": str(source_news_metadata_path) if source_news_metadata_path else None,
        "source_news_metadata_hash": source_news_metadata_hash,
        "source_row_count": source_row_count,
        "success_count": int(len(extracted_frame)),
        "failure_count": int(failure_count),
        "cache_hit_count": int(cache_hit_count),
        "fresh_call_count": int(fresh_call_count),
        "provider_request_count": int(provider_request_count),
        "retry_count": int(retry_count),
        "timing": build_stage_timing_payload(
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            elapsed_seconds=elapsed_seconds,
        ),
        "workload": build_stage6_workload_payload(
            source_row_count=source_row_count,
            success_count=int(len(extracted_frame)),
            failure_count=int(failure_count),
        ),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "extraction_mode_counts": count_series_values(extracted_frame, "extraction_mode"),
        "event_type_counts": count_series_values(extracted_frame, "event_type"),
        "sentiment_label_counts": count_series_values(extracted_frame, "sentiment_label"),
        "warning_flag_counts": count_series_values(extracted_frame, "warning_flag"),
        "llm_model_counts": count_series_values(extracted_frame, "llm_model"),
        "request_mode_counts": count_series_values(extracted_frame, "request_mode"),
        "recovery_status_counts": count_series_values(extracted_frame, "recovery_status"),
        "warnings": warnings,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def build_stage_timing_payload(
    *,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build one shared timing payload for Stage 6 artifacts."""

    return {
        "started_at_utc": started_at_utc.astimezone(timezone.utc).isoformat(),
        "finished_at_utc": finished_at_utc.astimezone(timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
    }


def build_stage6_workload_payload(
    *,
    source_row_count: int,
    success_count: int,
    failure_count: int,
) -> dict[str, int]:
    """Build the shared Stage 6 workload summary."""

    return {
        "source_row_count": int(source_row_count),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
    }


def process_source_row_extraction(
    *,
    source_row: Mapping[str, Any],
    runtime_settings: AppSettings,
    llm_provider: str,
    llm_model: str,
    prompt_version: str,
    effective_max_input_chars: int,
    client: StructuredNewsExtractionClient | None,
    cache: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    recovery_budget_tracker: RecoveryBudgetTracker,
    recovery_budget_lock: threading.Lock,
) -> SourceExtractionOutcome:
    """Process one Stage 5 row with optional recovery and return counters."""

    local_client = client
    local_provider = llm_provider
    local_cache = dict(cache)
    cache_hit_count = 0
    fresh_call_count = 0
    provider_request_count = 0
    retry_count = 0

    try:
        prepared_article = prepare_article_input(
            source_row,
            company_name=runtime_settings.ticker.company_name,
            max_input_chars=effective_max_input_chars,
        )
    except LlmExtractionError as exc:
        fallback_mode = normalize_enum(source_row.get("text_acquisition_mode"))
        prepared_article = PreparedArticleInput(
            article_id=sanitize_prompt_text(source_row.get("article_id")) or "unknown_article",
            source_row={
                key: source_row.get(key)
                for key in SOURCE_NEWS_REQUIRED_COLUMNS
            },
            prompt_article_text="",
            prompt_truncated=False,
            extraction_mode=fallback_mode if fallback_mode in CONTENT_QUALITY_BY_MODE else "headline_only",
            content_quality_score=CONTENT_QUALITY_BY_MODE.get(fallback_mode, 0.5),
            warning_flag=True,
            article_input_hash="",
        )
        failure = build_failure_entry(
            prepared_article=prepared_article,
            llm_provider=local_provider,
            llm_model=llm_model,
            prompt_version=prompt_version,
            failure_category="invalid_source_row",
            error_message=str(exc),
            attempt_logs=[],
        )
        return SourceExtractionOutcome(
            success_row=None,
            failure=failure,
            article_run={
                "article_id": prepared_article.article_id,
                "article_input_hash": prepared_article.article_input_hash,
                "extraction_mode": prepared_article.extraction_mode,
                "recovery_reason": determine_recovery_reason(prepared_article),
                "recovery_status": "skipped",
                "final_status": "failure",
                "failure_category": "invalid_source_row",
                "request_runs": [],
            },
            cache_hit_count=cache_hit_count,
            fresh_call_count=fresh_call_count,
            provider_request_count=provider_request_count,
            retry_count=retry_count,
            llm_provider=local_provider,
        )

    request_runs: list[dict[str, Any]] = []
    recovery_reason = determine_recovery_reason(prepared_article)
    recovery_status = "not_needed"

    try:
        plain_text_success, plain_text_failure, plain_text_run, plain_text_requests, plain_text_retries, plain_text_cache_hit = execute_extraction_request(
            prepared_article=prepared_article,
            company_name=runtime_settings.ticker.company_name,
            llm_provider=local_provider or (client.provider_name if client is not None else ""),
            llm_model=llm_model,
            prompt_version=prompt_version,
            request_mode=REQUEST_MODE_PLAIN_TEXT,
            recovery_reason=None,
            url_context_urls=(),
            retry_attempts=runtime_settings.llm_extraction.retry_attempts,
            retry_base_delay_seconds=runtime_settings.llm_extraction.retry_base_delay_seconds,
            client=local_client,
            cache=local_cache,
        )
    except LlmExtractionError:
        if local_client is None:
            local_client = resolve_extraction_client(runtime_settings, client=client)
            local_provider = local_client.provider_name
        plain_text_success, plain_text_failure, plain_text_run, plain_text_requests, plain_text_retries, plain_text_cache_hit = execute_extraction_request(
            prepared_article=prepared_article,
            company_name=runtime_settings.ticker.company_name,
            llm_provider=local_provider,
            llm_model=llm_model,
            prompt_version=prompt_version,
            request_mode=REQUEST_MODE_PLAIN_TEXT,
            recovery_reason=None,
            url_context_urls=(),
            retry_attempts=runtime_settings.llm_extraction.retry_attempts,
            retry_base_delay_seconds=runtime_settings.llm_extraction.retry_base_delay_seconds,
            client=local_client,
            cache=local_cache,
        )
    request_runs.append(plain_text_run)
    if plain_text_cache_hit:
        cache_hit_count += 1
    else:
        fresh_call_count += 1
    provider_request_count += plain_text_requests
    retry_count += plain_text_retries

    final_success_row = plain_text_success
    final_failure = plain_text_failure

    if is_weak_source_article(prepared_article):
        with recovery_budget_lock:
            recovery_options, blocked_status = build_recovery_request_options(
                settings=runtime_settings,
                prepared_article=prepared_article,
                budget_tracker=recovery_budget_tracker,
            )
            if recovery_options:
                recovery_budget_tracker.mark_article_attempt()
        if not recovery_options:
            recovery_status = blocked_status or "skipped"
        else:
            recovery_status = "exhausted"
            for recovery_option in recovery_options:
                try:
                    recovery_success, recovery_failure, recovery_run, recovery_requests, recovery_retries, recovery_cache_hit = execute_extraction_request(
                        prepared_article=prepared_article,
                        company_name=runtime_settings.ticker.company_name,
                        llm_provider=local_provider or (client.provider_name if client is not None else ""),
                        llm_model=recovery_option.model,
                        prompt_version=prompt_version,
                        request_mode=recovery_option.request_mode,
                        recovery_reason=recovery_reason,
                        url_context_urls=recovery_option.url_context_urls,
                        retry_attempts=runtime_settings.llm_extraction.retry_attempts,
                        retry_base_delay_seconds=runtime_settings.llm_extraction.retry_base_delay_seconds,
                        client=local_client,
                        cache=local_cache,
                    )
                except LlmExtractionError:
                    if local_client is None:
                        local_client = resolve_extraction_client(runtime_settings, client=client)
                        local_provider = local_client.provider_name
                    recovery_success, recovery_failure, recovery_run, recovery_requests, recovery_retries, recovery_cache_hit = execute_extraction_request(
                        prepared_article=prepared_article,
                        company_name=runtime_settings.ticker.company_name,
                        llm_provider=local_provider,
                        llm_model=recovery_option.model,
                        prompt_version=prompt_version,
                        request_mode=recovery_option.request_mode,
                        recovery_reason=recovery_reason,
                        url_context_urls=recovery_option.url_context_urls,
                        retry_attempts=runtime_settings.llm_extraction.retry_attempts,
                        retry_base_delay_seconds=runtime_settings.llm_extraction.retry_base_delay_seconds,
                        client=local_client,
                        cache=local_cache,
                    )
                request_runs.append(recovery_run)
                if recovery_cache_hit:
                    cache_hit_count += 1
                else:
                    fresh_call_count += 1
                    with recovery_budget_lock:
                        recovery_budget_tracker.note_model_requests(
                            recovery_option.model,
                            recovery_requests,
                        )
                provider_request_count += recovery_requests
                retry_count += recovery_retries
                if recovery_success is not None:
                    final_success_row = recovery_success
                    final_failure = None
                    recovery_status = "succeeded"
                    break
                if recovery_failure is not None and final_success_row is None:
                    final_failure = recovery_failure

    success_row: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    if final_success_row is not None:
        success_row = apply_recovery_metadata(
            final_success_row,
            recovery_reason=recovery_reason,
            recovery_status=recovery_status,
        )
    elif final_failure is not None:
        failure = final_failure

    final_request_mode = sanitize_prompt_text(
        final_success_row.get("request_mode") if final_success_row is not None else None
    ) or REQUEST_MODE_PLAIN_TEXT
    selected_model = sanitize_prompt_text(
        final_success_row.get("llm_model") if final_success_row is not None else None
    ) or sanitize_prompt_text(request_runs[-1].get("llm_model") if request_runs else llm_model)
    article_run = {
        "article_id": prepared_article.article_id,
        "article_input_hash": prepared_article.article_input_hash,
        "extraction_mode": prepared_article.extraction_mode,
        "recovery_reason": recovery_reason,
        "recovery_status": recovery_status,
        "final_status": "success" if final_success_row is not None else "failure",
        "final_request_mode": final_request_mode,
        "selected_model": selected_model,
        "failure_category": final_failure["failure_category"] if final_failure is not None else None,
        "request_runs": request_runs,
    }
    return SourceExtractionOutcome(
        success_row=success_row,
        failure=failure,
        article_run=article_run,
        cache_hit_count=cache_hit_count,
        fresh_call_count=fresh_call_count,
        provider_request_count=provider_request_count,
        retry_count=retry_count,
        llm_provider=local_provider,
    )


def extract_news(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    news_table_path: str | Path | None = None,
    force: bool = False,
    client: StructuredNewsExtractionClient | None = None,
    pilot_extraction: bool = False,
    max_input_chars_override: int | None = None,
) -> LlmExtractionRunResult:
    """Run the Stage 6 article-level extraction pipeline."""

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

    source_news_path = resolve_news_table_path(
        runtime_settings,
        path_manager=path_manager,
        news_table_path=news_table_path,
    )
    source_news_metadata_path = infer_news_metadata_path(source_news_path)
    source_frame = validate_processed_news_frame(
        read_processed_news(source_news_path),
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
    )
    source_news_hash = compute_file_sha256(source_news_path)
    source_news_metadata_hash = (
        compute_file_sha256(source_news_metadata_path)
        if source_news_metadata_path is not None
        else None
    )

    extraction_table_path = path_manager.build_processed_llm_extractions_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    metadata_path = path_manager.build_processed_llm_extractions_metadata_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    failure_log_path = path_manager.build_processed_llm_extraction_failures_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    raw_snapshot_path = path_manager.build_raw_llm_data_path(
        runtime_settings.ticker.symbol,
        run_context.run_id,
    )

    llm_model = sanitize_prompt_text(runtime_settings.llm_extraction.model)
    prompt_version = sanitize_prompt_text(runtime_settings.llm_extraction.prompt_version)
    llm_provider = sanitize_prompt_text(runtime_settings.providers.llm_provider) or (
        client.provider_name if client is not None else ""
    )
    cache = load_extraction_cache(extraction_table_path, force=force)
    active_client = client

    success_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    article_runs: list[dict[str, Any]] = []
    cache_hit_count = 0
    fresh_call_count = 0
    provider_request_count = 0
    retry_count = 0
    recovery_budget_tracker = RecoveryBudgetTracker(
        max_articles_per_run=runtime_settings.llm_extraction.recovery_max_articles_per_run
    )

    if max_input_chars_override is not None:
        effective_max_input_chars = max_input_chars_override
    elif (
        pilot_extraction
        and runtime_settings.llm_extraction.max_input_chars_pilot is not None
    ):
        effective_max_input_chars = runtime_settings.llm_extraction.max_input_chars_pilot
    else:
        effective_max_input_chars = runtime_settings.llm_extraction.max_input_chars

    source_rows = source_frame.to_dict(orient="records")
    recovery_budget_lock = threading.Lock()
    extraction_workers = max(1, int(runtime_settings.llm_extraction.extraction_workers))
    if extraction_workers == 1 or len(source_rows) <= 1:
        outcomes: list[SourceExtractionOutcome] = []
        for source_row in source_rows:
            outcomes.append(
                process_source_row_extraction(
                    source_row=source_row,
                    runtime_settings=runtime_settings,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    prompt_version=prompt_version,
                    effective_max_input_chars=effective_max_input_chars,
                    client=active_client,
                    cache=cache,
                    recovery_budget_tracker=recovery_budget_tracker,
                    recovery_budget_lock=recovery_budget_lock,
                )
            )
    else:
        max_workers = min(extraction_workers, len(source_rows))
        indexed_outcomes: dict[int, SourceExtractionOutcome] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_source_row_extraction,
                    source_row=source_row,
                    runtime_settings=runtime_settings,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    prompt_version=prompt_version,
                    effective_max_input_chars=effective_max_input_chars,
                    client=client,
                    cache=cache,
                    recovery_budget_tracker=recovery_budget_tracker,
                    recovery_budget_lock=recovery_budget_lock,
                ): index
                for index, source_row in enumerate(source_rows)
            }
            for future in as_completed(futures):
                index = futures[future]
                indexed_outcomes[index] = future.result()
        outcomes = [indexed_outcomes[index] for index in sorted(indexed_outcomes.keys())]

    for outcome in outcomes:
        if outcome.success_row is not None:
            success_rows.append(outcome.success_row)
        if outcome.failure is not None:
            failures.append(outcome.failure)
        article_runs.append(outcome.article_run)
        cache_hit_count += int(outcome.cache_hit_count)
        fresh_call_count += int(outcome.fresh_call_count)
        provider_request_count += int(outcome.provider_request_count)
        retry_count += int(outcome.retry_count)
        if outcome.llm_provider:
            llm_provider = outcome.llm_provider

    extracted_frame = pd.DataFrame(success_rows, columns=EXTRACTED_NEWS_COLUMNS)
    extraction_table_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_frame.to_csv(extraction_table_path, index=False)

    failure_payload = build_failure_log_payload(
        settings=runtime_settings,
        source_news_path=source_news_path,
        failures=failures,
        generated_at_utc=run_context.started_at_utc,
    )
    write_json_file(failure_log_path, failure_payload)

    raw_snapshot_payload = build_raw_snapshot_payload(
        settings=runtime_settings,
        source_news_path=source_news_path,
        source_news_hash=source_news_hash,
        source_news_metadata_path=source_news_metadata_path,
        source_news_metadata_hash=source_news_metadata_hash,
        run_id=run_context.run_id,
        generated_at_utc=run_context.started_at_utc,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_version=prompt_version,
        article_runs=article_runs,
        cache_hit_count=cache_hit_count,
        fresh_call_count=fresh_call_count,
        provider_request_count=provider_request_count,
        retry_count=retry_count,
        timing=build_stage_timing_payload(
            started_at_utc=run_context.started_at_utc,
            finished_at_utc=datetime.now(timezone.utc),
            elapsed_seconds=round(time.perf_counter() - stage_start, 6),
        ),
        workload=build_stage6_workload_payload(
            source_row_count=len(source_frame),
            success_count=len(extracted_frame),
            failure_count=len(failures),
        ),
    )
    write_json_file(raw_snapshot_path, raw_snapshot_payload)

    elapsed_seconds = round(time.perf_counter() - stage_start, 6)
    finished_at_utc = datetime.now(timezone.utc)
    metadata = build_extraction_metadata(
        settings=runtime_settings,
        extraction_table_path=extraction_table_path,
        failure_log_path=failure_log_path,
        raw_snapshot_path=raw_snapshot_path,
        source_news_path=source_news_path,
        source_news_hash=source_news_hash,
        source_news_metadata_path=source_news_metadata_path,
        source_news_metadata_hash=source_news_metadata_hash,
        extracted_frame=extracted_frame,
        source_row_count=len(source_frame),
        cache_hit_count=cache_hit_count,
        fresh_call_count=fresh_call_count,
        provider_request_count=provider_request_count,
        retry_count=retry_count,
        failure_count=len(failures),
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
        started_at_utc=run_context.started_at_utc,
        finished_at_utc=finished_at_utc,
        elapsed_seconds=elapsed_seconds,
    )
    write_json_file(metadata_path, metadata)

    logger.info(
        "LLM extraction ready | ticker=%s | exchange=%s | provider=%s | model=%s | source_rows=%s | successes=%s | failures=%s | cache_hits=%s | fresh_calls=%s | elapsed=%.3fs | extraction_csv=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        llm_provider or runtime_settings.providers.llm_provider,
        llm_model,
        len(source_frame),
        len(extracted_frame),
        len(failures),
        cache_hit_count,
        fresh_call_count,
        elapsed_seconds,
        extraction_table_path,
    )

    return LlmExtractionRunResult(
        extraction_table_path=extraction_table_path,
        metadata_path=metadata_path,
        failure_log_path=failure_log_path,
        raw_snapshot_path=raw_snapshot_path,
        source_row_count=len(source_frame),
        success_count=len(extracted_frame),
        failure_count=len(failures),
        cache_hit_count=cache_hit_count,
        fresh_call_count=fresh_call_count,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 6 extraction command arguments."""

    parser = argparse.ArgumentParser(description="Run Kubera Stage 6 LLM extraction.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--news-path",
        help="Use a specific processed news CSV file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the success cache and recompute all article extractions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 6 LLM extraction command."""

    args = parse_args(argv)
    settings = load_settings()
    extract_news(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        news_table_path=args.news_path,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
