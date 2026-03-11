"""Stage 6 LLM extraction helpers for Kubera."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA_VERSION = "1"
ARTICLE_TEXT_START_MARKER = "<article_text>"
ARTICLE_TEXT_END_MARKER = "</article_text>"
PROMPT_JSON_FIELDS = (
    "ticker",
    "company_name",
    "article_title",
    "published_at",
    "source",
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
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ENUM_NORMALIZER = re.compile(r"[^a-z0-9]+")


class LlmExtractionError(RuntimeError):
    """Raised when Stage 6 extraction cannot continue."""


class SchemaValidationError(LlmExtractionError):
    """Raised when a model payload fails schema validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


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


def sanitize_prompt_text(value: Any) -> str:
    """Remove control characters and collapse internal whitespace."""

    if value is None:
        return ""
    cleaned = _CONTROL_CHARACTERS.sub(" ", str(value))
    return " ".join(cleaned.split())


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
    prompt_text = sanitize_prompt_text(normalized_row.get("full_text"))
    if not prompt_text:
        prompt_text = build_article_fallback_text(
            sanitize_prompt_text(normalized_row.get("article_title")),
            sanitize_prompt_text(normalized_row.get("summary_snippet")),
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
        "provider_source": sanitize_prompt_text(normalized_row.get("provider_source")),
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

    return (
        "You extract structured stock-news signals for a machine learning pipeline.\n"
        "Return exactly one JSON object and no prose, markdown, or code fences.\n"
        "Do not invent missing facts. If the article does not support a stronger claim, choose neutral or other values.\n"
        "Echo the provided deterministic metadata exactly.\n"
        f"Prompt version: {sanitize_prompt_text(prompt_version)}\n"
        f"{retry_suffix}"
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
        "Deterministic metadata to echo exactly:\n"
        f"- ticker: {sanitize_prompt_text(source_row.get('ticker')).upper()}\n"
        f"- company_name: {sanitize_prompt_text(company_name)}\n"
        f"- article_title: {sanitize_prompt_text(source_row.get('article_title'))}\n"
        f"- published_at: {sanitize_prompt_text(source_row.get('published_at_utc'))}\n"
        f"- source: {sanitize_prompt_text(source_row.get('provider_source'))}\n"
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

    model_title = normalize_free_text(payload.get("article_title"))
    if model_title != normalize_free_text(source_row.get("article_title")):
        errors.append("article_title does not match the source row")

    try:
        published_at = normalize_timestamp(payload.get("published_at"))
    except ValueError:
        errors.append("published_at must be a valid timestamp")
        published_at = ""
    if published_at and published_at != sanitize_prompt_text(source_row.get("published_at_utc")):
        errors.append("published_at does not match the source row")

    model_source = normalize_free_text(payload.get("source"))
    if model_source != normalize_free_text(source_row.get("provider_source")):
        errors.append("source does not match the source row")

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
        "provider_source": sanitize_prompt_text(source_row.get("provider_source")),
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
