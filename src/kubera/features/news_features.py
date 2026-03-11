"""Stage 7 news feature engineering for Kubera."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from kubera.config import NewsFeatureSettings
from kubera.llm.extract_news import ALLOWED_EVENT_TYPES


FEATURE_FORMULA_VERSION = "1"
OUTPUT_IDENTITY_COLUMNS = ("date", "ticker", "exchange", "prediction_mode")
RAW_FEATURE_COLUMNS = (
    "news_article_count",
    "news_avg_sentiment",
    "news_max_severity",
    "news_avg_relevance",
    "news_avg_confidence",
    "news_bullish_article_count",
    "news_bearish_article_count",
    "news_neutral_article_count",
    "news_full_article_count",
    "news_headline_plus_snippet_count",
    "news_headline_only_count",
    "news_warning_article_count",
    "news_fallback_article_ratio",
    "news_avg_content_quality_score",
)
WEIGHTED_FEATURE_COLUMNS = (
    "news_weighted_sentiment_score",
    "news_weighted_relevance_score",
    "news_weighted_confidence_score",
    "news_weighted_bullish_score",
    "news_weighted_bearish_score",
)
EVENT_COUNT_COLUMNS = tuple(
    f"news_event_count_{event_type}" for event_type in sorted(ALLOWED_EVENT_TYPES)
)
NEWS_FEATURE_COLUMNS = (
    RAW_FEATURE_COLUMNS
    + WEIGHTED_FEATURE_COLUMNS
    + EVENT_COUNT_COLUMNS
)
OUTPUT_COLUMNS = OUTPUT_IDENTITY_COLUMNS + NEWS_FEATURE_COLUMNS
SUPPORTED_NEWS_PREDICTION_MODES = ("pre_market", "after_close")


class NewsFeatureError(RuntimeError):
    """Raised when Stage 7 news feature engineering cannot continue."""


@dataclass(frozen=True)
class NewsFeatureBuildResult:
    """Persisted Stage 7 feature artifact summary."""

    feature_table_path: Path
    metadata_path: Path
    raw_snapshot_path: Path
    row_count: int
    coverage_start: date | None
    coverage_end: date | None
    cache_hit: bool


def news_feature_settings_to_dict(settings: NewsFeatureSettings) -> dict[str, Any]:
    """Serialize Stage 7 weight settings into plain values."""

    return {
        "full_article_weight": settings.full_article_weight,
        "headline_plus_snippet_weight": settings.headline_plus_snippet_weight,
        "headline_only_weight": settings.headline_only_weight,
    }


def resolve_supported_prediction_modes(raw_modes: tuple[str, ...]) -> tuple[str, ...]:
    """Expand configured prediction modes into concrete Stage 7 row modes."""

    ordered_modes: list[str] = []
    for mode in raw_modes:
        if mode == "both":
            for concrete_mode in SUPPORTED_NEWS_PREDICTION_MODES:
                if concrete_mode not in ordered_modes:
                    ordered_modes.append(concrete_mode)
            continue
        if mode in SUPPORTED_NEWS_PREDICTION_MODES and mode not in ordered_modes:
            ordered_modes.append(mode)
    return tuple(ordered_modes)
