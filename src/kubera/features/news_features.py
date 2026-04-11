"""Stage 7 news feature engineering for Kubera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import pandas as pd

from kubera.config import (
    AppSettings,
    NewsFeatureSettings,
    load_settings,
    resolve_runtime_settings,
)
from kubera.llm.extract_news import (
    ALLOWED_DIRECTIONAL_BIAS,
    ALLOWED_EVENT_TYPES,
    ALLOWED_EXTRACTION_MODES,
    ALLOWED_SENTIMENT_LABELS,
    EXTRACTED_NEWS_COLUMNS,
    normalize_enum,
    sanitize_prompt_text,
)
from kubera.utils.calendar import (
    build_market_calendar,
    first_trading_day_after,
    first_trading_day_on_or_after,
)
from kubera.utils.hashing import compute_file_sha256 as _compute_file_sha256
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot
from kubera.utils.time_utils import is_after_close, is_intraday, is_pre_market


NEWS_CHANNELS = ("company", "sector", "macro")
SOURCE_PROVIDER_QUALITY_WEIGHTS = {
    "nse_announcements": 1.0,
    "marketaux": 1.0,
    "economic_times": 0.9,
    "alphavantage": 0.85,
    "google_news_rss": 0.75,
}
EVENT_TYPE_TO_CHANNEL = {
    "earnings": "company",
    "guidance": "company",
    "deal_win": "company",
    "product_launch": "company",
    "partnership": "company",
    "acquisition": "company",
    "lawsuit": "company",
    "management_change": "company",
    "supply_chain": "company",
    "other": "company",
    "analyst_commentary": "sector",
    "sector_development": "sector",
    "regulation": "sector",
    "market_reaction": "sector",
    "macro_spillover": "macro",
    "government_policy_impact": "macro",
    "rupee_commodity_sensitivity": "macro",
}

FEATURE_FORMULA_VERSION = "4"
OUTPUT_IDENTITY_COLUMNS = ("date", "ticker", "exchange", "prediction_mode")
NEWS_SIGNAL_STATE_COLUMN = "news_signal_state"
NEWS_SIGNAL_STATE_FRESH = "fresh_news"
NEWS_SIGNAL_STATE_CARRIED_FORWARD = "carried_forward_only"
NEWS_SIGNAL_STATE_ZERO = "zero_news"
NEWS_SIGNAL_STATE_FALLBACK_HEAVY = "fallback_heavy"
ALLOWED_NEWS_SIGNAL_STATES = (
    NEWS_SIGNAL_STATE_FRESH,
    NEWS_SIGNAL_STATE_CARRIED_FORWARD,
    NEWS_SIGNAL_STATE_ZERO,
    NEWS_SIGNAL_STATE_FALLBACK_HEAVY,
)
OUTPUT_AUXILIARY_COLUMNS = (NEWS_SIGNAL_STATE_COLUMN,)
REQUIRED_BOOLEAN_FLAGS = (
    "has_news_signal",
    "carried_forward_signal",
    "degraded_news_context",
    "has_fresh_news",
    "is_carried_forward",
    "is_fallback_heavy",
)
ROLLING_WINDOWS = (1, 3, 5)
ROLLING_FEATURE_COLUMNS = tuple(
    f"{prefix}_{w}d"
    for w in ROLLING_WINDOWS
    for prefix in ("news_sentiment", "news_event_count", "news_volume")
)
RAW_FEATURE_COLUMNS = (
    "news_article_count",
    "news_sentiment_dispersion_1d",
    "news_directional_agreement_rate",
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
    "news_avg_source_quality_score",
    "news_zero_flag",
    "news_carried_flag",
)
WEIGHTED_FEATURE_COLUMNS = (
    "news_weighted_sentiment_score",
    "news_weighted_relevance_score",
    "news_weighted_confidence_score",
    "news_weighted_source_quality_score",
    "news_weighted_bullish_score",
    "news_weighted_bearish_score",
)
CHANNEL_FEATURE_COLUMNS = tuple(
    f"news_{channel}_{metric}"
    for channel in NEWS_CHANNELS
    for metric in (
        "article_count",
        "article_share",
        "avg_sentiment",
        "weighted_sentiment_score",
        "max_severity",
    )
)
EVENT_COUNT_COLUMNS = tuple(
    f"news_event_count_{event_type}" for event_type in sorted(ALLOWED_EVENT_TYPES)
)
NEWS_FEATURE_COLUMNS = (
    RAW_FEATURE_COLUMNS
    + WEIGHTED_FEATURE_COLUMNS
    + CHANNEL_FEATURE_COLUMNS
    + EVENT_COUNT_COLUMNS
    + REQUIRED_BOOLEAN_FLAGS
    + ROLLING_FEATURE_COLUMNS
)
OUTPUT_COLUMNS = OUTPUT_IDENTITY_COLUMNS + OUTPUT_AUXILIARY_COLUMNS + NEWS_FEATURE_COLUMNS
COUNT_FEATURE_COLUMNS = (
    "news_article_count",
    "news_bullish_article_count",
    "news_bearish_article_count",
    "news_neutral_article_count",
    "news_full_article_count",
    "news_headline_plus_snippet_count",
    "news_headline_only_count",
    "news_warning_article_count",
    "news_zero_flag",
    "news_carried_flag",
    *tuple(f"news_{channel}_article_count" for channel in NEWS_CHANNELS),
    *EVENT_COUNT_COLUMNS,
    *REQUIRED_BOOLEAN_FLAGS,
)
FLOAT_FEATURE_COLUMNS = tuple(
    column for column in NEWS_FEATURE_COLUMNS if column not in COUNT_FEATURE_COLUMNS
)
SUPPORTED_NEWS_PREDICTION_MODES = ("pre_market", "after_close")
SOURCE_EXTRACTION_REQUIRED_COLUMNS = EXTRACTED_NEWS_COLUMNS
OPTIONAL_EXTRACTION_COLUMNS_WITH_DEFAULTS = {
    "request_mode": "plain_text",
    "recovery_reason": pd.NA,
    "recovery_status": "not_needed",
}
SOURCE_NUMERIC_RANGES = {
    "content_quality_score": (0.0, 1.0),
    "relevance_score": (0.0, 1.0),
    "sentiment_score": (-1.0, 1.0),
    "event_severity": (0.0, 1.0),
    "confidence_score": (0.0, 1.0),
}
FEATURE_NUMERIC_RANGES = {
    "news_avg_sentiment": (-1.0, 1.0),
    "news_max_severity": (0.0, 1.0),
    "news_avg_relevance": (0.0, 1.0),
    "news_avg_confidence": (0.0, 1.0),
    "news_fallback_article_ratio": (0.0, 1.0),
    "news_avg_content_quality_score": (0.0, 1.0),
    "news_avg_source_quality_score": (0.0, 1.0),
    "news_weighted_sentiment_score": (-1.0, 1.0),
    "news_weighted_relevance_score": (0.0, 1.0),
    "news_weighted_confidence_score": (0.0, 1.0),
    "news_weighted_source_quality_score": (0.0, 1.0),
    "news_weighted_bullish_score": (0.0, 1.0),
    "news_weighted_bearish_score": (0.0, 1.0),
}
for w in ROLLING_WINDOWS:
    FEATURE_NUMERIC_RANGES[f"news_sentiment_{w}d"] = (-1.0, 1.0)
    FEATURE_NUMERIC_RANGES[f"news_event_count_{w}d"] = (0.0, 5000.0)
    FEATURE_NUMERIC_RANGES[f"news_volume_{w}d"] = (0.0, 5000.0)
FEATURE_NUMERIC_RANGES["news_sentiment_dispersion_1d"] = (0.0, 2.0)
FEATURE_NUMERIC_RANGES["news_directional_agreement_rate"] = (0.0, 1.0)

for channel in NEWS_CHANNELS:
    FEATURE_NUMERIC_RANGES[f"news_{channel}_article_share"] = (0.0, 1.0)
    FEATURE_NUMERIC_RANGES[f"news_{channel}_avg_sentiment"] = (-1.0, 1.0)
    FEATURE_NUMERIC_RANGES[f"news_{channel}_weighted_sentiment_score"] = (-1.0, 1.0)
    FEATURE_NUMERIC_RANGES[f"news_{channel}_max_severity"] = (0.0, 1.0)

PREDICTION_MODE_ORDER = {
    prediction_mode: index
    for index, prediction_mode in enumerate(SUPPORTED_NEWS_PREDICTION_MODES)
}


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
        "use_confidence_in_article_weight": settings.use_confidence_in_article_weight,
        "carry_forward_days": settings.carry_forward_days,
        "carry_decay_factor": settings.carry_decay_factor,
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


def determine_news_signal_state(row: Mapping[str, Any]) -> str:
    """Collapse Stage 7 booleans into one stable primary news-state label."""

    def _coerce_flag(field_name: str) -> int:
        raw_value = row.get(field_name, 0)
        if raw_value is None or pd.isna(raw_value):
            return 0
        try:
            return int(float(raw_value))
        except (TypeError, ValueError):
            return 0

    if _coerce_flag("is_fallback_heavy") == 1 or _coerce_flag("degraded_news_context") == 1:
        return NEWS_SIGNAL_STATE_FALLBACK_HEAVY
    if _coerce_flag("has_fresh_news") == 1:
        return NEWS_SIGNAL_STATE_FRESH
    if _coerce_flag("is_carried_forward") == 1:
        return NEWS_SIGNAL_STATE_CARRIED_FORWARD
    if _coerce_flag("news_article_count") > 0:
        return NEWS_SIGNAL_STATE_FRESH
    return NEWS_SIGNAL_STATE_ZERO


def build_news_features(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    extraction_table_path: str | Path | None = None,
    force: bool = False,
    artifact_variant: str | None = None,
    news_feature_settings: NewsFeatureSettings | None = None,
    target_end_date: date | None = None,
) -> NewsFeatureBuildResult:
    """Build, validate, and persist the Stage 7 news feature table."""

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

    source_extractions_path = resolve_extraction_table_path(
        runtime_settings,
        path_manager=path_manager,
        extraction_table_path=extraction_table_path,
    )
    if not source_extractions_path.exists():
        raise NewsFeatureError(
            f"Processed LLM extraction file does not exist: {source_extractions_path}"
        )

    supported_prediction_modes = resolve_supported_prediction_modes(
        runtime_settings.market.supported_prediction_modes
    )
    if not supported_prediction_modes:
        raise NewsFeatureError("Stage 7 requires at least one concrete prediction mode.")

    effective_news_feature_settings = (
        news_feature_settings
        if news_feature_settings is not None
        else runtime_settings.news_features
    )
    effective_runtime_settings = replace(
        runtime_settings,
        news_features=effective_news_feature_settings,
    )

    source_extractions_metadata_path = infer_extraction_metadata_path(source_extractions_path)
    source_extractions_hash = compute_file_sha256(source_extractions_path)
    source_extractions_metadata_hash = (
        compute_file_sha256(source_extractions_metadata_path)
        if source_extractions_metadata_path is not None
        else None
    )

    feature_table_path = path_manager.build_news_feature_table_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        artifact_variant=artifact_variant,
    )
    metadata_path = path_manager.build_news_feature_metadata_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        artifact_variant=artifact_variant,
    )
    feature_config = news_feature_settings_to_dict(effective_news_feature_settings)

    cached_result = load_cached_result(
        feature_table_path=feature_table_path,
        metadata_path=metadata_path,
        source_extractions_hash=source_extractions_hash,
        source_extractions_metadata_path=source_extractions_metadata_path,
        source_extractions_metadata_hash=source_extractions_metadata_hash,
        feature_config=feature_config,
        supported_prediction_modes=supported_prediction_modes,
        force=force,
    )
    if cached_result is not None:
        logger.info(
            "News features ready from cache | ticker=%s | exchange=%s | rows=%s | feature_csv=%s",
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            cached_result.row_count,
            feature_table_path,
        )
        return cached_result

    source_frame = validate_extraction_frame(
        read_extraction_table(source_extractions_path),
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
    )
    calendar = build_market_calendar(runtime_settings.market)
    quality_weight_map = build_quality_weight_map(effective_news_feature_settings)

    enriched_frame, article_alignments = enrich_extraction_frame(
        source_frame,
        settings=effective_runtime_settings,
        quality_weight_map=quality_weight_map,
        calendar=calendar,
    )
    feature_frame, row_lineage = compute_news_feature_frame(
        enriched_frame,
        settings=effective_runtime_settings,
        supported_prediction_modes=supported_prediction_modes,
        calendar=calendar,
        target_end_date=target_end_date,
    )
    feature_frame = apply_news_carry_forward(
        feature_frame,
        carry_days=effective_news_feature_settings.carry_forward_days,
        decay_factor=effective_news_feature_settings.carry_decay_factor,
    )
    validated_feature_frame = validate_feature_frame(
        feature_frame,
        supported_prediction_modes=supported_prediction_modes,
    )

    raw_snapshot_path = path_manager.build_raw_news_feature_data_path(
        runtime_settings.ticker.symbol,
        run_context.run_id,
        artifact_variant=artifact_variant,
    )
    feature_table_path.parent.mkdir(parents=True, exist_ok=True)
    validated_feature_frame.to_csv(feature_table_path, index=False)

    raw_snapshot_payload = build_raw_snapshot_payload(
        settings=runtime_settings,
        source_extractions_path=source_extractions_path,
        source_extractions_hash=source_extractions_hash,
        source_extractions_metadata_path=source_extractions_metadata_path,
        source_extractions_metadata_hash=source_extractions_metadata_hash,
        generated_at_utc=run_context.started_at_utc,
        run_id=run_context.run_id,
        supported_prediction_modes=supported_prediction_modes,
        feature_config=feature_config,
        artifact_variant=artifact_variant,
        article_alignments=article_alignments,
        row_lineage=row_lineage,
    )
    write_json_file(raw_snapshot_path, raw_snapshot_payload)

    metadata = build_feature_metadata(
        settings=runtime_settings,
        feature_table_path=feature_table_path,
        raw_snapshot_path=raw_snapshot_path,
        source_extractions_path=source_extractions_path,
        source_extractions_hash=source_extractions_hash,
        source_extractions_metadata_path=source_extractions_metadata_path,
        source_extractions_metadata_hash=source_extractions_metadata_hash,
        feature_frame=validated_feature_frame,
        source_row_count=len(source_frame),
        supported_prediction_modes=supported_prediction_modes,
        feature_config=feature_config,
        artifact_variant=artifact_variant,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
        started_at_utc=run_context.started_at_utc,
        finished_at_utc=datetime.now(timezone.utc),
        elapsed_seconds=round(time.perf_counter() - stage_start, 6),
    )
    write_json_file(metadata_path, metadata)

    coverage_start = (
        date.fromisoformat(str(validated_feature_frame.iloc[0]["date"]))
        if not validated_feature_frame.empty
        else None
    )
    coverage_end = (
        date.fromisoformat(str(validated_feature_frame.iloc[-1]["date"]))
        if not validated_feature_frame.empty
        else None
    )

    logger.info(
        "News features ready | ticker=%s | exchange=%s | source_rows=%s | feature_rows=%s | coverage=%s..%s | elapsed=%.3fs | feature_csv=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        len(source_frame),
        len(validated_feature_frame),
        metadata["coverage_start"],
        metadata["coverage_end"],
        coerce_elapsed_seconds(metadata),
        feature_table_path,
    )

    return NewsFeatureBuildResult(
        feature_table_path=feature_table_path,
        metadata_path=metadata_path,
        raw_snapshot_path=raw_snapshot_path,
        row_count=len(validated_feature_frame),
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        cache_hit=False,
    )


def resolve_extraction_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    extraction_table_path: str | Path | None,
) -> Path:
    """Resolve the Stage 6 extraction table path for Stage 7."""

    if extraction_table_path is not None:
        return Path(extraction_table_path).expanduser().resolve()

    return path_manager.build_processed_llm_extractions_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def infer_extraction_metadata_path(extraction_table_path: Path) -> Path | None:
    """Infer the standard Stage 6 metadata path for one extraction table."""

    if extraction_table_path.suffix.lower() != ".csv":
        return None

    candidate = extraction_table_path.with_name(
        f"{extraction_table_path.stem}.metadata.json"
    )
    if candidate.exists():
        return candidate
    return None


def read_extraction_table(extraction_table_path: Path) -> pd.DataFrame:
    """Read the persisted Stage 6 extraction table."""

    try:
        return pd.read_csv(extraction_table_path)
    except FileNotFoundError as exc:
        raise NewsFeatureError(
            f"Processed LLM extraction file does not exist: {extraction_table_path}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise NewsFeatureError(
            f"Processed LLM extraction file is empty: {extraction_table_path}"
        ) from exc


def validate_extraction_frame(
    extraction_frame: pd.DataFrame,
    *,
    ticker: str,
    exchange: str,
) -> pd.DataFrame:
    """Validate the Stage 6 extraction table before Stage 7 aggregation."""

    working_frame = extraction_frame.copy()
    for column_name, default_value in OPTIONAL_EXTRACTION_COLUMNS_WITH_DEFAULTS.items():
        if column_name not in working_frame.columns:
            working_frame[column_name] = default_value
    missing_columns = [
        column
        for column in SOURCE_EXTRACTION_REQUIRED_COLUMNS
        if column not in working_frame.columns
    ]
    if missing_columns:
        raise NewsFeatureError(
            f"Processed LLM extraction table is missing required columns: {missing_columns}"
        )

    working_frame = working_frame.loc[:, SOURCE_EXTRACTION_REQUIRED_COLUMNS].copy()
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
        raise NewsFeatureError(
            f"Processed LLM extraction ticker values do not match the requested ticker: {sorted(source_tickers)}"
        )
    if source_exchanges != {exchange.upper()}:
        raise NewsFeatureError(
            f"Processed LLM extraction exchange values do not match the requested exchange: {sorted(source_exchanges)}"
        )

    working_frame["article_id"] = working_frame["article_id"].map(sanitize_prompt_text)
    if any(not article_id for article_id in working_frame["article_id"].tolist()):
        raise NewsFeatureError(
            "Processed LLM extraction table contains empty article_id values."
        )
    if working_frame["article_id"].duplicated().any():
        raise NewsFeatureError(
            "Processed LLM extraction table contains duplicate article_id values."
        )

    for column_name, allowed_values in (
        ("sentiment_label", ALLOWED_SENTIMENT_LABELS),
        ("directional_bias", ALLOWED_DIRECTIONAL_BIAS),
        ("event_type", ALLOWED_EVENT_TYPES),
        ("extraction_mode", ALLOWED_EXTRACTION_MODES),
    ):
        working_frame[column_name] = working_frame[column_name].map(normalize_enum)
        invalid_values = sorted(
            {
                value
                for value in working_frame[column_name].tolist()
                if value not in allowed_values
            }
        )
        if invalid_values:
            raise NewsFeatureError(
                f"Processed LLM extraction table contains unsupported {column_name} values: {invalid_values}"
            )

    for column_name, (minimum, maximum) in SOURCE_NUMERIC_RANGES.items():
        numeric_series = pd.to_numeric(working_frame[column_name], errors="coerce")
        if numeric_series.isna().any():
            raise NewsFeatureError(
                f"Processed LLM extraction table contains invalid {column_name} values."
            )
        if (~numeric_series.map(math.isfinite)).any():
            raise NewsFeatureError(
                f"Processed LLM extraction table contains non-finite {column_name} values."
            )
        if ((numeric_series < minimum) | (numeric_series > maximum)).any():
            raise NewsFeatureError(
                f"Processed LLM extraction table contains out-of-range {column_name} values."
            )
        working_frame[column_name] = numeric_series.astype(float)

    working_frame["warning_flag"] = coerce_bool_series(
        working_frame["warning_flag"],
        column_name="warning_flag",
    )
    working_frame["ticker"] = working_frame["ticker"].map(
        lambda value: sanitize_prompt_text(value).upper()
    )
    working_frame["exchange"] = working_frame["exchange"].map(
        lambda value: sanitize_prompt_text(value).upper()
    )
    return working_frame.reset_index(drop=True)


def build_quality_weight_map(settings: NewsFeatureSettings) -> dict[str, float]:
    """Build the Stage 7 quality-weight map from settings."""

    return {
        "full_article": settings.full_article_weight,
        "headline_plus_snippet": settings.headline_plus_snippet_weight,
        "headline_only": settings.headline_only_weight,
    }


def resolve_source_quality_weight(source_row: Mapping[str, Any]) -> float:
    """Resolve one deterministic source-quality weight from the upstream provider."""

    provider_name = sanitize_prompt_text(source_row.get("provider")).lower()
    return SOURCE_PROVIDER_QUALITY_WEIGHTS.get(provider_name, 0.85)


def enrich_extraction_frame(
    source_frame: pd.DataFrame,
    *,
    settings: AppSettings,
    quality_weight_map: Mapping[str, float],
    calendar: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Add Stage 7 timing and weighting columns to Stage 6 article rows."""

    if source_frame.empty:
        return pd.DataFrame(), []

    enriched_rows: list[dict[str, Any]] = []
    article_alignments: list[dict[str, Any]] = []
    for source_row in source_frame.to_dict(orient="records"):
        market_datetime = resolve_article_market_datetime(
            source_row,
            timezone_name=settings.market.timezone_name,
        )
        local_date = market_datetime.date()
        is_trading_day = calendar.is_trading_day(local_date)
        market_phase = determine_market_phase(
            market_datetime,
            settings=settings,
            is_trading_day=is_trading_day,
        )
        market_date = compute_market_date(
            local_date,
            market_phase=market_phase,
            calendar=calendar,
        )
        pre_market_target_date = compute_pre_market_target_date(
            local_date,
            market_phase=market_phase,
            calendar=calendar,
        )
        after_close_target_date = compute_after_close_target_date(
            local_date,
            calendar=calendar,
        )
        quality_weight = quality_weight_map[source_row["extraction_mode"]]
        source_quality_weight = resolve_source_quality_weight(source_row)
        relevance_score = float(source_row["relevance_score"])
        confidence_score = float(source_row["confidence_score"])
        confidence_weight = (
            confidence_score if settings.news_features.use_confidence_in_article_weight else 1.0
        )
        article_weight = (
            quality_weight
            * source_quality_weight
            * relevance_score
            * confidence_weight
        )
        is_fallback = source_row["extraction_mode"] != "full_article"

        enriched_row = dict(source_row)
        enriched_row.update(
            {
                "published_at_market": market_datetime.isoformat(),
                "published_at_market_dt": pd.Timestamp(market_datetime),
                "market_phase": market_phase,
                "market_date": market_date,
                "pre_market_target_date": pre_market_target_date,
                "after_close_target_date": after_close_target_date,
                "quality_weight": quality_weight,
                "source_quality_weight": source_quality_weight,
                "confidence_weight": confidence_weight,
                "article_weight": article_weight,
                "is_fallback": is_fallback,
                "bullish_indicator": int(source_row["directional_bias"] == "bullish"),
                "bearish_indicator": int(source_row["directional_bias"] == "bearish"),
                "neutral_indicator": int(source_row["directional_bias"] == "neutral"),
            }
        )
        enriched_rows.append(enriched_row)
        article_alignments.append(
            {
                "article_id": source_row["article_id"],
                "article_input_hash": sanitize_prompt_text(
                    source_row.get("article_input_hash")
                ),
                "published_at_market": market_datetime.isoformat(),
                "market_phase": market_phase,
                "market_date": market_date.isoformat(),
                "pre_market_target_date": pre_market_target_date.isoformat(),
                "after_close_target_date": after_close_target_date.isoformat(),
                "warning_flag": bool(source_row["warning_flag"]),
                "extraction_mode": source_row["extraction_mode"],
                "quality_weight": quality_weight,
                "source_quality_weight": source_quality_weight,
                "confidence_weight": confidence_weight,
                "article_weight": article_weight,
            }
        )

    enriched_frame = pd.DataFrame(enriched_rows)
    enriched_frame = enriched_frame.sort_values(
        by=["published_at_market_dt", "article_id"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return enriched_frame, article_alignments


def resolve_article_market_datetime(
    source_row: Mapping[str, Any],
    *,
    timezone_name: str,
) -> datetime:
    """Resolve one extraction timestamp into the configured market timezone."""

    for field_name, default_timezone in (
        ("published_at_ist", timezone_name),
        ("published_at_utc", "UTC"),
    ):
        raw_value = sanitize_prompt_text(source_row.get(field_name))
        if not raw_value:
            continue
        try:
            timestamp = pd.Timestamp(raw_value)
        except (TypeError, ValueError):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(default_timezone)
        return timestamp.tz_convert(timezone_name).to_pydatetime()

    article_id = sanitize_prompt_text(source_row.get("article_id")) or "unknown_article"
    raise NewsFeatureError(
        f"Stage 6 row {article_id} does not contain a usable published timestamp."
    )


def determine_market_phase(
    value: datetime,
    *,
    settings: AppSettings,
    is_trading_day: bool,
) -> str:
    """Classify one article into the Stage 7 market-time buckets."""

    if not is_trading_day:
        return "non_trading"
    if is_pre_market(value, settings.market):
        return "pre_market"
    if is_intraday(value, settings.market):
        return "intraday"
    if is_after_close(value, settings.market):
        return "after_close"
    raise NewsFeatureError("Failed to classify market-time phase for an article.")


def compute_market_date(
    local_date: date,
    *,
    market_phase: str,
    calendar: Any,
) -> date:
    """Compute the generic Stage 7 market_date traceability field."""

    if calendar.is_trading_day(local_date) and market_phase in {"pre_market", "intraday"}:
        return local_date
    return first_trading_day_after(local_date, calendar)


def compute_pre_market_target_date(
    local_date: date,
    *,
    market_phase: str,
    calendar: Any,
) -> date:
    """Compute the first target day whose pre-market snapshot could know the article."""

    if calendar.is_trading_day(local_date) and market_phase == "pre_market":
        return local_date
    return first_trading_day_after(local_date, calendar)


def compute_after_close_target_date(local_date: date, *, calendar: Any) -> date:
    """Compute the first target day whose after-close snapshot could know the article."""

    first_known_session_day = first_trading_day_on_or_after(local_date, calendar)
    return calendar.next_trading_day(first_known_session_day)


def compute_news_feature_frame(
    enriched_frame: pd.DataFrame,
    *,
    settings: AppSettings,
    supported_prediction_modes: tuple[str, ...],
    calendar: Any,
    target_end_date: date | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aggregate Stage 6 article rows into Stage 7 feature rows."""

    if enriched_frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), []

    target_dates = collect_target_dates(
        enriched_frame,
        supported_prediction_modes=supported_prediction_modes,
    )
    if not target_dates:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), []

    max_target_date = max(target_dates)
    if target_end_date is not None and target_end_date > max_target_date:
        max_target_date = target_end_date

    trading_days = build_trading_day_range(
        min(target_dates),
        max_target_date,
        calendar=calendar,
    )
    grouped_by_mode: dict[str, dict[date, pd.DataFrame]] = {}
    for prediction_mode in supported_prediction_modes:
        target_column = f"{prediction_mode}_target_date"
        mode_groups = {
            group_key: group_frame.copy()
            for group_key, group_frame in enriched_frame.groupby(target_column, sort=True)
        }
        grouped_by_mode[prediction_mode] = mode_groups

    feature_rows: list[dict[str, Any]] = []
    row_lineage: list[dict[str, Any]] = []
    for trading_day in trading_days:
        for prediction_mode in supported_prediction_modes:
            group = grouped_by_mode[prediction_mode].get(trading_day)
            feature_row, lineage_row = aggregate_feature_row(
                group,
                settings=settings,
                target_date=trading_day,
                prediction_mode=prediction_mode,
            )
            feature_rows.append(feature_row)
            row_lineage.append(lineage_row)

    feature_frame = pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS)

    # Apply rolling windows
    rolling_windows = settings.news_features.rolling_windows
    for mode in supported_prediction_modes:
        mode_mask = feature_frame["prediction_mode"] == mode
        mode_df = feature_frame[mode_mask].sort_values("date")

        for w in rolling_windows:
            # Volume: sum of article counts
            feature_frame.loc[mode_mask, f"news_volume_{w}d"] = (
                mode_df["news_article_count"].rolling(window=w, min_periods=1).sum()
            )

            # Event count: sum of all event type counts
            event_cols = [
                c
                for c in feature_frame.columns
                if c.startswith("news_event_count_") and not c.endswith("d")
            ]
            daily_event_sum = mode_df[event_cols].sum(axis=1)
            feature_frame.loc[mode_mask, f"news_event_count_{w}d"] = (
                daily_event_sum.rolling(window=w, min_periods=1).sum()
            )

            # Sentiment: Exponential Moving Average (EMA) of daily weighted sentiment scores
            # We use an EMA to apply steep decay to older news, preventing the model
            # from relying too heavily on stale sentiment on "zero news" days.
            if w == 1:
                feature_frame.loc[mode_mask, f"news_sentiment_{w}d"] = mode_df["news_weighted_sentiment_score"]
            else:
                feature_frame.loc[mode_mask, f"news_sentiment_{w}d"] = (
                    mode_df["news_weighted_sentiment_score"].ewm(span=w, adjust=False).mean().fillna(0.0)
                )

    feature_frame["prediction_mode_sort_key"] = feature_frame["prediction_mode"].map(
        PREDICTION_MODE_ORDER
    )
    feature_frame = feature_frame.sort_values(
        by=["date", "prediction_mode_sort_key"],
        ascending=[True, True],
    ).drop(columns=["prediction_mode_sort_key"])
    feature_frame = feature_frame.reset_index(drop=True)
    return feature_frame, row_lineage


def collect_target_dates(
    enriched_frame: pd.DataFrame,
    *,
    supported_prediction_modes: tuple[str, ...],
) -> list[date]:
    """Collect the target-day coverage implied by enriched article rows."""

    target_dates: list[date] = []
    for prediction_mode in supported_prediction_modes:
        column_name = f"{prediction_mode}_target_date"
        target_dates.extend(
            value
            for value in enriched_frame[column_name].tolist()
            if isinstance(value, date)
        )
    return target_dates


def build_trading_day_range(start: date, end: date, *, calendar: Any) -> tuple[date, ...]:
    """Build the inclusive trading-day range between two target dates."""

    current = first_trading_day_on_or_after(start, calendar)
    days: list[date] = []
    while current <= end:
        days.append(current)
        current = calendar.next_trading_day(current)
    return tuple(days)


def aggregate_feature_row(
    group: pd.DataFrame | None,
    *,
    settings: AppSettings,
    target_date: date,
    prediction_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate one target-day/prediction-mode article set into one feature row."""

    ticker = settings.ticker.symbol
    exchange = settings.ticker.exchange

    if group is None or group.empty:
        return (
            build_zero_feature_row(
                target_date=target_date,
                ticker=ticker,
                exchange=exchange,
                prediction_mode=prediction_mode,
            ),
            {
                "date": target_date.isoformat(),
                "prediction_mode": prediction_mode,
                "article_count": 0,
                "article_ids": [],
            },
        )

    sorted_group = group.sort_values(
        by=["published_at_market_dt", "article_id"],
        ascending=[True, True],
    ).reset_index(drop=True)
    quality_weights = sorted_group["quality_weight"].astype(float)
    article_weights = sorted_group["article_weight"].astype(float)
    article_ids = sorted_group["article_id"].astype(str).tolist()

    feature_row = build_zero_feature_row(
        target_date=target_date,
        ticker=ticker,
        exchange=exchange,
        prediction_mode=prediction_mode,
    )
    n_articles = int(len(sorted_group))
    bull_n = int(sorted_group["bullish_indicator"].sum())
    bear_n = int(sorted_group["bearish_indicator"].sum())
    neu_n = int(sorted_group["neutral_indicator"].sum())
    if n_articles <= 1:
        sentiment_dispersion = 0.0
    else:
        sentiment_dispersion = float(sorted_group["sentiment_score"].std(ddof=0))
    directional_agreement = (
        float(max(bull_n, bear_n, neu_n) / n_articles) if n_articles else 0.0
    )
    feature_row.update(
        {
            "news_article_count": n_articles,
            "news_sentiment_dispersion_1d": sentiment_dispersion,
            "news_directional_agreement_rate": directional_agreement,
            "news_avg_sentiment": float(sorted_group["sentiment_score"].mean()),
            "news_max_severity": float(sorted_group["event_severity"].max()),
            "news_avg_relevance": float(sorted_group["relevance_score"].mean()),
            "news_avg_confidence": float(sorted_group["confidence_score"].mean()),
            "news_bullish_article_count": int(sorted_group["bullish_indicator"].sum()),
            "news_bearish_article_count": int(sorted_group["bearish_indicator"].sum()),
            "news_neutral_article_count": int(sorted_group["neutral_indicator"].sum()),
            "news_full_article_count": int(
                (sorted_group["extraction_mode"] == "full_article").sum()
            ),
            "news_headline_plus_snippet_count": int(
                (sorted_group["extraction_mode"] == "headline_plus_snippet").sum()
            ),
            "news_headline_only_count": int(
                (sorted_group["extraction_mode"] == "headline_only").sum()
            ),
            "news_warning_article_count": int(sorted_group["warning_flag"].sum()),
            "news_fallback_article_ratio": float(sorted_group["is_fallback"].mean()),
            "news_avg_content_quality_score": float(
                sorted_group["content_quality_score"].mean()
            ),
            "news_avg_source_quality_score": float(
                sorted_group["source_quality_weight"].mean()
            ),
            "news_weighted_sentiment_score": weighted_mean(
                sorted_group["sentiment_score"],
                article_weights,
            ),
            "news_weighted_relevance_score": weighted_mean(
                sorted_group["relevance_score"],
                quality_weights,
            ),
            "news_weighted_confidence_score": weighted_mean(
                sorted_group["confidence_score"],
                quality_weights,
            ),
            "news_weighted_source_quality_score": weighted_mean(
                sorted_group["source_quality_weight"],
                article_weights,
            ),
            "news_weighted_bullish_score": weighted_mean(
                sorted_group["bullish_indicator"],
                article_weights,
            ),
            "news_weighted_bearish_score": weighted_mean(
                sorted_group["bearish_indicator"],
                article_weights,
            ),
            "news_zero_flag": 0,
            "news_carried_flag": 0,
            "has_news_signal": 1,
            "carried_forward_signal": 0,
            "degraded_news_context": int(
                float(sorted_group["is_fallback"].mean())
                > settings.pilot.fallback_heavy_ratio_threshold
            ),
            "has_fresh_news": 1,
            "is_carried_forward": 0,
            "is_fallback_heavy": int(
                float(sorted_group["is_fallback"].mean())
                > settings.pilot.fallback_heavy_ratio_threshold
            ),
        }
    )
    feature_row[NEWS_SIGNAL_STATE_COLUMN] = determine_news_signal_state(feature_row)

    # Channel aggregation
    sorted_group["channel"] = sorted_group["event_type"].map(EVENT_TYPE_TO_CHANNEL).fillna("company")
    for channel in NEWS_CHANNELS:
        channel_group = sorted_group[sorted_group["channel"] == channel]
        if not channel_group.empty:
            feature_row[f"news_{channel}_article_count"] = int(len(channel_group))
            feature_row[f"news_{channel}_article_share"] = float(
                len(channel_group) / len(sorted_group)
            )
            feature_row[f"news_{channel}_avg_sentiment"] = float(channel_group["sentiment_score"].mean())
            feature_row[f"news_{channel}_weighted_sentiment_score"] = weighted_mean(
                channel_group["sentiment_score"],
                channel_group["article_weight"],
            )
            feature_row[f"news_{channel}_max_severity"] = float(channel_group["event_severity"].max())
        else:
            feature_row[f"news_{channel}_article_count"] = 0
            feature_row[f"news_{channel}_article_share"] = 0.0
            feature_row[f"news_{channel}_avg_sentiment"] = 0.0
            feature_row[f"news_{channel}_weighted_sentiment_score"] = 0.0
            feature_row[f"news_{channel}_max_severity"] = 0.0
    for event_type in sorted(ALLOWED_EVENT_TYPES):
        feature_row[f"news_event_count_{event_type}"] = int(
            (sorted_group["event_type"] == event_type).sum()
        )

    return (
        feature_row,
        {
            "date": target_date.isoformat(),
            "prediction_mode": prediction_mode,
            "article_count": len(article_ids),
            "article_ids": article_ids,
        },
    )


def build_zero_feature_row(
    *,
    target_date: date,
    ticker: str,
    exchange: str,
    prediction_mode: str,
) -> dict[str, Any]:
    """Build a zero-filled Stage 7 feature row for a no-news trading day."""

    feature_row: dict[str, Any] = {
        "date": target_date.isoformat(),
        "ticker": ticker,
        "exchange": exchange,
        "prediction_mode": prediction_mode,
    }
    for column_name in COUNT_FEATURE_COLUMNS:
        feature_row[column_name] = 0
    for column_name in FLOAT_FEATURE_COLUMNS:
        feature_row[column_name] = 0.0
    # Mark this row as having had no articles on the target date.
    feature_row["news_zero_flag"] = 1
    feature_row["news_carried_flag"] = 0
    feature_row["has_news_signal"] = 0
    feature_row["carried_forward_signal"] = 0
    feature_row["degraded_news_context"] = 0
    feature_row["has_fresh_news"] = 0
    feature_row["is_carried_forward"] = 0
    feature_row["is_fallback_heavy"] = 0
    feature_row[NEWS_SIGNAL_STATE_COLUMN] = NEWS_SIGNAL_STATE_ZERO
    return feature_row


def apply_news_carry_forward(
    feature_frame: pd.DataFrame,
    *,
    carry_days: int,
    decay_factor: float,
) -> pd.DataFrame:
    """Propagate decayed news signals into zero-news trading days.

    For each row where news_zero_flag == 1, looks back up to carry_days rows
    (same prediction_mode) for the most recent row with news_zero_flag == 0.
    Copies its float/weighted feature values scaled by decay_factor^distance.
    Sets news_carried_flag=1 and news_zero_flag=0 on the destination row.
    Count columns (article counts, event counts) are intentionally left at 0
    so the model still knows no fresh articles were fetched.
    """

    if carry_days <= 0 or feature_frame.empty:
        return feature_frame

    working_frame = feature_frame.copy()
    # Columns that should be decayed (excludes identity, count, and flag cols).
    decay_columns = [
        col
        for col in FLOAT_FEATURE_COLUMNS
        if col not in {"news_zero_flag", "news_carried_flag"}
        and not col.endswith("d")
    ]

    # Process each prediction_mode independently so we never mix pre_market
    # signal into an after_close row.
    for mode in working_frame["prediction_mode"].unique():
        mode_mask = working_frame["prediction_mode"] == mode
        mode_indices = working_frame.index[mode_mask].tolist()

        for pos, idx in enumerate(mode_indices):
            if working_frame.at[idx, "news_zero_flag"] != 1:
                continue
            # Search backward within the same mode up to carry_days steps.
            for lookback in range(1, carry_days + 1):
                prior_pos = pos - lookback
                if prior_pos < 0:
                    break
                prior_idx = mode_indices[prior_pos]
                if (
                    working_frame.at[prior_idx, "news_zero_flag"] == 1
                    and working_frame.at[prior_idx, "news_carried_flag"] == 0
                ):
                    # That prior row also had no original news; skip it.
                    continue
                scale = decay_factor**lookback
                for col in decay_columns:
                    working_frame.at[idx, col] = (
                        working_frame.at[prior_idx, col] * scale
                    )
                working_frame.at[idx, "news_zero_flag"] = 0
                working_frame.at[idx, "news_carried_flag"] = 1
                working_frame.at[idx, "has_news_signal"] = 1
                working_frame.at[idx, "carried_forward_signal"] = 1
                working_frame.at[idx, "is_carried_forward"] = 1
                working_frame.at[idx, NEWS_SIGNAL_STATE_COLUMN] = determine_news_signal_state(
                    working_frame.loc[idx].to_dict()
                )
                break

    return working_frame


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a stable weighted mean that returns 0 for zero-denominator inputs."""

    value_series = pd.to_numeric(values, errors="coerce").astype(float)
    weight_series = pd.to_numeric(weights, errors="coerce").astype(float)
    total_weight = float(weight_series.sum())
    if total_weight <= 0:
        return 0.0
    return float((value_series * weight_series).sum() / total_weight)


def validate_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    supported_prediction_modes: tuple[str, ...],
) -> pd.DataFrame:
    """Validate the final Stage 7 feature table before persistence."""

    if feature_frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    working_frame = feature_frame.copy()
    working_frame[NEWS_SIGNAL_STATE_COLUMN] = working_frame.apply(
        determine_news_signal_state,
        axis=1,
    )
    working_frame = working_frame.loc[:, OUTPUT_COLUMNS].copy()
    if working_frame.loc[:, list(OUTPUT_IDENTITY_COLUMNS)].isna().any().any():
        raise NewsFeatureError("News feature table contains empty identity values.")

    date_series = pd.to_datetime(working_frame["date"], errors="coerce")
    if date_series.isna().any():
        raise NewsFeatureError("News feature table contains invalid date values.")
    working_frame["date"] = date_series.dt.strftime("%Y-%m-%d")

    prediction_modes = set(working_frame["prediction_mode"].tolist())
    unsupported_modes = prediction_modes - set(supported_prediction_modes)
    if unsupported_modes:
        raise NewsFeatureError(
            f"News feature table contains unsupported prediction modes: {sorted(unsupported_modes)}"
        )
    if working_frame.duplicated(subset=list(OUTPUT_IDENTITY_COLUMNS)).any():
        raise NewsFeatureError("News feature table contains duplicate identity rows.")
    invalid_news_states = sorted(
        set(working_frame[NEWS_SIGNAL_STATE_COLUMN].astype(str))
        - set(ALLOWED_NEWS_SIGNAL_STATES)
    )
    if invalid_news_states:
        raise NewsFeatureError(
            f"News feature table contains unsupported news_signal_state values: {invalid_news_states}"
        )

    for column_name in COUNT_FEATURE_COLUMNS:
        numeric_series = pd.to_numeric(working_frame[column_name], errors="coerce")
        if numeric_series.isna().any() or (~numeric_series.map(math.isfinite)).any():
            raise NewsFeatureError(
                f"News feature table contains invalid {column_name} values."
            )
        if (numeric_series < 0).any():
            raise NewsFeatureError(
                f"News feature table contains negative {column_name} values."
            )
        if not ((numeric_series % 1) == 0).all():
            raise NewsFeatureError(
                f"News feature table contains non-integer {column_name} values."
            )
        working_frame[column_name] = numeric_series.astype(int)

    for column_name in FLOAT_FEATURE_COLUMNS:
        numeric_series = pd.to_numeric(working_frame[column_name], errors="coerce")
        if numeric_series.isna().any() or (~numeric_series.map(math.isfinite)).any():
            raise NewsFeatureError(
                f"News feature table contains invalid {column_name} values."
            )
        minimum, maximum = FEATURE_NUMERIC_RANGES[column_name]
        if ((numeric_series < minimum) | (numeric_series > maximum)).any():
            raise NewsFeatureError(
                f"News feature table contains out-of-range {column_name} values."
            )
        working_frame[column_name] = numeric_series.astype(float)

    working_frame["prediction_mode_sort_key"] = working_frame["prediction_mode"].map(
        PREDICTION_MODE_ORDER
    )
    working_frame = working_frame.sort_values(
        by=["date", "prediction_mode_sort_key"],
        ascending=[True, True],
    ).drop(columns=["prediction_mode_sort_key"])
    return working_frame.reset_index(drop=True)


def coerce_bool_series(raw_values: pd.Series, *, column_name: str) -> pd.Series:
    """Coerce one DataFrame column into booleans."""

    parsed_values: list[bool] = []
    for raw_value in raw_values.tolist():
        if isinstance(raw_value, bool):
            parsed_values.append(raw_value)
            continue
        normalized = sanitize_prompt_text(raw_value).lower()
        if normalized in {"true", "1", "yes"}:
            parsed_values.append(True)
            continue
        if normalized in {"false", "0", "no"}:
            parsed_values.append(False)
            continue
        raise NewsFeatureError(f"Expected boolean values for {column_name}.")
    return pd.Series(parsed_values, index=raw_values.index, dtype=bool)


def build_raw_snapshot_payload(
    *,
    settings: AppSettings,
    source_extractions_path: Path,
    source_extractions_hash: str,
    source_extractions_metadata_path: Path | None,
    source_extractions_metadata_hash: str | None,
    generated_at_utc: datetime,
    run_id: str,
    supported_prediction_modes: tuple[str, ...],
    feature_config: dict[str, Any],
    artifact_variant: str | None,
    article_alignments: list[dict[str, Any]],
    row_lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the raw Stage 7 snapshot payload."""

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "company_name": settings.ticker.company_name,
        "source_extractions_path": str(source_extractions_path),
        "source_extractions_hash": source_extractions_hash,
        "source_extractions_metadata_path": (
            str(source_extractions_metadata_path)
            if source_extractions_metadata_path
            else None
        ),
        "source_extractions_metadata_hash": source_extractions_metadata_hash,
        "generated_at_utc": generated_at_utc.isoformat(),
        "run_id": run_id,
        "artifact_variant": artifact_variant,
        "formula_version": FEATURE_FORMULA_VERSION,
        "supported_prediction_modes": list(supported_prediction_modes),
        "feature_config": feature_config,
        "article_alignment_count": len(article_alignments),
        "feature_row_count": len(row_lineage),
        "article_alignments": article_alignments,
        "row_lineage": row_lineage,
    }


def build_feature_metadata(
    *,
    settings: AppSettings,
    feature_table_path: Path,
    raw_snapshot_path: Path,
    source_extractions_path: Path,
    source_extractions_hash: str,
    source_extractions_metadata_path: Path | None,
    source_extractions_metadata_hash: str | None,
    feature_frame: pd.DataFrame,
    source_row_count: int,
    supported_prediction_modes: tuple[str, ...],
    feature_config: dict[str, Any],
    artifact_variant: str | None,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the persisted Stage 7 metadata payload."""

    coverage_start = None
    coverage_end = None
    warnings: list[str] = []
    if not feature_frame.empty:
        coverage_start = str(feature_frame.iloc[0]["date"])
        coverage_end = str(feature_frame.iloc[-1]["date"])
    else:
        warnings.append("no_feature_rows")
    if source_row_count == 0:
        warnings.append("no_source_articles")

    zero_news_row_count = 0
    if not feature_frame.empty:
        zero_news_row_count = int((feature_frame["news_article_count"] == 0).sum())

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "company_name": settings.ticker.company_name,
        "feature_table_path": str(feature_table_path),
        "feature_table_hash": compute_file_sha256(feature_table_path),
        "raw_snapshot_path": str(raw_snapshot_path),
        "raw_snapshot_hash": compute_file_sha256(raw_snapshot_path),
        "source_extractions_path": str(source_extractions_path),
        "source_extractions_hash": source_extractions_hash,
        "source_extractions_metadata_path": (
            str(source_extractions_metadata_path)
            if source_extractions_metadata_path
            else None
        ),
        "source_extractions_metadata_hash": source_extractions_metadata_hash,
        "source_row_count": source_row_count,
        "output_row_count": int(len(feature_frame)),
        "zero_news_row_count": zero_news_row_count,
        "nonzero_news_row_count": int(len(feature_frame) - zero_news_row_count),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "supported_prediction_modes": list(supported_prediction_modes),
        "feature_config": feature_config,
        "artifact_variant": artifact_variant,
        "feature_columns": list(NEWS_FEATURE_COLUMNS),
        "event_columns": list(EVENT_COUNT_COLUMNS),
        "formula_version": FEATURE_FORMULA_VERSION,
        "prediction_mode_row_counts": count_series_values(
            feature_frame,
            "prediction_mode",
        ),
        "news_signal_state_counts": count_series_values(
            feature_frame,
            NEWS_SIGNAL_STATE_COLUMN,
        ),
        "timing": {
            "started_at_utc": started_at_utc.isoformat(),
            "finished_at_utc": finished_at_utc.isoformat(),
            "elapsed_seconds": float(elapsed_seconds),
        },
        "workload": {
            "source_row_count": int(source_row_count),
            "output_row_count": int(len(feature_frame)),
            "zero_news_row_count": int(zero_news_row_count),
        },
        "warnings": warnings,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def coerce_elapsed_seconds(metadata: dict[str, Any]) -> float:
    """Read the Stage 7 elapsed duration from metadata for logging."""

    timing = metadata.get("timing", {})
    try:
        return float((timing or {}).get("elapsed_seconds", 0.0))
    except (TypeError, ValueError):
        return 0.0


def count_series_values(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    """Count string values in one DataFrame column for metadata."""

    if frame.empty or column_name not in frame.columns:
        return {}
    counts = frame[column_name].fillna("null").value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def load_cached_result(
    *,
    feature_table_path: Path,
    metadata_path: Path,
    source_extractions_hash: str,
    source_extractions_metadata_path: Path | None,
    source_extractions_metadata_hash: str | None,
    feature_config: dict[str, Any],
    supported_prediction_modes: tuple[str, ...],
    force: bool,
) -> NewsFeatureBuildResult | None:
    """Return the cached Stage 7 result when the source and config still match."""

    if force or not feature_table_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    expected_metadata_path = (
        str(source_extractions_metadata_path) if source_extractions_metadata_path else None
    )
    if metadata.get("formula_version") != FEATURE_FORMULA_VERSION:
        return None
    if metadata.get("source_extractions_hash") != source_extractions_hash:
        return None
    if metadata.get("source_extractions_metadata_path") != expected_metadata_path:
        return None
    if metadata.get("source_extractions_metadata_hash") != source_extractions_metadata_hash:
        return None
    if metadata.get("feature_config") != feature_config:
        return None
    if metadata.get("supported_prediction_modes") != list(supported_prediction_modes):
        return None
    if not isinstance(metadata.get("news_signal_state_counts"), dict):
        return None

    raw_snapshot_value = metadata.get("raw_snapshot_path")
    if not raw_snapshot_value:
        return None
    raw_snapshot_path = Path(str(raw_snapshot_value))
    if not raw_snapshot_path.exists():
        return None

    try:
        coverage_start_value = metadata.get("coverage_start")
        coverage_end_value = metadata.get("coverage_end")
        return NewsFeatureBuildResult(
            feature_table_path=feature_table_path,
            metadata_path=metadata_path,
            raw_snapshot_path=raw_snapshot_path,
            row_count=int(metadata["output_row_count"]),
            coverage_start=(
                date.fromisoformat(str(coverage_start_value))
                if coverage_start_value
                else None
            ),
            coverage_end=(
                date.fromisoformat(str(coverage_end_value))
                if coverage_end_value
                else None
            ),
            cache_hit=True,
        )
    except (KeyError, TypeError, ValueError):
        return None


def compute_file_sha256(path: Path) -> str:
    """Hash one file so Stage 7 outputs can reuse cached artifacts."""

    return _compute_file_sha256(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 7 news-feature command arguments."""

    parser = argparse.ArgumentParser(description="Build Kubera Stage 7 news features.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--extractions-path",
        help="Use a specific Stage 6 extraction CSV file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild news features even if the cached artifact still matches the source table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 7 news-feature command."""

    args = parse_args(argv)
    settings = load_settings()
    build_news_features(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        extraction_table_path=args.extractions_path,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
