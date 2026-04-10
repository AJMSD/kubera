"""Live pilot workflow for Kubera."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from kubera.config import AppSettings, MarketSettings, load_settings, resolve_runtime_settings
from kubera.features.historical_features import (
    build_live_historical_feature_row,
    read_cleaned_market_data,
    validate_cleaned_market_data,
)
from kubera.features.news_features import (
    EVENT_COUNT_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    NEWS_SIGNAL_STATE_COLUMN,
    NEWS_SIGNAL_STATE_CARRIED_FORWARD,
    NEWS_SIGNAL_STATE_FALLBACK_HEAVY,
    NEWS_SIGNAL_STATE_ZERO,
    build_news_features,
    build_zero_feature_row,
    determine_news_signal_state,
)
from kubera.ingest.market_data import fetch_historical_market_data, slice_market_window
from kubera.ingest.news_data import fetch_company_news
from kubera.llm.extract_news import (
    GeminiApiExtractionClient,
    LlmExtractionError,
    extract_news,
    generate_plain_text_with_tiered_models,
)
from kubera.models.common import (
    blend_probabilities,
    compute_news_context_weight,
    explain_prediction_shap,
    resolve_selective_prediction,
)
from kubera.models.train_baseline import (
    load_saved_baseline_model,
    predict_with_saved_model_outputs,
)
from kubera.models.train_enhanced import (
    build_enhanced_feature_spec_from_metadata,
    build_live_enhanced_feature_row,
    load_saved_enhanced_model,
    predict_with_saved_enhanced_model_outputs,
)
from kubera.reporting.offline_evaluation import load_optional_json
from kubera.utils.calendar import (
    build_market_calendar,
    first_trading_day_on_or_after,
    format_live_pilot_cutoff_error,
    load_exchange_closures_as_of,
)
from kubera.utils.logging import configure_logging, sanitize_log_text
from kubera.utils.paths import PathManager
from kubera.utils.data_quality import (
    build_data_quality_payload,
    dedupe_quality_reasons as dedupe_shared_quality_reasons,
    grade_data_quality_score as grade_shared_data_quality_score,
)
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot
from kubera.utils.time_utils import (
    is_after_close,
    is_pre_market,
    market_time_to_utc,
    utc_to_market_time,
)
from kubera.utils.user_failure import describe_partial_failure_paths, describe_pilot_stage_failure


logger = logging.getLogger(__name__)

PILOT_PREDICTION_MODES = ("pre_market", "after_close")
PILOT_STATUS_SUCCESS = "success"
PILOT_STATUS_PARTIAL_FAILURE = "partial_failure"
PILOT_STATUS_FAILURE = "failure"
PILOT_STATUS_DRY_RUN = "dry_run"
PILOT_STATUS_ABSTAIN = "abstain"
PILOT_WINDOW_RESOLUTION_NATURAL = "natural"
PILOT_WINDOW_RESOLUTION_SNAPPED = "snapped"
PILOT_WINDOW_RESOLUTION_OVERRIDE = "override"
ACTUAL_STATUS_PENDING = "pending"
ACTUAL_STATUS_BACKFILLED = "backfilled"
ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE = "market_data_unavailable"
PILOT_WEEK_STATUS_PENDING = "pending"
PILOT_WEEK_STATUS_COMPLETED = "completed"
PILOT_WEEK_STATUS_PARTIAL_FAILURE = "partial_failure"
PILOT_WEEK_STATUS_FAILURE = "failure"
PILOT_LOG_COLUMNS = (
    "pilot_entry_id",
    "prediction_key",
    "prediction_attempt_number",
    "ticker",
    "exchange",
    "prediction_mode",
    "pilot_run_id",
    "pilot_timestamp_utc",
    "pilot_timestamp_market",
    "market_session_date",
    "historical_cutoff_date",
    "news_cutoff_timestamp_utc",
    "historical_date",
    "historical_market_gap_flag",
    "historical_market_gap_count_5d",
    "prediction_date",
    "baseline_predicted_next_day_direction",
    "baseline_raw_predicted_probability_up",
    "baseline_calibrated_predicted_probability_up",
    "baseline_predicted_probability_up",
    "enhanced_predicted_next_day_direction",
    "enhanced_raw_predicted_probability_up",
    "enhanced_calibrated_predicted_probability_up",
    "enhanced_predicted_probability_up",
    "blended_predicted_next_day_direction",
    "blended_raw_predicted_probability_up",
    "blended_calibrated_predicted_probability_up",
    "blended_predicted_probability_up",
    "selected_action",
    "abstain_flag",
    "selective_probability_margin",
    "selective_required_margin",
    "abstain_reason_codes_json",
    "news_context_weight",
    "disagreement_flag",
    "news_article_count",
    "news_warning_article_count",
    "news_fallback_article_ratio",
    "news_avg_confidence",
    "news_volume_3d",
    "news_sentiment_3d",
    "news_sentiment_dispersion_1d",
    "news_directional_agreement_rate",
    "has_fresh_news",
    "is_carried_forward",
    "is_fallback_heavy",
    "news_signal_state",
    "fallback_heavy_flag",
    "news_feature_synthetic_flag",
    "linked_article_ids_json",
    "top_event_counts_json",
    "warning_codes_json",
    "status",
    "failure_stage",
    "failure_message",
    "total_duration_seconds",
    "runtime_warning_flag",
    "runtime_warning_message",
    "data_quality_score",
    "data_quality_grade",
    "data_quality_reasons_json",
    "data_quality_components_json",
    "pilot_snapshot_path",
    "stage2_cleaned_path",
    "stage2_metadata_path",
    "stage2_run_id",
    "stage2_duration_seconds",
    "stage5_processed_news_path",
    "stage5_metadata_path",
    "stage5_run_id",
    "stage5_duration_seconds",
    "stage5_provider_request_count",
    "stage5_provider_request_retry_count",
    "stage5_article_fetch_attempt_count",
    "stage5_article_fetch_retry_count",
    "stage6_extraction_path",
    "stage6_metadata_path",
    "stage6_failure_log_path",
    "stage6_run_id",
    "stage6_duration_seconds",
    "stage6_provider_request_count",
    "stage6_retry_count",
    "stage7_feature_path",
    "stage7_metadata_path",
    "stage7_raw_snapshot_path",
    "stage7_run_id",
    "stage7_duration_seconds",
    "baseline_model_path",
    "baseline_model_metadata_path",
    "baseline_model_run_id",
    "baseline_duration_seconds",
    "enhanced_model_path",
    "enhanced_model_metadata_path",
    "enhanced_model_run_id",
    "enhanced_duration_seconds",
    "enhanced_feature_contributions_json",
    "recent_news_summaries_json",
    "actual_historical_close",
    "actual_prediction_close",
    "actual_next_day_direction",
    "actual_outcome_status",
    "actual_outcome_backfilled_at_utc",
    "actual_backfill_error",
    "baseline_correct",
    "enhanced_correct",
    "blended_correct",
    "news_quality_note",
    "market_shock_note",
    "source_outage_note",
    "manual_notes_updated_at_utc",
)


class LivePilotError(RuntimeError):
    """Raised when the live pilot workflow cannot continue."""


@dataclass(frozen=True)
class LivePredictionWindow:
    """Resolved live-pilot timing window for one run."""

    prediction_mode: str
    timestamp_utc: datetime
    timestamp_market: datetime
    market_session_date: date
    historical_cutoff_date: date
    prediction_date: date


@dataclass(frozen=True)
class LiveWindowResolution:
    """Resolved window plus user-facing resolution metadata."""

    prediction_window: LivePredictionWindow
    resolution_kind: str
    resolution_reason: str


@dataclass(frozen=True)
class PilotRunResult:
    """Persisted live pilot run summary."""

    log_path: Path
    snapshot_path: Path
    pilot_entry_id: str
    status: str
    market_session_date: date
    prediction_date: date
    prediction_mode: str
    historical_cutoff_date: date
    window_resolution_kind: str
    window_resolution_reason: str


@dataclass(frozen=True)
class PilotBackfillResult:
    """Summary of one actual-outcome backfill pass."""

    updated_row_count: int
    unresolved_row_count: int
    log_paths: tuple[Path, ...]


@dataclass(frozen=True)
class PilotPendingBackfillResult:
    """Summary of bounded pending actual-outcome backfill after ``kubera run`` / ``predict --backfill``."""

    updated_row_count: int
    unresolved_row_count: int
    error_count: int
    effective_as_of: date
    prediction_dates_attempted: tuple[date, ...]


@dataclass(frozen=True)
class PilotAnnotationResult:
    """Summary of one manual note update."""

    log_path: Path
    pilot_entry_id: str
    prediction_date: date
    prediction_mode: str


@dataclass(frozen=True)
class NewsFeatureResolution:
    """Resolved live Stage 7 row plus traceability details."""

    feature_row: pd.DataFrame
    metadata_path: Path
    metadata: dict[str, Any]
    raw_snapshot_path: Path | None
    linked_article_ids: list[str]
    top_event_counts: dict[str, int]
    synthetic: bool


@dataclass(frozen=True)
class PilotWeekPlanResult:
    """Summary of one generated pilot-week plan."""

    manifest_path: Path
    status_summary_path: Path
    slot_count: int


@dataclass(frozen=True)
class PilotWeekDueRunResult:
    """Summary of one due-slot execution pass."""

    manifest_path: Path
    status_summary_path: Path
    due_slot_count: int
    executed_slot_count: int
    dry_run: bool


@dataclass(frozen=True)
class PilotWeekOperatorResult:
    """Summary of one combined pilot-week operator pass."""

    manifest_path: Path
    status_summary_path: Path
    slot_count: int
    due_slot_count: int
    executed_slot_count: int
    updated_row_count: int
    unresolved_row_count: int
    dry_run: bool


def run_live_pilot(
    settings: AppSettings,
    *,
    prediction_mode: str,
    timestamp: datetime | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
    explain: bool = False,
    window_resolution_kind: str | None = None,
    window_resolution_reason: str | None = None,
) -> PilotRunResult:
    """Run one live pilot prediction and append it to the mode-specific log."""

    if prediction_mode not in PILOT_PREDICTION_MODES:
        raise LivePilotError(f"Unsupported pilot prediction mode: {prediction_mode}")

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    total_start = time.perf_counter()
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(runtime_settings, path_manager)
    write_settings_snapshot(runtime_settings, run_context.config_snapshot_path)
    logger = configure_logging(
        run_context,
        runtime_settings.run.log_level,
        logger_name="kubera.pilot",
    )

    calendar = build_market_calendar(runtime_settings.market)
    prediction_window = resolve_prediction_window(
        settings=runtime_settings,
        prediction_mode=prediction_mode,
        timestamp=timestamp,
        calendar=calendar,
    )
    if window_resolution_kind is None:
        window_resolution_kind = PILOT_WINDOW_RESOLUTION_OVERRIDE
    if window_resolution_reason is None:
        window_resolution_reason = "Used an explicit mode or timestamp override."
    pilot_log_path = path_manager.build_pilot_log_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        prediction_mode,
    )
    snapshot_path = path_manager.build_pilot_snapshot_path(
        runtime_settings.ticker.symbol,
        run_context.run_id,
        prediction_mode,
    )
    existing_log_frame = load_pilot_log_frame(pilot_log_path)

    pilot_row = build_empty_pilot_row()
    _apply_resolved_prediction_window_to_pilot_row(
        pilot_row,
        runtime_settings=runtime_settings,
        prediction_mode=prediction_mode,
        prediction_window=prediction_window,
        run_context=run_context,
        existing_log_frame=existing_log_frame,
    )
    pilot_row.update(
        {
            "ticker": runtime_settings.ticker.symbol,
            "exchange": runtime_settings.ticker.exchange,
            "prediction_mode": prediction_mode,
            "pilot_run_id": run_context.run_id,
            "pilot_timestamp_utc": prediction_window.timestamp_utc.isoformat(),
            "pilot_timestamp_market": prediction_window.timestamp_market.isoformat(),
            "news_cutoff_timestamp_utc": prediction_window.timestamp_utc.isoformat(),
            "status": PILOT_STATUS_FAILURE,
            "pilot_snapshot_path": str(snapshot_path),
            "actual_outcome_status": ACTUAL_STATUS_PENDING,
            "warning_codes_json": encode_json_cell([]),
            "linked_article_ids_json": encode_json_cell([]),
            "top_event_counts_json": encode_json_cell({}),
        }
    )

    warning_codes: list[str] = []
    stage_payloads: dict[str, Any] = {}
    stage2_metadata: dict[str, Any] | None = None
    stage5_metadata: dict[str, Any] | None = None
    stage6_metadata: dict[str, Any] | None = None
    stage7_metadata: dict[str, Any] | None = None
    failure_stage: str | None = None
    failure_message: str | None = None
    baseline_succeeded = False
    enhanced_succeeded = False
    historical_row: pd.DataFrame | None = None
    news_feature_resolution: NewsFeatureResolution | None = None

    try:
        stage2_start = time.perf_counter()
        try:
            for attempt in range(2):
                if attempt == 1:
                    calendar = build_market_calendar(runtime_settings.market)
                    prediction_window = resolve_prediction_window(
                        settings=runtime_settings,
                        prediction_mode=prediction_mode,
                        timestamp=timestamp,
                        calendar=calendar,
                    )
                    _apply_resolved_prediction_window_to_pilot_row(
                        pilot_row,
                        runtime_settings=runtime_settings,
                        prediction_mode=prediction_mode,
                        prediction_window=prediction_window,
                        run_context=run_context,
                        existing_log_frame=existing_log_frame,
                    )
                    pilot_row["news_cutoff_timestamp_utc"] = (
                        prediction_window.timestamp_utc.isoformat()
                    )
                market_result = fetch_historical_market_data(
                    runtime_settings,
                    end_date=prediction_window.historical_cutoff_date,
                )
                stage2_metadata = load_required_json(
                    market_result.metadata_path,
                    artifact_label="Stage 2 market-data metadata",
                )
                staged_market_frame = read_cleaned_market_data(market_result.cleaned_table_path)
                cutoff_market_frame = slice_market_window(
                    staged_market_frame,
                    start_date=date.min,
                    end_date=prediction_window.historical_cutoff_date,
                )
                validated_market_frame = validate_cleaned_market_data(
                    cutoff_market_frame,
                    ticker=runtime_settings.ticker.symbol,
                    exchange=runtime_settings.ticker.exchange,
                    feature_settings=runtime_settings.historical_features,
                    calendar=calendar,
                )
                latest_market_date = coerce_required_date(
                    validated_market_frame.iloc[-1]["date"],
                    field_label="latest Stage 2 market date",
                )
                if latest_market_date != prediction_window.historical_cutoff_date:
                    if (
                        prediction_mode == "after_close"
                        and latest_market_date < prediction_window.historical_cutoff_date
                    ):
                        requested_cutoff = prediction_window.historical_cutoff_date
                        prediction_window = live_prediction_window_after_close_for_session_date(
                            settings=runtime_settings,
                            calendar=calendar,
                            market_session_date=latest_market_date,
                        )
                        _apply_resolved_prediction_window_to_pilot_row(
                            pilot_row,
                            runtime_settings=runtime_settings,
                            prediction_mode=prediction_mode,
                            prediction_window=prediction_window,
                            run_context=run_context,
                            existing_log_frame=existing_log_frame,
                        )
                        pilot_row["pilot_timestamp_utc"] = (
                            prediction_window.timestamp_utc.isoformat()
                        )
                        pilot_row["pilot_timestamp_market"] = (
                            prediction_window.timestamp_market.isoformat()
                        )
                        pilot_row["news_cutoff_timestamp_utc"] = (
                            prediction_window.timestamp_utc.isoformat()
                        )
                        warning_codes.append("historical_cutoff_snapped_to_latest_bar")
                        logger.warning(
                            "Stage 2: historical_cutoff_snapped_to_latest_bar | "
                            "requested_cutoff=%s effective_cutoff=%s",
                            requested_cutoff.isoformat(),
                            prediction_window.historical_cutoff_date.isoformat(),
                        )
                    elif attempt == 0:
                        logger.warning(
                            "Stage 2: historical data cutoff mismatch; "
                            "refreshing exchange calendar and retrying once."
                        )
                        continue
                    else:
                        synced_as_of = load_exchange_closures_as_of(
                            runtime_settings.market.exchange_closures_path
                        )
                        raise LivePilotError(
                            format_live_pilot_cutoff_error(
                                calendar=calendar,
                                latest=latest_market_date,
                                cutoff=prediction_window.historical_cutoff_date,
                                synced_as_of=synced_as_of,
                            )
                        )
                historical_row = build_live_historical_feature_row(
                    validated_market_frame,
                    runtime_settings.historical_features,
                    calendar,
                    prediction_date=prediction_window.prediction_date,
                )
                pilot_row["historical_date"] = coerce_required_date(
                    historical_row.iloc[0]["date"],
                    field_label="live historical feature date",
                ).isoformat()
                pilot_row["historical_market_gap_flag"] = (
                    coerce_optional_int(historical_row.iloc[0].get("market_data_gap_flag")) or 0
                )
                pilot_row["historical_market_gap_count_5d"] = (
                    coerce_optional_int(historical_row.iloc[0].get("market_data_gap_count_5d")) or 0
                )
                pilot_row["stage2_cleaned_path"] = str(market_result.cleaned_table_path)
                pilot_row["stage2_metadata_path"] = str(market_result.metadata_path)
                pilot_row["stage2_run_id"] = stage2_metadata.get("run_id")
                stage_payloads["stage2"] = {
                    "cleaned_path": str(market_result.cleaned_table_path),
                    "metadata_path": str(market_result.metadata_path),
                    "run_id": stage2_metadata.get("run_id"),
                    "gap_filled_row_count": stage2_metadata.get("gap_filled_row_count"),
                    "max_recent_gap_count_5d": stage2_metadata.get("max_recent_gap_count_5d"),
                }
                break
        finally:
            pilot_row["stage2_duration_seconds"] = elapsed_seconds(stage2_start)
    except Exception as exc:
        failure_stage = "stage2"
        failure_message = sanitize_log_text(str(exc))

    if historical_row is not None:
        try:
            baseline_start = time.perf_counter()
            try:
                pilot_row.update(
                    predict_live_baseline(runtime_settings, path_manager, historical_row)
                )
                baseline_succeeded = True
                stage_payloads["baseline"] = {
                    "model_path": pilot_row["baseline_model_path"],
                    "metadata_path": pilot_row["baseline_model_metadata_path"],
                    "run_id": pilot_row["baseline_model_run_id"],
                }
            finally:
                pilot_row["baseline_duration_seconds"] = elapsed_seconds(baseline_start)
        except Exception as exc:
            if failure_stage is None:
                failure_stage = "baseline"
                failure_message = sanitize_log_text(str(exc))

    if historical_row is not None:
        try:
            stage5_start = time.perf_counter()
            try:
                news_result = fetch_company_news(
                    runtime_settings,
                    published_before=prediction_window.timestamp_utc,
                )
                stage5_metadata = load_required_json(
                    news_result.metadata_path,
                    artifact_label="Stage 5 news metadata",
                )
                warning_codes.extend(prefix_metadata_warnings(stage5_metadata, "stage5"))
                pilot_row["stage5_processed_news_path"] = str(news_result.cleaned_table_path)
                pilot_row["stage5_metadata_path"] = str(news_result.metadata_path)
                pilot_row["stage5_run_id"] = stage5_metadata.get("run_id")
                pilot_row["stage5_provider_request_count"] = stage5_metadata.get(
                    "provider_request_count",
                    pd.NA,
                )
                pilot_row["stage5_provider_request_retry_count"] = stage5_metadata.get(
                    "provider_request_retry_count",
                    pd.NA,
                )
                pilot_row["stage5_article_fetch_attempt_count"] = stage5_metadata.get(
                    "article_fetch_attempt_count",
                    pd.NA,
                )
                pilot_row["stage5_article_fetch_retry_count"] = stage5_metadata.get(
                    "article_fetch_retry_count",
                    pd.NA,
                )
                stage_payloads["stage5"] = {
                    "processed_news_path": str(news_result.cleaned_table_path),
                    "metadata_path": str(news_result.metadata_path),
                    "run_id": stage5_metadata.get("run_id"),
                    "retry_summary": {
                        "provider_request_count": stage5_metadata.get("provider_request_count"),
                        "provider_request_retry_count": stage5_metadata.get(
                            "provider_request_retry_count"
                        ),
                        "article_fetch_attempt_count": stage5_metadata.get(
                            "article_fetch_attempt_count"
                        ),
                        "article_fetch_retry_count": stage5_metadata.get(
                            "article_fetch_retry_count"
                        ),
                    },
                }
            finally:
                pilot_row["stage5_duration_seconds"] = elapsed_seconds(stage5_start)

            stage6_start = time.perf_counter()
            try:
                extraction_result = extract_news(runtime_settings, pilot_extraction=True)
                stage6_metadata = load_required_json(
                    extraction_result.metadata_path,
                    artifact_label="Stage 6 extraction metadata",
                )
                warning_codes.extend(prefix_metadata_warnings(stage6_metadata, "stage6"))
                pilot_row["stage6_extraction_path"] = str(extraction_result.extraction_table_path)
                pilot_row["stage6_metadata_path"] = str(extraction_result.metadata_path)
                pilot_row["stage6_failure_log_path"] = str(extraction_result.failure_log_path)
                pilot_row["stage6_run_id"] = stage6_metadata.get("run_id")
                pilot_row["stage6_provider_request_count"] = stage6_metadata.get(
                    "provider_request_count",
                    pd.NA,
                )
                pilot_row["stage6_retry_count"] = stage6_metadata.get("retry_count", pd.NA)
                stage_payloads["stage6"] = {
                    "extraction_path": str(extraction_result.extraction_table_path),
                    "metadata_path": str(extraction_result.metadata_path),
                    "failure_log_path": str(extraction_result.failure_log_path),
                    "run_id": stage6_metadata.get("run_id"),
                    "retry_summary": {
                        "provider_request_count": stage6_metadata.get("provider_request_count"),
                        "retry_count": stage6_metadata.get("retry_count"),
                    },
                }
            finally:
                pilot_row["stage6_duration_seconds"] = elapsed_seconds(stage6_start)

            stage7_start = time.perf_counter()
            try:
                news_feature_result = build_news_features(
                    runtime_settings,
                    target_end_date=prediction_window.prediction_date,
                )
                stage7_metadata = load_required_json(
                    news_feature_result.metadata_path,
                    artifact_label="Stage 7 news-feature metadata",
                )
                warning_codes.extend(prefix_metadata_warnings(stage7_metadata, "stage7"))
                news_feature_resolution = resolve_live_news_feature_row(
                    settings=runtime_settings,
                    path_manager=path_manager,
                    prediction_mode=prediction_mode,
                    prediction_date=prediction_window.prediction_date,
                )
                fallback_heavy_flag = bool(
                    float(
                        news_feature_resolution.feature_row.iloc[0]["news_fallback_article_ratio"]
                    )
                    >= runtime_settings.pilot.fallback_heavy_ratio_threshold
                    and float(news_feature_resolution.feature_row.iloc[0]["news_article_count"]) > 0
                )
                nf0 = news_feature_resolution.feature_row.iloc[0]
                pilot_row.update(
                    {
                        "news_article_count": int(
                            news_feature_resolution.feature_row.iloc[0]["news_article_count"]
                        ),
                        "news_warning_article_count": int(
                            news_feature_resolution.feature_row.iloc[0][
                                "news_warning_article_count"
                            ]
                        ),
                        "news_fallback_article_ratio": float(
                            news_feature_resolution.feature_row.iloc[0][
                                "news_fallback_article_ratio"
                            ]
                        ),
                        "news_avg_confidence": float(
                            news_feature_resolution.feature_row.iloc[0]["news_avg_confidence"]
                        ),
                        "has_fresh_news": int(news_feature_resolution.feature_row.iloc[0]["has_fresh_news"]),
                        "is_carried_forward": int(news_feature_resolution.feature_row.iloc[0]["is_carried_forward"]),
                        "is_fallback_heavy": int(news_feature_resolution.feature_row.iloc[0]["is_fallback_heavy"]),
                        "news_signal_state": (
                            clean_string(
                                news_feature_resolution.feature_row.iloc[0].get(
                                    NEWS_SIGNAL_STATE_COLUMN
                                )
                            )
                            or determine_news_signal_state(
                                news_feature_resolution.feature_row.iloc[0].to_dict()
                            )
                        ),
                        "fallback_heavy_flag": fallback_heavy_flag,
                        "news_feature_synthetic_flag": news_feature_resolution.synthetic,
                        "linked_article_ids_json": encode_json_cell(
                            news_feature_resolution.linked_article_ids
                        ),
                        "top_event_counts_json": encode_json_cell(
                            news_feature_resolution.top_event_counts
                        ),
                        "stage7_feature_path": str(news_feature_result.feature_table_path),
                        "stage7_metadata_path": str(news_feature_result.metadata_path),
                        "stage7_raw_snapshot_path": (
                            str(news_feature_resolution.raw_snapshot_path)
                            if news_feature_resolution.raw_snapshot_path
                            else pd.NA
                        ),
                        "stage7_run_id": stage7_metadata.get("run_id"),
                        "news_volume_3d": _coalesce_news_feature_float(nf0, "news_volume_3d"),
                        "news_sentiment_3d": _coalesce_news_feature_float(nf0, "news_sentiment_3d"),
                        "news_sentiment_dispersion_1d": _coalesce_news_feature_float(
                            nf0, "news_sentiment_dispersion_1d"
                        ),
                        "news_directional_agreement_rate": _coalesce_news_feature_float(
                            nf0, "news_directional_agreement_rate"
                        ),
                    }
                )
                if news_feature_resolution.synthetic:
                    warning_codes.append("zero_news_row_synthesized")
                if float(news_feature_resolution.feature_row.iloc[0]["news_article_count"]) == 0:
                    warning_codes.append("zero_news_available")
                if fallback_heavy_flag:
                    warning_codes.append("fallback_heavy")
                stage_payloads["stage7"] = {
                    "feature_path": str(news_feature_result.feature_table_path),
                    "metadata_path": str(news_feature_result.metadata_path),
                    "raw_snapshot_path": (
                        str(news_feature_resolution.raw_snapshot_path)
                        if news_feature_resolution.raw_snapshot_path
                        else None
                    ),
                    "run_id": stage7_metadata.get("run_id"),
                    "news_signal_state": clean_string(pilot_row.get("news_signal_state")),
                }
            finally:
                pilot_row["stage7_duration_seconds"] = elapsed_seconds(stage7_start)
        except Exception as exc:
            if failure_stage is None:
                failure_stage = determine_stage_failure_label(pilot_row)
                failure_message = sanitize_log_text(str(exc))

    if news_feature_resolution is not None:
        data_quality = build_run_data_quality_payload(
            pilot_row=pilot_row,
            stage2_metadata=stage2_metadata,
            stage5_metadata=stage5_metadata,
            stage6_metadata=stage6_metadata,
            stage7_metadata=stage7_metadata,
            news_feature_row=news_feature_resolution.feature_row.iloc[0].to_dict(),
        )
        pilot_row["data_quality_score"] = data_quality["score"]
        pilot_row["data_quality_grade"] = data_quality["grade"]
        pilot_row["data_quality_reasons_json"] = encode_json_cell(data_quality["reasons"])
        pilot_row["data_quality_components_json"] = encode_json_cell(data_quality["components"])

    if historical_row is not None and news_feature_resolution is not None:
        try:
            enhanced_start = time.perf_counter()
            try:
                pilot_row.update(
                    predict_live_enhanced(
                        runtime_settings,
                        path_manager,
                        prediction_mode=prediction_mode,
                        historical_row=historical_row,
                        news_feature_row=news_feature_resolution.feature_row,
                    )
                )
                enhanced_succeeded = True
                stage_payloads["enhanced"] = {
                    "model_path": pilot_row["enhanced_model_path"],
                    "metadata_path": pilot_row["enhanced_model_metadata_path"],
                    "run_id": pilot_row["enhanced_model_run_id"],
                }
            finally:
                pilot_row["enhanced_duration_seconds"] = elapsed_seconds(enhanced_start)
        except Exception as exc:
            if failure_stage is None:
                failure_stage = "enhanced"
                failure_message = sanitize_log_text(str(exc))

    if baseline_succeeded and enhanced_succeeded:
        pilot_row["disagreement_flag"] = bool(
            int(pilot_row["baseline_predicted_next_day_direction"])
            != int(pilot_row["enhanced_predicted_next_day_direction"])
        )
        news_weight = compute_news_context_weight(
            news_article_count=pilot_row["news_article_count"],
            news_avg_confidence=pilot_row["news_avg_confidence"],
            has_fresh_news=pilot_row["has_fresh_news"],
            is_fallback_heavy=pilot_row["is_fallback_heavy"],
            is_carried_forward=pilot_row["is_carried_forward"],
        )
        pilot_row["news_context_weight"] = news_weight
        baseline_raw_prob = coerce_optional_float(
            pilot_row.get("baseline_raw_predicted_probability_up")
        )
        if baseline_raw_prob is None:
            baseline_raw_prob = float(pilot_row["baseline_predicted_probability_up"])
        enhanced_raw_prob = coerce_optional_float(
            pilot_row.get("enhanced_raw_predicted_probability_up")
        )
        if enhanced_raw_prob is None:
            enhanced_raw_prob = float(pilot_row["enhanced_predicted_probability_up"])
        baseline_calibrated_prob = coerce_optional_float(
            pilot_row.get("baseline_calibrated_predicted_probability_up")
        )
        if baseline_calibrated_prob is None:
            baseline_calibrated_prob = float(pilot_row["baseline_predicted_probability_up"])
        enhanced_calibrated_prob = coerce_optional_float(
            pilot_row.get("enhanced_calibrated_predicted_probability_up")
        )
        if enhanced_calibrated_prob is None:
            enhanced_calibrated_prob = float(pilot_row["enhanced_predicted_probability_up"])
        blended_raw_prob = blend_probabilities(
            baseline_prob=baseline_raw_prob,
            enhanced_prob=enhanced_raw_prob,
            news_weight=news_weight,
        )
        blended_calibrated_prob = blend_probabilities(
            baseline_prob=baseline_calibrated_prob,
            enhanced_prob=enhanced_calibrated_prob,
            news_weight=news_weight,
        )
        pilot_row["blended_raw_predicted_probability_up"] = blended_raw_prob
        pilot_row["blended_calibrated_predicted_probability_up"] = blended_calibrated_prob
        pilot_row["blended_predicted_probability_up"] = blended_calibrated_prob
        baseline_threshold = 0.5
        try:
            baseline_metadata = load_required_json(
                Path(pilot_row["baseline_model_metadata_path"]),
                artifact_label="Baseline model metadata",
            )
            baseline_threshold = baseline_metadata.get("classification_threshold", 0.5)
        except Exception:
            pass
        model_suggested_direction = int(blended_calibrated_prob >= baseline_threshold)
        selective_decision = resolve_selective_prediction(
            probability_up=blended_calibrated_prob,
            classification_threshold=baseline_threshold,
            low_conviction_threshold=runtime_settings.pilot.abstain_low_conviction_threshold,
            news_signal_state=clean_string(pilot_row.get("news_signal_state")),
            data_quality_score=coerce_optional_float(pilot_row.get("data_quality_score")),
            data_quality_floor=runtime_settings.pilot.abstain_data_quality_floor,
            carried_forward_margin_penalty=(
                runtime_settings.pilot.abstain_carried_forward_margin_penalty
            ),
            degraded_margin_penalty=runtime_settings.pilot.abstain_degraded_margin_penalty,
        )
        pilot_row["blended_predicted_next_day_direction"] = model_suggested_direction
        pilot_row["selected_action"] = selective_decision.action
        pilot_row["abstain_flag"] = selective_decision.abstain
        pilot_row["selective_probability_margin"] = selective_decision.probability_margin
        pilot_row["selective_required_margin"] = selective_decision.required_margin
        pilot_row["abstain_reason_codes_json"] = encode_json_cell(
            list(selective_decision.reasons)
        )
        pilot_row["status"] = (
            PILOT_STATUS_ABSTAIN if selective_decision.abstain else PILOT_STATUS_SUCCESS
        )
    elif baseline_succeeded or enhanced_succeeded:
        pilot_row["status"] = PILOT_STATUS_PARTIAL_FAILURE
    else:
        pilot_row["status"] = PILOT_STATUS_FAILURE

    if news_feature_resolution is not None and clean_string(pilot_row.get("stage6_extraction_path")):
        recent_news_summaries = fetch_recent_news_summaries(
            news_feature_resolution.linked_article_ids,
            Path(str(pilot_row["stage6_extraction_path"])),
        )
        pilot_row["recent_news_summaries_json"] = encode_json_cell(recent_news_summaries)
    else:
        pilot_row["recent_news_summaries_json"] = encode_json_cell([])

    pilot_row["failure_stage"] = failure_stage if failure_stage is not None else pd.NA
    pilot_row["failure_message"] = failure_message if failure_message is not None else pd.NA
    pilot_row["total_duration_seconds"] = elapsed_seconds(total_start)
    runtime_warning_message = build_runtime_warning_message(
        total_duration_seconds=coerce_optional_float(pilot_row.get("total_duration_seconds")),
        runtime_warning_seconds=runtime_settings.pilot.runtime_warning_seconds,
    )
    if runtime_warning_message is not None:
        warning_codes.append("runtime_warning")
        pilot_row["runtime_warning_flag"] = True
        pilot_row["runtime_warning_message"] = runtime_warning_message
    else:
        pilot_row["runtime_warning_flag"] = False
        pilot_row["runtime_warning_message"] = pd.NA
    pilot_row["warning_codes_json"] = encode_json_cell(sorted(set(warning_codes)))

    append_pilot_row(pilot_log_path, pilot_row)
    prior_prediction_outcome = resolve_prior_prediction_outcome(
        log_path=pilot_log_path,
        prediction_date=prediction_window.prediction_date,
        market=runtime_settings.market,
    )
    snapshot_payload = build_pilot_snapshot_payload(
        settings=runtime_settings,
        pilot_entry_id=str(pilot_row["pilot_entry_id"]),
        prediction_key=str(pilot_row["prediction_key"]),
        prediction_window=prediction_window,
        window_resolution_kind=window_resolution_kind,
        window_resolution_reason=window_resolution_reason,
        pilot_row=pilot_row,
        stage_payloads=stage_payloads,
        prior_prediction_outcome=prior_prediction_outcome,
    )
    write_json_file(snapshot_path, snapshot_payload)
    logger.info(
        "Live pilot row recorded | ticker=%s | exchange=%s | mode=%s | prediction_date=%s | status=%s | runtime_warning=%s | log=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        prediction_mode,
        prediction_window.prediction_date,
        pilot_row["status"],
        bool(coerce_optional_bool(pilot_row.get("runtime_warning_flag"))),
        pilot_log_path,
    )
    print(format_pilot_summary(snapshot_payload))
    if explain:
        explanation_message = resolve_pilot_explanation_output(
            settings=runtime_settings,
            snapshot_payload=snapshot_payload,
        )
        print()
        print(explanation_message)
    return PilotRunResult(
        log_path=pilot_log_path,
        snapshot_path=snapshot_path,
        pilot_entry_id=str(pilot_row["pilot_entry_id"]),
        status=str(pilot_row["status"]),
        market_session_date=prediction_window.market_session_date,
        prediction_date=prediction_window.prediction_date,
        prediction_mode=prediction_mode,
        historical_cutoff_date=prediction_window.historical_cutoff_date,
        window_resolution_kind=window_resolution_kind,
        window_resolution_reason=window_resolution_reason,
    )


def backfill_pending_pilot_actuals_for_cli(
    settings: AppSettings,
    *,
    prediction_mode: str,
    current_prediction_date: date,
    historical_cutoff_date: date,
    as_of: date | None = None,
    limit: int | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
) -> PilotPendingBackfillResult:
    """Backfill pending ``actual_*`` columns for eligible prior rows in one mode-specific pilot log."""

    if prediction_mode not in PILOT_PREDICTION_MODES:
        raise LivePilotError(f"Unsupported pilot prediction mode: {prediction_mode}")

    effective_as_of = as_of or historical_cutoff_date
    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    log_path = path_manager.build_pilot_log_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        prediction_mode,
    )
    if not log_path.exists():
        return PilotPendingBackfillResult(
            updated_row_count=0,
            unresolved_row_count=0,
            error_count=0,
            effective_as_of=effective_as_of,
            prediction_dates_attempted=(),
        )

    log_frame = load_pilot_log_frame(log_path)
    if log_frame.empty:
        return PilotPendingBackfillResult(
            updated_row_count=0,
            unresolved_row_count=0,
            error_count=0,
            effective_as_of=effective_as_of,
            prediction_dates_attempted=(),
        )

    eligible_dates: set[date] = set()
    for _, row in log_frame.iterrows():
        pd_raw = row.get("prediction_date")
        if pd_raw is None or pd.isna(pd_raw):
            continue
        pd_str = clean_string(str(pd_raw).strip())
        if not pd_str:
            continue
        try:
            pd_d = date.fromisoformat(pd_str[:10])
        except ValueError:
            continue
        if pd_d >= current_prediction_date:
            continue
        if pd_d > effective_as_of:
            continue
        status = clean_string(row.get("actual_outcome_status")) or ACTUAL_STATUS_PENDING
        if status == ACTUAL_STATUS_BACKFILLED:
            continue
        eligible_dates.add(pd_d)

    sorted_dates = sorted(eligible_dates, reverse=True)
    if limit is not None and limit > 0:
        sorted_dates = sorted_dates[:limit]

    updated_total = 0
    unresolved_total = 0
    errors = 0
    attempted: list[date] = []

    for pred_d in sorted_dates:
        attempted.append(pred_d)
        try:
            one = backfill_pilot_actuals(
                settings,
                prediction_date=pred_d,
                prediction_mode=prediction_mode,
                ticker=ticker,
                exchange=exchange,
            )
        except Exception as exc:
            errors += 1
            logger.warning(
                "Pending pilot backfill failed for prediction_date=%s mode=%s: %s",
                pred_d.isoformat(),
                prediction_mode,
                sanitize_log_text(str(exc)),
            )
            continue
        updated_total += one.updated_row_count
        unresolved_total += one.unresolved_row_count

    return PilotPendingBackfillResult(
        updated_row_count=updated_total,
        unresolved_row_count=unresolved_total,
        error_count=errors,
        effective_as_of=effective_as_of,
        prediction_dates_attempted=tuple(attempted),
    )


def backfill_pilot_actuals(
    settings: AppSettings,
    *,
    prediction_date: date,
    prediction_mode: str | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
) -> PilotBackfillResult:
    """Backfill actual next-day outcomes for matching pilot rows."""

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    target_modes = (prediction_mode,) if prediction_mode is not None else PILOT_PREDICTION_MODES
    log_paths = tuple(
        path_manager.build_pilot_log_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            mode,
        )
        for mode in target_modes
        if path_manager.build_pilot_log_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            mode,
        ).exists()
    )
    if not log_paths:
        raise LivePilotError("No pilot log exists for the requested prediction mode selection.")

    market_result = fetch_historical_market_data(runtime_settings, end_date=prediction_date)
    cleaned_market = pd.read_csv(market_result.cleaned_table_path)
    updated_row_count = 0
    unresolved_row_count = 0
    backfilled_at = datetime.now(timezone.utc).isoformat()

    for log_path in log_paths:
        log_frame = load_pilot_log_frame(log_path)
        matching_indexes = log_frame.index[
            log_frame["prediction_date"].astype(str) == prediction_date.isoformat()
        ].tolist()
        for row_index in matching_indexes:
            actual_status = clean_string(log_frame.at[row_index, "actual_outcome_status"])
            if actual_status == ACTUAL_STATUS_BACKFILLED:
                continue

            historical_date_value = clean_string(log_frame.at[row_index, "historical_date"])
            if historical_date_value is None:
                log_frame.at[row_index, "actual_outcome_status"] = ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE
                log_frame.at[row_index, "actual_backfill_error"] = "missing_historical_date"
                unresolved_row_count += 1
                continue

            historical_close = lookup_close_for_date(cleaned_market, historical_date_value)
            prediction_close = lookup_close_for_date(cleaned_market, prediction_date.isoformat())
            if historical_close is None or prediction_close is None:
                log_frame.at[row_index, "actual_outcome_status"] = ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE
                log_frame.at[row_index, "actual_backfill_error"] = "prediction_window_market_data_unavailable"
                unresolved_row_count += 1
                continue

            actual_direction = int(prediction_close > historical_close)
            log_frame.at[row_index, "actual_historical_close"] = float(historical_close)
            log_frame.at[row_index, "actual_prediction_close"] = float(prediction_close)
            log_frame.at[row_index, "actual_next_day_direction"] = actual_direction
            log_frame.at[row_index, "actual_outcome_status"] = ACTUAL_STATUS_BACKFILLED
            log_frame.at[row_index, "actual_outcome_backfilled_at_utc"] = backfilled_at
            log_frame.at[row_index, "actual_backfill_error"] = pd.NA
            baseline_prediction = coerce_optional_int(
                log_frame.at[row_index, "baseline_predicted_next_day_direction"]
            )
            enhanced_prediction = coerce_optional_int(
                log_frame.at[row_index, "enhanced_predicted_next_day_direction"]
            )
            if baseline_prediction is not None:
                log_frame.at[row_index, "baseline_correct"] = bool(
                    baseline_prediction == actual_direction
                )
            if enhanced_prediction is not None:
                log_frame.at[row_index, "enhanced_correct"] = bool(
                    enhanced_prediction == actual_direction
                )
            updated_row_count += 1

        save_pilot_log_frame(log_path, log_frame)

    return PilotBackfillResult(
        updated_row_count=updated_row_count,
        unresolved_row_count=unresolved_row_count,
        log_paths=log_paths,
    )


def annotate_pilot_entry(
    settings: AppSettings,
    *,
    prediction_mode: str,
    prediction_date: date,
    news_quality_note: str | None = None,
    market_shock_note: str | None = None,
    source_outage_note: str | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
) -> PilotAnnotationResult:
    """Update manual review fields for the latest matching pilot row."""

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    log_path = path_manager.build_pilot_log_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        prediction_mode,
    )
    log_frame = load_pilot_log_frame(log_path)
    if log_frame.empty:
        raise LivePilotError(f"Pilot log does not exist or is empty: {log_path}")

    matching_frame = log_frame.loc[
        log_frame["prediction_date"].astype(str) == prediction_date.isoformat()
    ].copy()
    if matching_frame.empty:
        raise LivePilotError("No pilot log row matches the requested prediction date.")

    matching_frame = matching_frame.sort_values(
        by=["pilot_timestamp_utc", "pilot_entry_id"],
        ascending=[True, True],
        na_position="last",
    )
    row_index = int(matching_frame.index[-1])
    if news_quality_note is not None:
        log_frame.at[row_index, "news_quality_note"] = news_quality_note
    if market_shock_note is not None:
        log_frame.at[row_index, "market_shock_note"] = market_shock_note
    if source_outage_note is not None:
        log_frame.at[row_index, "source_outage_note"] = source_outage_note
    log_frame.at[row_index, "manual_notes_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    pilot_entry_id = str(log_frame.at[row_index, "pilot_entry_id"])
    save_pilot_log_frame(log_path, log_frame)
    return PilotAnnotationResult(
        log_path=log_path,
        pilot_entry_id=pilot_entry_id,
        prediction_date=prediction_date,
        prediction_mode=prediction_mode,
    )


def plan_pilot_week(
    settings: AppSettings,
    *,
    pilot_start_date: date,
    pilot_end_date: date,
    ticker: str | None = None,
    exchange: str | None = None,
) -> PilotWeekPlanResult:
    """Build the deterministic one-week pilot manifest and initial status summary."""

    if pilot_end_date < pilot_start_date:
        raise LivePilotError("Pilot end date must be on or after the pilot start date.")

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    calendar = build_market_calendar(runtime_settings.market)
    trading_dates = build_pilot_week_trading_dates(
        pilot_start_date=pilot_start_date,
        pilot_end_date=pilot_end_date,
        calendar=calendar,
    )
    manifest_path = path_manager.build_pilot_week_manifest_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        pilot_start_date,
        pilot_end_date,
    )
    status_summary_path = path_manager.build_pilot_week_status_summary_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        pilot_start_date,
        pilot_end_date,
    )
    slots: list[dict[str, Any]] = []
    for market_session_date in trading_dates:
        for prediction_mode in PILOT_PREDICTION_MODES:
            scheduled_market = build_pilot_slot_market_timestamp(
                runtime_settings=runtime_settings,
                market_session_date=market_session_date,
                prediction_mode=prediction_mode,
            )
            scheduled_utc = scheduled_market.astimezone(timezone.utc)
            prediction_window = resolve_prediction_window(
                settings=runtime_settings,
                prediction_mode=prediction_mode,
                timestamp=scheduled_utc,
                calendar=calendar,
            )
            slot_id = build_pilot_week_slot_id(
                market_session_date=market_session_date,
                prediction_mode=prediction_mode,
            )
            slot_status_path = path_manager.build_pilot_week_slot_status_path(
                runtime_settings.ticker.symbol,
                runtime_settings.ticker.exchange,
                pilot_start_date,
                pilot_end_date,
                slot_id,
            )
            slots.append(
                {
                    "slot_id": slot_id,
                    "market_session_date": market_session_date.isoformat(),
                    "prediction_mode": prediction_mode,
                    "scheduled_timestamp_market": scheduled_market.isoformat(),
                    "scheduled_timestamp_utc": scheduled_utc.isoformat(),
                    "prediction_date": prediction_window.prediction_date.isoformat(),
                    "prediction_key": build_prediction_key(
                        ticker=runtime_settings.ticker.symbol,
                        exchange=runtime_settings.ticker.exchange,
                        prediction_mode=prediction_mode,
                        prediction_date=prediction_window.prediction_date,
                    ),
                    "pilot_log_path": str(
                        path_manager.build_pilot_log_path(
                            runtime_settings.ticker.symbol,
                            runtime_settings.ticker.exchange,
                            prediction_mode,
                        )
                    ),
                    "slot_status_path": str(slot_status_path),
                }
            )

    manifest_payload = {
        "ticker": runtime_settings.ticker.symbol,
        "exchange": runtime_settings.ticker.exchange,
        "pilot_window": {
            "start_date": pilot_start_date.isoformat(),
            "end_date": pilot_end_date.isoformat(),
            "expected_market_session_dates": [value.isoformat() for value in trading_dates],
            "expected_market_session_count": int(len(trading_dates)),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "status_summary_path": str(status_summary_path),
        "slots": slots,
    }
    write_json_file(manifest_path, manifest_payload)
    write_json_file(
        status_summary_path,
        build_pilot_week_status_summary(
            manifest_payload=manifest_payload,
            generated_at_utc=datetime.now(timezone.utc),
        ),
    )
    return PilotWeekPlanResult(
        manifest_path=manifest_path,
        status_summary_path=status_summary_path,
        slot_count=len(slots),
    )


def run_due_pilot_week(
    settings: AppSettings,
    *,
    plan_path: str | Path,
    now: datetime | None = None,
    dry_run: bool = False,
) -> PilotWeekDueRunResult:
    """Execute due, incomplete pilot-week slots from a saved manifest."""

    manifest_payload = load_pilot_week_manifest(plan_path)
    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=clean_string(manifest_payload.get("ticker")),
        exchange=clean_string(manifest_payload.get("exchange")),
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    status_summary_path = Path(str(manifest_payload["status_summary_path"]))
    now_utc = normalize_timestamp(now)
    due_slots = [
        slot
        for slot in manifest_payload.get("slots", [])
        if slot_is_due(slot, now_utc=now_utc) and not pilot_week_slot_status_exists(slot)
    ]

    executed_slot_count = 0
    if not dry_run:
        for slot in due_slots:
            slot_status_path = Path(str(slot["slot_status_path"]))
            executed_at_utc = datetime.now(timezone.utc)
            try:
                result = run_live_pilot(
                    runtime_settings,
                    prediction_mode=str(slot["prediction_mode"]),
                    timestamp=parse_timestamp(str(slot["scheduled_timestamp_utc"])),
                    ticker=runtime_settings.ticker.symbol,
                    exchange=runtime_settings.ticker.exchange,
                )
                slot_status = map_pilot_run_status_to_week_status(result.status)
                status_payload = {
                    "slot_id": str(slot["slot_id"]),
                    "slot_status": slot_status,
                    "executed_at_utc": executed_at_utc.isoformat(),
                    "scheduled_timestamp_utc": str(slot["scheduled_timestamp_utc"]),
                    "prediction_mode": str(slot["prediction_mode"]),
                    "market_session_date": str(slot["market_session_date"]),
                    "prediction_date": result.prediction_date.isoformat(),
                    "pilot_entry_id": result.pilot_entry_id,
                    "pilot_log_path": str(result.log_path),
                    "pilot_snapshot_path": str(result.snapshot_path),
                }
            except Exception as exc:
                slot_status = PILOT_WEEK_STATUS_FAILURE
                status_payload = {
                    "slot_id": str(slot["slot_id"]),
                    "slot_status": slot_status,
                    "executed_at_utc": executed_at_utc.isoformat(),
                    "scheduled_timestamp_utc": str(slot["scheduled_timestamp_utc"]),
                    "prediction_mode": str(slot["prediction_mode"]),
                    "market_session_date": str(slot["market_session_date"]),
                    "prediction_date": clean_string(slot.get("prediction_date")),
                    "error_message": sanitize_log_text(str(exc)),
                }
            write_json_file(slot_status_path, status_payload)
            executed_slot_count += 1

        write_json_file(
            status_summary_path,
            build_pilot_week_status_summary(
                manifest_payload=manifest_payload,
                generated_at_utc=datetime.now(timezone.utc),
            ),
        )

    return PilotWeekDueRunResult(
        manifest_path=Path(str(manifest_payload["manifest_path"])),
        status_summary_path=status_summary_path,
        due_slot_count=len(due_slots),
        executed_slot_count=executed_slot_count,
        dry_run=dry_run,
    )


def backfill_due_pilot_week(
    settings: AppSettings,
    *,
    pilot_start_date: date,
    pilot_end_date: date,
    as_of: date | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
) -> PilotBackfillResult:
    """Backfill all eligible pilot rows across the requested pilot window."""

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    cutoff_date = as_of or datetime.now(timezone.utc).date()
    updated_row_count = 0
    unresolved_row_count = 0
    collected_log_paths: set[Path] = set()

    for prediction_mode in PILOT_PREDICTION_MODES:
        log_path = path_manager.build_pilot_log_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        if not log_path.exists():
            continue
        log_frame = load_pilot_log_frame(log_path)
        if log_frame.empty:
            continue
        pending_prediction_dates = sorted(
            {
                str(value)
                for value in log_frame.loc[
                    log_frame["market_session_date"].astype(str).between(
                        pilot_start_date.isoformat(),
                        pilot_end_date.isoformat(),
                    )
                    & (log_frame["actual_outcome_status"].astype(str) != ACTUAL_STATUS_BACKFILLED)
                    & (log_frame["prediction_date"].astype(str) <= cutoff_date.isoformat()),
                    "prediction_date",
                ].dropna()
            }
        )
        for prediction_date_value in pending_prediction_dates:
            result = backfill_pilot_actuals(
                runtime_settings,
                prediction_date=date.fromisoformat(prediction_date_value),
                prediction_mode=prediction_mode,
            )
            updated_row_count += result.updated_row_count
            unresolved_row_count += result.unresolved_row_count
            collected_log_paths.update(result.log_paths)

    return PilotBackfillResult(
        updated_row_count=updated_row_count,
        unresolved_row_count=unresolved_row_count,
        log_paths=tuple(sorted(collected_log_paths)),
    )


def operate_pilot_week(
    settings: AppSettings,
    *,
    pilot_start_date: date,
    pilot_end_date: date,
    now: datetime | None = None,
    as_of: date | None = None,
    dry_run: bool = False,
    ticker: str | None = None,
    exchange: str | None = None,
) -> PilotWeekOperatorResult:
    """Plan the week when needed, run due slots, and backfill eligible outcomes."""

    plan_result = plan_pilot_week(
        settings,
        pilot_start_date=pilot_start_date,
        pilot_end_date=pilot_end_date,
        ticker=ticker,
        exchange=exchange,
    )
    due_result = run_due_pilot_week(
        settings,
        plan_path=plan_result.manifest_path,
        now=now,
        dry_run=dry_run,
    )
    if dry_run:
        backfill_result = PilotBackfillResult(
            updated_row_count=0,
            unresolved_row_count=0,
            log_paths=(),
        )
    else:
        backfill_result = backfill_due_pilot_week(
            settings,
            pilot_start_date=pilot_start_date,
            pilot_end_date=pilot_end_date,
            as_of=as_of,
            ticker=ticker,
            exchange=exchange,
        )
    return PilotWeekOperatorResult(
        manifest_path=plan_result.manifest_path,
        status_summary_path=due_result.status_summary_path,
        slot_count=plan_result.slot_count,
        due_slot_count=due_result.due_slot_count,
        executed_slot_count=due_result.executed_slot_count,
        updated_row_count=backfill_result.updated_row_count,
        unresolved_row_count=backfill_result.unresolved_row_count,
        dry_run=dry_run,
    )


def resolve_prediction_window(
    *,
    settings: AppSettings,
    prediction_mode: str,
    timestamp: datetime | None,
    calendar: Any,
) -> LivePredictionWindow:
    """Resolve the live pilot timing cutoffs for one mode."""

    timestamp_utc = normalize_timestamp(timestamp)
    timestamp_market = utc_to_market_time(timestamp_utc, settings.market)
    raw_session_date = timestamp_market.date()
    raw_is_trading_day = calendar.is_trading_day(raw_session_date)

    if prediction_mode == "pre_market":
        if raw_is_trading_day and not is_pre_market(timestamp_market, settings.market):
            raise LivePilotError("Pre-market pilot runs must use a timestamp before the market open.")
        market_session_date = (
            raw_session_date
            if raw_is_trading_day
            else first_trading_day_on_or_after(raw_session_date, calendar)
        )
        historical_cutoff_date = calendar.previous_trading_day(market_session_date)
        prediction_date = market_session_date
    elif prediction_mode == "after_close":
        if not raw_is_trading_day:
            raise LivePilotError(
                "After-close pilot runs must use a timestamp on a trading day."
            )
        if not is_after_close(timestamp_market, settings.market):
            raise LivePilotError("After-close pilot runs must use a timestamp at or after the market close.")
        market_session_date = raw_session_date
        historical_cutoff_date = market_session_date
        prediction_date = calendar.next_trading_day(market_session_date)
    else:
        raise LivePilotError(f"Unsupported pilot prediction mode: {prediction_mode}")

    return LivePredictionWindow(
        prediction_mode=prediction_mode,
        timestamp_utc=timestamp_utc,
        timestamp_market=timestamp_market,
        market_session_date=market_session_date,
        historical_cutoff_date=historical_cutoff_date,
        prediction_date=prediction_date,
    )


def resolve_default_live_window(
    *,
    settings: AppSettings,
    timestamp: datetime | None,
    calendar: Any,
) -> LiveWindowResolution:
    """Resolve the default consumer-facing live window from the market phase."""

    timestamp_utc = normalize_timestamp(timestamp)
    timestamp_market = utc_to_market_time(timestamp_utc, settings.market)
    raw_session_date = timestamp_market.date()

    if not calendar.is_trading_day(raw_session_date):
        prediction_mode = "pre_market"
        market_session_date = first_trading_day_on_or_after(raw_session_date, calendar)
        resolution_kind = PILOT_WINDOW_RESOLUTION_SNAPPED
        resolution_reason = (
            "Snapped to the next trading day's pre-market window because today is not a trading day."
        )
    elif is_pre_market(timestamp_market, settings.market):
        prediction_mode = "pre_market"
        market_session_date = raw_session_date
        resolution_kind = PILOT_WINDOW_RESOLUTION_NATURAL
        resolution_reason = (
            "Used the same-day pre-market window because the market has not opened yet."
        )
    elif is_after_close(timestamp_market, settings.market):
        prediction_mode = "after_close"
        market_session_date = raw_session_date
        resolution_kind = PILOT_WINDOW_RESOLUTION_NATURAL
        resolution_reason = (
            "Used the same-day after-close window because the market session is complete."
        )
    else:
        prediction_mode = "pre_market"
        market_session_date = raw_session_date
        resolution_kind = PILOT_WINDOW_RESOLUTION_SNAPPED
        resolution_reason = (
            "Snapped to the same-day pre-market window because it is the latest completed scheduled window during market hours."
        )

    scheduled_market_timestamp = build_pilot_slot_market_timestamp(
        runtime_settings=settings,
        market_session_date=market_session_date,
        prediction_mode=prediction_mode,
    )
    prediction_window = resolve_prediction_window(
        settings=settings,
        prediction_mode=prediction_mode,
        timestamp=market_time_to_utc(scheduled_market_timestamp, settings.market),
        calendar=calendar,
    )
    return LiveWindowResolution(
        prediction_window=prediction_window,
        resolution_kind=resolution_kind,
        resolution_reason=resolution_reason,
    )


def live_prediction_window_after_close_for_session_date(
    *,
    settings: AppSettings,
    calendar: Any,
    market_session_date: date,
) -> LivePredictionWindow:
    """Build an after_close window using the default after-close clock on a session date."""

    timestamp_market = build_pilot_slot_market_timestamp(
        runtime_settings=settings,
        market_session_date=market_session_date,
        prediction_mode="after_close",
    )
    timestamp_utc = market_time_to_utc(timestamp_market, settings.market)
    timestamp_market = utc_to_market_time(timestamp_utc, settings.market)
    prediction_date = calendar.next_trading_day(market_session_date)
    return LivePredictionWindow(
        prediction_mode="after_close",
        timestamp_utc=timestamp_utc,
        timestamp_market=timestamp_market,
        market_session_date=market_session_date,
        historical_cutoff_date=market_session_date,
        prediction_date=prediction_date,
    )


def build_pilot_week_trading_dates(
    *,
    pilot_start_date: date,
    pilot_end_date: date,
    calendar: Any,
) -> tuple[date, ...]:
    """Build the trading-day window for a pilot-week plan."""

    current = pilot_start_date
    trading_dates: list[date] = []
    while current <= pilot_end_date:
        if calendar.is_trading_day(current):
            trading_dates.append(current)
        current += timedelta(days=1)
    if not trading_dates:
        raise LivePilotError("The requested pilot window does not contain any trading days.")
    return tuple(trading_dates)


def build_pilot_slot_market_timestamp(
    *,
    runtime_settings: AppSettings,
    market_session_date: date,
    prediction_mode: str,
) -> datetime:
    """Build the scheduled market timestamp for one pilot-week slot."""

    if prediction_mode == "pre_market":
        scheduled_time = runtime_settings.pilot.default_pre_market_run_time
    elif prediction_mode == "after_close":
        scheduled_time = runtime_settings.pilot.default_after_close_run_time
    else:
        raise LivePilotError(f"Unsupported pilot prediction mode: {prediction_mode}")

    return pd.Timestamp(
        f"{market_session_date.isoformat()}T{scheduled_time.isoformat(timespec='minutes')}"
    ).tz_localize(runtime_settings.market.timezone_name).to_pydatetime()


def build_pilot_week_slot_id(*, market_session_date: date, prediction_mode: str) -> str:
    """Build the stable identifier for one pilot-week slot."""

    return f"{market_session_date.isoformat()}_{prediction_mode}"


def load_pilot_week_manifest(plan_path: str | Path) -> dict[str, Any]:
    """Load one saved pilot-week manifest."""

    manifest_path = Path(plan_path).expanduser().resolve()
    payload = load_required_json(manifest_path, artifact_label="Pilot week manifest")
    slots = payload.get("slots")
    if not isinstance(slots, list):
        raise LivePilotError("Pilot week manifest does not contain a slots list.")
    payload["manifest_path"] = str(manifest_path)
    return payload


def slot_is_due(slot: dict[str, Any], *, now_utc: datetime) -> bool:
    """Return True when one pilot-week slot is due for execution."""

    scheduled_timestamp = parse_timestamp(str(slot["scheduled_timestamp_utc"]))
    return scheduled_timestamp <= now_utc


def pilot_week_slot_status_exists(slot: dict[str, Any]) -> bool:
    """Return True when a slot-status marker already exists."""

    return Path(str(slot["slot_status_path"])).exists()


def map_pilot_run_status_to_week_status(status: str) -> str:
    """Map a pilot log row status to the week-status summary vocabulary."""

    if status in {PILOT_STATUS_SUCCESS, PILOT_STATUS_ABSTAIN}:
        return PILOT_WEEK_STATUS_COMPLETED
    if status == PILOT_STATUS_PARTIAL_FAILURE:
        return PILOT_WEEK_STATUS_PARTIAL_FAILURE
    return PILOT_WEEK_STATUS_FAILURE


def build_pilot_week_status_summary(
    *,
    manifest_payload: dict[str, Any],
    generated_at_utc: datetime,
) -> dict[str, Any]:
    """Summarize slot completion state for one pilot-week manifest."""

    slots = manifest_payload.get("slots", [])
    slot_statuses: list[dict[str, Any]] = []
    completed_slot_count = 0
    partial_failure_count = 0
    failure_count = 0
    pending_slot_count = 0

    for slot in slots:
        slot_status_path = Path(str(slot["slot_status_path"]))
        status_payload: dict[str, Any] | None = None
        if slot_status_path.exists():
            try:
                loaded_payload = json.loads(slot_status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded_payload = None
            if isinstance(loaded_payload, dict):
                status_payload = loaded_payload
        slot_status = clean_string((status_payload or {}).get("slot_status")) or PILOT_WEEK_STATUS_PENDING
        if slot_status == PILOT_WEEK_STATUS_COMPLETED:
            completed_slot_count += 1
        elif slot_status == PILOT_WEEK_STATUS_PARTIAL_FAILURE:
            partial_failure_count += 1
        elif slot_status == PILOT_WEEK_STATUS_FAILURE:
            failure_count += 1
        else:
            pending_slot_count += 1
        slot_statuses.append(
            {
                "slot_id": str(slot["slot_id"]),
                "slot_status": slot_status,
                "status_path": str(slot_status_path),
                "prediction_mode": str(slot["prediction_mode"]),
                "market_session_date": str(slot["market_session_date"]),
                "prediction_date": clean_string(slot.get("prediction_date")),
                "executed_at_utc": clean_string((status_payload or {}).get("executed_at_utc")),
            }
        )

    return {
        "ticker": manifest_payload.get("ticker"),
        "exchange": manifest_payload.get("exchange"),
        "pilot_window": manifest_payload.get("pilot_window"),
        "manifest_path": manifest_payload.get("manifest_path"),
        "slot_count": int(len(slots)),
        "completed_slot_count": int(completed_slot_count),
        "partial_failure_count": int(partial_failure_count),
        "failure_count": int(failure_count),
        "pending_slot_count": int(pending_slot_count),
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "slot_statuses": slot_statuses,
    }


def format_plan_week_summary(result: PilotWeekPlanResult) -> str:
    """Render a compact terminal summary for the plan-week command."""

    status_summary = load_required_json(
        result.status_summary_path,
        artifact_label="Pilot week status summary",
    )
    pilot_window = status_summary.get("pilot_window") or {}
    return "\n".join(
        [
            "Pilot week plan ready",
            (
                f"Window: {clean_string(pilot_window.get('start_date'))}.."
                f"{clean_string(pilot_window.get('end_date'))} | "
                f"slots={result.slot_count} | pending={status_summary.get('pending_slot_count')}"
            ),
            f"Manifest: {result.manifest_path}",
        ]
    )


def format_run_due_summary(result: PilotWeekDueRunResult) -> str:
    """Render a compact terminal summary for the run-due command."""

    status_summary = load_required_json(
        result.status_summary_path,
        artifact_label="Pilot week status summary",
    )
    return "\n".join(
        [
            "Pilot week due-run summary",
            (
                f"Due={result.due_slot_count} | executed={result.executed_slot_count} | "
                f"dry_run={'yes' if result.dry_run else 'no'} | "
                f"pending={status_summary.get('pending_slot_count')}"
            ),
            (
                f"Completed={status_summary.get('completed_slot_count')} | "
                f"partial={status_summary.get('partial_failure_count')} | "
                f"failed={status_summary.get('failure_count')}"
            ),
        ]
    )


def format_backfill_due_summary(
    result: PilotBackfillResult,
    *,
    pilot_start_date: date,
    pilot_end_date: date,
) -> str:
    """Render a compact terminal summary for the backfill-due command."""

    return "\n".join(
        [
            "Pilot week backfill summary",
            (
                f"Window: {pilot_start_date.isoformat()}..{pilot_end_date.isoformat()} | "
                f"updated={result.updated_row_count} | unresolved={result.unresolved_row_count}"
            ),
            f"Logs touched: {len(result.log_paths)}",
        ]
    )


def format_week_operator_summary(result: PilotWeekOperatorResult) -> str:
    """Render the combined operator summary for one plan/run/backfill pass."""

    status_summary = load_required_json(
        result.status_summary_path,
        artifact_label="Pilot week status summary",
    )
    pilot_window = status_summary.get("pilot_window") or {}
    return "\n".join(
        [
            "Pilot week operator summary",
            (
                f"Window: {clean_string(pilot_window.get('start_date'))}.."
                f"{clean_string(pilot_window.get('end_date'))} | "
                f"slots={result.slot_count} | due={result.due_slot_count} | "
                f"executed={result.executed_slot_count} | dry_run={'yes' if result.dry_run else 'no'}"
            ),
            (
                f"Backfill updated={result.updated_row_count} | unresolved={result.unresolved_row_count} | "
                f"pending={status_summary.get('pending_slot_count')}"
            ),
            (
                f"Completed={status_summary.get('completed_slot_count')} | "
                f"partial={status_summary.get('partial_failure_count')} | "
                f"failed={status_summary.get('failure_count')}"
            ),
        ]
    )


def _coalesce_news_feature_float(feature_row: pd.Series, column: str) -> Any:
    """Read one optional float from a Stage 7 feature row for pilot logging."""

    if column not in feature_row.index:
        return pd.NA
    value = feature_row.get(column)
    if pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


_EXPLANATION_SNIPPET_MAX_CHARS = 280


def _truncate_explanation_snippet(text: str, *, max_chars: int = _EXPLANATION_SNIPPET_MAX_CHARS) -> str:
    cleaned = str(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def fetch_recent_news_summaries(
    article_ids: list[str],
    extraction_path: Path,
) -> list[dict[str, Any]]:
    """Fetch top 3-5 most relevant news summaries from Stage 6 extractions."""

    if not article_ids or not extraction_path.exists():
        return []

    try:
        extraction_frame = pd.read_csv(extraction_path)
        matching = extraction_frame.loc[
            extraction_frame["article_id"].astype(str).isin([str(aid) for aid in article_ids])
        ].copy()
        if matching.empty:
            return []

        # Sort by relevance score
        matching = matching.sort_values(by="relevance_score", ascending=False)
        top_n = matching.head(5)

        summaries = []
        for _, row in top_n.iterrows():
            raw_snippet = row.get("summary_snippet") or row.get("rationale_short") or ""
            snippet_text = _truncate_explanation_snippet(str(raw_snippet)) if str(raw_snippet).strip() else ""
            provider_src = row.get("provider_source")
            if provider_src is None or (isinstance(provider_src, float) and pd.isna(provider_src)):
                provider_label = ""
            else:
                provider_label = str(provider_src).strip()
            event_raw = row.get("event_type")
            event_label = str(event_raw).strip() if event_raw is not None and str(event_raw).strip() else ""
            summaries.append(
                {
                    "article_id": str(row["article_id"]),
                    "article_title": str(row["article_title"]),
                    "event_type": event_label,
                    "provider_source": provider_label,
                    "summary_snippet": snippet_text,
                    "sentiment_label": str(row["sentiment_label"]),
                    "sentiment_score": float(row["sentiment_score"]),
                    "relevance_score": float(row["relevance_score"]),
                }
            )
        return summaries
    except Exception:
        return []


def _top_shap_for_explanation(
    contributions: Any,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Reduce SHAP payload to the largest-magnitude features for LLM prompts."""

    if not isinstance(contributions, dict):
        return []
    shap_values = contributions.get("shap_values")
    if not isinstance(shap_values, dict):
        return []
    ranked = sorted(
        shap_values.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:limit]
    return [{"feature": str(name), "shap": float(value)} for name, value in ranked]


def build_pilot_explanation_context(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact JSON-safe context for Gemini pilot explanations (low token use)."""

    summary = snapshot_payload.get("summary_context") or {}
    enhanced = summary.get("enhanced_prediction") or {}
    contrib = enhanced.get("feature_contributions")
    nc = summary.get("news_context") or {}
    rolling = nc.get("rolling") if isinstance(nc.get("rolling"), dict) else {}

    slim: dict[str, Any] = {
        "ticker": summary.get("ticker"),
        "exchange": summary.get("exchange"),
        "prediction_mode": summary.get("prediction_mode"),
        "prediction_date": summary.get("prediction_date"),
        "run_timestamp_ist": summary.get("run_timestamp_ist"),
        "status": summary.get("status"),
        "baseline_prediction": {
            k: v
            for k, v in (summary.get("baseline_prediction") or {}).items()
            if k not in ("model_run_id",)
        },
        "enhanced_prediction": {
            **{
                k: v
                for k, v in (summary.get("enhanced_prediction") or {}).items()
                if k not in ("feature_contributions", "model_run_id")
            },
            "top_shap_features": _top_shap_for_explanation(contrib),
        },
        "blended_prediction": {
            k: v
            for k, v in (summary.get("blended_prediction") or {}).items()
            if k not in ("model_run_id",)
        },
        "model_agreement": summary.get("model_agreement"),
        "news_context": {
            "article_count": nc.get("article_count"),
            "avg_confidence": nc.get("avg_confidence"),
            "fallback_ratio": nc.get("fallback_ratio"),
            "signal_state": nc.get("signal_state"),
            "rolling": rolling,
            "top_event_types": nc.get("top_event_types"),
            "recent_news_summaries": nc.get("recent_news_summaries"),
        },
        "warnings": summary.get("warnings"),
        "data_quality": {
            "score": (summary.get("data_quality") or {}).get("score"),
            "grade": (summary.get("data_quality") or {}).get("grade"),
        },
        "prior_prediction_outcome": summary.get("prior_prediction_outcome"),
        "failure_stage": summary.get("failure_stage"),
        "failure_message": summary.get("failure_message"),
        "failure_reason_public": summary.get("failure_reason_public"),
        "failure_next_step": summary.get("failure_next_step"),
    }
    return slim


def predict_live_baseline(
    settings: AppSettings,
    path_manager: PathManager,
    historical_row: pd.DataFrame,
) -> dict[str, Any]:
    """Load the saved baseline artifact and predict one live row."""

    model_path = path_manager.build_baseline_model_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_baseline_model_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata = load_required_json(metadata_path, artifact_label="Baseline model metadata")
    saved_model = load_saved_baseline_model(model_path)
    feature_frame = build_numeric_feature_frame(
        row_mapping=historical_row.iloc[0].to_dict(),
        feature_columns=saved_model.feature_columns,
    )
    prediction_outputs = predict_with_saved_model_outputs(
        saved_model,
        feature_frame,
    )
    return {
        "baseline_predicted_next_day_direction": int(prediction_outputs.predicted_labels.iloc[0]),
        "baseline_raw_predicted_probability_up": float(
            prediction_outputs.raw_probabilities.iloc[0]
        ),
        "baseline_calibrated_predicted_probability_up": float(
            prediction_outputs.calibrated_probabilities.iloc[0]
        ),
        "baseline_predicted_probability_up": float(
            prediction_outputs.calibrated_probabilities.iloc[0]
        ),
        "baseline_model_path": str(model_path),
        "baseline_model_metadata_path": str(metadata_path),
        "baseline_model_run_id": metadata.get("run_id"),
    }


def predict_live_enhanced(
    settings: AppSettings,
    path_manager: PathManager,
    *,
    prediction_mode: str,
    historical_row: pd.DataFrame,
    news_feature_row: pd.DataFrame,
) -> dict[str, Any]:
    """Load the saved enhanced artifact and predict one live row."""

    model_path = path_manager.build_enhanced_model_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        prediction_mode,
    )
    metadata_path = path_manager.build_enhanced_model_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        prediction_mode,
    )
    metadata = load_required_json(metadata_path, artifact_label="Enhanced model metadata")
    if metadata.get("prediction_mode") != prediction_mode:
        raise LivePilotError("Enhanced model metadata does not match the requested pilot mode.")
    saved_model = load_saved_enhanced_model(model_path)
    feature_spec = build_enhanced_feature_spec_from_metadata(metadata)
    prediction_date_value = str(
        news_feature_row.iloc[0].get("date")
        or historical_row.iloc[0].get("prediction_date")
        or ""
    )
    news_history_frame = load_live_enhanced_news_history_frame(
        source_news_feature_path=Path(
            str(
                metadata.get("source_news_feature_path")
                or path_manager.build_news_feature_table_path(
                    settings.ticker.symbol,
                    settings.ticker.exchange,
                )
            )
        ),
        prediction_mode=prediction_mode,
        prediction_date=prediction_date_value,
        base_news_feature_columns=feature_spec.base_news_feature_columns,
    )
    merged_row = build_live_enhanced_feature_row(
        historical_row_mapping=historical_row.iloc[0].to_dict(),
        news_feature_row_mapping=news_feature_row.iloc[0].to_dict(),
        feature_spec=feature_spec,
        news_history_frame=news_history_frame,
    )

    feature_frame = build_numeric_feature_frame(
        row_mapping=merged_row,
        feature_columns=saved_model.feature_columns,
    )
    prediction_outputs = predict_with_saved_enhanced_model_outputs(
        saved_model,
        feature_frame,
    )

    shap_values = explain_prediction_shap(
        pipeline=saved_model.pipeline,
        feature_row=feature_frame,
        feature_columns=saved_model.feature_columns,
    )
    enhanced_feature_contributions = {"shap_values": shap_values}

    return {
        "enhanced_predicted_next_day_direction": int(prediction_outputs.predicted_labels.iloc[0]),
        "enhanced_raw_predicted_probability_up": float(
            prediction_outputs.raw_probabilities.iloc[0]
        ),
        "enhanced_calibrated_predicted_probability_up": float(
            prediction_outputs.calibrated_probabilities.iloc[0]
        ),
        "enhanced_predicted_probability_up": float(
            prediction_outputs.calibrated_probabilities.iloc[0]
        ),
        "enhanced_model_path": str(model_path),
        "enhanced_model_metadata_path": str(metadata_path),
        "enhanced_model_run_id": metadata.get("run_id"),
        "enhanced_feature_contributions_json": encode_json_cell(enhanced_feature_contributions),
    }


def load_live_enhanced_news_history_frame(
    *,
    source_news_feature_path: Path,
    prediction_mode: str,
    prediction_date: str,
    base_news_feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Load the saved Stage 7 history used to build one live enhanced row."""

    expected_columns = ["date", "prediction_mode", *base_news_feature_columns]
    if not source_news_feature_path.exists():
        return pd.DataFrame(columns=expected_columns)

    try:
        history_frame = pd.read_csv(source_news_feature_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=expected_columns)

    if "date" not in history_frame.columns or "prediction_mode" not in history_frame.columns:
        return pd.DataFrame(columns=expected_columns)

    working_frame = history_frame.copy()
    working_frame["date"] = pd.to_datetime(working_frame["date"], errors="coerce")
    target_date = pd.to_datetime(prediction_date, errors="coerce")
    if pd.isna(target_date):
        return pd.DataFrame(columns=expected_columns)

    working_frame = working_frame.loc[
        working_frame["prediction_mode"].astype(str) == prediction_mode
    ].copy()
    working_frame = working_frame.loc[working_frame["date"] < target_date].copy()
    working_frame = working_frame.sort_values("date").reset_index(drop=True)

    for column in base_news_feature_columns:
        if column not in working_frame.columns:
            working_frame[column] = 0.0
        working_frame[column] = pd.to_numeric(
            working_frame[column],
            errors="coerce",
        ).fillna(0.0)

    return working_frame.loc[:, expected_columns]


def resolve_live_news_feature_row(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    prediction_mode: str,
    prediction_date: date,
) -> NewsFeatureResolution:
    """Load or synthesize the live Stage 7 row for one prediction date."""

    feature_path = path_manager.build_news_feature_table_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_news_feature_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata = load_required_json(metadata_path, artifact_label="Stage 7 news-feature metadata")
    if not feature_path.exists():
        raise LivePilotError(f"Stage 7 news feature table does not exist: {feature_path}")

    try:
        feature_frame = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feature_frame = pd.DataFrame(
            columns=("date", "ticker", "exchange", "prediction_mode") + NEWS_FEATURE_COLUMNS
        )

    matching_frame = feature_frame.loc[
        (feature_frame["date"].astype(str) == prediction_date.isoformat())
        & (feature_frame["prediction_mode"].astype(str) == prediction_mode)
    ].copy()
    synthetic = matching_frame.empty
    if synthetic:
        matching_frame = pd.DataFrame(
            [
                build_zero_feature_row(
                    target_date=prediction_date,
                    ticker=settings.ticker.symbol,
                    exchange=settings.ticker.exchange,
                    prediction_mode=prediction_mode,
                )
            ]
        )

    raw_snapshot_path = resolve_stage7_raw_snapshot_path(metadata)
    linked_article_ids = load_stage7_article_ids(
        raw_snapshot_path=raw_snapshot_path,
        prediction_date=prediction_date,
        prediction_mode=prediction_mode,
    )
    top_event_counts = summarize_top_event_counts(matching_frame.iloc[0])
    return NewsFeatureResolution(
        feature_row=matching_frame.reset_index(drop=True),
        metadata_path=metadata_path,
        metadata=metadata,
        raw_snapshot_path=raw_snapshot_path,
        linked_article_ids=linked_article_ids,
        top_event_counts=top_event_counts,
        synthetic=synthetic,
    )


def resolve_stage7_raw_snapshot_path(metadata: dict[str, Any]) -> Path | None:
    """Resolve the Stage 7 raw snapshot path from saved metadata."""

    raw_snapshot_value = clean_string(metadata.get("raw_snapshot_path"))
    if raw_snapshot_value is None:
        return None
    raw_snapshot_path = Path(raw_snapshot_value)
    if not raw_snapshot_path.exists():
        return None
    return raw_snapshot_path


def load_stage7_article_ids(
    *,
    raw_snapshot_path: Path | None,
    prediction_date: date,
    prediction_mode: str,
) -> list[str]:
    """Load the linked article ids for one Stage 7 feature row."""

    if raw_snapshot_path is None:
        return []
    raw_snapshot = load_optional_json(raw_snapshot_path)
    if raw_snapshot is None:
        return []
    for row in raw_snapshot.get("row_lineage", []):
        if not isinstance(row, dict):
            continue
        if row.get("date") != prediction_date.isoformat():
            continue
        if row.get("prediction_mode") != prediction_mode:
            continue
        article_ids = row.get("article_ids", [])
        if not isinstance(article_ids, list):
            return []
        return [str(article_id) for article_id in article_ids if str(article_id).strip()]
    return []


def summarize_top_event_counts(row: pd.Series) -> dict[str, int]:
    """Summarize the non-zero event counts for one Stage 7 feature row."""

    event_counts: dict[str, int] = {}
    for column_name in EVENT_COUNT_COLUMNS:
        raw_value = pd.to_numeric(row.get(column_name), errors="coerce")
        if pd.isna(raw_value):
            value = 0
        else:
            value = int(raw_value)
        if value <= 0:
            continue
        event_counts[column_name.replace("news_event_count_", "")] = value
    return dict(sorted(event_counts.items(), key=lambda item: (-item[1], item[0])))


def build_numeric_feature_frame(
    *,
    row_mapping: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build a numeric feature frame in the exact saved column order."""

    values: dict[str, Any] = {}
    for column_name in feature_columns:
        if column_name not in row_mapping:
            raise LivePilotError(f"Live inference row is missing feature column: {column_name}")
        values[column_name] = row_mapping[column_name]
    feature_frame = pd.DataFrame([values], columns=list(feature_columns))
    for column_name in feature_columns:
        feature_frame[column_name] = pd.to_numeric(feature_frame[column_name], errors="coerce")
        if feature_frame[column_name].isna().any():
            raise LivePilotError(
                f"Live inference row contains a non-numeric value in feature column: {column_name}"
            )
    return feature_frame


def _apply_resolved_prediction_window_to_pilot_row(
    pilot_row: dict[str, Any],
    *,
    runtime_settings: AppSettings,
    prediction_mode: str,
    prediction_window: LivePredictionWindow,
    run_context: Any,
    existing_log_frame: pd.DataFrame,
) -> None:
    """Refresh pilot row identity fields when the prediction window changes (e.g. calendar refresh)."""

    prediction_key = build_prediction_key(
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
        prediction_mode=prediction_mode,
        prediction_date=prediction_window.prediction_date,
    )
    prediction_attempt_number = resolve_prediction_attempt_number(
        existing_log_frame,
        prediction_key=prediction_key,
    )
    pilot_entry_id = build_pilot_entry_id(prediction_key, run_context.run_id)
    pilot_row.update(
        {
            "pilot_entry_id": pilot_entry_id,
            "prediction_key": prediction_key,
            "prediction_attempt_number": prediction_attempt_number,
            "market_session_date": prediction_window.market_session_date.isoformat(),
            "historical_cutoff_date": prediction_window.historical_cutoff_date.isoformat(),
            "prediction_date": prediction_window.prediction_date.isoformat(),
        }
    )


def build_prediction_key(
    *,
    ticker: str,
    exchange: str,
    prediction_mode: str,
    prediction_date: date,
) -> str:
    """Build the reusable identity key for one pilot prediction window."""

    return (
        f"{ticker.strip().upper()}|{exchange.strip().upper()}|"
        f"{prediction_mode}|{prediction_date.isoformat()}"
    )


def build_pilot_entry_id(prediction_key: str, run_id: str) -> str:
    """Build the append-only unique row id for one pilot run."""

    return f"{prediction_key}|{run_id}"


def resolve_prediction_attempt_number(
    log_frame: pd.DataFrame,
    *,
    prediction_key: str,
) -> int:
    """Assign the next append-only attempt number for one prediction key."""

    if log_frame.empty or "prediction_key" not in log_frame.columns:
        return 1

    matching_frame = log_frame.loc[
        log_frame["prediction_key"].astype(str) == prediction_key
    ].copy()
    if matching_frame.empty:
        return 1

    attempt_numbers = pd.to_numeric(
        matching_frame.get("prediction_attempt_number"),
        errors="coerce",
    )
    attempt_numbers = attempt_numbers.dropna()
    if not attempt_numbers.empty:
        return int(attempt_numbers.max()) + 1
    return int(len(matching_frame)) + 1


def build_empty_pilot_row() -> dict[str, Any]:
    """Build one empty pilot-row payload in the canonical column order."""

    return {column_name: pd.NA for column_name in PILOT_LOG_COLUMNS}


def append_pilot_row(log_path: Path, row: dict[str, Any]) -> None:
    """Append one pilot row to the mode-specific CSV log."""

    log_frame = load_pilot_log_frame(log_path)
    if log_frame.empty:
        combined = pd.DataFrame([row], columns=PILOT_LOG_COLUMNS, dtype=object)
    else:
        combined = pd.DataFrame(
            log_frame.to_dict(orient="records") + [row],
            columns=PILOT_LOG_COLUMNS,
            dtype=object,
        )
    save_pilot_log_frame(log_path, combined)


def load_pilot_log_frame(log_path: Path) -> pd.DataFrame:
    """Load a pilot log CSV and normalize its column layout."""

    if not log_path.exists():
        return pd.DataFrame(columns=PILOT_LOG_COLUMNS, dtype=object)
    try:
        log_frame = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=PILOT_LOG_COLUMNS, dtype=object)
    return (
        log_frame.reindex(columns=PILOT_LOG_COLUMNS, fill_value=pd.NA)
        .astype(object)
        .copy()
    )


def save_pilot_log_frame(log_path: Path, log_frame: pd.DataFrame) -> None:
    """Persist a normalized pilot log frame."""

    normalized_frame = (
        log_frame.reindex(columns=PILOT_LOG_COLUMNS, fill_value=pd.NA)
        .astype(object)
        .copy()
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_frame.to_csv(log_path, index=False)


def load_required_json(path: Path, *, artifact_label: str) -> dict[str, Any]:
    """Load a JSON file or fail with a clear pilot-specific error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LivePilotError(f"{artifact_label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LivePilotError(f"{artifact_label} is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise LivePilotError(f"{artifact_label} does not contain a JSON object: {path}")
    return payload


def prefix_metadata_warnings(metadata: dict[str, Any], prefix: str) -> list[str]:
    """Prefix a saved metadata warning list for pilot-log traceability."""

    raw_warnings = metadata.get("warnings", [])
    if not isinstance(raw_warnings, list):
        return []
    return [
        f"{prefix}:{str(warning)}"
        for warning in raw_warnings
        if str(warning).strip()
    ]


def determine_stage_failure_label(pilot_row: dict[str, Any]) -> str:
    """Infer the most recent news stage reached before a failure."""

    if clean_string(pilot_row.get("stage6_extraction_path")) is not None:
        return "stage7"
    if clean_string(pilot_row.get("stage5_processed_news_path")) is not None:
        return "stage6"
    return "stage5"


def build_run_data_quality_payload(
    *,
    pilot_row: dict[str, Any],
    stage2_metadata: dict[str, Any] | None,
    stage5_metadata: dict[str, Any] | None,
    stage6_metadata: dict[str, Any] | None,
    stage7_metadata: dict[str, Any] | None,
    news_feature_row: dict[str, Any],
) -> dict[str, Any]:
    """Score one live run for operator-facing data quality review."""

    return build_data_quality_payload(
        row_mapping=pilot_row,
        news_feature_row=news_feature_row,
        stage2_metadata=stage2_metadata,
        stage5_metadata=stage5_metadata,
        stage6_metadata=stage6_metadata,
        stage7_metadata=stage7_metadata,
    )


def grade_data_quality_score(score: float) -> str:
    """Convert one numeric quality score into the operator-facing grade band."""

    return grade_shared_data_quality_score(score)


def dedupe_quality_reasons(reasons: list[str]) -> list[str]:
    """Keep the quality-reason list stable and compact."""

    return dedupe_shared_quality_reasons(reasons)


def build_pilot_snapshot_payload(
    *,
    settings: AppSettings,
    pilot_entry_id: str,
    prediction_key: str,
    prediction_window: LivePredictionWindow,
    window_resolution_kind: str,
    window_resolution_reason: str,
    pilot_row: dict[str, Any],
    stage_payloads: dict[str, Any],
    prior_prediction_outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the persisted pilot snapshot payload used by stdout and explanations."""

    return {
        "pilot_entry_id": pilot_entry_id,
        "prediction_key": prediction_key,
        "status": pilot_row["status"],
        "prediction_mode": prediction_window.prediction_mode,
        "prediction_date": prediction_window.prediction_date.isoformat(),
        "pilot_timestamp_utc": prediction_window.timestamp_utc.isoformat(),
        "window_resolution": {
            "kind": window_resolution_kind,
            "reason": window_resolution_reason,
        },
        "warning_codes": json.loads(str(pilot_row["warning_codes_json"])),
        "timing": build_pilot_timing_payload(pilot_row),
        "retry_summary": build_pilot_retry_payload(pilot_row),
        "runtime_warning": {
            "flag": bool(coerce_optional_bool(pilot_row.get("runtime_warning_flag"))),
            "message": clean_string(pilot_row.get("runtime_warning_message")),
            "threshold_seconds": settings.pilot.runtime_warning_seconds,
        },
        "prior_prediction_outcome": prior_prediction_outcome,
        "summary_context": build_pilot_summary_context(
            settings=settings,
            prediction_window=prediction_window,
            window_resolution_kind=window_resolution_kind,
            window_resolution_reason=window_resolution_reason,
            pilot_row=pilot_row,
            prior_prediction_outcome=prior_prediction_outcome,
        ),
        "stage_payloads": stage_payloads,
        "row": serialize_row_for_json(pilot_row),
    }


def build_pilot_summary_context(
    *,
    settings: AppSettings,
    prediction_window: LivePredictionWindow,
    window_resolution_kind: str,
    window_resolution_reason: str,
    pilot_row: dict[str, Any],
    prior_prediction_outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the normalized summary payload shared by stdout and explanations."""

    warning_codes = decode_json_cell(pilot_row.get("warning_codes_json"), default=[])
    baseline_direction = format_prediction_direction(
        coerce_optional_int(pilot_row.get("baseline_predicted_next_day_direction"))
    )
    enhanced_direction = format_prediction_direction(
        coerce_optional_int(pilot_row.get("enhanced_predicted_next_day_direction"))
    )
    blended_direction = format_prediction_direction(
        coerce_optional_int(pilot_row.get("blended_predicted_next_day_direction"))
    )
    top_event_counts = decode_json_cell(pilot_row.get("top_event_counts_json"), default={})
    top_event_types = [
        {
            "event_type": normalize_event_type_label(str(event_type)),
            "count": int(count),
        }
        for event_type, count in top_event_counts.items()
    ]
    rolling_context: dict[str, float] = {}
    for rolling_key in (
        "news_volume_3d",
        "news_sentiment_3d",
        "news_sentiment_dispersion_1d",
        "news_directional_agreement_rate",
    ):
        rolling_val = coerce_optional_float(pilot_row.get(rolling_key))
        if rolling_val is not None:
            rolling_context[rolling_key] = rolling_val
    news_context = {
        "article_count": coerce_optional_int(pilot_row.get("news_article_count")),
        "avg_confidence": coerce_optional_float(pilot_row.get("news_avg_confidence")),
        "fallback_ratio": coerce_optional_float(pilot_row.get("news_fallback_article_ratio")),
        "signal_state": clean_string(pilot_row.get("news_signal_state")),
        "historical_market_gap_count_5d": coerce_optional_int(
            pilot_row.get("historical_market_gap_count_5d")
        ),
        "top_event_types": top_event_types,
        "recent_news_summaries": decode_json_cell(
            pilot_row.get("recent_news_summaries_json"),
            default=[],
        ),
        "rolling": rolling_context,
    }
    raw_contributions = clean_string(pilot_row.get("enhanced_feature_contributions_json"))
    if raw_contributions and raw_contributions not in ("null", "nan"):
        parsed_contributions = json.loads(raw_contributions)
        if not parsed_contributions:
            parsed_contributions = None
    else:
        parsed_contributions = None

    def resolve_probability_value(*keys: str) -> float | None:
        for key in keys:
            value = coerce_optional_float(pilot_row.get(key))
            if value is not None:
                return value
        return None

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "market_session_date": prediction_window.market_session_date.isoformat(),
        "historical_cutoff_date": prediction_window.historical_cutoff_date.isoformat(),
        "prediction_mode": prediction_window.prediction_mode,
        "prediction_date": prediction_window.prediction_date.isoformat(),
        "window_resolution_kind": window_resolution_kind,
        "window_resolution_reason": window_resolution_reason,
        "run_timestamp_ist": prediction_window.timestamp_market.isoformat(),
        "status": str(pilot_row["status"]),
        "baseline_prediction": {
            "direction": baseline_direction,
            "raw_probability_up": resolve_probability_value(
                "baseline_raw_predicted_probability_up",
                "baseline_predicted_probability_up",
            ),
            "calibrated_probability_up": resolve_probability_value(
                "baseline_calibrated_predicted_probability_up",
                "baseline_predicted_probability_up",
            ),
            "probability_up": coerce_optional_float(pilot_row.get("baseline_predicted_probability_up")),
            "model_run_id": clean_string(pilot_row.get("baseline_model_run_id")),
        },
        "enhanced_prediction": {
            "direction": enhanced_direction,
            "raw_probability_up": resolve_probability_value(
                "enhanced_raw_predicted_probability_up",
                "enhanced_predicted_probability_up",
            ),
            "calibrated_probability_up": resolve_probability_value(
                "enhanced_calibrated_predicted_probability_up",
                "enhanced_predicted_probability_up",
            ),
            "probability_up": coerce_optional_float(pilot_row.get("enhanced_predicted_probability_up")),
            "model_run_id": clean_string(pilot_row.get("enhanced_model_run_id")),
            "feature_contributions": parsed_contributions,
        },
        "blended_prediction": {
            "direction": blended_direction,
            "action": clean_string(pilot_row.get("selected_action")),
            "abstain_flag": bool(coerce_optional_bool(pilot_row.get("abstain_flag"))),
            "raw_probability_up": resolve_probability_value(
                "blended_raw_predicted_probability_up",
                "blended_predicted_probability_up",
            ),
            "calibrated_probability_up": resolve_probability_value(
                "blended_calibrated_predicted_probability_up",
                "blended_predicted_probability_up",
            ),
            "probability_up": coerce_optional_float(pilot_row.get("blended_predicted_probability_up")),
            "news_weight": coerce_optional_float(pilot_row.get("news_context_weight")),
            "probability_margin": coerce_optional_float(
                pilot_row.get("selective_probability_margin")
            ),
            "required_margin": coerce_optional_float(
                pilot_row.get("selective_required_margin")
            ),
            "abstain_reasons": decode_json_cell(
                pilot_row.get("abstain_reason_codes_json"),
                default=[],
            ),
        },
        "model_agreement": determine_model_agreement(
            baseline_prediction=coerce_optional_int(
                pilot_row.get("baseline_predicted_next_day_direction")
            ),
            enhanced_prediction=coerce_optional_int(
                pilot_row.get("enhanced_predicted_next_day_direction")
            ),
        ),
        "news_context": news_context,
        "warnings": {
            "fired": bool(warning_codes),
            "codes": warning_codes,
        },
        "data_quality": {
            "score": coerce_optional_float(pilot_row.get("data_quality_score")),
            "grade": clean_string(pilot_row.get("data_quality_grade")),
            "reasons": decode_json_cell(
                pilot_row.get("data_quality_reasons_json"),
                default=[],
            ),
            "components": decode_json_cell(
                pilot_row.get("data_quality_components_json"),
                default={},
            ),
        },
        "prior_prediction_outcome": prior_prediction_outcome,
        "total_run_duration_seconds": coerce_optional_float(
            pilot_row.get("total_duration_seconds")
        ),
        "failure_stage": clean_string(pilot_row.get("failure_stage")),
        "failure_message": clean_string(pilot_row.get("failure_message")),
        **build_pilot_failure_public_fields(pilot_row),
    }


def build_pilot_failure_public_fields(pilot_row: dict[str, Any]) -> dict[str, Any]:
    """Derive user-facing failure fields for summary_context (snapshot + terminal)."""

    status_str = str(pilot_row.get("status", ""))
    if status_str not in (PILOT_STATUS_PARTIAL_FAILURE, PILOT_STATUS_FAILURE):
        return {
            "failure_reason_public": None,
            "failure_next_step": None,
            "partial_failure_path_summary": None,
        }
    fs = clean_string(pilot_row.get("failure_stage"))
    fm = clean_string(pilot_row.get("failure_message"))
    pub, nxt = describe_pilot_stage_failure(fs, fm)
    partial_line = (
        describe_partial_failure_paths(fs) if status_str == PILOT_STATUS_PARTIAL_FAILURE else None
    )
    return {
        "failure_reason_public": pub,
        "failure_next_step": nxt,
        "partial_failure_path_summary": partial_line,
    }


def resolve_prior_prediction_outcome(
    *,
    log_path: Path,
    prediction_date: date,
    market: MarketSettings,
) -> dict[str, Any] | None:
    """Resolve the latest earlier pilot row whose prediction_date is a trading day before the target."""

    log_frame = load_pilot_log_frame(log_path)
    if log_frame.empty:
        return None

    candidate_frame = log_frame.loc[
        pd.to_datetime(log_frame["prediction_date"], errors="coerce").dt.date < prediction_date
    ].copy()
    if candidate_frame.empty:
        return None

    calendar = build_market_calendar(market)
    pred_dt = pd.to_datetime(candidate_frame["prediction_date"], errors="coerce")
    trading_mask = pred_dt.map(
        lambda t: calendar.is_trading_day(t.date()) if pd.notna(t) else False
    )
    candidate_frame = candidate_frame.loc[trading_mask].copy()
    if candidate_frame.empty:
        return None

    candidate_frame["prediction_date_dt"] = pd.to_datetime(
        candidate_frame["prediction_date"],
        errors="coerce",
    )
    candidate_frame = candidate_frame.sort_values(
        by=["prediction_date_dt", "pilot_timestamp_utc", "pilot_entry_id"],
        ascending=[True, True, True],
        na_position="last",
    )
    row = candidate_frame.iloc[-1]
    actual_status = clean_string(row.get("actual_outcome_status")) or ACTUAL_STATUS_PENDING
    return {
        "prediction_date": clean_string(row.get("prediction_date")),
        "status": actual_status,
        "backfilled": actual_status == ACTUAL_STATUS_BACKFILLED,
        "baseline_correct": coerce_optional_bool(row.get("baseline_correct")),
        "enhanced_correct": coerce_optional_bool(row.get("enhanced_correct")),
        "actual_next_day_direction": format_prediction_direction(
            coerce_optional_int(row.get("actual_next_day_direction"))
        ),
        "historical_close": coerce_optional_float(row.get("actual_historical_close")),
        "prediction_close": coerce_optional_float(row.get("actual_prediction_close")),
        "backfilled_at_utc": clean_string(row.get("actual_outcome_backfilled_at_utc")),
        "error": clean_string(row.get("actual_backfill_error")),
    }


def resolve_pilot_explanation_output(
    *,
    settings: AppSettings,
    snapshot_payload: dict[str, Any],
    client: GeminiApiExtractionClient | None = None,
) -> str:
    """Build or skip the optional plain-English pilot explanation."""

    if not clean_string(settings.providers.llm_api_key):
        return "Pilot explanation skipped: KUBERA_LLM_API_KEY is not set."

    active_client = client or GeminiApiExtractionClient(
        str(settings.providers.llm_api_key),
        model=settings.llm_extraction.model,
        timeout_seconds=settings.llm_extraction.request_timeout_seconds,
    )
    try:
        explanation_text, model_used = generate_plain_text_with_tiered_models(
            settings=settings,
            api_key=str(settings.providers.llm_api_key),
            prompt=build_pilot_explanation_prompt(snapshot_payload),
            client=active_client,
        )
    except LlmExtractionError as exc:
        return f"Pilot explanation unavailable: {sanitize_log_text(str(exc))}"
    except Exception as exc:
        return f"Pilot explanation unavailable: {sanitize_log_text(str(exc))}"
    explanation = sanitize_log_text(explanation_text).strip()
    if not explanation:
        return "Pilot explanation unavailable: Gemini returned an empty response."
    return f"Pilot explanation (model={model_used}):\n{explanation}"


def build_pilot_explanation_prompt(snapshot_payload: dict[str, Any]) -> str:
    """Build the pilot explanation prompt sent to Gemini."""

    context = build_pilot_explanation_context(snapshot_payload)
    snapshot_json = json.dumps(context, ensure_ascii=True, sort_keys=True, indent=2)
    return (
        "You are summarizing one stock-direction pilot run for a human reader.\n"
        "Use only the JSON context below (do not invent tickers, numbers, or article titles).\n"
        "Write 8 to 12 plain-English sentences.\n"
        "Explain what happened in the run, what the baseline and enhanced models predict for the target date, "
        "and how the blended decision and abstention logic (if any) apply.\n"
        "Use 'enhanced_prediction.top_shap_features' to explain which inputs most influenced the enhanced model.\n"
        "Relate multi-day news context using 'news_context.rolling' (e.g. news_sentiment_3d, news_volume_3d) "
        "when present, and tie it to those SHAP highlights.\n"
        "Reference at least two distinct articles from 'news_context.recent_news_summaries' by article_title "
        "and event_type when at least two summaries exist; if only one exists, reference that one.\n"
        "Describe how accurate the prior backfilled prediction was when prior_prediction_outcome is available.\n"
        "If data is missing, warnings fired, or the prior day is not backfilled, say that plainly.\n"
        "Do not mention JSON, prompts, or hidden reasoning.\n"
        f"{snapshot_json}\n"
    )


NEWS_SIGNAL_STATE_HINTS: dict[str, str] = {
    "fallback_heavy": (
        "News signal is fallback-heavy: many articles use headline or snippet only, "
        "so sentiment is noisier than full-text coverage."
    ),
    "carried_forward": (
        "News context was carried forward from a prior day: today's row may not reflect fresh articles."
    ),
    "zero_news": "No qualifying news articles mapped to this prediction window.",
    "synthetic": (
        "News context is synthetic or placeholder: treat sentiment signals as low confidence."
    ),
}

def format_pilot_summary(snapshot_payload: dict[str, Any]) -> str:
    """Render the human-readable terminal summary for one completed pilot run."""

    summary = snapshot_payload["summary_context"]
    warning_codes = summary["warnings"]["codes"]
    warning_text = ", ".join(warning_codes) if warning_codes else "none"
    top_event_types = summary["news_context"]["top_event_types"]
    if top_event_types:
        top_event_text = ", ".join(
            f"{item['event_type']} ({item['count']})"
            for item in top_event_types[:3]
        )
    else:
        top_event_text = "none"
    prior_outcome = summary.get("prior_prediction_outcome")
    prior_outcome_text = format_prior_outcome_summary(prior_outcome)
    lines = [
        "=" * 72,
        "Kubera Live Pilot Summary",
        "=" * 72,
        f"Ticker: {summary['ticker']} | Exchange: {summary['exchange']} | Mode: {summary['prediction_mode']}",
        (
            "Selected action: "
            f"{summary['blended_prediction']['action'] or 'n/a'} | "
            f"Status: {summary['status']}"
        ),
        (
            "Resolved window: "
            f"market_session_date={summary['market_session_date']} | "
            f"historical_cutoff_date={summary['historical_cutoff_date']} | "
            f"resolution={summary['window_resolution_kind']}"
        ),
        f"Resolution reason: {summary['window_resolution_reason']}",
        f"Prediction date: {summary['prediction_date']} | Run timestamp (IST): {summary['run_timestamp_ist']}",
        (
            "Baseline: "
            f"{summary['baseline_prediction']['direction']} | "
            f"raw_up_probability={format_probability(summary['baseline_prediction']['raw_probability_up'])} | "
            f"calibrated_up_probability={format_probability(summary['baseline_prediction']['calibrated_probability_up'])} | "
            f"run_id={summary['baseline_prediction']['model_run_id'] or 'n/a'}"
        ),
        (
            "Enhanced: "
            f"{summary['enhanced_prediction']['direction']} | "
            f"raw_up_probability={format_probability(summary['enhanced_prediction']['raw_probability_up'])} | "
            f"calibrated_up_probability={format_probability(summary['enhanced_prediction']['calibrated_probability_up'])} | "
            f"run_id={summary['enhanced_prediction']['model_run_id'] or 'n/a'}"
        ),
        (
            "Blended: "
            f"{summary['blended_prediction']['direction']} | "
            f"raw_up_probability={format_probability(summary['blended_prediction']['raw_probability_up'])} | "
            f"calibrated_up_probability={format_probability(summary['blended_prediction']['calibrated_probability_up'])} | "
            f"news_weight={format_probability(summary['blended_prediction']['news_weight'])} | "
            f"margin={format_probability(summary['blended_prediction']['probability_margin'])} | "
            f"required_margin={format_probability(summary['blended_prediction']['required_margin'])}"
        ),
        f"Model agreement: {summary['model_agreement']}",
        (
            "News context: "
            f"articles={format_optional_int(summary['news_context']['article_count'])} | "
            f"avg_confidence={format_probability(summary['news_context']['avg_confidence'])} | "
            f"fallback_ratio={format_probability(summary['news_context']['fallback_ratio'])} | "
            f"state={summary['news_context']['signal_state'] or 'n/a'} | "
            f"recent_gap_5d={format_optional_int(summary['news_context']['historical_market_gap_count_5d'])} | "
            f"top_events={top_event_text}"
        ),
    ]
    signal_state = summary["news_context"].get("signal_state") or ""
    if signal_state in NEWS_SIGNAL_STATE_HINTS:
        lines.append(NEWS_SIGNAL_STATE_HINTS[signal_state])
    lines.extend(
        [
        (
            "Data quality: "
            f"grade={summary['data_quality']['grade'] or 'n/a'} | "
            f"score={format_probability(summary['data_quality']['score'])}"
        ),
        (
            "Selective decision: "
            f"abstain={'yes' if summary['blended_prediction']['abstain_flag'] else 'no'} | "
            f"reasons={', '.join(summary['blended_prediction']['abstain_reasons']) or 'none'}"
        ),
        f"Warnings fired: {'yes' if summary['warnings']['fired'] else 'no'} | codes={warning_text}",
        f"Prior day outcome: {prior_outcome_text}",
        f"Total run duration: {format_duration(summary['total_run_duration_seconds'])}",
    ]
    )
    status_key = str(summary.get("status") or "").strip().lower()
    if status_key == PILOT_STATUS_PARTIAL_FAILURE:
        path_line = summary.get("partial_failure_path_summary")
        if not path_line:
            path_line = describe_partial_failure_paths(summary.get("failure_stage"))
        if path_line:
            lines.append(path_line)
    if status_key in (PILOT_STATUS_PARTIAL_FAILURE, PILOT_STATUS_FAILURE):
        stage = summary.get("failure_stage") or "unknown"
        pub = summary.get("failure_reason_public")
        nxt = summary.get("failure_next_step")
        if not pub or not nxt:
            pub, nxt = describe_pilot_stage_failure(
                summary.get("failure_stage"),
                summary.get("failure_message"),
            )
        tech = summary.get("failure_message")
        lines.extend(
            [
                f"Failure stage: {stage}",
                f"What happened: {pub}",
                f"Suggestion: {nxt}",
            ]
        )
        if tech and str(tech).strip().lower() not in ("n/a", "none", "", "nan"):
            lines.append(f"Details (sanitized): {tech}")
    lines.append("=" * 72)
    return "\n".join(lines)


def format_prior_outcome_summary(prior_outcome: dict[str, Any] | None) -> str:
    """Format the prior-day outcome line for the pilot summary."""

    if prior_outcome is None:
        return "no earlier pilot row found"
    if not prior_outcome.get("backfilled"):
        status = clean_string(prior_outcome.get("status")) or ACTUAL_STATUS_PENDING
        return f"{prior_outcome.get('prediction_date') or 'unknown date'} | status={status}"
    return (
        f"{prior_outcome.get('prediction_date') or 'unknown date'} | "
        f"baseline_correct={format_boolean(prior_outcome.get('baseline_correct'))} | "
        f"enhanced_correct={format_boolean(prior_outcome.get('enhanced_correct'))} | "
        f"historical_close={format_price(prior_outcome.get('historical_close'))} | "
        f"prediction_close={format_price(prior_outcome.get('prediction_close'))} | "
        f"actual_direction={prior_outcome.get('actual_next_day_direction') or 'n/a'}"
    )


def determine_model_agreement(
    *,
    baseline_prediction: int | None,
    enhanced_prediction: int | None,
) -> str:
    """Describe whether the baseline and enhanced predictions agree."""

    if baseline_prediction is None or enhanced_prediction is None:
        return "unavailable"
    if baseline_prediction == enhanced_prediction:
        return "agree"
    return "disagree"


def format_prediction_direction(value: int | None) -> str | None:
    """Convert a binary prediction label into the user-facing direction string."""

    if value is None:
        return None
    return "UP" if value == 1 else "DOWN"


def normalize_event_type_label(value: str) -> str:
    """Render Stage 7 event-count keys as compact event labels."""

    if value.startswith("news_event_count_"):
        return value.replace("news_event_count_", "", 1)
    return value


def format_probability(value: float | None) -> str:
    """Format one optional probability or ratio for stdout."""

    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_optional_int(value: int | None) -> str:
    """Format one optional integer for stdout."""

    if value is None:
        return "n/a"
    return str(value)


def format_duration(value: float | None) -> str:
    """Format one optional duration for stdout."""

    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def format_price(value: Any) -> str:
    """Format one optional close price for stdout."""

    price = coerce_optional_float(value)
    if price is None:
        return "n/a"
    return f"{price:.2f}"


def format_boolean(value: Any) -> str:
    """Format one optional boolean for stdout."""

    normalized = coerce_optional_bool(value)
    if normalized is None:
        return "n/a"
    return "yes" if normalized else "no"


def serialize_row_for_json(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one pilot row into a JSON-safe mapping."""

    return {key: serialize_cell_value(value) for key, value in row.items()}


def serialize_cell_value(value: Any) -> Any:
    """Convert one CSV cell value into a JSON-safe value."""

    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def encode_json_cell(value: Any) -> str:
    """Encode structured pilot-log cell content as a compact JSON string."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def decode_json_cell(value: Any, *, default: Any) -> Any:
    """Decode one structured pilot-log cell and return a default on failure."""

    raw_value = clean_string(value)
    if raw_value is None or raw_value in {"null", "nan"}:
        return default
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return default


def elapsed_seconds(start_time: float) -> float:
    """Return one elapsed duration in seconds with stable precision."""

    return round(time.perf_counter() - start_time, 6)


def build_pilot_timing_payload(row: dict[str, Any]) -> dict[str, float | None]:
    """Build the compact timing payload saved in the pilot snapshot."""

    return {
        "total_duration_seconds": coerce_optional_float(row.get("total_duration_seconds")),
        "stage2_duration_seconds": coerce_optional_float(row.get("stage2_duration_seconds")),
        "baseline_duration_seconds": coerce_optional_float(row.get("baseline_duration_seconds")),
        "stage5_duration_seconds": coerce_optional_float(row.get("stage5_duration_seconds")),
        "stage6_duration_seconds": coerce_optional_float(row.get("stage6_duration_seconds")),
        "stage7_duration_seconds": coerce_optional_float(row.get("stage7_duration_seconds")),
        "enhanced_duration_seconds": coerce_optional_float(row.get("enhanced_duration_seconds")),
    }


def build_pilot_retry_payload(row: dict[str, Any]) -> dict[str, dict[str, int | None]]:
    """Build the compact retry summary saved in the pilot snapshot."""

    return {
        "stage5": {
            "provider_request_count": coerce_optional_int(row.get("stage5_provider_request_count")),
            "provider_request_retry_count": coerce_optional_int(
                row.get("stage5_provider_request_retry_count")
            ),
            "article_fetch_attempt_count": coerce_optional_int(
                row.get("stage5_article_fetch_attempt_count")
            ),
            "article_fetch_retry_count": coerce_optional_int(
                row.get("stage5_article_fetch_retry_count")
            ),
        },
        "stage6": {
            "provider_request_count": coerce_optional_int(row.get("stage6_provider_request_count")),
            "retry_count": coerce_optional_int(row.get("stage6_retry_count")),
        },
    }


def lookup_close_for_date(market_frame: pd.DataFrame, target_date: str) -> float | None:
    """Look up one close price from a cleaned market-data table."""

    matches = market_frame.loc[market_frame["date"].astype(str) == target_date]
    if matches.empty:
        return None
    close_value = pd.to_numeric(matches.iloc[-1]["close"], errors="coerce")
    if pd.isna(close_value):
        return None
    return float(close_value)


def build_runtime_warning_message(
    *,
    total_duration_seconds: float | None,
    runtime_warning_seconds: float,
) -> str | None:
    """Build the runtime warning message for slow pilot runs."""

    if total_duration_seconds is None or total_duration_seconds <= runtime_warning_seconds:
        return None
    return (
        "Pilot runtime exceeded the configured threshold: "
        f"{total_duration_seconds:.3f}s > {runtime_warning_seconds:.3f}s."
    )


def coerce_optional_int(value: Any) -> int | None:
    """Convert one optional CSV value into an integer when present."""

    if value is None or pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_optional_bool(value: Any) -> bool | None:
    """Convert one optional CSV value into a boolean when present."""

    if value is None or pd.isna(value) or value == "":
        return None
    if isinstance(value, bool):
        return value
    
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    return None


def coerce_optional_float(value: Any) -> float | None:
    """Convert one optional CSV value into a float when present."""

    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_string(value: Any) -> str | None:
    """Normalize one optional text value."""

    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def normalize_timestamp(value: datetime | None) -> datetime:
    """Normalize an optional timestamp into UTC."""

    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_timestamp(raw_value: str) -> datetime:
    """Parse a CLI timestamp into UTC."""

    timestamp = pd.Timestamp(raw_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def parse_prediction_date(raw_value: str) -> date:
    """Parse a CLI prediction date."""

    return date.fromisoformat(raw_value)


def coerce_required_date(value: Any, *, field_label: str) -> date:
    """Normalize one date-like value into a concrete date."""

    parsed_value = pd.Timestamp(value)
    if pd.isna(parsed_value):
        raise LivePilotError(f"{field_label} is missing or invalid.")
    return parsed_value.date()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse live-pilot command arguments."""

    parser = argparse.ArgumentParser(description="Run Kubera live pilot workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Append one live pilot prediction row.")
    run_parser.add_argument("--ticker", help="Override the configured ticker symbol for this run.")
    run_parser.add_argument("--exchange", help="Override the configured exchange for this run.")
    run_parser.add_argument(
        "--prediction-mode",
        required=True,
        choices=PILOT_PREDICTION_MODES,
        help="Pilot prediction mode to run.",
    )
    run_parser.add_argument(
        "--timestamp",
        help="Use a specific as-of timestamp in an ISO-style format supported by pandas Timestamp.",
    )
    run_parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate a short Gemini explanation after the pilot summary.",
    )

    backfill_parser = subparsers.add_parser(
        "backfill-actuals",
        help="Backfill actual outcomes for an existing pilot prediction date.",
    )
    backfill_parser.add_argument(
        "--ticker",
        help="Override the configured ticker symbol for this backfill.",
    )
    backfill_parser.add_argument(
        "--exchange",
        help="Override the configured exchange for this backfill.",
    )
    backfill_parser.add_argument(
        "--prediction-date",
        required=True,
        help="Prediction date to backfill in YYYY-MM-DD format.",
    )
    backfill_parser.add_argument(
        "--prediction-mode",
        choices=PILOT_PREDICTION_MODES,
        help="Limit the backfill to one pilot prediction mode.",
    )

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Update manual review notes for the latest matching pilot row.",
    )
    annotate_parser.add_argument(
        "--ticker",
        help="Override the configured ticker symbol for this annotation update.",
    )
    annotate_parser.add_argument(
        "--exchange",
        help="Override the configured exchange for this annotation update.",
    )
    annotate_parser.add_argument(
        "--prediction-mode",
        required=True,
        choices=PILOT_PREDICTION_MODES,
        help="Pilot prediction mode whose log row should be updated.",
    )
    annotate_parser.add_argument(
        "--prediction-date",
        required=True,
        help="Prediction date to annotate in YYYY-MM-DD format.",
    )
    annotate_parser.add_argument("--news-quality-note", help="Manual note about news quality.")
    annotate_parser.add_argument("--market-shock-note", help="Manual note about market-wide shocks.")
    annotate_parser.add_argument("--source-outage-note", help="Manual note about source outages.")

    week_plan_parser = subparsers.add_parser(
        "plan-week",
        help="Write a deterministic pilot-week manifest and status summary.",
    )
    week_plan_parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    week_plan_parser.add_argument("--exchange", help="Override the configured exchange code.")
    week_plan_parser.add_argument(
        "--pilot-start-date",
        required=True,
        help="Pilot window start date in YYYY-MM-DD format.",
    )
    week_plan_parser.add_argument(
        "--pilot-end-date",
        required=True,
        help="Pilot window end date in YYYY-MM-DD format.",
    )

    run_due_parser = subparsers.add_parser(
        "run-due",
        help="Execute all due, incomplete slots from a saved pilot-week plan.",
    )
    run_due_parser.add_argument(
        "--plan-path",
        required=True,
        help="Path to a saved pilot-week manifest JSON file.",
    )
    run_due_parser.add_argument(
        "--now",
        help="Optional UTC or offset-aware timestamp used to decide which slots are due.",
    )
    run_due_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report due slots without executing them or writing status markers.",
    )

    backfill_due_parser = subparsers.add_parser(
        "backfill-due",
        help="Backfill all eligible pilot rows across a planned pilot window.",
    )
    backfill_due_parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    backfill_due_parser.add_argument("--exchange", help="Override the configured exchange code.")
    backfill_due_parser.add_argument(
        "--pilot-start-date",
        required=True,
        help="Pilot window start date in YYYY-MM-DD format.",
    )
    backfill_due_parser.add_argument(
        "--pilot-end-date",
        required=True,
        help="Pilot window end date in YYYY-MM-DD format.",
    )
    backfill_due_parser.add_argument(
        "--as-of",
        help="Only backfill prediction dates on or before this YYYY-MM-DD date.",
    )

    operate_week_parser = subparsers.add_parser(
        "operate-week",
        help="Ensure the manifest exists, run due slots, and backfill eligible rows.",
    )
    operate_week_parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    operate_week_parser.add_argument("--exchange", help="Override the configured exchange code.")
    operate_week_parser.add_argument(
        "--pilot-start-date",
        required=True,
        help="Pilot window start date in YYYY-MM-DD format.",
    )
    operate_week_parser.add_argument(
        "--pilot-end-date",
        required=True,
        help="Pilot window end date in YYYY-MM-DD format.",
    )
    operate_week_parser.add_argument(
        "--now",
        help="Optional UTC or offset-aware timestamp used to decide which slots are due.",
    )
    operate_week_parser.add_argument(
        "--as-of",
        help="Only backfill prediction dates on or before this YYYY-MM-DD date.",
    )
    operate_week_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and report due work without executing runs or backfills.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the live-pilot CLI."""

    args = parse_args(argv)
    settings = load_settings()
    if args.command == "run":
        run_live_pilot(
            settings,
            prediction_mode=args.prediction_mode,
            timestamp=parse_timestamp(args.timestamp) if args.timestamp else None,
            ticker=args.ticker,
            exchange=args.exchange,
            explain=bool(args.explain),
        )
        return 0
    if args.command == "backfill-actuals":
        backfill_pilot_actuals(
            settings,
            prediction_date=parse_prediction_date(args.prediction_date),
            prediction_mode=args.prediction_mode,
            ticker=args.ticker,
            exchange=args.exchange,
        )
        return 0
    if args.command == "annotate":
        annotate_pilot_entry(
            settings,
            prediction_mode=args.prediction_mode,
            prediction_date=parse_prediction_date(args.prediction_date),
            news_quality_note=args.news_quality_note,
            market_shock_note=args.market_shock_note,
            source_outage_note=args.source_outage_note,
            ticker=args.ticker,
            exchange=args.exchange,
        )
        return 0
    if args.command == "plan-week":
        result = plan_pilot_week(
            settings,
            pilot_start_date=parse_prediction_date(args.pilot_start_date),
            pilot_end_date=parse_prediction_date(args.pilot_end_date),
            ticker=args.ticker,
            exchange=args.exchange,
        )
        print(format_plan_week_summary(result))
        return 0
    if args.command == "run-due":
        result = run_due_pilot_week(
            settings,
            plan_path=args.plan_path,
            now=parse_timestamp(args.now) if args.now else None,
            dry_run=bool(args.dry_run),
        )
        print(format_run_due_summary(result))
        return 0
    if args.command == "backfill-due":
        result = backfill_due_pilot_week(
            settings,
            pilot_start_date=parse_prediction_date(args.pilot_start_date),
            pilot_end_date=parse_prediction_date(args.pilot_end_date),
            as_of=parse_prediction_date(args.as_of) if args.as_of else None,
            ticker=args.ticker,
            exchange=args.exchange,
        )
        print(
            format_backfill_due_summary(
                result,
                pilot_start_date=parse_prediction_date(args.pilot_start_date),
                pilot_end_date=parse_prediction_date(args.pilot_end_date),
            )
        )
        return 0
    if args.command == "operate-week":
        result = operate_pilot_week(
            settings,
            pilot_start_date=parse_prediction_date(args.pilot_start_date),
            pilot_end_date=parse_prediction_date(args.pilot_end_date),
            now=parse_timestamp(args.now) if args.now else None,
            as_of=parse_prediction_date(args.as_of) if args.as_of else None,
            dry_run=bool(args.dry_run),
            ticker=args.ticker,
            exchange=args.exchange,
        )
        print(format_week_operator_summary(result))
        return 0
    raise LivePilotError(f"Unsupported pilot command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
