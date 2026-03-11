"""Live pilot workflow for Kubera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from kubera.config import AppSettings, load_settings, resolve_runtime_settings
from kubera.features.historical_features import (
    build_live_historical_feature_row,
    read_cleaned_market_data,
    validate_cleaned_market_data,
)
from kubera.features.news_features import (
    EVENT_COUNT_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    build_news_features,
    build_zero_feature_row,
)
from kubera.ingest.market_data import fetch_historical_market_data
from kubera.ingest.news_data import fetch_company_news
from kubera.llm.extract_news import extract_news
from kubera.models.train_baseline import load_saved_baseline_model, predict_with_saved_model
from kubera.models.train_enhanced import (
    load_saved_enhanced_model,
    predict_with_saved_enhanced_model,
)
from kubera.reporting.offline_evaluation import load_optional_json
from kubera.utils.calendar import build_market_calendar
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot
from kubera.utils.time_utils import is_after_close, is_pre_market, utc_to_market_time


PILOT_PREDICTION_MODES = ("pre_market", "after_close")
PILOT_STATUS_SUCCESS = "success"
PILOT_STATUS_PARTIAL_FAILURE = "partial_failure"
PILOT_STATUS_FAILURE = "failure"
ACTUAL_STATUS_PENDING = "pending"
ACTUAL_STATUS_BACKFILLED = "backfilled"
ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE = "market_data_unavailable"
PILOT_LOG_COLUMNS = (
    "pilot_entry_id",
    "prediction_key",
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
    "prediction_date",
    "baseline_predicted_next_day_direction",
    "baseline_predicted_probability_up",
    "enhanced_predicted_next_day_direction",
    "enhanced_predicted_probability_up",
    "disagreement_flag",
    "news_article_count",
    "news_warning_article_count",
    "news_fallback_article_ratio",
    "news_avg_confidence",
    "fallback_heavy_flag",
    "news_feature_synthetic_flag",
    "linked_article_ids_json",
    "top_event_counts_json",
    "warning_codes_json",
    "status",
    "failure_stage",
    "failure_message",
    "total_duration_seconds",
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
    "actual_historical_close",
    "actual_prediction_close",
    "actual_next_day_direction",
    "actual_outcome_status",
    "actual_outcome_backfilled_at_utc",
    "actual_backfill_error",
    "baseline_correct",
    "enhanced_correct",
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
class PilotRunResult:
    """Persisted live pilot run summary."""

    log_path: Path
    snapshot_path: Path
    pilot_entry_id: str
    status: str
    prediction_date: date
    prediction_mode: str


@dataclass(frozen=True)
class PilotBackfillResult:
    """Summary of one actual-outcome backfill pass."""

    updated_row_count: int
    unresolved_row_count: int
    log_paths: tuple[Path, ...]


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


def run_live_pilot(
    settings: AppSettings,
    *,
    prediction_mode: str,
    timestamp: datetime | None = None,
    ticker: str | None = None,
    exchange: str | None = None,
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
    prediction_key = build_prediction_key(
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
        prediction_mode=prediction_mode,
        prediction_date=prediction_window.prediction_date,
    )
    pilot_entry_id = build_pilot_entry_id(prediction_key, run_context.run_id)

    pilot_row = build_empty_pilot_row()
    pilot_row.update(
        {
            "pilot_entry_id": pilot_entry_id,
            "prediction_key": prediction_key,
            "ticker": runtime_settings.ticker.symbol,
            "exchange": runtime_settings.ticker.exchange,
            "prediction_mode": prediction_mode,
            "pilot_run_id": run_context.run_id,
            "pilot_timestamp_utc": prediction_window.timestamp_utc.isoformat(),
            "pilot_timestamp_market": prediction_window.timestamp_market.isoformat(),
            "market_session_date": prediction_window.market_session_date.isoformat(),
            "historical_cutoff_date": prediction_window.historical_cutoff_date.isoformat(),
            "news_cutoff_timestamp_utc": prediction_window.timestamp_utc.isoformat(),
            "prediction_date": prediction_window.prediction_date.isoformat(),
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
    failure_stage: str | None = None
    failure_message: str | None = None
    baseline_succeeded = False
    enhanced_succeeded = False
    historical_row: pd.DataFrame | None = None
    news_feature_resolution: NewsFeatureResolution | None = None

    try:
        stage2_start = time.perf_counter()
        try:
            market_result = fetch_historical_market_data(
                runtime_settings,
                end_date=prediction_window.historical_cutoff_date,
            )
            stage2_metadata = load_required_json(
                market_result.metadata_path,
                artifact_label="Stage 2 market-data metadata",
            )
            validated_market_frame = validate_cleaned_market_data(
                read_cleaned_market_data(market_result.cleaned_table_path),
                ticker=runtime_settings.ticker.symbol,
                exchange=runtime_settings.ticker.exchange,
                feature_settings=runtime_settings.historical_features,
            )
            latest_market_date = coerce_required_date(
                validated_market_frame.iloc[-1]["date"],
                field_label="latest Stage 2 market date",
            )
            if latest_market_date != prediction_window.historical_cutoff_date:
                raise LivePilotError(
                    "Historical market data does not cover the expected live cutoff date."
                )
            historical_row = build_live_historical_feature_row(
                validated_market_frame,
                runtime_settings.historical_features,
                prediction_date=prediction_window.prediction_date,
            )
            pilot_row["historical_date"] = coerce_required_date(
                historical_row.iloc[0]["date"],
                field_label="live historical feature date",
            ).isoformat()
            pilot_row["stage2_cleaned_path"] = str(market_result.cleaned_table_path)
            pilot_row["stage2_metadata_path"] = str(market_result.metadata_path)
            pilot_row["stage2_run_id"] = stage2_metadata.get("run_id")
            stage_payloads["stage2"] = {
                "cleaned_path": str(market_result.cleaned_table_path),
                "metadata_path": str(market_result.metadata_path),
                "run_id": stage2_metadata.get("run_id"),
            }
        finally:
            pilot_row["stage2_duration_seconds"] = elapsed_seconds(stage2_start)
    except Exception as exc:
        failure_stage = "stage2"
        failure_message = str(exc)

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
                failure_message = str(exc)

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
                extraction_result = extract_news(runtime_settings)
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
                news_feature_result = build_news_features(runtime_settings)
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
                }
            finally:
                pilot_row["stage7_duration_seconds"] = elapsed_seconds(stage7_start)
        except Exception as exc:
            if failure_stage is None:
                failure_stage = determine_stage_failure_label(pilot_row)
                failure_message = str(exc)

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
                failure_message = str(exc)

    if baseline_succeeded and enhanced_succeeded:
        pilot_row["status"] = PILOT_STATUS_SUCCESS
        pilot_row["disagreement_flag"] = bool(
            int(pilot_row["baseline_predicted_next_day_direction"])
            != int(pilot_row["enhanced_predicted_next_day_direction"])
        )
    elif baseline_succeeded or enhanced_succeeded:
        pilot_row["status"] = PILOT_STATUS_PARTIAL_FAILURE
    else:
        pilot_row["status"] = PILOT_STATUS_FAILURE

    pilot_row["failure_stage"] = failure_stage if failure_stage is not None else pd.NA
    pilot_row["failure_message"] = failure_message if failure_message is not None else pd.NA
    pilot_row["total_duration_seconds"] = elapsed_seconds(total_start)
    pilot_row["warning_codes_json"] = encode_json_cell(sorted(set(warning_codes)))

    append_pilot_row(pilot_log_path, pilot_row)
    write_json_file(
        snapshot_path,
        {
            "pilot_entry_id": pilot_entry_id,
            "prediction_key": prediction_key,
            "status": pilot_row["status"],
            "prediction_mode": prediction_mode,
            "prediction_date": prediction_window.prediction_date.isoformat(),
            "pilot_timestamp_utc": prediction_window.timestamp_utc.isoformat(),
            "warning_codes": json.loads(str(pilot_row["warning_codes_json"])),
            "timing": build_pilot_timing_payload(pilot_row),
            "retry_summary": build_pilot_retry_payload(pilot_row),
            "stage_payloads": stage_payloads,
            "row": serialize_row_for_json(pilot_row),
        },
    )
    logger.info(
        "Live pilot row recorded | ticker=%s | exchange=%s | mode=%s | prediction_date=%s | status=%s | log=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        prediction_mode,
        prediction_window.prediction_date,
        pilot_row["status"],
        pilot_log_path,
    )
    return PilotRunResult(
        log_path=pilot_log_path,
        snapshot_path=snapshot_path,
        pilot_entry_id=pilot_entry_id,
        status=str(pilot_row["status"]),
        prediction_date=prediction_window.prediction_date,
        prediction_mode=prediction_mode,
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
    market_session_date = timestamp_market.date()
    if not calendar.is_trading_day(market_session_date):
        raise LivePilotError("Live pilot runs must use a trading-day market timestamp.")

    if prediction_mode == "pre_market":
        if not is_pre_market(timestamp_market, settings.market):
            raise LivePilotError("Pre-market pilot runs must use a timestamp before the market open.")
        historical_cutoff_date = previous_trading_day(market_session_date, calendar)
        prediction_date = market_session_date
    elif prediction_mode == "after_close":
        if not is_after_close(timestamp_market, settings.market):
            raise LivePilotError("After-close pilot runs must use a timestamp at or after the market close.")
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


def previous_trading_day(value: date, calendar: Any) -> date:
    """Return the trading day immediately before the given date."""

    current = value - timedelta(days=1)
    while not calendar.is_trading_day(current):
        current -= timedelta(days=1)
    return current


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
    predicted_labels, predicted_probabilities = predict_with_saved_model(
        saved_model,
        feature_frame,
    )
    return {
        "baseline_predicted_next_day_direction": int(predicted_labels.iloc[0]),
        "baseline_predicted_probability_up": float(predicted_probabilities.iloc[0]),
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
    merged_row = {**historical_row.iloc[0].to_dict(), **news_feature_row.iloc[0].to_dict()}
    feature_frame = build_numeric_feature_frame(
        row_mapping=merged_row,
        feature_columns=saved_model.feature_columns,
    )
    predicted_labels, predicted_probabilities = predict_with_saved_enhanced_model(
        saved_model,
        feature_frame,
    )
    return {
        "enhanced_predicted_next_day_direction": int(predicted_labels.iloc[0]),
        "enhanced_predicted_probability_up": float(predicted_probabilities.iloc[0]),
        "enhanced_model_path": str(model_path),
        "enhanced_model_metadata_path": str(metadata_path),
        "enhanced_model_run_id": metadata.get("run_id"),
    }


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


def build_empty_pilot_row() -> dict[str, Any]:
    """Build one empty pilot-row payload in the canonical column order."""

    return {column_name: pd.NA for column_name in PILOT_LOG_COLUMNS}


def append_pilot_row(log_path: Path, row: dict[str, Any]) -> None:
    """Append one pilot row to the mode-specific CSV log."""

    log_frame = load_pilot_log_frame(log_path)
    row_frame = pd.DataFrame([row], columns=PILOT_LOG_COLUMNS)
    if log_frame.empty:
        combined = row_frame
    else:
        combined = pd.concat([log_frame, row_frame], ignore_index=True)
    save_pilot_log_frame(log_path, combined)


def load_pilot_log_frame(log_path: Path) -> pd.DataFrame:
    """Load a pilot log CSV and normalize its column layout."""

    if not log_path.exists():
        return pd.DataFrame(columns=PILOT_LOG_COLUMNS, dtype=object)
    try:
        log_frame = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=PILOT_LOG_COLUMNS, dtype=object)

    for column_name in PILOT_LOG_COLUMNS:
        if column_name not in log_frame.columns:
            log_frame[column_name] = pd.NA
    return log_frame.loc[:, PILOT_LOG_COLUMNS].astype(object).copy()


def save_pilot_log_frame(log_path: Path, log_frame: pd.DataFrame) -> None:
    """Persist a normalized pilot log frame."""

    normalized_frame = log_frame.copy()
    for column_name in PILOT_LOG_COLUMNS:
        if column_name not in normalized_frame.columns:
            normalized_frame[column_name] = pd.NA
    normalized_frame = normalized_frame.loc[:, PILOT_LOG_COLUMNS].astype(object)
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


def coerce_optional_int(value: Any) -> int | None:
    """Convert one optional CSV value into an integer when present."""

    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
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
    raise LivePilotError(f"Unsupported pilot command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
