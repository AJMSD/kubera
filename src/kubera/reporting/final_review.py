"""Stage 11 final review workflow for Kubera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kubera.config import AppSettings, load_settings
from kubera.pilot.live_pilot import (
    ACTUAL_STATUS_BACKFILLED,
    ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE,
    ACTUAL_STATUS_PENDING,
    PILOT_PREDICTION_MODES,
    PILOT_STATUS_FAILURE,
    PILOT_STATUS_PARTIAL_FAILURE,
    load_pilot_log_frame,
)
from kubera.reporting.offline_evaluation import (
    ALL_ROWS_SUBSET_NAME,
    BASELINE_VARIANT_NAME,
    ENHANCED_VARIANT_NAME,
    EVENT_ABLATION_VARIANT_NAME,
    MAJORITY_VARIANT_NAME,
    NEWS_HEAVY_SUBSET_NAME,
    NO_CONFIDENCE_VARIANT_NAME,
    NO_FALLBACK_VARIANT_NAME,
    PREVIOUS_DAY_VARIANT_NAME,
    SENTIMENT_ABLATION_VARIANT_NAME,
    ZERO_NEWS_SUBSET_NAME,
    evaluate_offline,
    load_optional_json,
)
from kubera.utils.calendar import build_market_calendar
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot


FINAL_REVIEW_VARIANT_ORDER = (
    ENHANCED_VARIANT_NAME,
    BASELINE_VARIANT_NAME,
    MAJORITY_VARIANT_NAME,
    PREVIOUS_DAY_VARIANT_NAME,
    SENTIMENT_ABLATION_VARIANT_NAME,
    EVENT_ABLATION_VARIANT_NAME,
    NO_CONFIDENCE_VARIANT_NAME,
    NO_FALLBACK_VARIANT_NAME,
)
PILOT_COVERAGE_UNAVAILABLE = "unavailable"
PILOT_COVERAGE_PARTIAL = "partial"
PILOT_COVERAGE_COMPLETE = "complete"


class FinalReviewError(RuntimeError):
    """Raised when the final review workflow cannot continue."""


@dataclass(frozen=True)
class OfflineEvaluationArtifacts:
    """Loaded Stage 9 artifacts reused by the final review workflow."""

    metrics_path: Path
    summary_json_path: Path
    summary_markdown_path: Path
    metrics_frame: pd.DataFrame
    summary_payload: dict[str, Any]
    refreshed: bool


@dataclass(frozen=True)
class FinalReviewResult:
    """Persisted Stage 11 review artifact summary."""

    summary_json_path: Path
    summary_markdown_path: Path
    offline_artifacts_refreshed: bool
    pilot_coverage_status: str


def generate_final_review(
    settings: AppSettings,
    *,
    pilot_start_date: date,
    pilot_end_date: date,
    refresh_offline_evaluation: bool = False,
) -> FinalReviewResult:
    """Build the Stage 11 final review package."""

    if pilot_end_date < pilot_start_date:
        raise FinalReviewError("Pilot end date must be on or after the pilot start date.")

    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(settings, path_manager)
    write_settings_snapshot(settings, run_context.config_snapshot_path)
    logger = configure_logging(run_context, settings.run.log_level, logger_name="kubera.final_review")

    offline_artifacts = resolve_offline_evaluation_artifacts(
        settings,
        refresh_offline_evaluation=refresh_offline_evaluation,
    )
    calendar = build_market_calendar(settings.market)
    expected_trading_dates = build_expected_trading_dates(
        pilot_start_date=pilot_start_date,
        pilot_end_date=pilot_end_date,
        calendar=calendar,
    )
    pilot_summary = build_pilot_summary(
        settings=settings,
        path_manager=path_manager,
        expected_trading_dates=expected_trading_dates,
    )
    final_review_json_path = path_manager.build_final_review_json_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    final_review_markdown_path = path_manager.build_final_review_markdown_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    summary_payload = build_final_review_payload(
        settings=settings,
        path_manager=path_manager,
        offline_artifacts=offline_artifacts,
        pilot_summary=pilot_summary,
        pilot_start_date=pilot_start_date,
        pilot_end_date=pilot_end_date,
        expected_trading_dates=expected_trading_dates,
        final_review_json_path=final_review_json_path,
        final_review_markdown_path=final_review_markdown_path,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
    )
    write_json_file(final_review_json_path, summary_payload)
    final_review_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    final_review_markdown_path.write_text(
        render_final_review_markdown(summary_payload),
        encoding="utf-8",
    )
    logger.info(
        "Final review ready | ticker=%s | exchange=%s | pilot_window=%s..%s | json=%s | markdown=%s",
        settings.ticker.symbol,
        settings.ticker.exchange,
        pilot_start_date,
        pilot_end_date,
        final_review_json_path,
        final_review_markdown_path,
    )
    return FinalReviewResult(
        summary_json_path=final_review_json_path,
        summary_markdown_path=final_review_markdown_path,
        offline_artifacts_refreshed=offline_artifacts.refreshed,
        pilot_coverage_status=str(pilot_summary["coverage_status"]),
    )


def resolve_offline_evaluation_artifacts(
    settings: AppSettings,
    *,
    refresh_offline_evaluation: bool = False,
) -> OfflineEvaluationArtifacts:
    """Load the Stage 9 artifacts, refreshing them once when needed."""

    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    metrics_path = path_manager.build_offline_metrics_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    summary_json_path = path_manager.build_offline_evaluation_summary_json_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    summary_markdown_path = path_manager.build_offline_evaluation_summary_markdown_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )

    refreshed = False
    if refresh_offline_evaluation or not offline_evaluation_artifacts_exist(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    ):
        evaluate_offline(settings)
        refreshed = True

    if not offline_evaluation_artifacts_exist(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    ):
        raise FinalReviewError(
            "Stage 9 offline evaluation artifacts are unavailable after refresh."
        )

    metrics_frame = load_required_csv(
        metrics_path,
        artifact_label="Stage 9 offline metrics",
    )
    summary_payload = load_required_json(
        summary_json_path,
        artifact_label="Stage 9 offline summary JSON",
    )
    return OfflineEvaluationArtifacts(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        metrics_frame=metrics_frame,
        summary_payload=summary_payload,
        refreshed=refreshed,
    )


def offline_evaluation_artifacts_exist(
    *,
    metrics_path: Path,
    summary_json_path: Path,
    summary_markdown_path: Path,
) -> bool:
    """Return True when the required Stage 9 outputs exist."""

    return metrics_path.exists() and summary_json_path.exists() and summary_markdown_path.exists()


def load_required_csv(path: Path, *, artifact_label: str) -> pd.DataFrame:
    """Load one required CSV artifact or fail clearly."""

    if not path.exists():
        raise FinalReviewError(f"{artifact_label} does not exist: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise FinalReviewError(f"{artifact_label} is empty: {path}") from exc


def load_required_json(path: Path, *, artifact_label: str) -> dict[str, Any]:
    """Load one required JSON artifact or fail clearly."""

    if not path.exists():
        raise FinalReviewError(f"{artifact_label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalReviewError(f"{artifact_label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalReviewError(f"{artifact_label} must contain a JSON object: {path}")
    return payload


def build_expected_trading_dates(
    *,
    pilot_start_date: date,
    pilot_end_date: date,
    calendar: Any,
) -> tuple[date, ...]:
    """Build the expected trading-day window for Stage 10 completeness checks."""

    current = pilot_start_date
    trading_dates: list[date] = []
    while current <= pilot_end_date:
        if calendar.is_trading_day(current):
            trading_dates.append(current)
        current += timedelta(days=1)
    if not trading_dates:
        raise FinalReviewError("The requested pilot window does not contain any trading days.")
    return tuple(trading_dates)


def build_pilot_summary(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    expected_trading_dates: tuple[date, ...],
) -> dict[str, Any]:
    """Summarize Stage 10 pilot evidence for the requested market-session window."""

    expected_date_strings = tuple(value.isoformat() for value in expected_trading_dates)
    per_mode: dict[str, dict[str, Any]] = {}
    daily_rows: list[dict[str, Any]] = []
    missing_expected_pairs: list[dict[str, str]] = []

    for prediction_mode in PILOT_PREDICTION_MODES:
        log_path = path_manager.build_pilot_log_path(
            settings.ticker.symbol,
            settings.ticker.exchange,
            prediction_mode,
        )
        raw_mode_frame = load_mode_pilot_window_frame(
            log_path,
            expected_date_strings=expected_date_strings,
        )
        latest_mode_frame = select_latest_mode_rows(raw_mode_frame)
        latest_rows_by_date = {
            str(row["market_session_date"]): row
            for row in latest_mode_frame.to_dict(orient="records")
        }

        for market_session_date in expected_trading_dates:
            market_session_string = market_session_date.isoformat()
            selected_row = latest_rows_by_date.get(market_session_string)
            if selected_row is None:
                missing_expected_pairs.append(
                    {
                        "market_session_date": market_session_string,
                        "prediction_mode": prediction_mode,
                    }
                )
            daily_rows.append(
                build_daily_pilot_row(
                    selected_row=selected_row,
                    market_session_date=market_session_string,
                    prediction_mode=prediction_mode,
                )
            )

        per_mode[prediction_mode] = build_mode_pilot_summary(
            prediction_mode=prediction_mode,
            log_path=log_path,
            raw_mode_frame=raw_mode_frame,
            latest_mode_frame=latest_mode_frame,
            expected_trading_dates=expected_trading_dates,
        )

    daily_rows = sort_daily_pilot_rows(daily_rows)
    overall = build_overall_pilot_summary(
        per_mode=per_mode,
        expected_pair_count=len(expected_trading_dates) * len(PILOT_PREDICTION_MODES),
        missing_expected_pairs=missing_expected_pairs,
    )
    return {
        "coverage_status": overall["coverage_status"],
        "expected_market_session_dates": list(expected_date_strings),
        "expected_market_session_count": int(len(expected_trading_dates)),
        "expected_pair_count": overall["expected_pair_count"],
        "available_pair_count": overall["available_pair_count"],
        "missing_expected_pairs": missing_expected_pairs,
        "per_mode": per_mode,
        "daily_prediction_rows": daily_rows,
        "operational_issues": build_pilot_operational_issues(
            overall=overall,
            missing_expected_pairs=missing_expected_pairs,
        ),
        "overall": overall,
    }


def load_mode_pilot_window_frame(
    log_path: Path,
    *,
    expected_date_strings: tuple[str, ...],
) -> pd.DataFrame:
    """Load one pilot log and filter it to the expected market-session window."""

    if not log_path.exists():
        return pd.DataFrame()

    log_frame = load_pilot_log_frame(log_path)
    if log_frame.empty:
        return pd.DataFrame()

    filtered_frame = log_frame.loc[
        log_frame["market_session_date"].astype(str).isin(expected_date_strings)
    ].copy()
    if filtered_frame.empty:
        return pd.DataFrame()

    filtered_frame["market_session_date_sort_key"] = pd.to_datetime(
        filtered_frame["market_session_date"],
        errors="coerce",
    )
    filtered_frame["pilot_timestamp_sort_key"] = pd.to_datetime(
        filtered_frame["pilot_timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    return filtered_frame


def select_latest_mode_rows(raw_mode_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse one append-only pilot log to the latest row per market session."""

    if raw_mode_frame.empty:
        return pd.DataFrame()

    ordered_frame = raw_mode_frame.sort_values(
        by=["market_session_date_sort_key", "pilot_timestamp_sort_key", "pilot_entry_id"],
        ascending=[True, True, True],
        na_position="last",
    )
    latest_frame = ordered_frame.drop_duplicates(
        subset=["market_session_date"],
        keep="last",
    )
    return latest_frame.sort_values(
        by=["market_session_date_sort_key", "pilot_entry_id"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_mode_pilot_summary(
    *,
    prediction_mode: str,
    log_path: Path,
    raw_mode_frame: pd.DataFrame,
    latest_mode_frame: pd.DataFrame,
    expected_trading_dates: tuple[date, ...],
) -> dict[str, Any]:
    """Build one per-mode pilot summary."""

    expected_date_strings = tuple(value.isoformat() for value in expected_trading_dates)
    latest_rows = latest_mode_frame.to_dict(orient="records") if not latest_mode_frame.empty else []
    latest_dates = {str(row["market_session_date"]) for row in latest_rows}
    missing_dates = [
        market_session_date
        for market_session_date in expected_date_strings
        if market_session_date not in latest_dates
    ]

    success_row_count = 0
    partial_failure_count = 0
    failure_count = 0
    disagreement_count = 0
    disagreement_denominator = 0
    fallback_heavy_count = 0
    zero_news_count = 0
    manual_note_row_count = 0
    source_outage_note_count = 0
    backfilled_row_count = 0
    pending_actual_count = 0
    market_data_unavailable_count = 0
    baseline_correct_values: list[int] = []
    enhanced_correct_values: list[int] = []

    for row in latest_rows:
        status = clean_string(row.get("status")) or "unknown"
        if status == "success":
            success_row_count += 1
        elif status == PILOT_STATUS_PARTIAL_FAILURE:
            partial_failure_count += 1
        elif status == PILOT_STATUS_FAILURE:
            failure_count += 1

        disagreement_value = coerce_optional_bool(row.get("disagreement_flag"))
        if disagreement_value is not None:
            disagreement_denominator += 1
            disagreement_count += int(disagreement_value)

        fallback_heavy_count += int(coerce_optional_bool(row.get("fallback_heavy_flag")) is True)
        news_article_count = coerce_optional_int(row.get("news_article_count"))
        zero_news_count += int(news_article_count == 0 if news_article_count is not None else False)

        news_quality_note = clean_string(row.get("news_quality_note"))
        market_shock_note = clean_string(row.get("market_shock_note"))
        source_outage_note = clean_string(row.get("source_outage_note"))
        manual_note_row_count += int(
            any(note is not None for note in (news_quality_note, market_shock_note, source_outage_note))
        )
        source_outage_note_count += int(source_outage_note is not None)

        actual_status = clean_string(row.get("actual_outcome_status")) or ACTUAL_STATUS_PENDING
        if actual_status == ACTUAL_STATUS_BACKFILLED:
            backfilled_row_count += 1
        elif actual_status == ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE:
            market_data_unavailable_count += 1
        else:
            pending_actual_count += 1

        baseline_correct = coerce_optional_bool(row.get("baseline_correct"))
        if baseline_correct is not None:
            baseline_correct_values.append(int(baseline_correct))
        enhanced_correct = coerce_optional_bool(row.get("enhanced_correct"))
        if enhanced_correct is not None:
            enhanced_correct_values.append(int(enhanced_correct))

    return {
        "prediction_mode": prediction_mode,
        "log_path": str(log_path),
        "log_exists": bool(log_path.exists()),
        "expected_market_session_count": int(len(expected_trading_dates)),
        "observed_raw_row_count": int(len(raw_mode_frame)),
        "selected_row_count": int(len(latest_rows)),
        "rerun_row_count": int(max(len(raw_mode_frame) - len(latest_rows), 0)),
        "missing_market_session_dates": missing_dates,
        "success_row_count": success_row_count,
        "partial_failure_count": partial_failure_count,
        "failure_row_count": failure_count,
        "backfilled_row_count": backfilled_row_count,
        "pending_actual_row_count": pending_actual_count,
        "market_data_unavailable_row_count": market_data_unavailable_count,
        "disagreement_count": disagreement_count,
        "disagreement_rate": (
            float(disagreement_count / disagreement_denominator)
            if disagreement_denominator > 0
            else None
        ),
        "fallback_heavy_count": fallback_heavy_count,
        "zero_news_count": zero_news_count,
        "manual_note_row_count": manual_note_row_count,
        "source_outage_note_count": source_outage_note_count,
        "baseline_accuracy": calculate_optional_accuracy(baseline_correct_values),
        "enhanced_accuracy": calculate_optional_accuracy(enhanced_correct_values),
    }


def build_daily_pilot_row(
    *,
    selected_row: dict[str, Any] | None,
    market_session_date: str,
    prediction_mode: str,
) -> dict[str, Any]:
    """Build one Stage 11 daily pilot table row."""

    if selected_row is None:
        return {
            "market_session_date": market_session_date,
            "prediction_mode": prediction_mode,
            "prediction_date": None,
            "status": "missing",
            "pilot_entry_id": None,
            "prediction_key": None,
            "baseline_predicted_next_day_direction": None,
            "baseline_predicted_probability_up": None,
            "enhanced_predicted_next_day_direction": None,
            "enhanced_predicted_probability_up": None,
            "actual_next_day_direction": None,
            "actual_outcome_status": "missing",
            "disagreement_flag": None,
            "fallback_heavy_flag": None,
            "news_article_count": None,
            "warning_codes": [],
            "linked_article_ids": [],
            "top_event_counts": {},
            "notes": ["missing_expected_run"],
        }

    return {
        "market_session_date": market_session_date,
        "prediction_mode": prediction_mode,
        "prediction_date": clean_string(selected_row.get("prediction_date")),
        "status": clean_string(selected_row.get("status")) or "unknown",
        "pilot_entry_id": clean_string(selected_row.get("pilot_entry_id")),
        "prediction_key": clean_string(selected_row.get("prediction_key")),
        "baseline_predicted_next_day_direction": coerce_optional_int(
            selected_row.get("baseline_predicted_next_day_direction")
        ),
        "baseline_predicted_probability_up": coerce_optional_float(
            selected_row.get("baseline_predicted_probability_up")
        ),
        "enhanced_predicted_next_day_direction": coerce_optional_int(
            selected_row.get("enhanced_predicted_next_day_direction")
        ),
        "enhanced_predicted_probability_up": coerce_optional_float(
            selected_row.get("enhanced_predicted_probability_up")
        ),
        "actual_next_day_direction": coerce_optional_int(
            selected_row.get("actual_next_day_direction")
        ),
        "actual_outcome_status": clean_string(selected_row.get("actual_outcome_status"))
        or ACTUAL_STATUS_PENDING,
        "disagreement_flag": coerce_optional_bool(selected_row.get("disagreement_flag")),
        "fallback_heavy_flag": coerce_optional_bool(selected_row.get("fallback_heavy_flag")),
        "news_article_count": coerce_optional_int(selected_row.get("news_article_count")),
        "warning_codes": decode_json_cell(selected_row.get("warning_codes_json"), default=[]),
        "linked_article_ids": decode_json_cell(
            selected_row.get("linked_article_ids_json"),
            default=[],
        ),
        "top_event_counts": decode_json_cell(
            selected_row.get("top_event_counts_json"),
            default={},
        ),
        "notes": build_daily_pilot_notes(selected_row),
    }


def build_daily_pilot_notes(row: dict[str, Any]) -> list[str]:
    """Build a compact note list for one pilot row."""

    notes: list[str] = []
    status = clean_string(row.get("status")) or "unknown"
    if status in {PILOT_STATUS_PARTIAL_FAILURE, PILOT_STATUS_FAILURE}:
        failure_stage = clean_string(row.get("failure_stage"))
        notes.append(f"{status}:{failure_stage}" if failure_stage is not None else status)

    if coerce_optional_bool(row.get("fallback_heavy_flag")) is True:
        notes.append("fallback_heavy")
    if coerce_optional_int(row.get("news_article_count")) == 0:
        notes.append("zero_news")

    for label, field_name in (
        ("news_quality_note", "news_quality_note"),
        ("market_shock_note", "market_shock_note"),
        ("source_outage_note", "source_outage_note"),
    ):
        field_value = clean_string(row.get(field_name))
        if field_value is not None:
            notes.append(f"{label}:{field_value}")

    actual_status = clean_string(row.get("actual_outcome_status"))
    if actual_status == ACTUAL_STATUS_PENDING:
        notes.append("actual_pending")
    elif actual_status == ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE:
        notes.append("actual_market_data_unavailable")
    return notes


def sort_daily_pilot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort Stage 11 daily pilot rows by date then mode."""

    mode_order = {prediction_mode: index for index, prediction_mode in enumerate(PILOT_PREDICTION_MODES)}
    return sorted(
        rows,
        key=lambda row: (
            row["market_session_date"],
            mode_order.get(str(row["prediction_mode"]), len(mode_order)),
        ),
    )


def build_overall_pilot_summary(
    *,
    per_mode: dict[str, dict[str, Any]],
    expected_pair_count: int,
    missing_expected_pairs: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the overall Stage 10 evidence summary."""

    available_pair_count = sum(
        int(mode_summary["selected_row_count"])
        for mode_summary in per_mode.values()
    )
    if available_pair_count == 0:
        coverage_status = PILOT_COVERAGE_UNAVAILABLE
    elif available_pair_count < expected_pair_count:
        coverage_status = PILOT_COVERAGE_PARTIAL
    else:
        coverage_status = PILOT_COVERAGE_COMPLETE

    return {
        "coverage_status": coverage_status,
        "expected_pair_count": int(expected_pair_count),
        "available_pair_count": int(available_pair_count),
        "missing_pair_count": int(len(missing_expected_pairs)),
        "success_row_count": sum(int(mode_summary["success_row_count"]) for mode_summary in per_mode.values()),
        "partial_failure_count": sum(int(mode_summary["partial_failure_count"]) for mode_summary in per_mode.values()),
        "failure_row_count": sum(int(mode_summary["failure_row_count"]) for mode_summary in per_mode.values()),
        "backfilled_row_count": sum(int(mode_summary["backfilled_row_count"]) for mode_summary in per_mode.values()),
        "pending_actual_row_count": sum(int(mode_summary["pending_actual_row_count"]) for mode_summary in per_mode.values()),
        "market_data_unavailable_row_count": sum(
            int(mode_summary["market_data_unavailable_row_count"]) for mode_summary in per_mode.values()
        ),
        "disagreement_count": sum(int(mode_summary["disagreement_count"]) for mode_summary in per_mode.values()),
        "fallback_heavy_count": sum(int(mode_summary["fallback_heavy_count"]) for mode_summary in per_mode.values()),
        "zero_news_count": sum(int(mode_summary["zero_news_count"]) for mode_summary in per_mode.values()),
        "manual_note_row_count": sum(int(mode_summary["manual_note_row_count"]) for mode_summary in per_mode.values()),
        "source_outage_note_count": sum(
            int(mode_summary["source_outage_note_count"]) for mode_summary in per_mode.values()
        ),
    }


def build_pilot_operational_issues(
    *,
    overall: dict[str, Any],
    missing_expected_pairs: list[dict[str, str]],
) -> list[str]:
    """Build deterministic operational notes for the pilot section."""

    issues: list[str] = []
    if overall["coverage_status"] == PILOT_COVERAGE_UNAVAILABLE:
        issues.append("No pilot log rows were found for the requested market-session window.")
    elif missing_expected_pairs:
        issues.append(
            f"{len(missing_expected_pairs)} expected pilot mode-day pairs are missing from the requested window."
        )
    if int(overall["partial_failure_count"]) > 0:
        issues.append(f"{overall['partial_failure_count']} pilot rows recorded partial failures.")
    if int(overall["failure_row_count"]) > 0:
        issues.append(f"{overall['failure_row_count']} pilot rows recorded full failures.")
    if int(overall["pending_actual_row_count"]) > 0:
        issues.append(f"{overall['pending_actual_row_count']} pilot rows still need actual-outcome backfill.")
    if int(overall["market_data_unavailable_row_count"]) > 0:
        issues.append(
            f"{overall['market_data_unavailable_row_count']} pilot rows could not backfill actuals because market data was unavailable."
        )
    if int(overall["source_outage_note_count"]) > 0:
        issues.append(f"{overall['source_outage_note_count']} pilot rows include source outage notes.")
    if int(overall["fallback_heavy_count"]) > 0:
        issues.append(f"{overall['fallback_heavy_count']} pilot rows were fallback-heavy.")
    return issues


def build_final_review_payload(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    offline_artifacts: OfflineEvaluationArtifacts,
    pilot_summary: dict[str, Any],
    pilot_start_date: date,
    pilot_end_date: date,
    expected_trading_dates: tuple[date, ...],
    final_review_json_path: Path,
    final_review_markdown_path: Path,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
) -> dict[str, Any]:
    """Build the Stage 11 summary JSON payload."""

    evaluation_summary = build_evaluation_summary(
        settings=settings,
        path_manager=path_manager,
        offline_artifacts=offline_artifacts,
    )
    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "pilot_window": {
            "start_date": pilot_start_date.isoformat(),
            "end_date": pilot_end_date.isoformat(),
            "expected_market_session_dates": [value.isoformat() for value in expected_trading_dates],
            "expected_market_session_count": int(len(expected_trading_dates)),
        },
        "offline_evaluation": evaluation_summary,
        "pilot_summary": pilot_summary,
        "limitations": build_limitations(pilot_summary),
        "lessons_learned": build_lessons_learned(pilot_summary),
        "claim_checks": build_claim_checks(
            offline_summary_payload=offline_artifacts.summary_payload,
            pilot_summary=pilot_summary,
        ),
        "traceability": build_traceability_summary(
            settings=settings,
            path_manager=path_manager,
            offline_summary_payload=offline_artifacts.summary_payload,
            pilot_summary=pilot_summary,
        ),
        "manual_actions": [
            "Run real Stage 10 pre-market and after-close commands across the intended pilot window.",
            "Backfill actual outcomes for those pilot rows once the relevant closes are available.",
            "Add manual outage, sparse-news, or market-shock notes where needed.",
            "Review provider and publisher terms before broader unattended automation.",
        ],
        "summary_json_path": str(final_review_json_path),
        "summary_markdown_path": str(final_review_markdown_path),
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def build_evaluation_summary(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    offline_artifacts: OfflineEvaluationArtifacts,
) -> dict[str, Any]:
    """Build the Stage 9 evidence section used by Stage 11."""

    summary_payload = offline_artifacts.summary_payload
    article_coverage = summary_payload.get("article_coverage", {}) or {}
    extraction_summary = summary_payload.get("extraction_summary", {}) or {}
    news_feature_summary = summary_payload.get("news_feature_summary", {}) or {}
    return {
        "reused_saved_outputs": not offline_artifacts.refreshed,
        "offline_artifacts_refreshed": offline_artifacts.refreshed,
        "headline_split": summary_payload.get("headline_split"),
        "news_heavy_min_article_count": summary_payload.get("news_heavy_min_article_count"),
        "metric_materiality_threshold": summary_payload.get("metric_materiality_threshold"),
        "historical_features": build_historical_feature_summary(summary_payload),
        "article_coverage": {
            "available": bool(article_coverage.get("available")),
            "row_count": coerce_optional_int(article_coverage.get("row_count")),
            "coverage_start": clean_string(article_coverage.get("coverage_start")),
            "coverage_end": clean_string(article_coverage.get("coverage_end")),
            "cache_hit_count": coerce_optional_int(article_coverage.get("cache_hit_count")),
            "fresh_fetch_count": coerce_optional_int(article_coverage.get("fresh_fetch_count")),
            "content_origin_counts": article_coverage.get("content_origin_counts", {}) or {},
            "source_name_counts": article_coverage.get("source_name_counts", {}) or {},
        },
        "extraction_summary": {
            "available": bool(extraction_summary.get("available")),
            "source_row_count": coerce_optional_int(extraction_summary.get("source_row_count")),
            "success_count": coerce_optional_int(extraction_summary.get("success_count")),
            "failure_count": coerce_optional_int(extraction_summary.get("failure_count")),
            "success_rate": coerce_optional_float(extraction_summary.get("success_rate")),
            "fallback_usage_rate": coerce_optional_float(
                extraction_summary.get("fallback_usage_rate")
            ),
            "extraction_mode_counts": extraction_summary.get("extraction_mode_counts", {}) or {},
        },
        "news_feature_summary": {
            "available": bool(news_feature_summary.get("available")),
            "output_row_count": coerce_optional_int(news_feature_summary.get("output_row_count")),
            "zero_news_row_count": coerce_optional_int(
                news_feature_summary.get("zero_news_row_count")
            ),
            "nonzero_news_row_count": coerce_optional_int(
                news_feature_summary.get("nonzero_news_row_count")
            ),
            "coverage_start": clean_string(news_feature_summary.get("coverage_start")),
            "coverage_end": clean_string(news_feature_summary.get("coverage_end")),
            "prediction_mode_row_counts": news_feature_summary.get("prediction_mode_row_counts", {})
            or {},
        },
        "per_mode": {
            prediction_mode: build_evaluation_mode_summary(
                prediction_mode=prediction_mode,
                metrics_frame=offline_artifacts.metrics_frame,
                mode_summary=mode_summary,
            )
            for prediction_mode, mode_summary in (
                summary_payload.get("mode_summaries", {}) or {}
            ).items()
        },
        "artifact_paths": {
            "metrics_csv": str(offline_artifacts.metrics_path),
            "summary_json": str(offline_artifacts.summary_json_path),
            "summary_markdown": str(offline_artifacts.summary_markdown_path),
            "source_historical_feature_path": clean_string(
                summary_payload.get("source_historical_feature_path")
            ),
            "source_historical_metadata_path": clean_string(
                summary_payload.get("source_historical_metadata_path")
            ),
            "source_news_feature_path": clean_string(summary_payload.get("source_news_feature_path")),
            "source_news_metadata_path": clean_string(
                summary_payload.get("source_news_metadata_path")
            ),
            "baseline_model_metadata_path": str(
                path_manager.build_baseline_model_metadata_path(
                    settings.ticker.symbol,
                    settings.ticker.exchange,
                )
            ),
            "enhanced_model_metadata_paths": {
                prediction_mode: str(
                    path_manager.build_enhanced_model_metadata_path(
                        settings.ticker.symbol,
                        settings.ticker.exchange,
                        prediction_mode,
                    )
                )
                for prediction_mode in PILOT_PREDICTION_MODES
            },
        },
        "offline_run_id": clean_string(summary_payload.get("run_id")),
    }


def build_historical_feature_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize Stage 3 coverage from saved metadata and feature tables."""

    historical_feature_path = coerce_optional_path(summary_payload.get("source_historical_feature_path"))
    historical_metadata = load_optional_json_object(
        coerce_optional_path(summary_payload.get("source_historical_metadata_path"))
    )
    historical_frame = pd.read_csv(historical_feature_path) if historical_feature_path and historical_feature_path.exists() else pd.DataFrame()
    if historical_frame.empty:
        return {
            "available": False,
            "row_count": None,
            "historical_start": None,
            "historical_end": None,
            "prediction_start": None,
            "prediction_end": None,
            "feature_columns": historical_metadata.get("feature_columns") if historical_metadata else None,
            "formula_version": clean_string(historical_metadata.get("formula_version")) if historical_metadata else None,
            "run_id": clean_string(historical_metadata.get("run_id")) if historical_metadata else None,
        }

    return {
        "available": True,
        "row_count": int(len(historical_frame)),
        "historical_start": clean_string(historical_frame["date"].min()),
        "historical_end": clean_string(historical_frame["date"].max()),
        "prediction_start": clean_string(historical_frame["target_date"].min()),
        "prediction_end": clean_string(historical_frame["target_date"].max()),
        "feature_columns": historical_metadata.get("feature_columns") if historical_metadata else None,
        "formula_version": clean_string(historical_metadata.get("formula_version")) if historical_metadata else None,
        "run_id": clean_string(historical_metadata.get("run_id")) if historical_metadata else None,
    }


def build_evaluation_mode_summary(
    *,
    prediction_mode: str,
    metrics_frame: pd.DataFrame,
    mode_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build one offline-evaluation mode summary from saved Stage 9 outputs."""

    mode_metrics = metrics_frame.loc[metrics_frame["prediction_mode"] == prediction_mode].copy()
    metrics_by_subset = mode_summary.get("metrics_by_subset", {}) or {}
    subset_notes = {
        subset_name: build_subset_note_payload(metrics_by_subset.get(subset_name, {}) or {})
        for subset_name in (
            ALL_ROWS_SUBSET_NAME,
            NEWS_HEAVY_SUBSET_NAME,
            ZERO_NEWS_SUBSET_NAME,
        )
    }
    return {
        "prediction_mode": prediction_mode,
        "predictions_path": clean_string(mode_summary.get("predictions_path")),
        "headline_row_count": coerce_optional_int(mode_summary.get("headline_row_count")),
        "news_heavy_row_count": coerce_optional_int(mode_summary.get("news_heavy_row_count")),
        "zero_news_row_count": coerce_optional_int(mode_summary.get("zero_news_row_count")),
        "variants": {
            variant_name: build_metric_variant_snapshot(
                prediction_mode=prediction_mode,
                variant_name=variant_name,
                metrics_frame=mode_metrics,
            )
            for variant_name in FINAL_REVIEW_VARIANT_ORDER
        },
        "subset_notes": subset_notes,
        "evidence_summary": mode_summary.get("evidence_summary", {}) or {},
    }


def build_metric_variant_snapshot(
    *,
    prediction_mode: str,
    variant_name: str,
    metrics_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Extract one variant snapshot from the saved Stage 9 metrics CSV."""

    variant_column_name = "variant_name" if "variant_name" in metrics_frame.columns else "model_variant"
    variant_rows = metrics_frame.loc[metrics_frame[variant_column_name] == variant_name].copy()
    subsets: dict[str, dict[str, Any]] = {}
    for subset_name in (ALL_ROWS_SUBSET_NAME, NEWS_HEAVY_SUBSET_NAME, ZERO_NEWS_SUBSET_NAME):
        subset_row_frame = variant_rows.loc[variant_rows["subset_name"] == subset_name]
        if subset_row_frame.empty:
            subsets[subset_name] = {"available": False}
            continue
        row = subset_row_frame.iloc[0]
        subsets[subset_name] = {
            "available": True,
            "row_count": coerce_optional_int(row.get("row_count")),
            "positive_rate": coerce_optional_float(row.get("positive_rate")),
            "accuracy": coerce_optional_float(row.get("accuracy")),
            "precision": coerce_optional_float(row.get("precision")),
            "recall": coerce_optional_float(row.get("recall")),
            "f1": coerce_optional_float(row.get("f1")),
            "roc_auc": coerce_optional_float(row.get("roc_auc")),
            "log_loss": coerce_optional_float(row.get("log_loss")),
            "brier_score": coerce_optional_float(row.get("brier_score")),
        }
    return {
        "prediction_mode": prediction_mode,
        "variant_name": variant_name,
        "subsets": subsets,
    }


def build_subset_note_payload(subset_summary: dict[str, Any]) -> dict[str, Any]:
    """Keep a compact, report-friendly copy of one Stage 9 subset note."""

    return {
        "note": clean_string(subset_summary.get("note")),
        "comparison": subset_summary.get("comparison"),
    }


def build_claim_checks(
    *,
    offline_summary_payload: dict[str, Any],
    pilot_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build explicit report guardrails tied to saved evidence."""

    coverage_status = clean_string(pilot_summary.get("coverage_status")) or PILOT_COVERAGE_UNAVAILABLE
    expected_pair_count = coerce_optional_int(pilot_summary.get("expected_pair_count")) or 0
    available_pair_count = coerce_optional_int(pilot_summary.get("available_pair_count")) or 0
    claim_notes = [
        "This report summarizes saved offline evaluation artifacts and any saved pilot logs in the requested window.",
        "The output is a prototype research review and is not trading advice.",
    ]
    if coverage_status == PILOT_COVERAGE_UNAVAILABLE:
        claim_notes.append("Operational pilot reliability cannot be claimed because no pilot rows were available.")
    elif coverage_status == PILOT_COVERAGE_PARTIAL:
        claim_notes.append(
            "Operational pilot reliability cannot be claimed because the requested pilot window is incomplete."
        )
    else:
        claim_notes.append(
            "Pilot evidence exists for the requested window, but the scope remains single-stock and limited."
        )
    claim_notes.extend(
        f"Stage 9 manual follow-up: {action}"
        for action in (offline_summary_payload.get("manual_actions", []) or [])
        if clean_string(action) is not None
    )
    return {
        "offline_claims_are_artifact_backed": True,
        "pilot_claims_are_artifact_backed": available_pair_count > 0,
        "operational_reliability_supported": coverage_status == PILOT_COVERAGE_COMPLETE,
        "coverage_status": coverage_status,
        "expected_pair_count": expected_pair_count,
        "available_pair_count": available_pair_count,
        "notes": claim_notes,
    }


def build_limitations(pilot_summary: dict[str, Any]) -> list[str]:
    """Return deterministic Stage 11 limitations."""

    limitations = [
        "Single-stock v1 scope limits generalization beyond the configured ticker and exchange.",
        "Offline evaluation is historical and does not remove market-noise or regime-shift risk.",
        "Pilot evidence is observational and limited to the requested trading window.",
        "Free-data ingestion and publisher access can be incomplete or delayed, especially on sparse-news days.",
        "This workflow is a prototype review package and not a production trading system or investment advice.",
    ]
    if clean_string(pilot_summary.get("coverage_status")) != PILOT_COVERAGE_COMPLETE:
        limitations.append(
            "Pilot completeness is not yet full for the requested window, so operational claims must stay conservative."
        )
    return limitations


def build_lessons_learned(pilot_summary: dict[str, Any]) -> list[str]:
    """Build evidence-backed lessons from the pilot summary."""

    lessons = [
        "Saved artifact traceability makes the final review readable without opening raw run folders.",
        "Disagreement tracking between baseline and enhanced models is operationally useful even when correctness is still pending.",
    ]
    if int((pilot_summary.get("overall", {}) or {}).get("fallback_heavy_count", 0) or 0) > 0:
        lessons.append(
            "Fallback-heavy days should be reviewed manually because degraded article detail can reduce news-signal quality."
        )
    if int((pilot_summary.get("overall", {}) or {}).get("zero_news_count", 0) or 0) > 0:
        lessons.append(
            "Zero-news handling keeps inference running, but those days still deserve separate interpretation."
        )
    if clean_string(pilot_summary.get("coverage_status")) != PILOT_COVERAGE_COMPLETE:
        lessons.append(
            "The Stage 10 pilot still needs more real trading-window runs and actual-outcome backfills."
        )
    return lessons


def build_traceability_summary(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    offline_summary_payload: dict[str, Any],
    pilot_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build artifact-path and run-id traceability for the final report."""

    baseline_metadata_path = path_manager.build_baseline_model_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    baseline_metadata = load_optional_json_object(baseline_metadata_path)
    enhanced_metadata_by_mode: dict[str, dict[str, Any] | None] = {}
    for prediction_mode in PILOT_PREDICTION_MODES:
        enhanced_metadata_by_mode[prediction_mode] = load_optional_json_object(
            path_manager.build_enhanced_model_metadata_path(
                settings.ticker.symbol,
                settings.ticker.exchange,
                prediction_mode,
            )
        )
    return {
        "offline_evaluation": {
            "run_id": clean_string(offline_summary_payload.get("run_id")),
            "metrics_path": clean_string(offline_summary_payload.get("metrics_path")),
            "summary_markdown_path": clean_string(offline_summary_payload.get("summary_markdown_path")),
            "source_historical_feature_path": clean_string(
                offline_summary_payload.get("source_historical_feature_path")
            ),
            "source_historical_metadata_path": clean_string(
                offline_summary_payload.get("source_historical_metadata_path")
            ),
            "source_news_feature_path": clean_string(
                offline_summary_payload.get("source_news_feature_path")
            ),
            "source_news_metadata_path": clean_string(
                offline_summary_payload.get("source_news_metadata_path")
            ),
        },
        "models": {
            "baseline": {
                "metadata_path": str(baseline_metadata_path),
                "run_id": clean_string(baseline_metadata.get("run_id")) if baseline_metadata else None,
            },
            "enhanced": {
                prediction_mode: {
                    "metadata_path": str(
                        path_manager.build_enhanced_model_metadata_path(
                            settings.ticker.symbol,
                            settings.ticker.exchange,
                            prediction_mode,
                        )
                    ),
                    "run_id": clean_string(metadata.get("run_id")) if metadata else None,
                }
                for prediction_mode, metadata in enhanced_metadata_by_mode.items()
            },
        },
        "pilot_logs": {
            prediction_mode: {
                "log_path": str(
                    path_manager.build_pilot_log_path(
                        settings.ticker.symbol,
                        settings.ticker.exchange,
                        prediction_mode,
                    )
                ),
                "log_exists": bool(
                    path_manager.build_pilot_log_path(
                        settings.ticker.symbol,
                        settings.ticker.exchange,
                        prediction_mode,
                    ).exists()
                ),
                "selected_row_count": int(
                    ((pilot_summary.get("per_mode", {}) or {}).get(prediction_mode, {}) or {}).get(
                        "selected_row_count",
                        0,
                    )
                    or 0
                ),
            }
            for prediction_mode in PILOT_PREDICTION_MODES
        },
    }


def render_final_review_markdown(summary_payload: dict[str, Any]) -> str:
    """Render the human-readable Stage 11 review."""

    offline_summary = summary_payload["offline_evaluation"]
    pilot_summary = summary_payload["pilot_summary"]
    lines = [
        "# Kubera Final Review",
        "",
        f"- Ticker: {summary_payload['ticker']}",
        f"- Exchange: {summary_payload['exchange']}",
        f"- Pilot window: {summary_payload['pilot_window']['start_date']} to {summary_payload['pilot_window']['end_date']}",
        f"- Final review JSON: {summary_payload['summary_json_path']}",
        f"- Final review Markdown: {summary_payload['summary_markdown_path']}",
        "",
        "## Evaluation Summary",
    ]

    historical_summary = offline_summary["historical_features"]
    if historical_summary["available"]:
        lines.append(
            f"- Stage 3 historical rows: {historical_summary['row_count']} from {historical_summary['historical_start']} to {historical_summary['historical_end']}"
        )
        lines.append(
            f"- Stage 3 prediction coverage: {historical_summary['prediction_start']} to {historical_summary['prediction_end']}"
        )
    else:
        lines.append("- Stage 3 historical feature coverage: unavailable")

    article_coverage = offline_summary["article_coverage"]
    if article_coverage["available"]:
        lines.append(
            f"- Stage 5 article coverage: {article_coverage['row_count']} rows from {article_coverage['coverage_start']} to {article_coverage['coverage_end']}"
        )
    else:
        lines.append("- Stage 5 article coverage metadata: unavailable")

    extraction_summary = offline_summary["extraction_summary"]
    if extraction_summary["available"]:
        lines.append(
            f"- Stage 6 extraction success rate: {format_optional_metric(extraction_summary['success_rate'])}"
        )
        lines.append(
            f"- Stage 6 fallback usage rate: {format_optional_metric(extraction_summary['fallback_usage_rate'])}"
        )
    else:
        lines.append("- Stage 6 extraction metadata: unavailable")

    news_feature_summary = offline_summary["news_feature_summary"]
    if news_feature_summary["available"]:
        lines.append(
            f"- Stage 7 zero-news rows: {news_feature_summary['zero_news_row_count']} of {news_feature_summary['output_row_count']}"
        )
    else:
        lines.append("- Stage 7 news feature metadata: unavailable")
    lines.append("")

    for prediction_mode, mode_summary in offline_summary["per_mode"].items():
        lines.append(f"## Offline {prediction_mode}")
        lines.append(f"- Held-out rows: {mode_summary['headline_row_count']}")
        lines.append(f"- News-heavy rows: {mode_summary['news_heavy_row_count']}")
        lines.append(f"- Zero-news rows: {mode_summary['zero_news_row_count']}")
        for variant_name in (ENHANCED_VARIANT_NAME, BASELINE_VARIANT_NAME, MAJORITY_VARIANT_NAME):
            variant_summary = mode_summary["variants"][variant_name]["subsets"][ALL_ROWS_SUBSET_NAME]
            if variant_summary["available"]:
                lines.append(
                    f"- {variant_name}: accuracy {format_optional_metric(variant_summary['accuracy'])}, f1 {format_optional_metric(variant_summary['f1'])}"
                )
        for subset_name, subset_payload in mode_summary["subset_notes"].items():
            if subset_payload["note"] is not None:
                lines.append(f"- {subset_name}: {subset_payload['note']}")
        lines.append("")

    lines.extend(
        [
            "## Pilot Summary",
            f"- Coverage status: {pilot_summary['coverage_status']}",
            f"- Expected mode-day pairs: {pilot_summary['expected_pair_count']}",
            f"- Available mode-day pairs: {pilot_summary['available_pair_count']}",
            f"- Missing mode-day pairs: {len(pilot_summary['missing_expected_pairs'])}",
        ]
    )
    for issue in pilot_summary["operational_issues"]:
        lines.append(f"- Operational issue: {issue}")
    lines.append("")

    for prediction_mode, mode_summary in pilot_summary["per_mode"].items():
        lines.append(f"## Pilot {prediction_mode}")
        lines.append(f"- Selected rows: {mode_summary['selected_row_count']}")
        lines.append(f"- Backfilled rows: {mode_summary['backfilled_row_count']}")
        lines.append(f"- Baseline accuracy: {format_optional_metric(mode_summary['baseline_accuracy'])}")
        lines.append(f"- Enhanced accuracy: {format_optional_metric(mode_summary['enhanced_accuracy'])}")
        lines.append(f"- Disagreement rate: {format_optional_metric(mode_summary['disagreement_rate'])}")
        lines.append(f"- Fallback-heavy rows: {mode_summary['fallback_heavy_count']}")
        lines.append(f"- Zero-news rows: {mode_summary['zero_news_count']}")
        if mode_summary["missing_market_session_dates"]:
            lines.append(
                f"- Missing market sessions: {', '.join(mode_summary['missing_market_session_dates'])}"
            )
        lines.append("")

    lines.extend(
        [
            "## Pilot Daily Table",
            "",
            "| Market Session | Mode | Prediction Date | Status | Baseline | Enhanced | Actual | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in pilot_summary["daily_prediction_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["market_session_date"]),
                    str(row["prediction_mode"]),
                    str(row["prediction_date"] or "-"),
                    str(row["status"]),
                    format_direction_with_probability(
                        row["baseline_predicted_next_day_direction"],
                        row["baseline_predicted_probability_up"],
                    ),
                    format_direction_with_probability(
                        row["enhanced_predicted_next_day_direction"],
                        row["enhanced_predicted_probability_up"],
                    ),
                    format_actual_outcome(
                        row["actual_next_day_direction"],
                        row["actual_outcome_status"],
                    ),
                    ", ".join(row["notes"]) if row["notes"] else "-",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Limitations")
    for limitation in summary_payload["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")

    lines.append("## Lessons Learned")
    for lesson in summary_payload["lessons_learned"]:
        lines.append(f"- {lesson}")
    lines.append("")

    lines.append("## Claim Checks")
    for note in summary_payload["claim_checks"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Traceability")
    traceability = summary_payload["traceability"]
    lines.append(
        f"- Offline metrics: {traceability['offline_evaluation']['metrics_path']}"
    )
    lines.append(
        f"- Offline summary: {traceability['offline_evaluation']['summary_markdown_path']}"
    )
    lines.append(
        f"- Baseline model metadata: {traceability['models']['baseline']['metadata_path']}"
    )
    for prediction_mode, pilot_log in traceability["pilot_logs"].items():
        lines.append(f"- Pilot log {prediction_mode}: {pilot_log['log_path']}")
    lines.append("")

    lines.append("## Manual Follow-Up")
    for action in summary_payload["manual_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def load_optional_json_object(path: Path | None) -> dict[str, Any] | None:
    """Load one optional JSON object when it exists and is valid."""

    if path is None:
        return None
    return load_optional_json(path)


def decode_json_cell(value: Any, *, default: Any) -> Any:
    """Decode one structured CSV cell or fall back to the provided default."""

    if value is None or pd.isna(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def clean_string(value: Any) -> str | None:
    """Normalize one optional text value."""

    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def coerce_optional_path(value: Any) -> Path | None:
    """Resolve one optional path-like value."""

    cleaned = clean_string(value)
    if cleaned is None:
        return None
    return Path(cleaned).expanduser().resolve()


def coerce_optional_int(value: Any) -> int | None:
    """Coerce one optional numeric value to int."""

    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_float(value: Any) -> float | None:
    """Coerce one optional numeric value to float."""

    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_bool(value: Any) -> bool | None:
    """Coerce one optional CSV cell to bool."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    cleaned = str(value).strip().lower()
    if cleaned in {"true", "1", "yes"}:
        return True
    if cleaned in {"false", "0", "no"}:
        return False
    return None


def calculate_optional_accuracy(values: list[int]) -> float | None:
    """Calculate an accuracy-style mean when at least one observation exists."""

    if not values:
        return None
    return float(sum(values) / len(values))


def format_optional_metric(value: float | int | None) -> str:
    """Format one optional metric for markdown output."""

    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def format_direction_with_probability(
    direction: int | None,
    probability_up: float | None,
) -> str:
    """Format one prediction direction and probability pair."""

    if direction is None:
        return "-"
    label = "up" if int(direction) == 1 else "down"
    if probability_up is None:
        return label
    return f"{label} ({probability_up:.3f})"


def format_actual_outcome(direction: int | None, actual_status: str | None) -> str:
    """Format one actual outcome cell for markdown."""

    if actual_status == ACTUAL_STATUS_BACKFILLED and direction is not None:
        return "up" if int(direction) == 1 else "down"
    if actual_status == ACTUAL_STATUS_MARKET_DATA_UNAVAILABLE:
        return "market_data_unavailable"
    if actual_status == "missing":
        return "missing"
    return "pending"


def parse_review_date(raw_value: str) -> date:
    """Parse one Stage 11 CLI date."""

    return date.fromisoformat(raw_value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 11 command arguments."""

    parser = argparse.ArgumentParser(description="Run Kubera Stage 11 final review.")
    parser.add_argument(
        "--pilot-start-date",
        required=True,
        help="Pilot market-session window start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--pilot-end-date",
        required=True,
        help="Pilot market-session window end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--refresh-offline-evaluation",
        action="store_true",
        help="Rebuild Stage 9 artifacts before generating the final review.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 11 final-review CLI."""

    args = parse_args(argv)
    settings = load_settings()
    generate_final_review(
        settings,
        pilot_start_date=parse_review_date(args.pilot_start_date),
        pilot_end_date=parse_review_date(args.pilot_end_date),
        refresh_offline_evaluation=args.refresh_offline_evaluation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
