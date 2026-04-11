"""Stage 9 offline evaluation and benchmark comparison for Kubera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kubera.config import (
    AppSettings,
    NewsFeatureSettings,
    load_settings,
    resolve_runtime_settings,
)
from kubera.features.news_features import (
    NEWS_SIGNAL_STATE_CARRIED_FORWARD,
    NEWS_SIGNAL_STATE_FALLBACK_HEAVY,
    NEWS_SIGNAL_STATE_FRESH,
    NEWS_SIGNAL_STATE_ZERO,
    build_news_features,
    determine_news_signal_state,
    resolve_supported_prediction_modes,
)
from kubera.models.common import (
    blend_probabilities,
    compute_news_context_weight,
    compute_prediction_metrics,
    resolve_selective_prediction,
)
from kubera.models.train_baseline import (
    BaselineDataset,
    build_split_summary as build_baseline_split_summary,
    infer_feature_metadata_path,
    load_baseline_dataset,
    split_baseline_dataset,
)
from kubera.models.train_enhanced import (
    COMPARISON_NEWS_CONTEXT_COLUMNS,
    EnhancedDataset,
    NewsFeatureDataset,
    build_merged_enhanced_dataset,
    infer_news_feature_metadata_path,
    load_cached_merged_enhanced_dataset,
    load_news_feature_dataset,
    save_merged_enhanced_dataset,
    split_enhanced_dataset,
    train_enhanced_models,
)
from kubera.utils.hashing import compute_file_sha256
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.data_quality import build_data_quality_payload
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot


IDENTITY_COLUMNS = (
    "historical_date",
    "prediction_date",
    "ticker",
    "exchange",
    "prediction_mode",
)
BASELINE_VARIANT_NAME = "baseline_historical_only"
ENHANCED_VARIANT_NAME = "enhanced_full"
BLENDED_VARIANT_NAME = "blended_enhanced"
MAJORITY_VARIANT_NAME = "naive_majority_class"
PREVIOUS_DAY_VARIANT_NAME = "naive_previous_day_direction"
SENTIMENT_ABLATION_VARIANT_NAME = "ablation_sentiment_only"
EVENT_ABLATION_VARIANT_NAME = "ablation_event_counts_only"
NO_CONFIDENCE_VARIANT_NAME = "ablation_full_without_confidence"
NO_FALLBACK_VARIANT_NAME = "ablation_full_without_fallback_penalties"
ALL_ROWS_SUBSET_NAME = "all_rows"
NEWS_HEAVY_SUBSET_NAME = "news_heavy_rows"
ZERO_NEWS_SUBSET_NAME = "zero_news_rows"
FRESH_NEWS_SUBSET_NAME = "fresh_news_rows"
NO_FRESH_NEWS_SUBSET_NAME = "no_fresh_news_rows"
CARRIED_FORWARD_SUBSET_NAME = "carried_forward_rows"
NOT_CARRIED_FORWARD_SUBSET_NAME = "not_carried_forward_rows"
FALLBACK_HEAVY_SUBSET_NAME = "fallback_heavy_rows"
NOT_FALLBACK_HEAVY_SUBSET_NAME = "not_fallback_heavy_rows"
HAS_NEWS_SUBSET_NAME = "has_news_rows"
ABSTAIN_ELIMINATED_SUBSET_NAME = "abstain_eliminated_rows"
HIGH_CONFIDENCE_SUBSET_NAME = "high_confidence_rows"
HIGH_QUALITY_SUBSET_NAME = "quality_grade_a_or_b_rows"
LOW_QUALITY_SUBSET_NAME = "quality_grade_c_or_worse_rows"

SENTIMENT_NEWS_COLUMNS = (
    "news_avg_sentiment",
    "news_bullish_article_count",
    "news_bearish_article_count",
    "news_neutral_article_count",
    "news_weighted_sentiment_score",
    "news_weighted_bullish_score",
    "news_weighted_bearish_score",
)
DIRECT_CONFIDENCE_COLUMNS = (
    "news_avg_confidence",
    "news_weighted_confidence_score",
)
METRIC_DIRECTION = {
    "accuracy": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    "roc_auc": "higher",
    "log_loss": "lower",
    "brier_score": "lower",
}
SUMMARY_METRIC_ORDER = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "brier_score",
)


class OfflineEvaluationError(RuntimeError):
    """Raised when the offline evaluation pipeline cannot continue."""


@dataclass(frozen=True)
class OfflineEvaluationModeResult:
    prediction_mode: str
    predictions_path: Path
    headline_row_count: int
    news_heavy_row_count: int
    zero_news_row_count: int


@dataclass(frozen=True)
class OfflineEvaluationResult:
    metrics_path: Path
    summary_json_path: Path
    summary_markdown_path: Path
    mode_results: dict[str, OfflineEvaluationModeResult]


def resolve_historical_feature_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    historical_feature_path: str | Path | None,
) -> Path:
    """Resolve the active Stage 3 historical feature table."""

    if historical_feature_path is not None:
        return Path(historical_feature_path).expanduser().resolve()
    return path_manager.build_historical_feature_table_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def resolve_news_feature_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    news_feature_path: str | Path | None,
) -> Path:
    """Resolve the active default Stage 7 news feature table."""

    if news_feature_path is not None:
        return Path(news_feature_path).expanduser().resolve()
    return path_manager.build_news_feature_table_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def resolve_extraction_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    extraction_table_path: str | Path | None,
) -> Path:
    """Resolve the Stage 6 extraction table used for formula-level ablations."""

    if extraction_table_path is not None:
        return Path(extraction_table_path).expanduser().resolve()
    return path_manager.build_processed_llm_extractions_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def load_prediction_frame(path: Path, artifact_label: str) -> pd.DataFrame:
    """Load one saved prediction CSV and fail clearly when it is unavailable."""

    if not path.exists():
        raise OfflineEvaluationError(f"{artifact_label} file does not exist: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise OfflineEvaluationError(f"{artifact_label} file is empty: {path}") from exc


def load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load one JSON file when it exists and is valid."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def compute_optional_hash(path: Path) -> str | None:
    """Hash one artifact when it exists and otherwise return None."""

    if not path.exists():
        return None
    return compute_file_sha256(path)


def split_mode_dataset(
    dataset: EnhancedDataset,
    *,
    prediction_mode: str,
    settings: AppSettings,
) -> Any:
    """Return the train, validation, and test rows for one mode."""

    mode_frame = dataset.dataset_frame.loc[
        dataset.dataset_frame["prediction_mode"] == prediction_mode
    ].copy()
    if mode_frame.empty:
        raise OfflineEvaluationError(
            f"Enhanced dataset does not contain rows for prediction mode: {prediction_mode}"
        )
    return split_enhanced_dataset(mode_frame, settings.enhanced_model)


def load_or_build_merged_dataset(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    historical_dataset: BaselineDataset,
    news_dataset: NewsFeatureDataset,
    run_context: Any,
    logger: Any,
    artifact_variant: str | None,
) -> EnhancedDataset:
    """Load a cached merged dataset or persist a fresh one."""

    merged_dataset_path = path_manager.build_merged_enhanced_dataset_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        artifact_variant=artifact_variant,
    )
    merged_metadata_path = path_manager.build_merged_enhanced_dataset_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        artifact_variant=artifact_variant,
    )
    cached_dataset = load_cached_merged_enhanced_dataset(
        path=merged_dataset_path,
        metadata_path=merged_metadata_path,
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        lag_windows=settings.historical_features.lag_windows,
    )
    if cached_dataset is not None:
        logger.info(
            "Merged dataset ready from cache | ticker=%s | exchange=%s | variant=%s | rows=%s | merged_csv=%s",
            settings.ticker.symbol,
            settings.ticker.exchange,
            artifact_variant or "default",
            len(cached_dataset.dataset_frame),
            merged_dataset_path,
        )
        return cached_dataset

    merged_dataset = build_merged_enhanced_dataset(
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        lag_windows=settings.historical_features.lag_windows,
    )
    save_merged_enhanced_dataset(
        path=merged_dataset_path,
        metadata_path=merged_metadata_path,
        settings=settings,
        dataset=merged_dataset,
        artifact_variant=artifact_variant,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
    )
    logger.info(
        "Merged dataset built for offline evaluation | ticker=%s | exchange=%s | variant=%s | rows=%s | merged_csv=%s",
        settings.ticker.symbol,
        settings.ticker.exchange,
        artifact_variant or "default",
        len(merged_dataset.dataset_frame),
        merged_dataset_path,
    )
    return merged_dataset


def stage8_artifacts_are_aligned(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    historical_dataset: BaselineDataset,
    news_dataset: NewsFeatureDataset,
    merged_dataset: EnhancedDataset,
    logger: Any,
) -> bool:
    """Return True when the saved Stage 8 artifacts match the current inputs and splits."""

    baseline_metadata = load_optional_json(
        path_manager.build_baseline_model_metadata_path(
            settings.ticker.symbol,
            settings.ticker.exchange,
        )
    )
    if baseline_metadata is None:
        return False

    baseline_split = split_baseline_dataset(
        historical_dataset.dataset_frame,
        settings.baseline_model,
    )
    expected_baseline_summary = {
        "train": build_baseline_split_summary(baseline_split.train_frame),
        "validation": build_baseline_split_summary(baseline_split.validation_frame),
        "test": build_baseline_split_summary(baseline_split.test_frame),
    }
    if baseline_metadata.get("source_feature_table_hash") != historical_dataset.source_feature_table_hash:
        return False
    if baseline_metadata.get("source_feature_metadata_hash") != historical_dataset.source_feature_metadata_hash:
        return False
    if baseline_metadata.get("split_summary") != expected_baseline_summary:
        return False
    if not path_manager.build_baseline_predictions_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    ).exists():
        return False

    for prediction_mode in news_dataset.supported_prediction_modes:
        mode_split = split_mode_dataset(
            merged_dataset,
            prediction_mode=prediction_mode,
            settings=settings,
        )
        expected_mode_summary = {
            "train": {
                "row_count": int(len(mode_split.train_frame)),
                "date_start": str(mode_split.train_frame.iloc[0]["prediction_date"]),
                "date_end": str(mode_split.train_frame.iloc[-1]["prediction_date"]),
            },
            "validation": {
                "row_count": int(len(mode_split.validation_frame)),
                "date_start": str(mode_split.validation_frame.iloc[0]["prediction_date"]),
                "date_end": str(mode_split.validation_frame.iloc[-1]["prediction_date"]),
            },
            "test": {
                "row_count": int(len(mode_split.test_frame)),
                "date_start": str(mode_split.test_frame.iloc[0]["prediction_date"]),
                "date_end": str(mode_split.test_frame.iloc[-1]["prediction_date"]),
            },
        }
        enhanced_metadata = load_optional_json(
            path_manager.build_enhanced_model_metadata_path(
                settings.ticker.symbol,
                settings.ticker.exchange,
                prediction_mode,
            )
        )
        if enhanced_metadata is None:
            return False
        if enhanced_metadata.get("source_historical_feature_hash") != historical_dataset.source_feature_table_hash:
            return False
        if enhanced_metadata.get("source_historical_metadata_hash") != historical_dataset.source_feature_metadata_hash:
            return False
        if enhanced_metadata.get("source_news_feature_hash") != news_dataset.source_feature_table_hash:
            return False
        if enhanced_metadata.get("source_news_metadata_hash") != news_dataset.source_feature_metadata_hash:
            return False
        if enhanced_metadata.get("prediction_mode") != prediction_mode:
            return False
        if enhanced_metadata.get("split_summary") != expected_mode_summary:
            return False
        if not path_manager.build_enhanced_predictions_path(
            settings.ticker.symbol,
            settings.ticker.exchange,
            prediction_mode,
        ).exists():
            return False

    logger.info(
        "Stage 8 artifacts already align with the current Stage 3 and Stage 7 inputs."
    )
    return True


def should_run_training_for_current_features(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    historical_feature_path: str | Path | None = None,
    news_feature_path: str | Path | None = None,
) -> tuple[bool, str]:
    """Return whether the full training pipeline is needed for the current Stage 3/7 inputs.

    When Stage 4/8 saved artifacts match hashes and splits for the loaded feature tables,
    returns ``(False, ...)`` so callers can skip ``kubera train``. On missing inputs,
    load failures, or misalignment, returns ``(True, ...)`` with a short reason.

    Uses :func:`stage8_artifacts_are_aligned` (same policy as offline evaluation).
    """

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(runtime_settings, path_manager)
    logger = configure_logging(run_context, runtime_settings.run.log_level)

    try:
        historical_feature_table_path = resolve_historical_feature_table_path(
            runtime_settings,
            path_manager=path_manager,
            historical_feature_path=historical_feature_path,
        )
        if not historical_feature_table_path.exists():
            return (
                True,
                f"historical feature table not found: {historical_feature_table_path}",
            )

        historical_dataset = load_baseline_dataset(
            feature_table_path=historical_feature_table_path,
            feature_metadata_path=infer_feature_metadata_path(historical_feature_table_path),
            ticker=runtime_settings.ticker.symbol,
            exchange=runtime_settings.ticker.exchange,
        )

        news_feature_table_path = resolve_news_feature_table_path(
            runtime_settings,
            path_manager=path_manager,
            news_feature_path=news_feature_path,
        )
        if not news_feature_table_path.exists():
            return (
                True,
                f"news feature table not found: {news_feature_table_path}",
            )

        news_dataset = load_news_feature_dataset(
            news_feature_table_path=news_feature_table_path,
            news_feature_metadata_path=infer_news_feature_metadata_path(news_feature_table_path),
            ticker=runtime_settings.ticker.symbol,
            exchange=runtime_settings.ticker.exchange,
            supported_prediction_modes=resolve_supported_prediction_modes(
                runtime_settings.market.supported_prediction_modes
            ),
        )

        merged_dataset = load_or_build_merged_dataset(
            settings=runtime_settings,
            path_manager=path_manager,
            historical_dataset=historical_dataset,
            news_dataset=news_dataset,
            run_context=run_context,
            logger=logger,
            artifact_variant=None,
        )
    except Exception as exc:
        return (True, f"cannot verify training need ({type(exc).__name__}): {exc}")

    aligned = stage8_artifacts_are_aligned(
        settings=runtime_settings,
        path_manager=path_manager,
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        merged_dataset=merged_dataset,
        logger=logger,
    )
    if aligned:
        return (False, "stage 4/8 artifacts match current feature tables")
    return (True, "stage 4/8 artifacts missing or out of date vs current feature tables")


def build_mode_evaluation_frame(
    *,
    settings: AppSettings,
    enhanced_predictions_frame: pd.DataFrame,
    baseline_predictions_frame: pd.DataFrame,
    prediction_mode: str,
    target_column: str,
    news_heavy_min_article_count: int,
    stage5_metadata: dict[str, Any] | None,
    stage6_metadata: dict[str, Any] | None,
    stage7_metadata: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build the wide per-mode evaluation table on held-out headline rows."""

    required_enhanced_columns = (
        "split",
        "historical_date",
        "prediction_date",
        "ticker",
        "exchange",
        "prediction_mode",
        target_column,
        "predicted_next_day_direction",
        "raw_predicted_probability_up",
        "calibrated_predicted_probability_up",
        "predicted_probability_up",
    ) + COMPARISON_NEWS_CONTEXT_COLUMNS
    missing_enhanced_columns = [
        column
        for column in required_enhanced_columns
        if column not in enhanced_predictions_frame.columns
    ]
    if missing_enhanced_columns:
        raise OfflineEvaluationError(
            f"Stage 8 enhanced predictions are missing required columns for offline evaluation: {missing_enhanced_columns}"
        )

    enhanced_test_frame = enhanced_predictions_frame.loc[
        enhanced_predictions_frame["split"] == "test",
        required_enhanced_columns,
    ].copy()
    if enhanced_test_frame.empty:
        raise OfflineEvaluationError(
            f"Stage 8 enhanced predictions do not contain any held-out test rows for {prediction_mode}."
        )
    if set(enhanced_test_frame["prediction_mode"].tolist()) != {prediction_mode}:
        raise OfflineEvaluationError(
            f"Stage 8 enhanced predictions mix modes inside the {prediction_mode} evaluation file."
        )

    evaluation_frame = enhanced_test_frame.rename(
        columns={
            "predicted_next_day_direction": f"{ENHANCED_VARIANT_NAME}_predicted_next_day_direction",
            "raw_predicted_probability_up": f"{ENHANCED_VARIANT_NAME}_raw_predicted_probability_up",
            "calibrated_predicted_probability_up": (
                f"{ENHANCED_VARIANT_NAME}_calibrated_predicted_probability_up"
            ),
            "predicted_probability_up": f"{ENHANCED_VARIANT_NAME}_predicted_probability_up",
        }
    )

    # Add Blended Variant
    news_weights = evaluation_frame.apply(
        lambda row: compute_news_context_weight(
            news_article_count=row["news_article_count"],
            news_avg_confidence=row["news_avg_confidence"],
            has_fresh_news=row["has_fresh_news"],
            is_fallback_heavy=row["is_fallback_heavy"],
            is_carried_forward=row["is_carried_forward"],
        ),
        axis=1,
    )
    evaluation_frame["news_context_weight"] = news_weights

    required_baseline_columns = (
        "split",
        "date",
        "target_date",
        "ticker",
        "exchange",
        target_column,
        "predicted_next_day_direction",
        "raw_predicted_probability_up",
        "calibrated_predicted_probability_up",
        "predicted_probability_up",
    )
    missing_baseline_columns = [
        column
        for column in required_baseline_columns
        if column not in baseline_predictions_frame.columns
    ]
    if missing_baseline_columns:
        raise OfflineEvaluationError(
            f"Stage 4 baseline predictions are missing required columns for offline evaluation: {missing_baseline_columns}"
        )

    baseline_test_frame = baseline_predictions_frame.loc[
        baseline_predictions_frame["split"] == "test",
        required_baseline_columns,
    ].copy()
    baseline_test_frame = baseline_test_frame.rename(
        columns={
            "date": "historical_date",
            "target_date": "prediction_date",
            "predicted_next_day_direction": f"{BASELINE_VARIANT_NAME}_predicted_next_day_direction",
            "raw_predicted_probability_up": f"{BASELINE_VARIANT_NAME}_raw_predicted_probability_up",
            "calibrated_predicted_probability_up": (
                f"{BASELINE_VARIANT_NAME}_calibrated_predicted_probability_up"
            ),
            "predicted_probability_up": f"{BASELINE_VARIANT_NAME}_predicted_probability_up",
            target_column: f"{BASELINE_VARIANT_NAME}_{target_column}",
        }
    )

    evaluation_frame = evaluation_frame.merge(
        baseline_test_frame,
        how="left",
        on=("historical_date", "prediction_date", "ticker", "exchange"),
        validate="one_to_one",
    )
    if evaluation_frame[f"{BASELINE_VARIANT_NAME}_{target_column}"].isna().any():
        raise OfflineEvaluationError(
            f"Stage 4 baseline predictions do not align to the Stage 8 held-out rows for {prediction_mode}."
        )

    # Compute blended probabilities
    evaluation_frame[f"{BLENDED_VARIANT_NAME}_raw_predicted_probability_up"] = blend_probabilities(
        evaluation_frame[f"{BASELINE_VARIANT_NAME}_raw_predicted_probability_up"],
        evaluation_frame[f"{ENHANCED_VARIANT_NAME}_raw_predicted_probability_up"],
        evaluation_frame["news_context_weight"],
    )
    evaluation_frame[f"{BLENDED_VARIANT_NAME}_calibrated_predicted_probability_up"] = (
        blend_probabilities(
            evaluation_frame[f"{BASELINE_VARIANT_NAME}_calibrated_predicted_probability_up"],
            evaluation_frame[f"{ENHANCED_VARIANT_NAME}_calibrated_predicted_probability_up"],
            evaluation_frame["news_context_weight"],
        )
    )
    evaluation_frame[f"{BLENDED_VARIANT_NAME}_predicted_probability_up"] = evaluation_frame[
        f"{BLENDED_VARIANT_NAME}_calibrated_predicted_probability_up"
    ]
    evaluation_frame[f"{BLENDED_VARIANT_NAME}_predicted_next_day_direction"] = (
        evaluation_frame[f"{BLENDED_VARIANT_NAME}_calibrated_predicted_probability_up"] >= 0.5
    ).astype(int)

    if (
        evaluation_frame[f"{BASELINE_VARIANT_NAME}_{target_column}"].astype(int)
        != evaluation_frame[target_column].astype(int)
    ).any():
        raise OfflineEvaluationError(
            f"Stage 4 and Stage 8 targets disagree on the held-out rows for {prediction_mode}."
        )

    evaluation_frame = evaluation_frame.drop(
        columns=[f"{BASELINE_VARIANT_NAME}_{target_column}", "split_y"],
        errors="ignore",
    )
    evaluation_frame = evaluation_frame.rename(columns={"split_x": "split"})
    evaluation_frame["news_heavy_flag"] = (
        evaluation_frame["news_article_count"].astype(float) >= news_heavy_min_article_count
    )
    evaluation_frame["zero_news_flag"] = (
        evaluation_frame["news_article_count"].astype(float) == 0.0
    )
    evaluation_frame["news_signal_state"] = evaluation_frame.apply(
        lambda row: determine_news_signal_state(row.to_dict()),
        axis=1,
    )

    quality_payloads = [
        build_data_quality_payload(
            row_mapping=row_mapping,
            news_feature_row=row_mapping,
            stage5_metadata=stage5_metadata,
            stage6_metadata=stage6_metadata,
            stage7_metadata=stage7_metadata,
        )
        for row_mapping in evaluation_frame.to_dict(orient="records")
    ]
    evaluation_frame["data_quality_score"] = [
        float(payload["score"]) for payload in quality_payloads
    ]
    evaluation_frame["data_quality_grade"] = [
        str(payload["grade"]) for payload in quality_payloads
    ]

    selective_decisions = [
        resolve_selective_prediction(
            probability_up=float(
                row_mapping[f"{BLENDED_VARIANT_NAME}_calibrated_predicted_probability_up"]
            ),
            classification_threshold=settings.baseline_model.classification_threshold,
            low_conviction_threshold=settings.pilot.abstain_low_conviction_threshold,
            news_signal_state=str(row_mapping.get("news_signal_state") or "").strip() or None,
            data_quality_score=float(row_mapping["data_quality_score"]),
            data_quality_floor=settings.pilot.abstain_data_quality_floor,
            carried_forward_margin_penalty=(
                settings.pilot.abstain_carried_forward_margin_penalty
            ),
            degraded_margin_penalty=settings.pilot.abstain_degraded_margin_penalty,
        )
        for row_mapping in evaluation_frame.to_dict(orient="records")
    ]
    evaluation_frame["blended_selective_action"] = [
        decision.action for decision in selective_decisions
    ]
    evaluation_frame["abstain_flag"] = [decision.abstain for decision in selective_decisions]
    evaluation_frame["selective_probability_margin"] = [
        decision.probability_margin for decision in selective_decisions
    ]
    evaluation_frame["selective_required_margin"] = [
        decision.required_margin for decision in selective_decisions
    ]
    evaluation_frame["abstain_reason_codes_json"] = [
        json.dumps(list(decision.reasons)) for decision in selective_decisions
    ]
    evaluation_frame["high_confidence_flag"] = (
        evaluation_frame["selective_probability_margin"].astype(float)
        >= evaluation_frame["selective_required_margin"].astype(float).clip(lower=0.10)
    )
    return evaluation_frame.reset_index(drop=True)


def align_mode_frame(
    *,
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    target_column: str,
    artifact_label: str,
) -> pd.DataFrame:
    """Align one candidate mode frame to the saved Stage 8 test row order."""

    keyed_reference = reference_frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    keyed_reference["row_order"] = range(len(keyed_reference))
    candidate_columns = list(IDENTITY_COLUMNS) + list(
        candidate_frame.columns.difference(IDENTITY_COLUMNS)
    )
    keyed_candidate = candidate_frame.loc[:, candidate_columns].copy()
    merged = keyed_reference.merge(
        keyed_candidate,
        how="left",
        on=IDENTITY_COLUMNS,
        validate="one_to_one",
    )
    if merged[target_column].isna().any():
        raise OfflineEvaluationError(
            f"{artifact_label} do not align to the saved Stage 8 test rows."
        )
    if (
        merged[target_column].astype(int).tolist()
        != reference_frame[target_column].astype(int).tolist()
    ):
        raise OfflineEvaluationError(
            f"{artifact_label} do not share the same held-out labels as the saved Stage 8 test rows."
        )
    merged = merged.sort_values("row_order", ascending=True).drop(columns=["row_order"])
    return merged.reset_index(drop=True)


def add_majority_baseline_predictions(
    *,
    base_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    target_column: str,
) -> None:
    """Add the deterministic majority-class baseline predictions."""

    positive_rate = float(train_frame[target_column].astype(int).mean())
    predicted_label = int(positive_rate >= 0.5)
    base_frame[f"{MAJORITY_VARIANT_NAME}_predicted_next_day_direction"] = predicted_label
    base_frame[f"{MAJORITY_VARIANT_NAME}_predicted_probability_up"] = positive_rate


def add_previous_day_direction_predictions(
    *,
    base_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    """Add the deterministic previous-day-direction heuristic predictions."""

    base_frame[f"{PREVIOUS_DAY_VARIANT_NAME}_predicted_next_day_direction"] = (
        test_frame["ret_1d"].astype(float) >= 0.0
    ).astype(int)


def build_sentiment_feature_columns(dataset: EnhancedDataset) -> tuple[str, ...]:
    """Resolve the sentiment-only ablation feature set."""

    desired_columns = tuple(
        column for column in SENTIMENT_NEWS_COLUMNS if column in dataset.news_feature_columns
    )
    if not desired_columns:
        raise OfflineEvaluationError(
            "The merged dataset does not contain any sentiment ablation columns."
        )
    return dataset.historical_feature_columns + desired_columns


def build_event_feature_columns(dataset: EnhancedDataset) -> tuple[str, ...]:
    """Resolve the event-count-only ablation feature set."""

    event_columns = tuple(
        column for column in dataset.news_feature_columns if column.startswith("news_event_count_")
    )
    if not event_columns:
        raise OfflineEvaluationError(
            "The merged dataset does not contain any event-count columns."
        )
    return dataset.historical_feature_columns + event_columns


def build_no_confidence_feature_columns(dataset: EnhancedDataset) -> tuple[str, ...]:
    """Resolve the no-confidence ablation feature set."""

    news_columns = tuple(
        column for column in dataset.news_feature_columns if column not in DIRECT_CONFIDENCE_COLUMNS
    )
    if not news_columns:
        raise OfflineEvaluationError(
            "The no-confidence ablation removed every news feature column."
        )
    return dataset.historical_feature_columns + news_columns


def add_trained_ablation_predictions(
    *,
    base_frame: pd.DataFrame,
    aligned_test_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    settings: AppSettings,
    prediction_mode: str,
    variant_name: str,
) -> None:
    """Train one logistic-regression ablation and append its held-out predictions."""

    from kubera.models.train_enhanced import build_enhanced_prediction_frame, fit_enhanced_model

    saved_model = fit_enhanced_model(
        train_frame=train_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        enhanced_settings=settings.enhanced_model,
        random_seed=settings.run.random_seed,
        prediction_mode=prediction_mode,
    )
    prediction_frame = build_enhanced_prediction_frame(
        split_name="test",
        split_frame=aligned_test_frame,
        saved_model=saved_model,
    )
    base_frame[f"{variant_name}_predicted_next_day_direction"] = prediction_frame[
        "predicted_next_day_direction"
    ].astype(int)
    base_frame[f"{variant_name}_raw_predicted_probability_up"] = prediction_frame[
        "raw_predicted_probability_up"
    ].astype(float)
    base_frame[f"{variant_name}_calibrated_predicted_probability_up"] = prediction_frame[
        "calibrated_predicted_probability_up"
    ].astype(float)
    base_frame[f"{variant_name}_predicted_probability_up"] = prediction_frame[
        "predicted_probability_up"
    ].astype(float)


def build_variant_dataset(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    historical_dataset: BaselineDataset,
    extraction_table_path: Path,
    run_context: Any,
    logger: Any,
    artifact_variant: str,
    news_feature_settings: NewsFeatureSettings,
) -> EnhancedDataset:
    """Build or reuse one variant-specific Stage 7 and merged Stage 8 dataset."""

    feature_result = build_news_features(
        settings,
        extraction_table_path=extraction_table_path,
        artifact_variant=artifact_variant,
        news_feature_settings=news_feature_settings,
    )
    news_dataset = load_news_feature_dataset(
        news_feature_table_path=feature_result.feature_table_path,
        news_feature_metadata_path=feature_result.metadata_path,
        ticker=settings.ticker.symbol,
        exchange=settings.ticker.exchange,
        supported_prediction_modes=resolve_supported_prediction_modes(
            settings.market.supported_prediction_modes
        ),
    )
    return load_or_build_merged_dataset(
        settings=settings,
        path_manager=path_manager,
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        run_context=run_context,
        logger=logger,
        artifact_variant=artifact_variant,
    )


def build_mode_metrics_rows(
    *,
    prediction_mode: str,
    evaluation_frame: pd.DataFrame,
    target_column: str,
    logger: Any,
) -> list[dict[str, Any]]:
    """Compute one long-form metric table for a single prediction mode."""

    rows: list[dict[str, Any]] = []
    subsets = {
        ALL_ROWS_SUBSET_NAME: evaluation_frame,
        NEWS_HEAVY_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_heavy_flag"]
        ].copy(),
        ZERO_NEWS_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] == NEWS_SIGNAL_STATE_ZERO
        ].copy(),
        FRESH_NEWS_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] == NEWS_SIGNAL_STATE_FRESH
        ].copy(),
        NO_FRESH_NEWS_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] != NEWS_SIGNAL_STATE_FRESH
        ].copy(),
        CARRIED_FORWARD_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] == NEWS_SIGNAL_STATE_CARRIED_FORWARD
        ].copy(),
        NOT_CARRIED_FORWARD_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] != NEWS_SIGNAL_STATE_CARRIED_FORWARD
        ].copy(),
        FALLBACK_HEAVY_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] == NEWS_SIGNAL_STATE_FALLBACK_HEAVY
        ].copy(),
        NOT_FALLBACK_HEAVY_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_signal_state"] != NEWS_SIGNAL_STATE_FALLBACK_HEAVY
        ].copy(),
        HAS_NEWS_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["news_article_count"].astype(float) > 0.0
        ].copy(),
        ABSTAIN_ELIMINATED_SUBSET_NAME: evaluation_frame.loc[
            ~evaluation_frame["abstain_flag"]
        ].copy(),
        HIGH_CONFIDENCE_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["high_confidence_flag"]
        ].copy(),
        HIGH_QUALITY_SUBSET_NAME: evaluation_frame.loc[
            evaluation_frame["data_quality_grade"].isin({"A", "B"})
        ].copy(),
        LOW_QUALITY_SUBSET_NAME: evaluation_frame.loc[
            ~evaluation_frame["data_quality_grade"].isin({"A", "B"})
        ].copy(),
    }
    probability_support = {
        BASELINE_VARIANT_NAME: True,
        ENHANCED_VARIANT_NAME: True,
        BLENDED_VARIANT_NAME: True,
        MAJORITY_VARIANT_NAME: True,
        PREVIOUS_DAY_VARIANT_NAME: False,
        SENTIMENT_ABLATION_VARIANT_NAME: True,
        EVENT_ABLATION_VARIANT_NAME: True,
        NO_CONFIDENCE_VARIANT_NAME: True,
        NO_FALLBACK_VARIANT_NAME: True,
    }

    for subset_name, subset_frame in subsets.items():
        for variant_name, has_probabilities in probability_support.items():
            metrics = compute_prediction_metrics(
                split_name=subset_name,
                prediction_frame=subset_frame,
                target_column=target_column,
                predicted_column=f"{variant_name}_predicted_next_day_direction",
                probability_column=(
                    f"{variant_name}_predicted_probability_up"
                    if has_probabilities
                    else None
                ),
                raw_probability_column=(
                    f"{variant_name}_raw_predicted_probability_up"
                    if has_probabilities
                    and f"{variant_name}_raw_predicted_probability_up" in subset_frame.columns
                    else None
                ),
                logger=logger,
                date_column="prediction_date",
            )
            metrics.update(
                {
                    "prediction_mode": prediction_mode,
                    "subset_name": subset_name,
                    "model_variant": variant_name,
                    "abstained_row_count": int(
                        subset_frame["abstain_flag"].astype(bool).sum()
                    )
                    if "abstain_flag" in subset_frame.columns
                    else 0,
                    "selective_coverage": (
                        float((~subset_frame["abstain_flag"].astype(bool)).mean())
                        if len(subset_frame) > 0 and "abstain_flag" in subset_frame.columns
                        else None
                    ),
                }
            )
            rows.append(metrics)
    return rows


def reshape_metrics_by_subset(metrics_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reshape one flat metric list into subset then variant dictionaries."""

    metrics_by_subset: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        subset_name = str(row["subset_name"])
        variant_name = str(row["model_variant"])
        metrics_by_subset.setdefault(subset_name, {})[variant_name] = {
            key: value
            for key, value in row.items()
            if key not in {"subset_name", "model_variant", "prediction_mode"}
        }
    return metrics_by_subset


def compare_metric_sets(
    *,
    enhanced_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    materiality_threshold: float,
) -> dict[str, Any]:
    """Compare enhanced and baseline metrics with a conservative threshold."""

    improved: list[str] = []
    worsened: list[str] = []
    unchanged: list[str] = []
    deltas: dict[str, float | None] = {}

    for metric_name in SUMMARY_METRIC_ORDER:
        enhanced_value = enhanced_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        if enhanced_value is None or baseline_value is None:
            deltas[metric_name] = None
            continue
        delta = float(enhanced_value) - float(baseline_value)
        deltas[metric_name] = delta
        direction = METRIC_DIRECTION[metric_name]
        if direction == "higher":
            if delta >= materiality_threshold:
                improved.append(metric_name)
            elif delta <= -materiality_threshold:
                worsened.append(metric_name)
            else:
                unchanged.append(metric_name)
        else:
            if delta <= -materiality_threshold:
                improved.append(metric_name)
            elif delta >= materiality_threshold:
                worsened.append(metric_name)
            else:
                unchanged.append(metric_name)

    return {
        "improved_metrics": improved,
        "worsened_metrics": worsened,
        "unchanged_metrics": unchanged,
        "metric_deltas": deltas,
    }


def render_evidence_note(
    *,
    subset_name: str,
    enhanced_metrics: dict[str, Any],
    comparison: dict[str, Any],
    materiality_threshold: float,
) -> str:
    """Render one conservative summary sentence for a subset comparison."""

    row_count = int(enhanced_metrics.get("row_count", 0))
    if row_count == 0:
        return f"No rows were available for the {subset_name} subset."

    improved = comparison["improved_metrics"]
    worsened = comparison["worsened_metrics"]
    unchanged = comparison["unchanged_metrics"]
    if not improved and not worsened:
        return (
            f"On {row_count} {subset_name} rows, enhanced and baseline were effectively tied at the "
            f"{materiality_threshold:.2f} materiality threshold."
        )

    improved_text = ", ".join(improved) if improved else "none"
    worsened_text = ", ".join(worsened) if worsened else "none"
    unchanged_text = ", ".join(unchanged) if unchanged else "none"
    return (
        f"On {row_count} {subset_name} rows, enhanced improved on {improved_text}, worsened on "
        f"{worsened_text}, and stayed within the {materiality_threshold:.2f} materiality threshold on "
        f"{unchanged_text}."
    )


def build_mode_evidence_summary(
    *,
    prediction_mode: str,
    metrics_by_subset: dict[str, dict[str, Any]],
    materiality_threshold: float,
) -> dict[str, Any]:
    """Build the plain-language evidence summary for one mode."""

    summary: dict[str, Any] = {"prediction_mode": prediction_mode, "subsets": {}}
    for subset_name in (
        ALL_ROWS_SUBSET_NAME,
        NEWS_HEAVY_SUBSET_NAME,
        ZERO_NEWS_SUBSET_NAME,
        FRESH_NEWS_SUBSET_NAME,
        NO_FRESH_NEWS_SUBSET_NAME,
        CARRIED_FORWARD_SUBSET_NAME,
        NOT_CARRIED_FORWARD_SUBSET_NAME,
        FALLBACK_HEAVY_SUBSET_NAME,
        NOT_FALLBACK_HEAVY_SUBSET_NAME,
        HAS_NEWS_SUBSET_NAME,
        ABSTAIN_ELIMINATED_SUBSET_NAME,
        HIGH_CONFIDENCE_SUBSET_NAME,
    ):
        subset_metrics = metrics_by_subset.get(subset_name, {})
        enhanced_metrics = subset_metrics.get(ENHANCED_VARIANT_NAME)
        baseline_metrics = subset_metrics.get(BASELINE_VARIANT_NAME)
        if enhanced_metrics is None or baseline_metrics is None:
            summary["subsets"][subset_name] = {
                "note": "This subset could not be compared because required model metrics were missing."
            }
            continue
        comparison = compare_metric_sets(
            enhanced_metrics=enhanced_metrics,
            baseline_metrics=baseline_metrics,
            materiality_threshold=materiality_threshold,
        )
        summary["subsets"][subset_name] = {
            "comparison": comparison,
            "note": render_evidence_note(
                subset_name=subset_name,
                enhanced_metrics=enhanced_metrics,
                comparison=comparison,
                materiality_threshold=materiality_threshold,
            ),
        }
    return summary


def build_mode_diagnostics(
    *,
    prediction_mode: str,
    metrics_by_subset: dict[str, dict[str, Any]],
    feature_importance_summary: dict[str, Any],
    materiality_threshold: float,
) -> list[str]:
    """Surface direct diagnostic notes for one saved offline-evaluation mode."""

    diagnostics: list[str] = []
    subset_metrics = metrics_by_subset.get(ALL_ROWS_SUBSET_NAME, {})
    enhanced_metrics = subset_metrics.get(ENHANCED_VARIANT_NAME)
    baseline_metrics = subset_metrics.get(BASELINE_VARIANT_NAME)
    if enhanced_metrics is None or baseline_metrics is None:
        return diagnostics

    comparison = compare_metric_sets(
        enhanced_metrics=enhanced_metrics,
        baseline_metrics=baseline_metrics,
        materiality_threshold=materiality_threshold,
    )
    if (
        not comparison["improved_metrics"]
        and not comparison["worsened_metrics"]
        and feature_importance_summary.get("news_features_contributed") is False
    ):
        diagnostics.append(
            f"{prediction_mode}: enhanced matched baseline at the "
            f"{materiality_threshold:.2f} materiality threshold, and the saved Stage 8 "
            "importance summary shows no news-feature contribution."
        )
    return diagnostics


def build_cross_mode_diagnostics(mode_summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Summarize when both prediction modes behaved identically in offline evaluation."""

    if len(mode_summaries) < 2:
        return []

    serialized_metrics = {
        prediction_mode: json.dumps(
            mode_summary.get("metrics_by_subset", {}),
            sort_keys=True,
        )
        for prediction_mode, mode_summary in mode_summaries.items()
    }
    unique_metrics = set(serialized_metrics.values())
    if len(unique_metrics) != 1:
        return []

    return [
        "Pre-market and after-close enhanced-vs-baseline results were identical across the "
        "saved evaluation subsets, so the observed news signal was not mode-separating in "
        "this evaluation window."
    ]


def build_article_coverage_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize Stage 5 article coverage for the Stage 9 report."""

    if metadata is None:
        return {"available": False}
    return {
        "available": True,
        "row_count": metadata.get("row_count"),
        "coverage_start": metadata.get("coverage_start"),
        "coverage_end": metadata.get("coverage_end"),
        "cache_hit_count": metadata.get("cache_hit_count"),
        "fresh_fetch_count": metadata.get("fresh_fetch_count"),
        "content_origin_counts": metadata.get("content_origin_counts"),
        "source_name_counts": metadata.get("source_name_counts"),
    }


def build_extraction_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize Stage 6 extraction behavior for the Stage 9 report."""

    if metadata is None:
        return {"available": False}
    success_count = int(metadata.get("success_count", 0) or 0)
    source_row_count = int(metadata.get("source_row_count", 0) or 0)
    extraction_mode_counts = metadata.get("extraction_mode_counts", {}) or {}
    fallback_count = int(extraction_mode_counts.get("headline_plus_snippet", 0)) + int(
        extraction_mode_counts.get("headline_only", 0)
    )
    return {
        "available": True,
        "source_row_count": source_row_count,
        "success_count": success_count,
        "failure_count": metadata.get("failure_count"),
        "success_rate": (
            float(success_count / source_row_count) if source_row_count > 0 else None
        ),
        "fallback_usage_rate": (
            float(fallback_count / success_count) if success_count > 0 else None
        ),
        "extraction_mode_counts": extraction_mode_counts,
    }


def build_news_feature_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize Stage 7 feature coverage for the Stage 9 report."""

    if metadata is None:
        return {"available": False}
    return {
        "available": True,
        "output_row_count": metadata.get("output_row_count"),
        "zero_news_row_count": metadata.get("zero_news_row_count"),
        "nonzero_news_row_count": metadata.get("nonzero_news_row_count"),
        "coverage_start": metadata.get("coverage_start"),
        "coverage_end": metadata.get("coverage_end"),
        "prediction_mode_row_counts": metadata.get("prediction_mode_row_counts"),
    }


def build_summary_payload(
    *,
    settings: AppSettings,
    path_manager: PathManager,
    metrics_path: Path,
    summary_markdown_path: Path,
    historical_dataset: BaselineDataset,
    default_news_dataset: NewsFeatureDataset,
    merged_dataset: EnhancedDataset,
    stage5_metadata: dict[str, Any] | None,
    stage6_metadata: dict[str, Any] | None,
    stage7_metadata: dict[str, Any] | None,
    mode_summaries: dict[str, dict[str, Any]],
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
) -> dict[str, Any]:
    """Build the persisted Stage 9 summary payload."""

    baseline_model_metadata_path = path_manager.build_baseline_model_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    enhanced_model_metadata_paths = {
        prediction_mode: path_manager.build_enhanced_model_metadata_path(
            settings.ticker.symbol,
            settings.ticker.exchange,
            prediction_mode,
        )
        for prediction_mode in default_news_dataset.supported_prediction_modes
    }
    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "headline_split": settings.offline_evaluation.headline_split,
        "news_heavy_min_article_count": settings.offline_evaluation.news_heavy_min_article_count,
        "metric_materiality_threshold": settings.offline_evaluation.metric_materiality_threshold,
        "metrics_path": str(metrics_path),
        "summary_markdown_path": str(summary_markdown_path),
        "source_historical_feature_path": str(historical_dataset.source_feature_table_path),
        "source_historical_feature_hash": historical_dataset.source_feature_table_hash,
        "source_historical_metadata_path": str(historical_dataset.source_feature_metadata_path),
        "source_historical_metadata_hash": historical_dataset.source_feature_metadata_hash,
        "source_historical_formula_version": historical_dataset.source_metadata.get(
            "formula_version"
        ),
        "source_news_feature_path": str(default_news_dataset.source_feature_table_path),
        "source_news_feature_hash": default_news_dataset.source_feature_table_hash,
        "source_news_metadata_path": str(default_news_dataset.source_feature_metadata_path),
        "source_news_metadata_hash": default_news_dataset.source_feature_metadata_hash,
        "source_news_formula_version": default_news_dataset.source_metadata.get(
            "formula_version"
        ),
        "baseline_model_metadata_path": str(baseline_model_metadata_path),
        "baseline_model_metadata_hash": compute_optional_hash(baseline_model_metadata_path),
        "enhanced_model_metadata_paths": {
            prediction_mode: str(path)
            for prediction_mode, path in enhanced_model_metadata_paths.items()
        },
        "enhanced_model_metadata_hashes": {
            prediction_mode: compute_optional_hash(path)
            for prediction_mode, path in enhanced_model_metadata_paths.items()
        },
        "merged_row_count": int(len(merged_dataset.dataset_frame)),
        "article_coverage": build_article_coverage_summary(stage5_metadata),
        "extraction_summary": build_extraction_summary(stage6_metadata),
        "news_feature_summary": build_news_feature_summary(stage7_metadata),
        "mode_summaries": mode_summaries,
        "cross_mode_diagnostics": build_cross_mode_diagnostics(mode_summaries),
        "manual_actions": [
            "Provider/source terms review remains manual for the Stage 5 source-terms checkbox."
        ],
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def render_summary_markdown(summary_payload: dict[str, Any]) -> str:
    """Render a short human-readable Stage 9 summary."""

    lines = [
        "# Kubera Offline Evaluation",
        "",
        f"- Ticker: {summary_payload['ticker']}",
        f"- Exchange: {summary_payload['exchange']}",
        f"- Headline split: {summary_payload['headline_split']}",
        f"- Metrics CSV: {summary_payload['metrics_path']}",
        "",
        "## Coverage",
    ]
    article_summary = summary_payload["article_coverage"]
    extraction_summary = summary_payload["extraction_summary"]
    news_feature_summary = summary_payload["news_feature_summary"]
    if article_summary["available"]:
        lines.append(
            f"- Stage 5 articles: {article_summary['row_count']} rows from {article_summary['coverage_start']} to {article_summary['coverage_end']}"
        )
    else:
        lines.append("- Stage 5 metadata: unavailable")
    if extraction_summary["available"]:
        success_rate = extraction_summary["success_rate"]
        fallback_rate = extraction_summary["fallback_usage_rate"]
        lines.append(
            f"- Stage 6 extraction success rate: {success_rate:.3f}"
            if success_rate is not None
            else "- Stage 6 extraction success rate: unavailable"
        )
        lines.append(
            f"- Stage 6 fallback usage rate: {fallback_rate:.3f}"
            if fallback_rate is not None
            else "- Stage 6 fallback usage rate: unavailable"
        )
    else:
        lines.append("- Stage 6 metadata: unavailable")
    if news_feature_summary["available"]:
        lines.append(
            f"- Stage 7 zero-news rows: {news_feature_summary['zero_news_row_count']} of {news_feature_summary['output_row_count']}"
        )
    else:
        lines.append("- Stage 7 metadata: unavailable")
    lines.append("")

    for prediction_mode, mode_summary in summary_payload["mode_summaries"].items():
        lines.append(f"## {prediction_mode}")
        lines.append(f"- Held-out rows: {mode_summary['headline_row_count']}")
        lines.append(f"- News-heavy rows: {mode_summary['news_heavy_row_count']}")
        lines.append(f"- Zero-news rows: {mode_summary['zero_news_row_count']}")
        lines.append(f"- Predictions CSV: {mode_summary['predictions_path']}")
        evidence_summary = mode_summary["evidence_summary"]["subsets"]
        for subset_name in (
            ALL_ROWS_SUBSET_NAME,
            NEWS_HEAVY_SUBSET_NAME,
            ZERO_NEWS_SUBSET_NAME,
            FRESH_NEWS_SUBSET_NAME,
            NO_FRESH_NEWS_SUBSET_NAME,
            CARRIED_FORWARD_SUBSET_NAME,
            NOT_CARRIED_FORWARD_SUBSET_NAME,
            FALLBACK_HEAVY_SUBSET_NAME,
            NOT_FALLBACK_HEAVY_SUBSET_NAME,
            HAS_NEWS_SUBSET_NAME,
            ABSTAIN_ELIMINATED_SUBSET_NAME,
            HIGH_CONFIDENCE_SUBSET_NAME,
        ):
            note = evidence_summary.get(subset_name, {}).get("note")
            if note:
                lines.append(f"- {subset_name}: {note}")
        for diagnostic in mode_summary.get("diagnostics", []):
            lines.append(f"- Diagnostic: {diagnostic}")
        
        # Add a small calibration table for the blended model
        blended_metrics = mode_summary["metrics_by_subset"].get(ALL_ROWS_SUBSET_NAME, {}).get(BLENDED_VARIANT_NAME, {})
        bins = blended_metrics.get("calibration_bins", [])
        if bins:
            lines.append("")
            lines.append("### Blended Model Calibration")
            lines.append("| Prob Bin | Count | Avg Prob | Actual Freq |")
            lines.append("| --- | --- | --- | --- |")
            for b in bins:
                lines.append(f"| {b['bin_range'][0]:.1f}-{b['bin_range'][1]:.1f} | {b['row_count']} | {b['avg_probability']:.3f} | {b['actual_frequency']:.3f} |")
        
        lines.append("")

    for diagnostic in summary_payload.get("cross_mode_diagnostics", []):
        lines.append(f"- Cross-mode diagnostic: {diagnostic}")
    if summary_payload.get("cross_mode_diagnostics"):
        lines.append("")

    for manual_action in summary_payload["manual_actions"]:
        lines.append(f"- Manual follow-up: {manual_action}")
    lines.append("")
    return "\n".join(lines)


def evaluate_offline(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    historical_feature_path: str | Path | None = None,
    news_feature_path: str | Path | None = None,
    extraction_table_path: str | Path | None = None,
    force_stage8_refresh: bool = False,
) -> OfflineEvaluationResult:
    """Run the Stage 9 offline evaluation workflow."""

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

    historical_feature_table_path = resolve_historical_feature_table_path(
        runtime_settings,
        path_manager=path_manager,
        historical_feature_path=historical_feature_path,
    )
    historical_dataset = load_baseline_dataset(
        feature_table_path=historical_feature_table_path,
        feature_metadata_path=infer_feature_metadata_path(historical_feature_table_path),
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
    )

    default_news_feature_table_path = resolve_news_feature_table_path(
        runtime_settings,
        path_manager=path_manager,
        news_feature_path=news_feature_path,
    )
    default_news_dataset = load_news_feature_dataset(
        news_feature_table_path=default_news_feature_table_path,
        news_feature_metadata_path=infer_news_feature_metadata_path(
            default_news_feature_table_path
        ),
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
        supported_prediction_modes=resolve_supported_prediction_modes(
            runtime_settings.market.supported_prediction_modes
        ),
    )
    default_merged_dataset = load_or_build_merged_dataset(
        settings=runtime_settings,
        path_manager=path_manager,
        historical_dataset=historical_dataset,
        news_dataset=default_news_dataset,
        run_context=run_context,
        logger=logger,
        artifact_variant=None,
    )

    if force_stage8_refresh or not stage8_artifacts_are_aligned(
        settings=runtime_settings,
        path_manager=path_manager,
        historical_dataset=historical_dataset,
        news_dataset=default_news_dataset,
        merged_dataset=default_merged_dataset,
        logger=logger,
    ):
        logger.info(
            "Refreshing Stage 8 artifacts before offline evaluation | ticker=%s | exchange=%s",
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
        )
        train_enhanced_models(
            runtime_settings,
            historical_feature_path=historical_feature_table_path,
            news_feature_path=default_news_feature_table_path,
            force_baseline_refresh=force_stage8_refresh,
        )
        default_merged_dataset = load_or_build_merged_dataset(
            settings=runtime_settings,
            path_manager=path_manager,
            historical_dataset=historical_dataset,
            news_dataset=default_news_dataset,
            run_context=run_context,
            logger=logger,
            artifact_variant=None,
        )

    baseline_predictions_frame = load_prediction_frame(
        path_manager.build_baseline_predictions_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
        ),
        artifact_label="Stage 4 baseline predictions",
    )
    stage5_metadata = load_optional_json(
        path_manager.build_processed_news_metadata_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
        )
    )
    stage6_metadata = load_optional_json(
        path_manager.build_processed_llm_extractions_metadata_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
        )
    )
    stage7_metadata = load_optional_json(
        path_manager.build_news_feature_metadata_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
        )
    )
    extraction_source_path = resolve_extraction_table_path(
        runtime_settings,
        path_manager=path_manager,
        extraction_table_path=extraction_table_path,
    )
    no_confidence_dataset = build_variant_dataset(
        settings=runtime_settings,
        path_manager=path_manager,
        historical_dataset=historical_dataset,
        extraction_table_path=extraction_source_path,
        run_context=run_context,
        logger=logger,
        artifact_variant="no_confidence",
        news_feature_settings=replace(
            runtime_settings.news_features,
            use_confidence_in_article_weight=False,
        ),
    )
    no_fallback_dataset = build_variant_dataset(
        settings=runtime_settings,
        path_manager=path_manager,
        historical_dataset=historical_dataset,
        extraction_table_path=extraction_source_path,
        run_context=run_context,
        logger=logger,
        artifact_variant="no_fallback_penalties",
        news_feature_settings=replace(
            runtime_settings.news_features,
            full_article_weight=1.0,
            headline_plus_snippet_weight=1.0,
            headline_only_weight=1.0,
        ),
    )

    metrics_rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, dict[str, Any]] = {}
    mode_results: dict[str, OfflineEvaluationModeResult] = {}

    for prediction_mode in default_news_dataset.supported_prediction_modes:
        mode_split = split_mode_dataset(
            default_merged_dataset,
            prediction_mode=prediction_mode,
            settings=runtime_settings,
        )
        enhanced_predictions_frame = load_prediction_frame(
            path_manager.build_enhanced_predictions_path(
                runtime_settings.ticker.symbol,
                runtime_settings.ticker.exchange,
                prediction_mode,
            ),
            artifact_label=f"Stage 8 enhanced predictions for {prediction_mode}",
        )
        base_frame = build_mode_evaluation_frame(
            settings=runtime_settings,
            enhanced_predictions_frame=enhanced_predictions_frame,
            baseline_predictions_frame=baseline_predictions_frame,
            prediction_mode=prediction_mode,
            target_column=default_merged_dataset.target_column,
            news_heavy_min_article_count=runtime_settings.offline_evaluation.news_heavy_min_article_count,
            stage5_metadata=stage5_metadata,
            stage6_metadata=stage6_metadata,
            stage7_metadata=stage7_metadata,
        )
        aligned_default_test_frame = align_mode_frame(
            reference_frame=base_frame,
            candidate_frame=mode_split.test_frame,
            target_column=default_merged_dataset.target_column,
            artifact_label=f"default merged test rows for {prediction_mode}",
        )

        add_majority_baseline_predictions(
            base_frame=base_frame,
            train_frame=mode_split.train_frame,
            target_column=default_merged_dataset.target_column,
        )
        add_previous_day_direction_predictions(
            base_frame=base_frame,
            test_frame=aligned_default_test_frame,
        )
        add_trained_ablation_predictions(
            base_frame=base_frame,
            aligned_test_frame=aligned_default_test_frame,
            train_frame=mode_split.train_frame,
            feature_columns=build_sentiment_feature_columns(default_merged_dataset),
            target_column=default_merged_dataset.target_column,
            settings=runtime_settings,
            prediction_mode=prediction_mode,
            variant_name=SENTIMENT_ABLATION_VARIANT_NAME,
        )
        add_trained_ablation_predictions(
            base_frame=base_frame,
            aligned_test_frame=aligned_default_test_frame,
            train_frame=mode_split.train_frame,
            feature_columns=build_event_feature_columns(default_merged_dataset),
            target_column=default_merged_dataset.target_column,
            settings=runtime_settings,
            prediction_mode=prediction_mode,
            variant_name=EVENT_ABLATION_VARIANT_NAME,
        )

        no_confidence_split = split_mode_dataset(
            no_confidence_dataset,
            prediction_mode=prediction_mode,
            settings=runtime_settings,
        )
        aligned_no_confidence_test_frame = align_mode_frame(
            reference_frame=base_frame,
            candidate_frame=no_confidence_split.test_frame,
            target_column=no_confidence_dataset.target_column,
            artifact_label=f"no-confidence merged test rows for {prediction_mode}",
        )
        add_trained_ablation_predictions(
            base_frame=base_frame,
            aligned_test_frame=aligned_no_confidence_test_frame,
            train_frame=no_confidence_split.train_frame,
            feature_columns=build_no_confidence_feature_columns(no_confidence_dataset),
            target_column=no_confidence_dataset.target_column,
            settings=runtime_settings,
            prediction_mode=prediction_mode,
            variant_name=NO_CONFIDENCE_VARIANT_NAME,
        )

        no_fallback_split = split_mode_dataset(
            no_fallback_dataset,
            prediction_mode=prediction_mode,
            settings=runtime_settings,
        )
        aligned_no_fallback_test_frame = align_mode_frame(
            reference_frame=base_frame,
            candidate_frame=no_fallback_split.test_frame,
            target_column=no_fallback_dataset.target_column,
            artifact_label=f"no-fallback merged test rows for {prediction_mode}",
        )
        add_trained_ablation_predictions(
            base_frame=base_frame,
            aligned_test_frame=aligned_no_fallback_test_frame,
            train_frame=no_fallback_split.train_frame,
            feature_columns=no_fallback_dataset.feature_columns,
            target_column=no_fallback_dataset.target_column,
            settings=runtime_settings,
            prediction_mode=prediction_mode,
            variant_name=NO_FALLBACK_VARIANT_NAME,
        )

        predictions_path = path_manager.build_offline_evaluation_predictions_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        base_frame.to_csv(predictions_path, index=False)

        mode_metrics_rows = build_mode_metrics_rows(
            prediction_mode=prediction_mode,
            evaluation_frame=base_frame,
            target_column=default_merged_dataset.target_column,
            logger=logger,
        )
        metrics_rows.extend(mode_metrics_rows)

        metrics_by_subset = reshape_metrics_by_subset(mode_metrics_rows)
        enhanced_metadata = load_optional_json(
            path_manager.build_enhanced_model_metadata_path(
                runtime_settings.ticker.symbol,
                runtime_settings.ticker.exchange,
                prediction_mode,
            )
        ) or {}
        feature_importance_summary = (
            enhanced_metadata.get("feature_importance_summary", {}) or {}
        )
        evidence_summary = build_mode_evidence_summary(
            prediction_mode=prediction_mode,
            metrics_by_subset=metrics_by_subset,
            materiality_threshold=runtime_settings.offline_evaluation.metric_materiality_threshold,
        )
        mode_summaries[prediction_mode] = {
            "prediction_mode": prediction_mode,
            "predictions_path": str(predictions_path),
            "headline_row_count": int(len(base_frame)),
            "news_heavy_row_count": int(base_frame["news_heavy_flag"].sum()),
            "zero_news_row_count": int(base_frame["zero_news_flag"].sum()),
            "metrics_by_subset": metrics_by_subset,
            "evidence_summary": evidence_summary,
            "diagnostics": build_mode_diagnostics(
                prediction_mode=prediction_mode,
                metrics_by_subset=metrics_by_subset,
                feature_importance_summary=feature_importance_summary,
                materiality_threshold=runtime_settings.offline_evaluation.metric_materiality_threshold,
            ),
        }
        mode_results[prediction_mode] = OfflineEvaluationModeResult(
            prediction_mode=prediction_mode,
            predictions_path=predictions_path,
            headline_row_count=int(len(base_frame)),
            news_heavy_row_count=int(base_frame["news_heavy_flag"].sum()),
            zero_news_row_count=int(base_frame["zero_news_flag"].sum()),
        )

    metrics_path = path_manager.build_offline_metrics_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(metrics_path, index=False)

    summary_json_path = path_manager.build_offline_evaluation_summary_json_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    summary_markdown_path = path_manager.build_offline_evaluation_summary_markdown_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    summary_payload = build_summary_payload(
        settings=runtime_settings,
        path_manager=path_manager,
        metrics_path=metrics_path,
        summary_markdown_path=summary_markdown_path,
        historical_dataset=historical_dataset,
        default_news_dataset=default_news_dataset,
        merged_dataset=default_merged_dataset,
        stage5_metadata=stage5_metadata,
        stage6_metadata=stage6_metadata,
        stage7_metadata=stage7_metadata,
        mode_summaries=mode_summaries,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
    )
    write_json_file(summary_json_path, summary_payload)
    summary_markdown_path.write_text(
        render_summary_markdown(summary_payload),
        encoding="utf-8",
    )

    logger.info(
        "Offline evaluation ready | ticker=%s | exchange=%s | modes=%s | metrics_csv=%s | summary_json=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        list(mode_results),
        metrics_path,
        summary_json_path,
    )

    return OfflineEvaluationResult(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        mode_results=mode_results,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 9 offline-evaluation command arguments."""

    parser = argparse.ArgumentParser(description="Run Kubera Stage 9 offline evaluation.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--historical-feature-path",
        help="Use a specific Stage 3 historical feature CSV file.",
    )
    parser.add_argument(
        "--news-feature-path",
        help="Use a specific default Stage 7 news feature CSV file.",
    )
    parser.add_argument(
        "--extractions-path",
        help="Use a specific Stage 6 extraction CSV file for formula-level ablations.",
    )
    parser.add_argument(
        "--force-stage8-refresh",
        action="store_true",
        help="Retrain Stage 8 artifacts before running the offline evaluation report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 9 offline evaluation command."""

    args = parse_args(argv)
    settings = load_settings()
    evaluate_offline(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        historical_feature_path=args.historical_feature_path,
        news_feature_path=args.news_feature_path,
        extraction_table_path=args.extractions_path,
        force_stage8_refresh=args.force_stage8_refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
