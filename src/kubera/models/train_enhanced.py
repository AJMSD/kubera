"""Stage 8 enhanced model training for Kubera."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.pipeline import Pipeline

from kubera.config import (
    AppSettings,
    EnhancedModelSettings,
    load_settings,
    resolve_runtime_settings,
)
from kubera.features.news_features import (
    NEWS_FEATURE_COLUMNS,
    OUTPUT_IDENTITY_COLUMNS as NEWS_OUTPUT_IDENTITY_COLUMNS,
    PREDICTION_MODE_ORDER,
    resolve_supported_prediction_modes,
)
from kubera.models.artifact_validation import validate_news_feature_artifact_metadata
from kubera.models.common import (
    BinaryPredictionOutputs,
    ProbabilityCalibrator,
    TemporalDatasetSplit,
    apply_probability_calibrator,
    blend_probabilities,
    build_logistic_regression_pipeline,
    build_split_summary,
    compute_news_context_weight,
    compute_sample_weights,
    compute_split_metrics,
    fit_probability_calibrator,
    load_pickle_artifact,
    optimize_blend_alpha,
    optimize_classification_threshold,
    predict_binary_classifier,
    predict_binary_classifier_outputs,
    save_pickle_artifact,
    split_temporal_dataset,
    tune_model_hyperparameters,
)
from kubera.models.train_baseline import (
    BaselineDataset,
    PersistedBaselineModel,
    build_model_params as build_baseline_model_params,
    infer_feature_metadata_path,
    load_baseline_dataset,
    load_saved_baseline_model,
    predict_with_saved_model,
    predict_with_saved_model_outputs,
    split_baseline_dataset,
    train_baseline_model,
)
from kubera.utils.hashing import compute_file_sha256
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot


MERGED_IDENTITY_COLUMNS = (
    "historical_date",
    "prediction_date",
    "ticker",
    "exchange",
    "prediction_mode",
    "close",
    "volume",
)
ENHANCED_PREDICTION_IDENTITY_COLUMNS = (
    "historical_date",
    "prediction_date",
    "ticker",
    "exchange",
    "prediction_mode",
    "close",
    "volume",
)
HISTORICAL_FEATURE_GROUP_KEY = "historical_features"
NEWS_SENTIMENT_GROUP_KEY = "news_sentiment_features"
NEWS_EVENT_GROUP_KEY = "news_event_count_features"
NEWS_QUALITY_GROUP_KEY = "news_quality_fallback_features"
COMPARISON_NEWS_CONTEXT_COLUMNS = (
    "news_article_count",
    "news_fallback_article_ratio",
    "news_warning_article_count",
    "news_weighted_sentiment_score",
    "news_avg_confidence",
    "has_fresh_news",
    "is_carried_forward",
    "is_fallback_heavy",
)
OPTIONAL_EVALUATION_CONTEXT_COLUMNS = (
    "news_signal_state",
    "news_feature_synthetic_flag",
    "market_data_gap_flag",
    "market_data_gap_count_5d",
)


class EnhancedModelError(RuntimeError):
    """Raised when Stage 8 training or comparison cannot continue."""


@dataclass(frozen=True)
class NewsFeatureDataset:
    dataset_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    source_feature_table_path: Path
    source_feature_metadata_path: Path
    source_feature_table_hash: str
    source_feature_metadata_hash: str
    source_metadata: dict[str, Any]
    supported_prediction_modes: tuple[str, ...]


@dataclass(frozen=True)
class EnhancedDataset:
    dataset_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    historical_feature_columns: tuple[str, ...]
    news_feature_columns: tuple[str, ...]
    target_column: str
    feature_groups: dict[str, tuple[str, ...]]
    historical_dataset: BaselineDataset
    news_dataset: NewsFeatureDataset
    missing_news_row_count: int


@dataclass(frozen=True)
class EnhancedInteractionSpec:
    name: str
    left_column: str
    right_column: str
    left_center: float = 0.0


@dataclass(frozen=True)
class EnhancedFeatureSpec:
    base_news_feature_columns: tuple[str, ...]
    lag_windows: tuple[int, ...]
    lagged_news_feature_columns: tuple[str, ...]
    interaction_specs: tuple[EnhancedInteractionSpec, ...]
    extended_news_feature_columns: tuple[str, ...]

    @property
    def cross_feature_columns(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.interaction_specs)


@dataclass(frozen=True)
class PersistedEnhancedModel:
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    target_column: str
    model_type: str
    classification_threshold: float
    prediction_mode: str
    calibrator: ProbabilityCalibrator | None = None


CANONICAL_ENHANCED_MODULE_NAME = "kubera.models.train_enhanced"
if __name__ == "__main__":
    sys.modules.setdefault(CANONICAL_ENHANCED_MODULE_NAME, sys.modules[__name__])
PersistedEnhancedModel.__module__ = CANONICAL_ENHANCED_MODULE_NAME


@dataclass(frozen=True)
class EnhancedModeTrainingResult:
    prediction_mode: str
    model_path: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path
    comparison_path: Path
    comparison_summary_path: Path
    train_row_count: int
    validation_row_count: int
    test_row_count: int


@dataclass(frozen=True)
class EnhancedTrainingResult:
    merged_dataset_path: Path
    merged_dataset_metadata_path: Path
    baseline_artifact_status: str
    mode_results: dict[str, EnhancedModeTrainingResult]


@dataclass(frozen=True)
class BaselineComparisonArtifacts:
    saved_model: PersistedBaselineModel
    metadata: dict[str, Any]
    status: str


def train_enhanced_models(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    historical_feature_path: str | Path | None = None,
    news_feature_path: str | Path | None = None,
    force_baseline_refresh: bool = False,
    tune: bool = False,
) -> EnhancedTrainingResult:
    """Train separate Stage 8 enhanced models for each prediction mode."""

    started_at_utc = datetime.now(timezone.utc)
    stage_start = time.perf_counter()
    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    validate_split_alignment(runtime_settings)

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
    historical_feature_metadata_path = infer_feature_metadata_path(
        historical_feature_table_path
    )
    historical_dataset = load_baseline_dataset(
        feature_table_path=historical_feature_table_path,
        feature_metadata_path=historical_feature_metadata_path,
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
    )

    news_feature_table_path = resolve_news_feature_table_path(
        runtime_settings,
        path_manager=path_manager,
        news_feature_path=news_feature_path,
    )
    news_feature_metadata_path = infer_news_feature_metadata_path(news_feature_table_path)
    news_dataset = load_news_feature_dataset(
        news_feature_table_path=news_feature_table_path,
        news_feature_metadata_path=news_feature_metadata_path,
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
        supported_prediction_modes=resolve_supported_prediction_modes(
            runtime_settings.market.supported_prediction_modes
        ),
    )
    if not news_dataset.supported_prediction_modes:
        raise EnhancedModelError("Stage 8 requires at least one concrete prediction mode.")

    merged_dataset_path = path_manager.build_merged_enhanced_dataset_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    merged_dataset_metadata_path = path_manager.build_merged_enhanced_dataset_metadata_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    enhanced_dataset = load_cached_merged_enhanced_dataset(
        path=merged_dataset_path,
        metadata_path=merged_dataset_metadata_path,
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        lag_windows=runtime_settings.historical_features.lag_windows,
    )
    if enhanced_dataset is None:
        enhanced_dataset = build_merged_enhanced_dataset(
            historical_dataset=historical_dataset,
            news_dataset=news_dataset,
            lag_windows=runtime_settings.historical_features.lag_windows,
        )
        save_merged_enhanced_dataset(
            path=merged_dataset_path,
            metadata_path=merged_dataset_metadata_path,
            settings=runtime_settings,
            dataset=enhanced_dataset,
            run_id=run_context.run_id,
            git_commit=run_context.git_commit,
            git_is_dirty=run_context.git_is_dirty,
        )
    else:
        logger.info(
            "Stage 8 merged dataset ready from cache | ticker=%s | exchange=%s | rows=%s | merged_csv=%s",
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            len(enhanced_dataset.dataset_frame),
            merged_dataset_path,
        )

    baseline_artifacts = ensure_baseline_artifacts(
        settings=runtime_settings,
        historical_dataset=historical_dataset,
        feature_table_path=historical_feature_table_path,
        force_refresh=force_baseline_refresh,
    )

    mode_results: dict[str, EnhancedModeTrainingResult] = {}
    mode_elapsed_seconds: dict[str, float] = {}

    def _train_mode(prediction_mode: str) -> tuple[str, EnhancedModeTrainingResult, float]:
        mode_start = time.perf_counter()
        mode_frame = enhanced_dataset.dataset_frame.loc[
            enhanced_dataset.dataset_frame["prediction_mode"] == prediction_mode
        ].copy()
        split = split_enhanced_dataset(mode_frame, runtime_settings.enhanced_model)
        tuned_params: dict[str, Any] = {}
        if tune:
            logger.info(
                "Tuning enhanced model hyperparameters for mode=%s (this may take a few minutes)...",
                prediction_mode,
            )
            tuned_params = tune_model_hyperparameters(
                train_frame=split.train_frame,
                feature_columns=enhanced_dataset.feature_columns,
                target_column=enhanced_dataset.target_column,
                model_type=runtime_settings.enhanced_model.model_type,
                random_seed=runtime_settings.run.random_seed,
            )
            if tuned_params:
                logger.info("Enhanced tuned params for mode=%s: %s", prediction_mode, tuned_params)
            else:
                logger.warning(
                    "Enhanced hyperparameter tuning failed for mode=%s; using configured defaults.",
                    prediction_mode,
                )
        persisted_model = fit_enhanced_model(
            train_frame=split.train_frame,
            feature_columns=enhanced_dataset.feature_columns,
            target_column=enhanced_dataset.target_column,
            enhanced_settings=runtime_settings.enhanced_model,
            random_seed=runtime_settings.run.random_seed,
            prediction_mode=prediction_mode,
            tuned_params=tuned_params,
        )
        initial_validation_outputs = predict_with_saved_enhanced_model_outputs(
            persisted_model,
            split.validation_frame.loc[:, persisted_model.feature_columns],
        )
        calibrator, calibration_summary = fit_probability_calibrator(
            raw_probabilities=initial_validation_outputs.raw_probabilities,
            actual=split.validation_frame[enhanced_dataset.target_column],
            enabled=runtime_settings.enhanced_model.enable_calibration,
            random_seed=runtime_settings.run.random_seed,
            timestamps=(
                split.validation_frame["prediction_date"]
                if "prediction_date" in split.validation_frame.columns
                else None
            ),
        )
        persisted_model = replace(persisted_model, calibrator=calibrator)
        val_calibrated = apply_probability_calibrator(
            initial_validation_outputs.raw_probabilities,
            calibrator=calibrator,
        )
        optimal_threshold = optimize_classification_threshold(
            val_calibrated,
            split.validation_frame[enhanced_dataset.target_column],
        )
        logger.info(
            "Enhanced optimized classification threshold | mode=%s | threshold=%.4f",
            prediction_mode,
            optimal_threshold,
        )
        persisted_model = replace(persisted_model, classification_threshold=optimal_threshold)

        validation_predictions = build_enhanced_prediction_frame(
            split_name="validation",
            split_frame=split.validation_frame,
            saved_model=persisted_model,
        )
        test_predictions = build_enhanced_prediction_frame(
            split_name="test",
            split_frame=split.test_frame,
            saved_model=persisted_model,
        )
        predictions_frame = pd.concat(
            [validation_predictions, test_predictions],
            ignore_index=True,
        )
        mode_elapsed = round(time.perf_counter() - mode_start, 6)
        metrics_payload = {
            "model_type": persisted_model.model_type,
            "prediction_mode": prediction_mode,
            "classification_threshold": persisted_model.classification_threshold,
            "calibration": calibration_summary,
            "validation": compute_split_metrics(
                split_name="validation",
                prediction_frame=validation_predictions,
                target_column=enhanced_dataset.target_column,
                logger=logger,
                date_column="prediction_date",
                raw_probability_column="raw_predicted_probability_up",
            ),
            "test": compute_split_metrics(
                split_name="test",
                prediction_frame=test_predictions,
                target_column=enhanced_dataset.target_column,
                logger=logger,
                date_column="prediction_date",
                raw_probability_column="raw_predicted_probability_up",
            ),
            "feature_importance": summarize_feature_importance(
                persisted_model=persisted_model,
                feature_groups=enhanced_dataset.feature_groups,
            ),
            "timing": {"elapsed_seconds": mode_elapsed},
        }
        comparison_frame = build_baseline_comparison_frame(
            evaluation_frame=predictions_frame,
            baseline_model=baseline_artifacts.saved_model,
            historical_feature_columns=enhanced_dataset.historical_feature_columns,
            enhanced_target_column=enhanced_dataset.target_column,
        )
        comparison_summary = build_baseline_comparison_summary(
            comparison_frame=comparison_frame,
            prediction_mode=prediction_mode,
            baseline_artifact_status=baseline_artifacts.status,
            baseline_metadata=baseline_artifacts.metadata,
            feature_importance_summary=metrics_payload["feature_importance"],
        )
        val_comparison = comparison_frame[comparison_frame["split"] == "validation"]
        if len(val_comparison) > 0:
            _, blend_summary = optimize_blend_alpha(
                baseline_probs=val_comparison["baseline_calibrated_predicted_probability_up"],
                enhanced_probs=val_comparison["enhanced_calibrated_predicted_probability_up"],
                actual=val_comparison[enhanced_dataset.target_column],
            )
        else:
            blend_summary = {"status": "no_validation_rows", "best_alpha": 0.5}
        metrics_payload["blend_alpha_optimization"] = blend_summary

        model_path = path_manager.build_enhanced_model_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        metadata_path = path_manager.build_enhanced_model_metadata_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        predictions_path = path_manager.build_enhanced_predictions_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        metrics_path = path_manager.build_enhanced_metrics_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        comparison_path = path_manager.build_enhanced_comparison_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )
        comparison_summary_path = path_manager.build_enhanced_comparison_summary_path(
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
        )

        save_enhanced_model(model_path, persisted_model)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_frame.to_csv(predictions_path, index=False)
        comparison_frame.to_csv(comparison_path, index=False)
        write_json_file(metrics_path, metrics_payload)
        write_json_file(comparison_summary_path, comparison_summary)
        write_json_file(
            metadata_path,
            build_enhanced_model_metadata(
                settings=runtime_settings,
                dataset=enhanced_dataset,
                split=split,
                persisted_model=persisted_model,
                prediction_mode=prediction_mode,
                model_path=model_path,
                metadata_path=metadata_path,
                predictions_path=predictions_path,
                metrics_path=metrics_path,
                comparison_path=comparison_path,
                comparison_summary_path=comparison_summary_path,
                merged_dataset_path=merged_dataset_path,
                merged_dataset_metadata_path=merged_dataset_metadata_path,
                metrics_payload=metrics_payload,
                baseline_artifact_status=baseline_artifacts.status,
                baseline_metadata=baseline_artifacts.metadata,
                mode_elapsed_seconds=mode_elapsed,
                run_id=run_context.run_id,
                git_commit=run_context.git_commit,
                git_is_dirty=run_context.git_is_dirty,
            ),
        )
        logger.info(
            "Enhanced model ready | ticker=%s | exchange=%s | mode=%s | train_rows=%s | validation_rows=%s | test_rows=%s | model=%s | metrics=%s",
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            prediction_mode,
            len(split.train_frame),
            len(split.validation_frame),
            len(split.test_frame),
            model_path,
            metrics_path,
        )
        return (
            prediction_mode,
            EnhancedModeTrainingResult(
                prediction_mode=prediction_mode,
                model_path=model_path,
                metadata_path=metadata_path,
                predictions_path=predictions_path,
                metrics_path=metrics_path,
                comparison_path=comparison_path,
                comparison_summary_path=comparison_summary_path,
                train_row_count=len(split.train_frame),
                validation_row_count=len(split.validation_frame),
                test_row_count=len(split.test_frame),
            ),
            mode_elapsed,
        )

    supported_modes = tuple(news_dataset.supported_prediction_modes)
    max_workers = min(
        int(runtime_settings.enhanced_model.mode_training_workers),
        len(supported_modes),
    )
    if max_workers > 1 and len(supported_modes) > 1:
        mode_output: dict[str, tuple[EnhancedModeTrainingResult, float]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_train_mode, mode): mode
                for mode in supported_modes
            }
            for future in as_completed(futures):
                prediction_mode, mode_result, elapsed = future.result()
                mode_output[prediction_mode] = (mode_result, elapsed)
        for prediction_mode in supported_modes:
            mode_result, elapsed = mode_output[prediction_mode]
            mode_results[prediction_mode] = mode_result
            mode_elapsed_seconds[prediction_mode] = elapsed
    else:
        for prediction_mode in supported_modes:
            mode_name, mode_result, elapsed = _train_mode(prediction_mode)
            mode_results[mode_name] = mode_result
            mode_elapsed_seconds[mode_name] = elapsed

    finished_at_utc = datetime.now(timezone.utc)
    elapsed_seconds = round(time.perf_counter() - stage_start, 6)
    logger.info(
        "Stage 8 enhanced training finished | ticker=%s | exchange=%s | elapsed=%.3fs | started=%s | finished=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        elapsed_seconds,
        started_at_utc.isoformat(),
        finished_at_utc.isoformat(),
    )
    return EnhancedTrainingResult(
        merged_dataset_path=merged_dataset_path,
        merged_dataset_metadata_path=merged_dataset_metadata_path,
        baseline_artifact_status=baseline_artifacts.status,
        mode_results=mode_results,
    )


def validate_split_alignment(settings: AppSettings) -> None:
    """Require the enhanced split windows to stay aligned with the baseline."""

    baseline_split = (
        settings.baseline_model.train_ratio,
        settings.baseline_model.validation_ratio,
        settings.baseline_model.test_ratio,
    )
    enhanced_split = (
        settings.enhanced_model.train_ratio,
        settings.enhanced_model.validation_ratio,
        settings.enhanced_model.test_ratio,
    )
    if any(
        not math.isclose(baseline_value, enhanced_value, rel_tol=0.0, abs_tol=1e-9)
        for baseline_value, enhanced_value in zip(baseline_split, enhanced_split)
    ):
        raise EnhancedModelError(
            "Enhanced model split ratios must match the baseline split ratios for fair Stage 8 comparison."
        )


def resolve_historical_feature_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    historical_feature_path: str | Path | None,
) -> Path:
    """Resolve the Stage 3 historical feature table for Stage 8."""

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
    """Resolve the Stage 7 news feature table for Stage 8."""

    if news_feature_path is not None:
        return Path(news_feature_path).expanduser().resolve()
    return path_manager.build_news_feature_table_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def infer_news_feature_metadata_path(news_feature_table_path: Path) -> Path:
    """Infer the Stage 7 news feature metadata path from the CSV path."""

    if news_feature_table_path.suffix.lower() != ".csv":
        raise EnhancedModelError(
            "News feature table path must point to a CSV file so the matching metadata can be resolved."
        )

    metadata_path = news_feature_table_path.with_name(
        f"{news_feature_table_path.stem}.metadata.json"
    )
    if not metadata_path.exists():
        raise EnhancedModelError(
            f"News feature metadata file does not exist: {metadata_path}"
        )
    return metadata_path


def load_news_feature_dataset(
    *,
    news_feature_table_path: Path,
    news_feature_metadata_path: Path,
    ticker: str,
    exchange: str,
    supported_prediction_modes: tuple[str, ...],
) -> NewsFeatureDataset:
    """Read and validate the Stage 7 feature artifact for Stage 8."""

    if not news_feature_table_path.exists():
        raise EnhancedModelError(
            f"News feature table does not exist: {news_feature_table_path}"
        )

    try:
        feature_frame = pd.read_csv(news_feature_table_path)
    except pd.errors.EmptyDataError as exc:
        raise EnhancedModelError(
            f"News feature table is empty: {news_feature_table_path}"
        ) from exc

    try:
        source_metadata = json.loads(news_feature_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnhancedModelError(
            f"News feature metadata is not valid JSON: {news_feature_metadata_path}"
        ) from exc
    validate_news_feature_artifact_metadata(
        source_metadata,
        metadata_path=news_feature_metadata_path,
        error_factory=EnhancedModelError,
    )

    raw_feature_columns = source_metadata.get("feature_columns")
    if not isinstance(raw_feature_columns, list) or not raw_feature_columns:
        raw_feature_columns = list(NEWS_FEATURE_COLUMNS)
    feature_columns = tuple(str(column) for column in raw_feature_columns)

    missing_columns = [
        column
        for column in NEWS_OUTPUT_IDENTITY_COLUMNS + feature_columns
        if column not in feature_frame.columns
    ]
    if missing_columns:
        raise EnhancedModelError(
            f"News feature table is missing required columns: {missing_columns}"
        )

    working_frame = feature_frame.loc[
        :,
        NEWS_OUTPUT_IDENTITY_COLUMNS + feature_columns,
    ].copy()
    if working_frame.empty:
        raise EnhancedModelError("News feature table does not contain any rows for Stage 8.")

    working_frame["date"] = pd.to_datetime(working_frame["date"], errors="coerce")
    if working_frame["date"].isna().any():
        raise EnhancedModelError("News feature table contains invalid date values.")
    working_frame["date"] = working_frame["date"].dt.strftime("%Y-%m-%d")

    source_tickers = {
        str(value).strip().upper()
        for value in working_frame["ticker"].dropna().unique().tolist()
    }
    source_exchanges = {
        str(value).strip().upper()
        for value in working_frame["exchange"].dropna().unique().tolist()
    }
    if source_tickers != {ticker.upper()}:
        raise EnhancedModelError(
            f"News feature table ticker values do not match the requested ticker: {sorted(source_tickers)}"
        )
    if source_exchanges != {exchange.upper()}:
        raise EnhancedModelError(
            f"News feature table exchange values do not match the requested exchange: {sorted(source_exchanges)}"
        )

    prediction_modes = tuple(
        str(value).strip()
        for value in working_frame["prediction_mode"].dropna().unique().tolist()
    )
    unsupported_modes = sorted(set(prediction_modes) - set(supported_prediction_modes))
    if unsupported_modes:
        raise EnhancedModelError(
            f"News feature table contains unsupported prediction modes: {unsupported_modes}"
        )
    if working_frame.duplicated(
        subset=["date", "ticker", "exchange", "prediction_mode"]
    ).any():
        raise EnhancedModelError("News feature table contains duplicate identity rows.")

    for column in feature_columns:
        numeric_series = pd.to_numeric(working_frame[column], errors="coerce")
        if numeric_series.isna().any():
            raise EnhancedModelError(
                f"News feature table contains non-numeric or missing values in column: {column}"
            )
        if not np.isfinite(numeric_series.to_numpy(dtype=float)).all():
            raise EnhancedModelError(
                f"News feature table contains non-finite values in column: {column}"
            )
        working_frame[column] = numeric_series.astype(float)

    working_frame = working_frame.sort_values(
        by=["date", "prediction_mode"],
        ascending=[True, True],
    ).reset_index(drop=True)

    resolved_modes = tuple(
        mode
        for mode in supported_prediction_modes
        if mode in set(working_frame["prediction_mode"].tolist())
    )
    return NewsFeatureDataset(
        dataset_frame=working_frame,
        feature_columns=feature_columns,
        source_feature_table_path=news_feature_table_path,
        source_feature_metadata_path=news_feature_metadata_path,
        source_feature_table_hash=compute_file_sha256(news_feature_table_path),
        source_feature_metadata_hash=compute_file_sha256(news_feature_metadata_path),
        source_metadata=source_metadata,
        supported_prediction_modes=resolved_modes,
    )


def build_enhanced_feature_spec(
    *,
    historical_feature_columns: tuple[str, ...],
    base_news_feature_columns: tuple[str, ...],
    lag_windows: tuple[int, ...],
) -> EnhancedFeatureSpec:
    """Build the shared Stage 8 lag and interaction feature specification."""

    normalized_lag_windows = tuple(sorted({int(window) for window in lag_windows if int(window) > 0}))
    lagged_news_feature_columns = tuple(
        f"{column}_lag{window}"
        for window in normalized_lag_windows
        for column in base_news_feature_columns
    )

    interaction_specs: list[EnhancedInteractionSpec] = []
    sentiment_col = "news_sentiment_3d"
    confidence_col = "news_weighted_confidence_score"
    company_sent_col = "news_company_weighted_sentiment_score"
    sector_sent_col = "news_sector_weighted_sentiment_score"
    momentum_col = "ret_5d"
    rsi_col = next((c for c in historical_feature_columns if c.startswith("rsi_")), None)

    if rsi_col and sentiment_col in base_news_feature_columns:
        interaction_specs.append(
            EnhancedInteractionSpec(
                name=f"cross_{rsi_col}_sentiment",
                left_column=rsi_col,
                right_column=sentiment_col,
                left_center=50.0,
            )
        )
    if momentum_col in historical_feature_columns and confidence_col in base_news_feature_columns:
        interaction_specs.append(
            EnhancedInteractionSpec(
                name="cross_momentum_confidence",
                left_column=momentum_col,
                right_column=confidence_col,
            )
        )
    if (
        company_sent_col in base_news_feature_columns
        and sector_sent_col in base_news_feature_columns
    ):
        interaction_specs.append(
            EnhancedInteractionSpec(
                name="cross_company_sector_sentiment",
                left_column=company_sent_col,
                right_column=sector_sent_col,
            )
        )
    if "macd" in historical_feature_columns and sentiment_col in base_news_feature_columns:
        interaction_specs.append(
            EnhancedInteractionSpec(
                name="cross_macd_sentiment",
                left_column="macd",
                right_column=sentiment_col,
            )
        )
    if "price_vs_52w_high" in historical_feature_columns and sentiment_col in base_news_feature_columns:
        interaction_specs.append(
            EnhancedInteractionSpec(
                name="cross_price_vs_52w_high_sentiment",
                left_column="price_vs_52w_high",
                right_column=sentiment_col,
            )
        )

    extended_news_feature_columns = (
        base_news_feature_columns
        + lagged_news_feature_columns
        + tuple(spec.name for spec in interaction_specs)
    )
    return EnhancedFeatureSpec(
        base_news_feature_columns=base_news_feature_columns,
        lag_windows=normalized_lag_windows,
        lagged_news_feature_columns=lagged_news_feature_columns,
        interaction_specs=tuple(interaction_specs),
        extended_news_feature_columns=extended_news_feature_columns,
    )


def serialize_enhanced_feature_spec(feature_spec: EnhancedFeatureSpec) -> dict[str, Any]:
    """Convert the shared Stage 8 feature specification into JSON-safe values."""

    return {
        "base_news_feature_columns": list(feature_spec.base_news_feature_columns),
        "lag_windows": list(feature_spec.lag_windows),
        "lagged_news_feature_columns": list(feature_spec.lagged_news_feature_columns),
        "cross_feature_columns": list(feature_spec.cross_feature_columns),
        "interaction_specs": [
            {
                "name": spec.name,
                "left_column": spec.left_column,
                "right_column": spec.right_column,
                "left_center": spec.left_center,
            }
            for spec in feature_spec.interaction_specs
        ],
        "extended_news_feature_columns": list(feature_spec.extended_news_feature_columns),
    }


def build_enhanced_feature_spec_from_metadata(metadata: dict[str, Any]) -> EnhancedFeatureSpec:
    """Reconstruct the saved Stage 8 feature spec from model metadata."""

    feature_spec_payload = metadata.get("feature_spec")
    if isinstance(feature_spec_payload, dict):
        interaction_specs = tuple(
            EnhancedInteractionSpec(
                name=str(spec.get("name", "")).strip(),
                left_column=str(spec.get("left_column", "")).strip(),
                right_column=str(spec.get("right_column", "")).strip(),
                left_center=float(spec.get("left_center", 0.0)),
            )
            for spec in feature_spec_payload.get("interaction_specs", [])
            if str(spec.get("name", "")).strip()
        )
        base_news_feature_columns = tuple(
            str(column)
            for column in feature_spec_payload.get("base_news_feature_columns", [])
        )
        lag_windows = tuple(
            int(window) for window in feature_spec_payload.get("lag_windows", [])
        )
        lagged_news_feature_columns = tuple(
            str(column)
            for column in feature_spec_payload.get("lagged_news_feature_columns", [])
        )
        extended_news_feature_columns = tuple(
            str(column)
            for column in feature_spec_payload.get("extended_news_feature_columns", [])
        )
        if (
            base_news_feature_columns
            and lagged_news_feature_columns
            and extended_news_feature_columns
        ):
            return EnhancedFeatureSpec(
                base_news_feature_columns=base_news_feature_columns,
                lag_windows=lag_windows,
                lagged_news_feature_columns=lagged_news_feature_columns,
                interaction_specs=interaction_specs,
                extended_news_feature_columns=extended_news_feature_columns,
            )

    historical_feature_columns = tuple(
        str(column) for column in metadata.get("historical_feature_columns", [])
    )
    base_news_feature_columns = tuple(
        str(column) for column in metadata.get("base_news_feature_columns", [])
    )
    if not base_news_feature_columns:
        base_news_feature_columns = tuple(
            str(column)
            for column in metadata.get("news_feature_columns", [])
            if "_lag" not in str(column) and not str(column).startswith("cross_")
        )
    lag_windows = extract_lag_windows_from_feature_columns(
        tuple(str(column) for column in metadata.get("feature_columns", []))
    )
    return build_enhanced_feature_spec(
        historical_feature_columns=historical_feature_columns,
        base_news_feature_columns=base_news_feature_columns,
        lag_windows=lag_windows,
    )


def extract_lag_windows_from_feature_columns(feature_columns: tuple[str, ...]) -> tuple[int, ...]:
    """Infer lag windows from one persisted feature-column list."""

    lag_windows: set[int] = set()
    for column in feature_columns:
        if "_lag" not in column:
            continue
        _base_name, lag_suffix = column.rsplit("_lag", 1)
        if lag_suffix.isdigit():
            lag_windows.add(int(lag_suffix))
    return tuple(sorted(lag_windows))


def apply_enhanced_feature_spec_to_frame(
    frame: pd.DataFrame,
    *,
    feature_spec: EnhancedFeatureSpec,
    group_columns: tuple[str, ...] = ("ticker", "exchange", "prediction_mode"),
) -> pd.DataFrame:
    """Add the shared lag and interaction features to one merged frame."""

    if frame.empty:
        return frame.copy()

    new_columns: dict[str, pd.Series] = {}
    group_values = [column for column in group_columns if column in frame.columns]
    if group_values:
        grouped_frame = frame.groupby(group_values)
        for window in feature_spec.lag_windows:
            for column in feature_spec.base_news_feature_columns:
                lagged_name = f"{column}_lag{window}"
                new_columns[lagged_name] = grouped_frame[column].shift(window)

    for interaction_spec in feature_spec.interaction_specs:
        left_series = pd.to_numeric(
            frame.get(interaction_spec.left_column, 0.0),
            errors="coerce",
        ).fillna(0.0)
        right_series = pd.to_numeric(
            frame.get(interaction_spec.right_column, 0.0),
            errors="coerce",
        ).fillna(0.0)
        new_columns[interaction_spec.name] = (
            left_series - interaction_spec.left_center
        ) * right_series

    if new_columns:
        feature_frame = pd.DataFrame(new_columns, index=frame.index)
        enriched_frame = pd.concat([frame, feature_frame], axis=1)
    else:
        enriched_frame = frame.copy()

    derived_columns = (
        list(feature_spec.lagged_news_feature_columns)
        + list(feature_spec.cross_feature_columns)
    )
    for column in derived_columns:
        if column not in enriched_frame.columns:
            enriched_frame[column] = 0.0
    if derived_columns:
        enriched_frame.loc[:, derived_columns] = (
            enriched_frame.loc[:, derived_columns].fillna(0.0).astype(float)
        )
    return enriched_frame


def build_live_enhanced_feature_row(
    *,
    historical_row_mapping: Mapping[str, Any],
    news_feature_row_mapping: Mapping[str, Any],
    feature_spec: EnhancedFeatureSpec,
    news_history_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build one live enhanced feature row from the shared Stage 8 feature spec."""

    merged_row: dict[str, Any] = {
        **dict(historical_row_mapping),
        **dict(news_feature_row_mapping),
    }
    for window in feature_spec.lag_windows:
        if len(news_history_frame) >= window:
            lag_row = news_history_frame.iloc[-window].to_dict()
        else:
            lag_row = {}
        for column in feature_spec.base_news_feature_columns:
            lagged_name = f"{column}_lag{window}"
            merged_row[lagged_name] = float(lag_row.get(column, 0.0) or 0.0)

    for interaction_spec in feature_spec.interaction_specs:
        left_value = float(merged_row.get(interaction_spec.left_column, 0.0) or 0.0)
        right_value = float(merged_row.get(interaction_spec.right_column, 0.0) or 0.0)
        merged_row[interaction_spec.name] = (
            left_value - interaction_spec.left_center
        ) * right_value

    for column in feature_spec.cross_feature_columns:
        merged_row.setdefault(column, 0.0)
    return merged_row


def build_merged_enhanced_dataset(
    *,
    historical_dataset: BaselineDataset,
    news_dataset: NewsFeatureDataset,
    lag_windows: tuple[int, ...],
) -> EnhancedDataset:
    """Build the Stage 8 merged dataset keyed by prediction date and mode."""

    historical_frame = historical_dataset.dataset_frame.copy().rename(
        columns={
            "date": "historical_date",
            "target_date": "prediction_date",
        }
    )
    expanded_frames: list[pd.DataFrame] = []
    for prediction_mode in news_dataset.supported_prediction_modes:
        mode_frame = historical_frame.copy()
        mode_frame["prediction_mode"] = prediction_mode
        expanded_frames.append(mode_frame)
    expanded_historical_frame = pd.concat(expanded_frames, ignore_index=True)

    news_frame = news_dataset.dataset_frame.copy().rename(columns={"date": "prediction_date"})
    news_frame = news_frame.loc[
        :,
        ("prediction_date", "ticker", "exchange", "prediction_mode")
        + news_dataset.feature_columns,
    ]

    merged_frame = expanded_historical_frame.merge(
        news_frame,
        how="left",
        on=["prediction_date", "ticker", "exchange", "prediction_mode"],
        validate="one_to_one",
    )
    missing_news_mask = merged_frame.loc[:, list(news_dataset.feature_columns)].isna().all(axis=1)
    merged_frame.loc[:, list(news_dataset.feature_columns)] = (
        merged_frame.loc[:, list(news_dataset.feature_columns)].fillna(0.0)
    )
    feature_spec = build_enhanced_feature_spec(
        historical_feature_columns=historical_dataset.feature_columns,
        base_news_feature_columns=news_dataset.feature_columns,
        lag_windows=lag_windows,
    )
    merged_frame = apply_enhanced_feature_spec_to_frame(
        merged_frame,
        feature_spec=feature_spec,
    )

    if merged_frame.duplicated(subset=list(MERGED_IDENTITY_COLUMNS[:5])).any():
        raise EnhancedModelError("Merged Stage 8 dataset contains duplicate identity rows.")
    if merged_frame["prediction_date"].isna().any() or merged_frame["historical_date"].isna().any():
        raise EnhancedModelError("Merged Stage 8 dataset contains invalid date values.")

    merged_frame["prediction_mode_sort_key"] = merged_frame["prediction_mode"].map(
        PREDICTION_MODE_ORDER
    )
    merged_frame = merged_frame.sort_values(
        by=["prediction_date", "prediction_mode_sort_key", "historical_date"],
        ascending=[True, True, True],
    ).drop(columns=["prediction_mode_sort_key"]).reset_index(drop=True)

    feature_groups = build_feature_groups(
        historical_feature_columns=historical_dataset.feature_columns,
        news_feature_columns=feature_spec.extended_news_feature_columns,
    )
    feature_columns = historical_dataset.feature_columns + feature_spec.extended_news_feature_columns
    return EnhancedDataset(
        dataset_frame=merged_frame,
        feature_columns=feature_columns,
        historical_feature_columns=historical_dataset.feature_columns,
        news_feature_columns=feature_spec.extended_news_feature_columns,
        target_column=historical_dataset.target_column,
        feature_groups=feature_groups,
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        missing_news_row_count=int(missing_news_mask.sum()),
    )


def build_feature_groups(
    *,
    historical_feature_columns: tuple[str, ...],
    news_feature_columns: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Group Stage 8 features into the metadata buckets used for analysis."""

    event_columns = tuple(
        column for column in news_feature_columns if "news_event_count_" in column
    )
    quality_columns = tuple(
        column
        for column in news_feature_columns
        if any(
            column.startswith(base) for base in (
                "news_article_count",
                "news_max_severity",
                "news_full_article_count",
                "news_headline_plus_snippet_count",
                "news_headline_only_count",
                "news_warning_article_count",
                "news_fallback_article_ratio",
                "news_avg_content_quality_score",
            )
        )
    )
    sentiment_columns = tuple(
        column
        for column in news_feature_columns
        if column not in event_columns and column not in quality_columns
    )
    return {
        HISTORICAL_FEATURE_GROUP_KEY: historical_feature_columns,
        NEWS_SENTIMENT_GROUP_KEY: sentiment_columns,
        NEWS_EVENT_GROUP_KEY: event_columns,
        NEWS_QUALITY_GROUP_KEY: quality_columns,
    }


def save_merged_enhanced_dataset(
    *,
    path: Path,
    metadata_path: Path,
    settings: AppSettings,
    dataset: EnhancedDataset,
    artifact_variant: str | None = None,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
) -> None:
    """Persist the Stage 8 merged dataset and its metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.dataset_frame.to_csv(path, index=False)
    metadata = {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "merged_dataset_path": str(path),
        "merged_dataset_hash": compute_file_sha256(path),
        "source_historical_feature_path": str(dataset.historical_dataset.source_feature_table_path),
        "source_historical_feature_hash": dataset.historical_dataset.source_feature_table_hash,
        "source_historical_metadata_path": str(dataset.historical_dataset.source_feature_metadata_path),
        "source_historical_metadata_hash": dataset.historical_dataset.source_feature_metadata_hash,
        "source_news_feature_path": str(dataset.news_dataset.source_feature_table_path),
        "source_news_feature_hash": dataset.news_dataset.source_feature_table_hash,
        "source_news_metadata_path": str(dataset.news_dataset.source_feature_metadata_path),
        "source_news_metadata_hash": dataset.news_dataset.source_feature_metadata_hash,
        "source_news_formula_version": dataset.news_dataset.source_metadata.get("formula_version"),
        "base_news_feature_columns": list(dataset.news_dataset.feature_columns),
        "feature_columns": list(dataset.feature_columns),
        "historical_feature_columns": list(dataset.historical_feature_columns),
        "news_feature_columns": list(dataset.news_feature_columns),
        "feature_spec": serialize_enhanced_feature_spec(
            build_enhanced_feature_spec(
                historical_feature_columns=dataset.historical_feature_columns,
                base_news_feature_columns=dataset.news_dataset.feature_columns,
                lag_windows=settings.historical_features.lag_windows,
            )
        ),
        "feature_groups": {
            group_name: list(columns)
            for group_name, columns in dataset.feature_groups.items()
        },
        "artifact_variant": artifact_variant,
        "target_column": dataset.target_column,
        "row_count": int(len(dataset.dataset_frame)),
        "supported_prediction_modes": list(dataset.news_dataset.supported_prediction_modes),
        "prediction_mode_row_counts": count_series_values(
            dataset.dataset_frame,
            "prediction_mode",
        ),
        "coverage_start": str(dataset.dataset_frame.iloc[0]["prediction_date"]),
        "coverage_end": str(dataset.dataset_frame.iloc[-1]["prediction_date"]),
        "missing_news_row_count": dataset.missing_news_row_count,
        "zero_filled_news_row_count": dataset.missing_news_row_count,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }
    write_json_file(metadata_path, metadata)


def load_cached_merged_enhanced_dataset(
    *,
    path: Path,
    metadata_path: Path,
    historical_dataset: BaselineDataset,
    news_dataset: NewsFeatureDataset,
    lag_windows: tuple[int, ...],
) -> EnhancedDataset | None:
    """Reuse the merged Stage 8 dataset when the saved inputs still match."""

    if not path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if metadata.get("source_historical_feature_hash") != historical_dataset.source_feature_table_hash:
        return None
    if metadata.get("source_historical_metadata_hash") != historical_dataset.source_feature_metadata_hash:
        return None
    if metadata.get("source_news_feature_hash") != news_dataset.source_feature_table_hash:
        return None
    if metadata.get("source_news_metadata_hash") != news_dataset.source_feature_metadata_hash:
        return None
    if metadata.get("supported_prediction_modes") != list(news_dataset.supported_prediction_modes):
        return None

    feature_spec = build_enhanced_feature_spec(
        historical_feature_columns=historical_dataset.feature_columns,
        base_news_feature_columns=news_dataset.feature_columns,
        lag_windows=lag_windows,
    )

    expected_feature_groups = build_feature_groups(
        historical_feature_columns=historical_dataset.feature_columns,
        news_feature_columns=feature_spec.extended_news_feature_columns,
    )
    expected_feature_columns = list(
        historical_dataset.feature_columns + feature_spec.extended_news_feature_columns
    )

    if metadata.get("feature_columns") != expected_feature_columns:
        return None
    if metadata.get("base_news_feature_columns") not in (
        None,
        list(news_dataset.feature_columns),
    ):
        return None

    try:
        dataset_frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None

    required_columns = list(MERGED_IDENTITY_COLUMNS) + list(
        historical_dataset.feature_columns + feature_spec.extended_news_feature_columns
    ) + [historical_dataset.target_column]
    missing_columns = [column for column in required_columns if column not in dataset_frame.columns]
    if missing_columns:
        return None

    dataset_frame["historical_date"] = pd.to_datetime(
        dataset_frame["historical_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    dataset_frame["prediction_date"] = pd.to_datetime(
        dataset_frame["prediction_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    if dataset_frame["historical_date"].isna().any() or dataset_frame["prediction_date"].isna().any():
        return None

    for column in historical_dataset.feature_columns + feature_spec.extended_news_feature_columns:
        numeric_series = pd.to_numeric(dataset_frame[column], errors="coerce")
        if numeric_series.isna().any() or not np.isfinite(numeric_series.to_numpy(dtype=float)).all():
            return None
        dataset_frame[column] = numeric_series.astype(float)
    dataset_frame[historical_dataset.target_column] = pd.to_numeric(
        dataset_frame[historical_dataset.target_column],
        errors="coerce",
    ).astype(int)

    return EnhancedDataset(
        dataset_frame=dataset_frame,
        feature_columns=historical_dataset.feature_columns + feature_spec.extended_news_feature_columns,
        historical_feature_columns=historical_dataset.feature_columns,
        news_feature_columns=feature_spec.extended_news_feature_columns,
        target_column=historical_dataset.target_column,
        feature_groups=expected_feature_groups,
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        missing_news_row_count=int(metadata.get("missing_news_row_count", 0)),
    )


def ensure_baseline_artifacts(
    *,
    settings: AppSettings,
    historical_dataset: BaselineDataset,
    feature_table_path: Path,
    force_refresh: bool,
) -> BaselineComparisonArtifacts:
    """Reuse aligned baseline artifacts or refresh them before Stage 8 comparison."""

    path_manager = PathManager(settings.paths)
    model_path = path_manager.build_baseline_model_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_baseline_model_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    predictions_path = path_manager.build_baseline_predictions_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metrics_path = path_manager.build_baseline_metrics_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )

    metadata: dict[str, Any] | None = None
    status = "reused"
    if not force_refresh:
        metadata = load_baseline_metadata_if_aligned(
            settings=settings,
            historical_dataset=historical_dataset,
            metadata_path=metadata_path,
            model_path=model_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
        )
    if metadata is None:
        train_baseline_model(
            settings,
            feature_table_path=feature_table_path,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = "refreshed"

    return BaselineComparisonArtifacts(
        saved_model=load_saved_baseline_model(model_path),
        metadata=metadata,
        status=status,
    )


def load_baseline_metadata_if_aligned(
    *,
    settings: AppSettings,
    historical_dataset: BaselineDataset,
    metadata_path: Path,
    model_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> dict[str, Any] | None:
    """Return baseline metadata when the current artifacts still match Stage 8 inputs."""

    if not (
        metadata_path.exists()
        and model_path.exists()
        and predictions_path.exists()
        and metrics_path.exists()
    ):
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    expected_split = split_baseline_dataset(
        historical_dataset.dataset_frame,
        settings.baseline_model,
    )
    expected_model_params = build_baseline_model_params(settings)
    expected_split_summary = {
        "train": build_split_summary(expected_split.train_frame, date_column="date"),
        "validation": build_split_summary(
            expected_split.validation_frame,
            date_column="date",
        ),
        "test": build_split_summary(expected_split.test_frame, date_column="date"),
    }
    if metadata.get("source_feature_table_hash") != historical_dataset.source_feature_table_hash:
        return None
    if metadata.get("source_feature_metadata_hash") != historical_dataset.source_feature_metadata_hash:
        return None
    if metadata.get("feature_columns") != list(historical_dataset.feature_columns):
        return None
    if metadata.get("model_params") != expected_model_params:
        return None
    if metadata.get("split_summary") != expected_split_summary:
        return None
    return metadata


def split_enhanced_dataset(
    dataset_frame: pd.DataFrame,
    enhanced_settings: EnhancedModelSettings,
) -> TemporalDatasetSplit:
    """Split one prediction-mode dataset into train, validation, and test windows."""

    return split_temporal_dataset(
        dataset_frame,
        train_ratio=enhanced_settings.train_ratio,
        validation_ratio=enhanced_settings.validation_ratio,
        test_ratio=enhanced_settings.test_ratio,
        error_factory=EnhancedModelError,
        dataset_label="Enhanced mode dataset",
    )


def fit_enhanced_model(
    *,
    train_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    enhanced_settings: EnhancedModelSettings,
    random_seed: int,
    prediction_mode: str,
    tuned_params: dict[str, Any] | None = None,
) -> PersistedEnhancedModel:
    """Fit the configured enhanced model on training rows only."""

    if train_frame[target_column].nunique() < 2:
        raise EnhancedModelError(
            "Enhanced training split must contain both target classes."
        )

    tp = tuned_params or {}
    pipeline = build_logistic_regression_pipeline(
        model_type=enhanced_settings.model_type,
        logistic_c=tp.get("C", enhanced_settings.logistic_c),
        logistic_max_iter=enhanced_settings.logistic_max_iter,
        random_seed=random_seed,
        gbm_n_estimators=tp.get("n_estimators", enhanced_settings.gbm_n_estimators),
        gbm_max_depth=tp.get("max_depth", enhanced_settings.gbm_max_depth),
        gbm_learning_rate=tp.get("learning_rate", enhanced_settings.gbm_learning_rate),
        gbm_subsample=enhanced_settings.gbm_subsample,
        gbm_min_samples_leaf=tp.get("min_samples_leaf", enhanced_settings.gbm_min_samples_leaf),
        rf_n_estimators=tp.get("n_estimators", enhanced_settings.rf_n_estimators),
        rf_max_depth=tp.get("max_depth", enhanced_settings.rf_max_depth),
        rf_min_samples_leaf=tp.get("min_samples_leaf", enhanced_settings.rf_min_samples_leaf),
        enable_calibration=False,
    )
    fit_kwargs: dict[str, Any] = {}
    if enhanced_settings.enable_class_weight:
        fit_kwargs["classifier__sample_weight"] = compute_sample_weights(
            train_frame[target_column],
            enhanced_settings.class_weight_strategy,
        )
    pipeline.fit(
        train_frame.loc[:, feature_columns],
        train_frame[target_column],
        **fit_kwargs,
    )
    return PersistedEnhancedModel(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_column=target_column,
        model_type=enhanced_settings.model_type,
        classification_threshold=enhanced_settings.classification_threshold,
        prediction_mode=prediction_mode,
    )


def save_enhanced_model(model_path: Path, saved_model: PersistedEnhancedModel) -> Path:
    """Persist one trained Stage 8 enhanced model bundle."""

    return save_pickle_artifact(model_path, saved_model)


def load_saved_enhanced_model(model_path: Path) -> PersistedEnhancedModel:
    """Load a persisted Kubera enhanced model bundle."""

    return load_pickle_artifact(
        model_path,
        expected_type=PersistedEnhancedModel,
        error_factory=EnhancedModelError,
        artifact_label="Enhanced model",
    )


def predict_with_saved_enhanced_model(
    saved_model: PersistedEnhancedModel,
    feature_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Generate predicted classes and positive-class probabilities for Stage 8."""

    return predict_binary_classifier(
        pipeline=saved_model.pipeline,
        feature_frame=feature_frame,
        expected_feature_columns=saved_model.feature_columns,
        classification_threshold=saved_model.classification_threshold,
        error_factory=EnhancedModelError,
        calibrator=saved_model.calibrator,
    )


def predict_with_saved_enhanced_model_outputs(
    saved_model: PersistedEnhancedModel,
    feature_frame: pd.DataFrame,
) -> BinaryPredictionOutputs:
    """Generate raw and calibrated probabilities for a saved enhanced model."""

    return predict_binary_classifier_outputs(
        pipeline=saved_model.pipeline,
        feature_frame=feature_frame,
        expected_feature_columns=saved_model.feature_columns,
        classification_threshold=saved_model.classification_threshold,
        error_factory=EnhancedModelError,
        calibrator=saved_model.calibrator,
    )


def build_enhanced_prediction_frame(
    *,
    split_name: str,
    split_frame: pd.DataFrame,
    saved_model: PersistedEnhancedModel,
) -> pd.DataFrame:
    """Build the persisted Stage 8 prediction rows for one evaluation split."""

    prediction_outputs = predict_with_saved_enhanced_model_outputs(
        saved_model,
        split_frame.loc[:, saved_model.feature_columns],
    )
    optional_context_columns = tuple(
        column for column in OPTIONAL_EVALUATION_CONTEXT_COLUMNS if column in split_frame.columns
    )
    prediction_frame = split_frame.loc[
        :,
        ENHANCED_PREDICTION_IDENTITY_COLUMNS
        + optional_context_columns
        + saved_model.feature_columns
        + (saved_model.target_column,),
    ].copy()
    prediction_frame.insert(0, "split", split_name)
    prediction_frame["predicted_next_day_direction"] = prediction_outputs.predicted_labels
    prediction_frame["raw_predicted_probability_up"] = prediction_outputs.raw_probabilities
    prediction_frame["calibrated_predicted_probability_up"] = (
        prediction_outputs.calibrated_probabilities
    )
    prediction_frame["predicted_probability_up"] = prediction_outputs.calibrated_probabilities
    return prediction_frame


def build_baseline_comparison_frame(
    *,
    evaluation_frame: pd.DataFrame,
    baseline_model: PersistedBaselineModel,
    historical_feature_columns: tuple[str, ...],
    enhanced_target_column: str,
) -> pd.DataFrame:
    """Build the aligned Stage 8 baseline-versus-enhanced comparison rows."""

    baseline_outputs = predict_with_saved_model_outputs(
        baseline_model,
        evaluation_frame.loc[:, historical_feature_columns],
    )
    comparison_frame = evaluation_frame.loc[
        :,
        (
            "split",
            "historical_date",
            "prediction_date",
            "ticker",
            "exchange",
            "prediction_mode",
            enhanced_target_column,
        )
        + COMPARISON_NEWS_CONTEXT_COLUMNS
        + tuple(
            column
            for column in OPTIONAL_EVALUATION_CONTEXT_COLUMNS
            if column in evaluation_frame.columns
        ),
    ].copy()
    comparison_frame["enhanced_predicted_next_day_direction"] = (
        evaluation_frame["predicted_next_day_direction"].astype(int)
    )
    comparison_frame["enhanced_raw_predicted_probability_up"] = (
        evaluation_frame["raw_predicted_probability_up"].astype(float)
    )
    comparison_frame["enhanced_calibrated_predicted_probability_up"] = (
        evaluation_frame["calibrated_predicted_probability_up"].astype(float)
    )
    comparison_frame["enhanced_predicted_probability_up"] = (
        evaluation_frame["calibrated_predicted_probability_up"].astype(float)
    )
    comparison_frame["baseline_predicted_next_day_direction"] = baseline_outputs.predicted_labels
    comparison_frame["baseline_raw_predicted_probability_up"] = baseline_outputs.raw_probabilities
    comparison_frame["baseline_calibrated_predicted_probability_up"] = (
        baseline_outputs.calibrated_probabilities
    )
    comparison_frame["baseline_predicted_probability_up"] = baseline_outputs.calibrated_probabilities

    # Compute Blended Prediction
    news_weights = comparison_frame.apply(
        lambda row: compute_news_context_weight(
            news_article_count=row["news_article_count"],
            news_avg_confidence=row["news_avg_confidence"],
            has_fresh_news=row["has_fresh_news"],
            is_fallback_heavy=row["is_fallback_heavy"],
            is_carried_forward=row["is_carried_forward"],
        ),
        axis=1,
    )
    comparison_frame["news_context_weight"] = news_weights
    comparison_frame["blended_raw_predicted_probability_up"] = blend_probabilities(
        comparison_frame["baseline_raw_predicted_probability_up"],
        comparison_frame["enhanced_raw_predicted_probability_up"],
        news_weights,
    )
    comparison_frame["blended_calibrated_predicted_probability_up"] = blend_probabilities(
        comparison_frame["baseline_calibrated_predicted_probability_up"],
        comparison_frame["enhanced_calibrated_predicted_probability_up"],
        news_weights,
    )
    comparison_frame["blended_predicted_probability_up"] = blend_probabilities(
        comparison_frame["baseline_calibrated_predicted_probability_up"],
        comparison_frame["enhanced_calibrated_predicted_probability_up"],
        news_weights,
    )
    comparison_frame["blended_predicted_next_day_direction"] = (
        comparison_frame["blended_predicted_probability_up"] >= baseline_model.classification_threshold
    ).astype(int)

    comparison_frame["disagreement_flag"] = (
        comparison_frame["enhanced_predicted_next_day_direction"]
        != comparison_frame["baseline_predicted_next_day_direction"]
    )
    comparison_frame["enhanced_correct"] = (
        comparison_frame["enhanced_predicted_next_day_direction"]
        == comparison_frame[enhanced_target_column]
    )
    comparison_frame["baseline_correct"] = (
        comparison_frame["baseline_predicted_next_day_direction"]
        == comparison_frame[enhanced_target_column]
    )
    comparison_frame["blended_correct"] = (
        comparison_frame["blended_predicted_next_day_direction"]
        == comparison_frame[enhanced_target_column]
    )
    return comparison_frame


def build_baseline_comparison_summary(
    *,
    comparison_frame: pd.DataFrame,
    prediction_mode: str,
    baseline_artifact_status: str,
    baseline_metadata: dict[str, Any],
    feature_importance_summary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize one Stage 8 comparison table."""

    disagreement_frame = comparison_frame.loc[comparison_frame["disagreement_flag"]].copy()
    return {
        "prediction_mode": prediction_mode,
        "row_count": int(len(comparison_frame)),
        "baseline_artifact_status": baseline_artifact_status,
        "baseline_run_id": baseline_metadata.get("run_id"),
        "disagreement_count": int(len(disagreement_frame)),
        "disagreement_rate": float(disagreement_frame.shape[0] / len(comparison_frame))
        if len(comparison_frame)
        else 0.0,
        "enhanced_correct_count": int(comparison_frame["enhanced_correct"].sum()),
        "baseline_correct_count": int(comparison_frame["baseline_correct"].sum()),
        "blended_correct_count": int(comparison_frame["blended_correct"].sum()),
        "enhanced_better_count": int(
            ((comparison_frame["enhanced_correct"]) & (~comparison_frame["baseline_correct"])).sum()
        ),
        "baseline_better_count": int(
            ((comparison_frame["baseline_correct"]) & (~comparison_frame["enhanced_correct"])).sum()
        ),
        "blended_better_than_baseline_count": int(
            ((comparison_frame["blended_correct"]) & (~comparison_frame["baseline_correct"])).sum()
        ),
        "tied_count": int(
            (comparison_frame["baseline_correct"] == comparison_frame["enhanced_correct"]).sum()
        ),
        "news_heavy_disagreement_count": int(
            (disagreement_frame["news_article_count"].astype(float) > 0).sum()
        ),
        "feature_importance_summary": feature_importance_summary,
    }


def summarize_feature_importance(
    *,
    persisted_model: PersistedEnhancedModel,
    feature_groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Summarize model-agnostic feature importance for one enhanced model."""

    importance_frame, importance_metric = build_feature_importance_frame(persisted_model)
    top_features = importance_frame.sort_values(
        by=["importance", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    news_feature_columns = (
        feature_groups[NEWS_SENTIMENT_GROUP_KEY]
        + feature_groups[NEWS_EVENT_GROUP_KEY]
        + feature_groups[NEWS_QUALITY_GROUP_KEY]
    )
    top_news_features = top_features.loc[
        top_features["feature_name"].isin(news_feature_columns)
    ].reset_index(drop=True)

    group_summaries: dict[str, dict[str, Any]] = {}
    for group_name, columns in feature_groups.items():
        group_frame = importance_frame.loc[
            importance_frame["feature_name"].isin(columns)
        ]
        importance_sum = float(group_frame["importance"].sum())
        group_summaries[group_name] = {
            "feature_count": int(len(group_frame)),
            "importance_sum": importance_sum,
            "top_features": group_frame.sort_values(
                by=["importance", "feature_name"],
                ascending=[False, True],
            )
            .head(5)
            .to_dict(orient="records"),
        }

    news_importance_sum = (
        group_summaries[NEWS_SENTIMENT_GROUP_KEY]["importance_sum"]
        + group_summaries[NEWS_EVENT_GROUP_KEY]["importance_sum"]
        + group_summaries[NEWS_QUALITY_GROUP_KEY]["importance_sum"]
    )
    historical_importance_sum = group_summaries[HISTORICAL_FEATURE_GROUP_KEY]["importance_sum"]
    total_importance = historical_importance_sum + news_importance_sum
    return {
        "prediction_mode": persisted_model.prediction_mode,
        "model_type": persisted_model.model_type,
        "importance_metric": importance_metric,
        "news_features_contributed": bool(
            news_importance_sum > 0.0 and not top_news_features.empty
        ),
        "historical_importance_sum": historical_importance_sum,
        "news_importance_sum": news_importance_sum,
        "news_share_of_importance": (
            float(news_importance_sum / total_importance) if total_importance > 0 else 0.0
        ),
        "top_features": top_features.head(10).to_dict(orient="records"),
        "top_news_features": top_news_features.head(5).to_dict(orient="records"),
        "group_summaries": group_summaries,
    }


def build_enhanced_model_metadata(
    *,
    settings: AppSettings,
    dataset: EnhancedDataset,
    split: TemporalDatasetSplit,
    persisted_model: PersistedEnhancedModel,
    prediction_mode: str,
    model_path: Path,
    metadata_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    comparison_path: Path,
    comparison_summary_path: Path,
    merged_dataset_path: Path,
    merged_dataset_metadata_path: Path,
    metrics_payload: dict[str, Any],
    baseline_artifact_status: str,
    baseline_metadata: dict[str, Any],
    mode_elapsed_seconds: float,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
) -> dict[str, Any]:
    """Build the metadata payload for a persisted Stage 8 model run."""

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "prediction_mode": prediction_mode,
        "model_type": settings.enhanced_model.model_type,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "comparison_path": str(comparison_path),
        "comparison_summary_path": str(comparison_summary_path),
        "merged_dataset_path": str(merged_dataset_path),
        "merged_dataset_hash": compute_file_sha256(merged_dataset_path),
        "merged_dataset_metadata_path": str(merged_dataset_metadata_path),
        "source_historical_feature_path": str(dataset.historical_dataset.source_feature_table_path),
        "source_historical_feature_hash": dataset.historical_dataset.source_feature_table_hash,
        "source_historical_metadata_path": str(dataset.historical_dataset.source_feature_metadata_path),
        "source_historical_metadata_hash": dataset.historical_dataset.source_feature_metadata_hash,
        "source_news_feature_path": str(dataset.news_dataset.source_feature_table_path),
        "source_news_feature_hash": dataset.news_dataset.source_feature_table_hash,
        "source_news_metadata_path": str(dataset.news_dataset.source_feature_metadata_path),
        "source_news_metadata_hash": dataset.news_dataset.source_feature_metadata_hash,
        "source_news_formula_version": dataset.news_dataset.source_metadata.get("formula_version"),
        "base_news_feature_columns": list(dataset.news_dataset.feature_columns),
        "feature_columns": list(dataset.feature_columns),
        "historical_feature_columns": list(dataset.historical_feature_columns),
        "news_feature_columns": list(dataset.news_feature_columns),
        "feature_spec": serialize_enhanced_feature_spec(
            build_enhanced_feature_spec(
                historical_feature_columns=dataset.historical_feature_columns,
                base_news_feature_columns=dataset.news_dataset.feature_columns,
                lag_windows=settings.historical_features.lag_windows,
            )
        ),
        "feature_groups": {
            group_name: list(columns)
            for group_name, columns in dataset.feature_groups.items()
        },
        "target_column": dataset.target_column,
        "classification_threshold": persisted_model.classification_threshold,
        "calibration": metrics_payload.get("calibration"),
        "calibration_method": (
            persisted_model.calibrator.method if persisted_model.calibrator is not None else None
        ),
        "model_params": build_model_params(settings),
        "split_summary": {
            "train": build_split_summary(split.train_frame, date_column="prediction_date"),
            "validation": build_split_summary(
                split.validation_frame,
                date_column="prediction_date",
            ),
            "test": build_split_summary(split.test_frame, date_column="prediction_date"),
        },
        "baseline_artifact_status": baseline_artifact_status,
        "baseline_run_id": baseline_metadata.get("run_id"),
        "baseline_model_path": baseline_metadata.get("model_path"),
        "feature_importance_summary": metrics_payload["feature_importance"],
        "missing_news_row_count": dataset.missing_news_row_count,
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
        },
        "timing": {
            "elapsed_seconds": float(mode_elapsed_seconds),
        },
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def build_feature_importance_frame(
    persisted_model: PersistedEnhancedModel,
) -> tuple[pd.DataFrame, str]:
    """Build the model-agnostic feature-importance frame for one classifier."""

    classifier = persisted_model.pipeline.named_steps["classifier"]
    
    if hasattr(classifier, "calibrated_classifiers_"):
        base_estimator = classifier.calibrated_classifiers_[0].estimator
    else:
        base_estimator = classifier

    if hasattr(base_estimator, "coef_"):
        raw_values = np.abs(np.asarray(base_estimator.coef_[0], dtype=float))
        metric = "absolute_coefficient"
    elif hasattr(base_estimator, "feature_importances_"):
        raw_values = np.asarray(base_estimator.feature_importances_, dtype=float)
        metric = "feature_importances"
    else:
        raise EnhancedModelError(
            f"Enhanced model type {persisted_model.model_type} does not expose a supported feature-importance interface."
        )
    return (
        pd.DataFrame(
            {
                "feature_name": list(persisted_model.feature_columns),
                "importance": raw_values,
            }
        ),
        metric,
    )


def build_model_params(settings: AppSettings) -> dict[str, Any]:
    """Build the persisted model-parameter payload for one enhanced run."""

    if settings.enhanced_model.model_type == "logistic_regression":
        return {
            "logistic_c": settings.enhanced_model.logistic_c,
            "logistic_max_iter": settings.enhanced_model.logistic_max_iter,
            "random_seed": settings.run.random_seed,
            "enable_calibration": settings.enhanced_model.enable_calibration,
            "enable_class_weight": settings.enhanced_model.enable_class_weight,
            "class_weight_strategy": settings.enhanced_model.class_weight_strategy,
        }
    if settings.enhanced_model.model_type == "gradient_boosting":
        return {
            "n_estimators": settings.enhanced_model.gbm_n_estimators,
            "max_depth": settings.enhanced_model.gbm_max_depth,
            "learning_rate": settings.enhanced_model.gbm_learning_rate,
            "subsample": settings.enhanced_model.gbm_subsample,
            "min_samples_leaf": settings.enhanced_model.gbm_min_samples_leaf,
            "random_seed": settings.run.random_seed,
            "enable_calibration": settings.enhanced_model.enable_calibration,
            "enable_class_weight": settings.enhanced_model.enable_class_weight,
            "class_weight_strategy": settings.enhanced_model.class_weight_strategy,
        }
    if settings.enhanced_model.model_type in {"xgboost", "lightgbm"}:
        return {
            "n_estimators": settings.enhanced_model.gbm_n_estimators,
            "max_depth": settings.enhanced_model.gbm_max_depth,
            "learning_rate": settings.enhanced_model.gbm_learning_rate,
            "subsample": settings.enhanced_model.gbm_subsample,
            "random_seed": settings.run.random_seed,
            "enable_calibration": settings.enhanced_model.enable_calibration,
            "enable_class_weight": settings.enhanced_model.enable_class_weight,
            "class_weight_strategy": settings.enhanced_model.class_weight_strategy,
        }
    if settings.enhanced_model.model_type == "random_forest":
        return {
            "n_estimators": settings.enhanced_model.rf_n_estimators,
            "max_depth": settings.enhanced_model.rf_max_depth,
            "min_samples_leaf": settings.enhanced_model.rf_min_samples_leaf,
            "random_seed": settings.run.random_seed,
            "enable_calibration": settings.enhanced_model.enable_calibration,
            "enable_class_weight": settings.enhanced_model.enable_class_weight,
            "class_weight_strategy": settings.enhanced_model.class_weight_strategy,
        }
    raise EnhancedModelError(f"Unsupported enhanced model type: {settings.enhanced_model.model_type}")


def count_series_values(frame: pd.DataFrame, column_name: str) -> dict[str, int]:
    """Count string values in one DataFrame column for metadata."""

    if frame.empty or column_name not in frame.columns:
        return {}
    counts = frame[column_name].fillna("null").value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 8 enhanced-model command arguments."""

    parser = argparse.ArgumentParser(description="Train Kubera Stage 8 enhanced models.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--historical-feature-path",
        help="Use a specific Stage 3 historical feature CSV file.",
    )
    parser.add_argument(
        "--news-feature-path",
        help="Use a specific Stage 7 news feature CSV file.",
    )
    parser.add_argument(
        "--force-baseline-refresh",
        action="store_true",
        help="Retrain the baseline artifact before Stage 8 comparison.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 8 enhanced-model training command."""

    args = parse_args(argv)
    settings = load_settings()
    train_enhanced_models(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        historical_feature_path=args.historical_feature_path,
        news_feature_path=args.news_feature_path,
        force_baseline_refresh=args.force_baseline_refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
