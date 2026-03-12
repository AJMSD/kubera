"""Baseline model training for Kubera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.pipeline import Pipeline

from kubera.config import (
    AppSettings,
    BaselineModelSettings,
    load_settings,
    resolve_runtime_settings,
)
from kubera.models.common import (
    TemporalDatasetSplit,
    build_logistic_regression_pipeline,
    build_split_summary as build_common_split_summary,
    compute_split_metrics as compute_common_split_metrics,
    load_pickle_artifact,
    predict_binary_classifier,
    save_pickle_artifact,
    split_temporal_dataset,
    validate_feature_order as validate_common_feature_order,
)
from kubera.utils.hashing import compute_file_sha256
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file, write_settings_snapshot


REQUIRED_BASELINE_COLUMNS = ("date", "target_date", "ticker", "exchange", "close", "volume")
PREDICTION_IDENTITY_COLUMNS = (
    "date",
    "target_date",
    "ticker",
    "exchange",
    "close",
    "volume",
)


class BaselineModelError(RuntimeError):
    """Raised when baseline training or inference cannot continue."""


@dataclass(frozen=True)
class BaselineDataset:
    dataset_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    source_feature_table_path: Path
    source_feature_metadata_path: Path
    source_feature_table_hash: str
    source_feature_metadata_hash: str
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class PersistedBaselineModel:
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    target_column: str
    model_type: str
    classification_threshold: float


CANONICAL_BASELINE_MODULE_NAME = "kubera.models.train_baseline"
if __name__ == "__main__":
    sys.modules.setdefault(CANONICAL_BASELINE_MODULE_NAME, sys.modules[__name__])
PersistedBaselineModel.__module__ = CANONICAL_BASELINE_MODULE_NAME


@dataclass(frozen=True)
class BaselineTrainingResult:
    model_path: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path
    train_row_count: int
    validation_row_count: int
    test_row_count: int


def train_baseline_model(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    feature_table_path: str | Path | None = None,
) -> BaselineTrainingResult:
    """Train and persist the Stage 4 historical-only baseline model."""

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

    source_feature_table_path = resolve_feature_table_path(
        runtime_settings,
        path_manager=path_manager,
        feature_table_path=feature_table_path,
    )
    source_feature_metadata_path = infer_feature_metadata_path(source_feature_table_path)
    dataset = load_baseline_dataset(
        feature_table_path=source_feature_table_path,
        feature_metadata_path=source_feature_metadata_path,
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
    )
    split = split_baseline_dataset(dataset.dataset_frame, runtime_settings.baseline_model)
    persisted_model = fit_baseline_model(
        train_frame=split.train_frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        baseline_settings=runtime_settings.baseline_model,
        random_seed=runtime_settings.run.random_seed,
    )

    validation_predictions = build_prediction_frame(
        split_name="validation",
        split_frame=split.validation_frame,
        saved_model=persisted_model,
    )
    test_predictions = build_prediction_frame(
        split_name="test",
        split_frame=split.test_frame,
        saved_model=persisted_model,
    )
    predictions_frame = pd.concat(
        [validation_predictions, test_predictions],
        ignore_index=True,
    )

    metrics_payload = {
        "model_type": persisted_model.model_type,
        "classification_threshold": persisted_model.classification_threshold,
        "validation": compute_split_metrics(
            split_name="validation",
            prediction_frame=validation_predictions,
            target_column=dataset.target_column,
            logger=logger,
        ),
        "test": compute_split_metrics(
            split_name="test",
            prediction_frame=test_predictions,
            target_column=dataset.target_column,
            logger=logger,
        ),
    }

    model_path = path_manager.build_baseline_model_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    metadata_path = path_manager.build_baseline_model_metadata_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    predictions_path = path_manager.build_baseline_predictions_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    metrics_path = path_manager.build_baseline_metrics_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )

    save_baseline_model(model_path, persisted_model)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(predictions_path, index=False)
    write_json_file(metrics_path, metrics_payload)
    write_json_file(
        metadata_path,
        build_model_metadata(
            settings=runtime_settings,
            dataset=dataset,
            split=split,
            model_path=model_path,
            metadata_path=metadata_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            run_id=run_context.run_id,
            git_commit=run_context.git_commit,
            git_is_dirty=run_context.git_is_dirty,
        ),
    )

    logger.info(
        "Baseline model ready | ticker=%s | exchange=%s | train_rows=%s | validation_rows=%s | test_rows=%s | model=%s | metrics=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        len(split.train_frame),
        len(split.validation_frame),
        len(split.test_frame),
        model_path,
        metrics_path,
    )

    return BaselineTrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        train_row_count=len(split.train_frame),
        validation_row_count=len(split.validation_frame),
        test_row_count=len(split.test_frame),
    )


def resolve_feature_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    feature_table_path: str | Path | None,
) -> Path:
    """Resolve the Stage 3 feature-table path for baseline training."""

    if feature_table_path is not None:
        return Path(feature_table_path).expanduser().resolve()

    return path_manager.build_historical_feature_table_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def infer_feature_metadata_path(feature_table_path: Path) -> Path:
    """Infer the Stage 3 feature metadata path from the feature CSV path."""

    if feature_table_path.suffix.lower() != ".csv":
        raise BaselineModelError(
            "Historical feature table path must point to a CSV file so the matching metadata can be resolved."
        )

    metadata_path = feature_table_path.with_name(f"{feature_table_path.stem}.metadata.json")
    if not metadata_path.exists():
        raise BaselineModelError(
            f"Historical feature metadata file does not exist: {metadata_path}"
        )
    return metadata_path


def load_baseline_dataset(
    *,
    feature_table_path: Path,
    feature_metadata_path: Path,
    ticker: str,
    exchange: str,
) -> BaselineDataset:
    """Read and validate the Stage 3 feature artifact for baseline training."""

    if not feature_table_path.exists():
        raise BaselineModelError(
            f"Historical feature table does not exist: {feature_table_path}"
        )

    try:
        feature_frame = pd.read_csv(feature_table_path)
    except pd.errors.EmptyDataError as exc:
        raise BaselineModelError(
            f"Historical feature table is empty: {feature_table_path}"
        ) from exc

    try:
        source_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineModelError(
            f"Historical feature metadata is not valid JSON: {feature_metadata_path}"
        ) from exc

    raw_feature_columns = source_metadata.get("feature_columns")
    if not isinstance(raw_feature_columns, list) or not raw_feature_columns:
        raise BaselineModelError(
            "Historical feature metadata must include a non-empty feature_columns list."
        )
    feature_columns = tuple(str(column) for column in raw_feature_columns)
    target_column = str(source_metadata.get("target_column", "")).strip()
    if not target_column:
        raise BaselineModelError(
            "Historical feature metadata must include a target_column value."
        )

    missing_columns = [
        column
        for column in REQUIRED_BASELINE_COLUMNS + feature_columns + (target_column,)
        if column not in feature_frame.columns
    ]
    if missing_columns:
        raise BaselineModelError(
            f"Historical feature table is missing required columns: {missing_columns}"
        )

    working_frame = feature_frame.copy()
    working_frame["date"] = pd.to_datetime(working_frame["date"], errors="coerce")
    working_frame["target_date"] = pd.to_datetime(working_frame["target_date"], errors="coerce")
    if working_frame["date"].isna().any() or working_frame["target_date"].isna().any():
        raise BaselineModelError("Historical feature table contains invalid date values.")

    working_frame = working_frame.sort_values("date").reset_index(drop=True)
    if working_frame["date"].duplicated().any():
        raise BaselineModelError("Historical feature table contains duplicate dates.")

    source_tickers = {
        str(value).strip().upper()
        for value in working_frame["ticker"].dropna().unique().tolist()
    }
    source_exchanges = {
        str(value).strip().upper()
        for value in working_frame["exchange"].dropna().unique().tolist()
    }
    if source_tickers != {ticker.upper()}:
        raise BaselineModelError(
            f"Historical feature table ticker values do not match the requested ticker: {sorted(source_tickers)}"
        )
    if source_exchanges != {exchange.upper()}:
        raise BaselineModelError(
            f"Historical feature table exchange values do not match the requested exchange: {sorted(source_exchanges)}"
        )

    numeric_columns = ("close", "volume", target_column) + feature_columns
    for column in numeric_columns:
        working_frame[column] = pd.to_numeric(working_frame[column], errors="coerce")
        if working_frame[column].isna().any():
            raise BaselineModelError(
                f"Historical feature table contains non-numeric or missing values in column: {column}"
            )
        if not np.isfinite(working_frame[column].to_numpy(dtype=float)).all():
            raise BaselineModelError(
                f"Historical feature table contains non-finite values in column: {column}"
            )

    if (working_frame["close"] <= 0).any():
        raise BaselineModelError("Historical feature table close values must be positive.")
    if (working_frame["volume"] < 0).any():
        raise BaselineModelError("Historical feature table volume values must be non-negative.")

    target_values = set(working_frame[target_column].astype(int).tolist())
    if target_values - {0, 1}:
        raise BaselineModelError(
            f"Historical feature table target values must stay binary, got: {sorted(target_values)}"
        )

    working_frame[target_column] = working_frame[target_column].astype(int)
    working_frame["date"] = working_frame["date"].dt.strftime("%Y-%m-%d")
    working_frame["target_date"] = working_frame["target_date"].dt.strftime("%Y-%m-%d")

    return BaselineDataset(
        dataset_frame=working_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        source_feature_table_path=feature_table_path,
        source_feature_metadata_path=feature_metadata_path,
        source_feature_table_hash=compute_file_sha256(feature_table_path),
        source_feature_metadata_hash=compute_file_sha256(feature_metadata_path),
        source_metadata=source_metadata,
    )


def split_baseline_dataset(
    dataset_frame: pd.DataFrame,
    baseline_settings: BaselineModelSettings,
) -> TemporalDatasetSplit:
    """Split the baseline dataset by row order into train, validation, and test."""

    return split_temporal_dataset(
        dataset_frame,
        train_ratio=baseline_settings.train_ratio,
        validation_ratio=baseline_settings.validation_ratio,
        test_ratio=baseline_settings.test_ratio,
        error_factory=BaselineModelError,
        dataset_label="Historical feature table",
    )


def fit_baseline_model(
    *,
    train_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    baseline_settings: BaselineModelSettings,
    random_seed: int,
) -> PersistedBaselineModel:
    """Fit the configured baseline model on training rows only."""

    if train_frame[target_column].nunique() < 2:
        raise BaselineModelError(
            "Baseline training split must contain both target classes."
        )

    pipeline = build_logistic_regression_pipeline(
        model_type=baseline_settings.model_type,
        logistic_c=baseline_settings.logistic_c,
        logistic_max_iter=baseline_settings.logistic_max_iter,
        random_seed=random_seed,
    )
    pipeline.fit(
        train_frame.loc[:, feature_columns],
        train_frame[target_column],
    )
    return PersistedBaselineModel(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_column=target_column,
        model_type=baseline_settings.model_type,
        classification_threshold=baseline_settings.classification_threshold,
    )


def save_baseline_model(model_path: Path, saved_model: PersistedBaselineModel) -> Path:
    """Persist the fitted baseline model bundle."""

    return save_pickle_artifact(model_path, saved_model)


def load_saved_baseline_model(model_path: Path) -> PersistedBaselineModel:
    """Load a persisted Kubera baseline model bundle."""

    return load_pickle_artifact(
        model_path,
        expected_type=PersistedBaselineModel,
        error_factory=BaselineModelError,
        artifact_label="Baseline model",
    )


def validate_feature_order(
    feature_frame: pd.DataFrame,
    expected_feature_columns: tuple[str, ...],
) -> None:
    """Require the exact saved feature order before inference."""

    validate_common_feature_order(
        feature_frame,
        expected_feature_columns,
        error_factory=BaselineModelError,
    )


def predict_with_saved_model(
    saved_model: PersistedBaselineModel,
    feature_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Generate predicted classes and positive-class probabilities."""

    return predict_binary_classifier(
        pipeline=saved_model.pipeline,
        feature_frame=feature_frame,
        expected_feature_columns=saved_model.feature_columns,
        classification_threshold=saved_model.classification_threshold,
        error_factory=BaselineModelError,
    )


def build_prediction_frame(
    *,
    split_name: str,
    split_frame: pd.DataFrame,
    saved_model: PersistedBaselineModel,
) -> pd.DataFrame:
    """Build the persisted prediction rows for one evaluation split."""

    predicted_labels, predicted_probabilities = predict_with_saved_model(
        saved_model,
        split_frame.loc[:, saved_model.feature_columns],
    )
    prediction_frame = split_frame.loc[
        :,
        PREDICTION_IDENTITY_COLUMNS + (saved_model.target_column,),
    ].copy()
    prediction_frame.insert(0, "split", split_name)
    prediction_frame["predicted_next_day_direction"] = predicted_labels
    prediction_frame["predicted_probability_up"] = predicted_probabilities
    return prediction_frame


def compute_split_metrics(
    *,
    split_name: str,
    prediction_frame: pd.DataFrame,
    target_column: str,
    logger: Any,
) -> dict[str, Any]:
    """Compute evaluation metrics for a validation or test split."""

    return compute_common_split_metrics(
        split_name=split_name,
        prediction_frame=prediction_frame,
        target_column=target_column,
        logger=logger,
        date_column="date",
    )


def build_model_metadata(
    *,
    settings: AppSettings,
    dataset: BaselineDataset,
    split: TemporalDatasetSplit,
    model_path: Path,
    metadata_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
) -> dict[str, Any]:
    """Build the metadata payload for a persisted baseline model run."""

    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "model_type": settings.baseline_model.model_type,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "source_feature_table_path": str(dataset.source_feature_table_path),
        "source_feature_metadata_path": str(dataset.source_feature_metadata_path),
        "source_feature_table_hash": dataset.source_feature_table_hash,
        "source_feature_metadata_hash": dataset.source_feature_metadata_hash,
        "source_feature_formula_version": dataset.source_metadata.get("formula_version"),
        "source_feature_run_id": dataset.source_metadata.get("run_id"),
        "feature_columns": list(dataset.feature_columns),
        "target_column": dataset.target_column,
        "classification_threshold": settings.baseline_model.classification_threshold,
        "model_params": build_model_params(settings),
        "split_summary": {
            "train": build_split_summary(split.train_frame),
            "validation": build_split_summary(split.validation_frame),
            "test": build_split_summary(split.test_frame),
        },
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
        },
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def build_model_params(settings: AppSettings) -> dict[str, Any]:
    """Build the persisted model-parameter payload for one baseline run."""

    if settings.baseline_model.model_type == "logistic_regression":
        return {
            "logistic_c": settings.baseline_model.logistic_c,
            "logistic_max_iter": settings.baseline_model.logistic_max_iter,
            "random_seed": settings.run.random_seed,
        }
    if settings.baseline_model.model_type == "gradient_boosting":
        return {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_seed": settings.run.random_seed,
        }
    raise BaselineModelError(f"Unsupported baseline model type: {settings.baseline_model.model_type}")


def build_split_summary(split_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize one temporal split for persisted metadata."""

    return build_common_split_summary(split_frame, date_column="date")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse baseline training command arguments."""

    parser = argparse.ArgumentParser(description="Train the Kubera baseline model.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--feature-path",
        help="Use a specific historical feature CSV file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the baseline training command."""

    args = parse_args(argv)
    settings = load_settings()
    train_baseline_model(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        feature_table_path=args.feature_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
