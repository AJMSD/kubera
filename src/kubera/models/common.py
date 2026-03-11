"""Shared modeling helpers for Kubera."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ErrorFactory = Callable[[str], Exception]


@dataclass(frozen=True)
class TemporalDatasetSplit:
    """One temporal train/validation/test split."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame


def split_temporal_dataset(
    dataset_frame: pd.DataFrame,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    error_factory: ErrorFactory,
    dataset_label: str,
) -> TemporalDatasetSplit:
    """Split a dataset by row order into train, validation, and test windows."""

    del test_ratio
    row_count = len(dataset_frame)
    train_end = int(row_count * train_ratio)
    validation_end = int(row_count * (train_ratio + validation_ratio))

    train_frame = dataset_frame.iloc[:train_end].copy()
    validation_frame = dataset_frame.iloc[train_end:validation_end].copy()
    test_frame = dataset_frame.iloc[validation_end:].copy()

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise error_factory(
            f"{dataset_label} does not contain enough rows for the configured temporal split."
        )

    return TemporalDatasetSplit(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
    )


def build_logistic_regression_pipeline(
    *,
    logistic_c: float,
    logistic_max_iter: int,
    random_seed: int,
) -> Pipeline:
    """Build the standard Kubera logistic-regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=logistic_c,
                    max_iter=logistic_max_iter,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def save_pickle_artifact(path: Path, payload: Any) -> Path:
    """Persist one trusted local model artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle_artifact(
    path: Path,
    *,
    expected_type: type[Any],
    error_factory: ErrorFactory,
    artifact_label: str,
) -> Any:
    """Load and validate one trusted local pickled artifact."""

    try:
        with path.open("rb") as file_handle:
            loaded = pickle.load(file_handle)
    except FileNotFoundError as exc:
        raise error_factory(f"{artifact_label} file does not exist: {path}") from exc

    if not isinstance(loaded, expected_type):
        raise error_factory(
            f"{artifact_label} artifact does not contain a supported Kubera payload: {path}"
        )
    return loaded


def validate_feature_order(
    feature_frame: pd.DataFrame,
    expected_feature_columns: tuple[str, ...],
    *,
    error_factory: ErrorFactory,
) -> None:
    """Require the exact saved feature order before inference."""

    actual_feature_columns = tuple(feature_frame.columns.tolist())
    if actual_feature_columns != expected_feature_columns:
        raise error_factory(
            f"Feature order mismatch. Expected {list(expected_feature_columns)}, got {list(actual_feature_columns)}"
        )


def predict_binary_classifier(
    *,
    pipeline: Pipeline,
    feature_frame: pd.DataFrame,
    expected_feature_columns: tuple[str, ...],
    classification_threshold: float,
    error_factory: ErrorFactory,
) -> tuple[pd.Series, pd.Series]:
    """Generate binary labels and positive-class probabilities."""

    validate_feature_order(
        feature_frame,
        expected_feature_columns,
        error_factory=error_factory,
    )
    probabilities = pipeline.predict_proba(feature_frame)[:, 1]
    predicted_labels = (probabilities >= classification_threshold).astype(int)
    return (
        pd.Series(predicted_labels, index=feature_frame.index, dtype="int64"),
        pd.Series(probabilities, index=feature_frame.index, dtype="float64"),
    )


def compute_split_metrics(
    *,
    split_name: str,
    prediction_frame: pd.DataFrame,
    target_column: str,
    logger: Any,
    date_column: str,
) -> dict[str, Any]:
    """Compute the standard Kubera evaluation metrics for one split."""

    actual = prediction_frame[target_column].astype(int)
    predicted = prediction_frame["predicted_next_day_direction"].astype(int)
    predicted_probabilities = prediction_frame["predicted_probability_up"].astype(float)

    metrics = {
        "row_count": int(len(prediction_frame)),
        "date_start": str(prediction_frame.iloc[0][date_column]),
        "date_end": str(prediction_frame.iloc[-1][date_column]),
        "target_positive_count": int((actual == 1).sum()),
        "target_negative_count": int((actual == 0).sum()),
        "predicted_positive_count": int((predicted == 1).sum()),
        "predicted_negative_count": int((predicted == 0).sum()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }

    if actual.nunique() < 2:
        logger.warning(
            "Probability metrics are undefined on a single-class evaluation split | split=%s | rows=%s",
            split_name,
            len(prediction_frame),
        )
        metrics["roc_auc"] = None
        metrics["log_loss"] = None
        metrics["brier_score"] = None
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(actual, predicted_probabilities))
    metrics["log_loss"] = float(log_loss(actual, predicted_probabilities, labels=[0, 1]))
    metrics["brier_score"] = float(brier_score_loss(actual, predicted_probabilities))
    return metrics


def build_split_summary(
    split_frame: pd.DataFrame,
    *,
    date_column: str,
) -> dict[str, Any]:
    """Summarize one temporal split for persisted metadata."""

    return {
        "row_count": int(len(split_frame)),
        "date_start": str(split_frame.iloc[0][date_column]),
        "date_end": str(split_frame.iloc[-1][date_column]),
    }
