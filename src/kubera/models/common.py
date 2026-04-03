"""Shared modeling helpers for Kubera."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except (ImportError, TypeError):
    # Handle Numpy 2.0 compatibility issues with some shap versions
    SHAP_AVAILABLE = False

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ErrorFactory = Callable[[str], Exception]


@dataclass(frozen=True)
class TemporalDatasetSplit:
    """One temporal train/validation/test split."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Persisted probability calibrator fitted on held-out validation data."""

    method: str
    estimator: Any


@dataclass(frozen=True)
class BinaryPredictionOutputs:
    """Raw and calibrated probabilities plus thresholded labels."""

    predicted_labels: pd.Series
    raw_probabilities: pd.Series
    calibrated_probabilities: pd.Series


@dataclass(frozen=True)
class SelectivePredictionDecision:
    """Calibrated selective prediction decision for one probability score."""

    action: str
    abstain: bool
    predicted_label: int | None
    probability_up: float
    probability_margin: float
    required_margin: float
    reasons: tuple[str, ...]


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
    model_type: str,
    logistic_c: float,
    logistic_max_iter: int,
    random_seed: int,
    gbm_n_estimators: int = 100,
    gbm_max_depth: int = 3,
    gbm_learning_rate: float = 0.05,
    gbm_subsample: float = 1.0,
    gbm_min_samples_leaf: int = 1,
    rf_n_estimators: int = 100,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int = 1,
    enable_calibration: bool = False,
) -> Pipeline:
    """Build the configured Kubera classifier pipeline."""

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=logistic_c,
            max_iter=logistic_max_iter,
            random_state=random_seed,
        )
        pipeline_steps = [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=gbm_n_estimators,
            max_depth=gbm_max_depth,
            learning_rate=gbm_learning_rate,
            subsample=gbm_subsample,
            min_samples_leaf=gbm_min_samples_leaf,
            random_state=random_seed,
        )
        pipeline_steps = [("classifier", classifier)]
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_seed,
            n_jobs=-1,
        )
        pipeline_steps = [("classifier", classifier)]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if enable_calibration:
        calibrated = CalibratedClassifierCV(
            estimator=classifier,
            method="isotonic",
            cv=3,
        )
        for i, (name, _) in enumerate(pipeline_steps):
            if name == "classifier":
                pipeline_steps[i] = ("classifier", calibrated)

    return Pipeline(steps=pipeline_steps)


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
    calibrator: ProbabilityCalibrator | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Generate binary labels and positive-class probabilities."""

    outputs = predict_binary_classifier_outputs(
        pipeline=pipeline,
        feature_frame=feature_frame,
        expected_feature_columns=expected_feature_columns,
        classification_threshold=classification_threshold,
        error_factory=error_factory,
        calibrator=calibrator,
    )
    return outputs.predicted_labels, outputs.calibrated_probabilities


def predict_binary_classifier_outputs(
    *,
    pipeline: Pipeline,
    feature_frame: pd.DataFrame,
    expected_feature_columns: tuple[str, ...],
    classification_threshold: float,
    error_factory: ErrorFactory,
    calibrator: ProbabilityCalibrator | None = None,
) -> BinaryPredictionOutputs:
    """Generate raw probabilities, calibrated probabilities, and labels."""

    validate_feature_order(
        feature_frame,
        expected_feature_columns,
        error_factory=error_factory,
    )
    raw_probabilities = pd.Series(
        pipeline.predict_proba(feature_frame)[:, 1],
        index=feature_frame.index,
        dtype="float64",
    )
    calibrated_probabilities = apply_probability_calibrator(
        raw_probabilities,
        calibrator=calibrator,
    )
    predicted_labels = (
        calibrated_probabilities >= classification_threshold
    ).astype(int)
    return BinaryPredictionOutputs(
        predicted_labels=pd.Series(
            predicted_labels,
            index=feature_frame.index,
            dtype="int64",
        ),
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
    )


def fit_probability_calibrator(
    *,
    raw_probabilities: pd.Series,
    actual: pd.Series,
    enabled: bool,
    random_seed: int,
) -> tuple[ProbabilityCalibrator | None, dict[str, Any]]:
    """Fit and select a probability calibrator from validation-only data."""

    if not enabled:
        return None, {"enabled": False, "selected_method": None, "candidate_metrics": {}}

    actual_series = actual.astype(int).reset_index(drop=True)
    probability_series = pd.Series(raw_probabilities, dtype="float64").reset_index(drop=True)
    if actual_series.nunique() < 2 or len(actual_series) < 8:
        return None, {
            "enabled": True,
            "selected_method": None,
            "candidate_metrics": {},
            "status": "insufficient_validation_data",
        }

    candidate_models: dict[str, ProbabilityCalibrator] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    probability_values = probability_series.to_numpy(dtype=float)
    actual_values = actual_series.to_numpy(dtype=int)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(probability_values, actual_values)
    isotonic_predictions = pd.Series(
        np.clip(isotonic.predict(probability_values), 0.0, 1.0),
        dtype="float64",
    )
    candidate_models["isotonic"] = ProbabilityCalibrator("isotonic", isotonic)
    candidate_metrics["isotonic"] = compute_probability_score_metrics(
        actual=actual_series,
        predicted_probabilities=isotonic_predictions,
    )

    sigmoid = LogisticRegression(random_state=random_seed)
    sigmoid.fit(probability_values.reshape(-1, 1), actual_values)
    sigmoid_predictions = pd.Series(
        sigmoid.predict_proba(probability_values.reshape(-1, 1))[:, 1],
        dtype="float64",
    )
    candidate_models["sigmoid"] = ProbabilityCalibrator("sigmoid", sigmoid)
    candidate_metrics["sigmoid"] = compute_probability_score_metrics(
        actual=actual_series,
        predicted_probabilities=sigmoid_predictions,
    )

    selected_method = min(
        candidate_metrics,
        key=lambda method: (
            candidate_metrics[method]["brier_score"],
            candidate_metrics[method]["log_loss"],
            -candidate_metrics[method]["roc_auc"],
        ),
    )
    return candidate_models[selected_method], {
        "enabled": True,
        "selected_method": selected_method,
        "candidate_metrics": candidate_metrics,
        "status": "fitted",
    }


def apply_probability_calibrator(
    raw_probabilities: pd.Series | np.ndarray,
    *,
    calibrator: ProbabilityCalibrator | None,
) -> pd.Series:
    """Transform raw positive-class probabilities with an optional calibrator."""

    probability_series = pd.Series(raw_probabilities, dtype="float64")
    if calibrator is None:
        return probability_series

    raw_values = probability_series.to_numpy(dtype=float)
    if calibrator.method == "isotonic":
        calibrated_values = calibrator.estimator.predict(raw_values)
    elif calibrator.method == "sigmoid":
        calibrated_values = calibrator.estimator.predict_proba(raw_values.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unsupported calibrator method: {calibrator.method}")
    return pd.Series(
        np.clip(calibrated_values, 0.0, 1.0),
        index=probability_series.index,
        dtype="float64",
    )


def compute_calibration_bins(
    actual: pd.Series,
    probabilities: pd.Series,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute calibration data points (bins) for a probability model."""
    
    if actual.empty or probabilities.empty:
        return []
        
    df = pd.DataFrame({"actual": actual.astype(int), "prob": probabilities.astype(float)})
    df["bin"] = pd.cut(df["prob"], bins=np.linspace(0, 1, n_bins + 1), labels=False, include_lowest=True)
    
    bin_results = []
    for i in range(n_bins):
        bin_df = df[df["bin"] == i]
        if bin_df.empty:
            continue
            
        bin_results.append({
            "bin_index": i,
            "bin_range": (float(i / n_bins), float((i + 1) / n_bins)),
            "row_count": int(len(bin_df)),
            "avg_probability": float(bin_df["prob"].mean()),
            "actual_frequency": float(bin_df["actual"].mean()),
        })
        
    return bin_results


def compute_prediction_metrics(
    *,
    split_name: str,
    prediction_frame: pd.DataFrame,
    target_column: str,
    logger: Any,
    date_column: str,
    predicted_column: str = "predicted_next_day_direction",
    probability_column: str | None = "predicted_probability_up",
    raw_probability_column: str | None = None,
) -> dict[str, Any]:
    """Compute Kubera classification metrics with optional probability scores."""

    if prediction_frame.empty:
        return {
            "row_count": 0,
            "date_start": None,
            "date_end": None,
            "target_positive_count": 0,
            "target_negative_count": 0,
            "predicted_positive_count": 0,
            "predicted_negative_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "has_probability_scores": False,
            "roc_auc": None,
            "log_loss": None,
            "brier_score": None,
        }

    actual = prediction_frame[target_column].astype(int)
    predicted = prediction_frame[predicted_column].astype(int)

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
        "has_probability_scores": False,
    }

    if probability_column is None or probability_column not in prediction_frame.columns:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None
        metrics["brier_score"] = None
        metrics["calibration_bins"] = []
        metrics["raw_roc_auc"] = None
        metrics["raw_log_loss"] = None
        metrics["raw_brier_score"] = None
        metrics["raw_calibration_bins"] = []
        return metrics

    predicted_probabilities = pd.to_numeric(
        prediction_frame[probability_column],
        errors="coerce",
    )
    if predicted_probabilities.isna().any():
        logger.warning(
            "Probability metrics are unavailable because predicted probabilities are missing | split=%s | rows=%s",
            split_name,
            len(prediction_frame),
        )
        metrics["roc_auc"] = None
        metrics["log_loss"] = None
        metrics["brier_score"] = None
        metrics["calibration_bins"] = []
        metrics["raw_roc_auc"] = None
        metrics["raw_log_loss"] = None
        metrics["raw_brier_score"] = None
        metrics["raw_calibration_bins"] = []
        return metrics

    metrics["has_probability_scores"] = True
    if actual.nunique() < 2:
        logger.warning(
            "Probability metrics are undefined on a single-class evaluation split | split=%s | rows=%s",
            split_name,
            len(prediction_frame),
        )
        metrics["roc_auc"] = None
        metrics["log_loss"] = None
        metrics["brier_score"] = None
        metrics["calibration_bins"] = []
        metrics["raw_roc_auc"] = None
        metrics["raw_log_loss"] = None
        metrics["raw_brier_score"] = None
        metrics["raw_calibration_bins"] = []
        return metrics

    calibrated_metrics = compute_probability_score_metrics(
        actual=actual,
        predicted_probabilities=predicted_probabilities,
    )
    metrics["roc_auc"] = calibrated_metrics["roc_auc"]
    metrics["log_loss"] = calibrated_metrics["log_loss"]
    metrics["brier_score"] = calibrated_metrics["brier_score"]
    metrics["calibration_bins"] = calibrated_metrics["calibration_bins"]

    metrics["raw_roc_auc"] = None
    metrics["raw_log_loss"] = None
    metrics["raw_brier_score"] = None
    metrics["raw_calibration_bins"] = []
    if raw_probability_column is not None and raw_probability_column in prediction_frame.columns:
        raw_probabilities = pd.to_numeric(
            prediction_frame[raw_probability_column],
            errors="coerce",
        )
        if not raw_probabilities.isna().any():
            raw_metrics = compute_probability_score_metrics(
                actual=actual,
                predicted_probabilities=raw_probabilities,
            )
            metrics["raw_roc_auc"] = raw_metrics["roc_auc"]
            metrics["raw_log_loss"] = raw_metrics["log_loss"]
            metrics["raw_brier_score"] = raw_metrics["brier_score"]
            metrics["raw_calibration_bins"] = raw_metrics["calibration_bins"]
    return metrics


def compute_probability_score_metrics(
    *,
    actual: pd.Series,
    predicted_probabilities: pd.Series,
) -> dict[str, Any]:
    """Compute probability-only metrics for one aligned actual and score series."""

    actual_series = actual.astype(int)
    probability_series = pd.to_numeric(predicted_probabilities, errors="coerce").astype(float)
    return {
        "roc_auc": float(roc_auc_score(actual_series, probability_series)),
        "log_loss": float(log_loss(actual_series, probability_series, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual_series, probability_series)),
        "calibration_bins": compute_calibration_bins(actual_series, probability_series),
    }


def compute_split_metrics(
    *,
    split_name: str,
    prediction_frame: pd.DataFrame,
    target_column: str,
    logger: Any,
    date_column: str,
    raw_probability_column: str | None = None,
) -> dict[str, Any]:
    """Compute the standard Kubera evaluation metrics for one split."""

    return compute_prediction_metrics(
        split_name=split_name,
        prediction_frame=prediction_frame,
        target_column=target_column,
        logger=logger,
        date_column=date_column,
        raw_probability_column=raw_probability_column,
    )


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


def compute_news_context_weight(
    news_article_count: int | float,
    news_avg_confidence: float,
    has_fresh_news: bool | int,
    is_fallback_heavy: bool | int,
    is_carried_forward: bool | int,
) -> float:
    """Compute a news-signal strength weight [0, 1] for probability blending."""

    if int(news_article_count) == 0 and not int(is_carried_forward):
        return 0.0

    if int(has_fresh_news):
        base_weight = 0.8
    elif int(is_carried_forward):
        base_weight = 0.3
    else:
        base_weight = 0.0

    weight = base_weight * float(news_avg_confidence)

    if int(is_fallback_heavy):
        weight *= 0.6

    count_boost = min(float(news_article_count) / 3.0, 1.0)
    weight = weight * (0.4 + 0.6 * count_boost)

    return min(max(weight, 0.0), 1.0)


def resolve_selective_prediction(
    *,
    probability_up: float,
    classification_threshold: float,
    low_conviction_threshold: float,
    news_signal_state: str | None = None,
    data_quality_score: float | None = None,
    data_quality_floor: float | None = None,
    carried_forward_margin_penalty: float = 0.0,
    degraded_margin_penalty: float = 0.0,
) -> SelectivePredictionDecision:
    """Turn one calibrated probability into an up, down, or abstain action."""

    bounded_probability = float(np.clip(probability_up, 0.0, 1.0))
    required_margin = max(0.0, float(low_conviction_threshold))
    reasons: list[str] = []

    normalized_state = (news_signal_state or "").strip()
    if normalized_state == "carried_forward_only":
        required_margin += max(0.0, float(carried_forward_margin_penalty))
        reasons.append("carried_forward_only")
    elif normalized_state in {"zero_news", "fallback_heavy"}:
        required_margin += max(0.0, float(degraded_margin_penalty))
        reasons.append(normalized_state)

    if data_quality_floor is not None and data_quality_score is not None:
        if float(data_quality_score) < float(data_quality_floor):
            reasons.append("data_quality_below_floor")

    probability_margin = abs(bounded_probability - classification_threshold)
    if probability_margin < required_margin:
        reasons.append("low_conviction_margin")

    if reasons:
        return SelectivePredictionDecision(
            action="abstain",
            abstain=True,
            predicted_label=None,
            probability_up=bounded_probability,
            probability_margin=float(probability_margin),
            required_margin=float(required_margin),
            reasons=tuple(reasons),
        )

    predicted_label = int(bounded_probability >= classification_threshold)
    return SelectivePredictionDecision(
        action="up" if predicted_label == 1 else "down",
        abstain=False,
        predicted_label=predicted_label,
        probability_up=bounded_probability,
        probability_margin=float(probability_margin),
        required_margin=float(required_margin),
        reasons=(),
    )


def blend_probabilities(
    baseline_prob: float | pd.Series,
    enhanced_prob: float | pd.Series,
    news_weight: float | pd.Series,
) -> float | pd.Series:
    """Blend baseline and enhanced probabilities based on news weight."""
    return (1.0 - news_weight) * baseline_prob + news_weight * enhanced_prob


def tune_model_hyperparameters(
    *,
    train_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    model_type: str,
    random_seed: int,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Grid-search key hyperparameters using time-series cross-validation.

    Returns a dict of best params (same keys as BaselineModelSettings /
    EnhancedModelSettings).  If the search fails for any reason the function
    returns an empty dict so callers can fall back to their configured defaults.
    """

    X = train_frame.loc[:, feature_columns]
    y = train_frame[target_column]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    try:
        if model_type == "logistic_regression":
            param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed)),
            ])
        elif model_type == "gradient_boosting":
            param_grid = {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
            }
            pipeline = Pipeline([
                ("classifier", GradientBoostingClassifier(random_state=random_seed)),
            ])
        elif model_type == "random_forest":
            param_grid = {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 5, 10],
            }
            pipeline = Pipeline([
                ("classifier", RandomForestClassifier(random_state=random_seed, n_jobs=-1)),
            ])
        else:
            return {}

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=tscv,
            scoring="neg_log_loss",
            n_jobs=-1,
            refit=False,
        )
        search.fit(X, y)

        # Strip the "classifier__" prefix from param names
        best = {
            key.replace("classifier__", ""): value
            for key, value in search.best_params_.items()
        }
        return best
    except Exception:
        return {}


def optimize_blend_alpha(
    *,
    baseline_probs: pd.Series,
    enhanced_probs: pd.Series,
    actual: pd.Series,
    n_steps: int = 21,
) -> tuple[float, dict[str, Any]]:
    """Find alpha in [0, 1] that minimises log-loss when blending two prob series.

    Returns (best_alpha, search_summary).
    alpha=0 means pure baseline; alpha=1 means pure enhanced.
    """

    actual_arr = actual.astype(int).to_numpy()
    base_arr = baseline_probs.to_numpy(dtype=float)
    enh_arr = enhanced_probs.to_numpy(dtype=float)

    if len(actual_arr) < 4 or len(np.unique(actual_arr)) < 2:
        return 0.5, {"status": "insufficient_data", "best_alpha": 0.5}

    alphas = np.linspace(0.0, 1.0, n_steps)
    losses = []
    for alpha in alphas:
        blended = np.clip((1.0 - alpha) * base_arr + alpha * enh_arr, 1e-7, 1 - 1e-7)
        try:
            loss = log_loss(actual_arr, blended, labels=[0, 1])
        except Exception:
            loss = float("inf")
        losses.append(loss)

    best_idx = int(np.argmin(losses))
    best_alpha = float(alphas[best_idx])
    return best_alpha, {
        "status": "fitted",
        "best_alpha": best_alpha,
        "best_log_loss": float(losses[best_idx]),
        "n_steps": n_steps,
    }


def explain_prediction_shap(
    pipeline: Pipeline,
    feature_row: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    """Compute SHAP values for a single prediction row (local explanation)."""

    if not SHAP_AVAILABLE:
        return _explain_prediction_fallback(pipeline, feature_row, feature_columns)

    try:
        data_processed = feature_row.copy()
        for name, step in pipeline.steps:
            if name == "classifier":
                break
            data_processed = step.transform(data_processed)

        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "calibrated_classifiers_"):
            actual_model = classifier.calibrated_classifiers_[0].estimator
        else:
            actual_model = classifier

        if hasattr(actual_model, "coef_"):
            explainer = shap.LinearExplainer(actual_model, data_processed)
            shap_values = explainer.shap_values(data_processed)
        elif hasattr(actual_model, "feature_importances_"):
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(data_processed)
        else:
            return _explain_prediction_fallback(pipeline, feature_row, feature_columns)

        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
            
        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        return dict(zip(feature_columns, [float(v) for v in vals]))
    except Exception:
        return _explain_prediction_fallback(pipeline, feature_row, feature_columns)


def _explain_prediction_fallback(
    pipeline: Pipeline,
    feature_row: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    """Provide a simple linear contribution proxy when SHAP is unavailable."""
    try:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "calibrated_classifiers_"):
            actual_model = classifier.calibrated_classifiers_[0].estimator
        else:
            actual_model = classifier

        if hasattr(actual_model, "coef_"):
            data_processed = feature_row.copy()
            for name, step in pipeline.steps:
                if name == "classifier":
                    break
                data_processed = step.transform(data_processed)

            coefs = actual_model.coef_[0]
            # Convert to float to avoid numpy types in JSON
            if isinstance(data_processed, pd.DataFrame):
                row_data = data_processed.iloc[0].values
            else:
                row_data = data_processed[0]
            contributions = coefs * row_data
            return dict(zip(feature_columns, [float(c) for c in contributions]))

        if hasattr(actual_model, "feature_importances_"):
            importances = actual_model.feature_importances_
            return dict(zip(feature_columns, [float(i) for i in importances]))

        return {}
    except Exception:
        return {}
