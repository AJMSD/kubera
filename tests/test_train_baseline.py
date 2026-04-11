from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.historical_features import build_historical_features
from kubera.models.train_baseline import (
    BaselineModelError,
    PersistedBaselineModel,
    fit_baseline_model,
    load_baseline_dataset,
    load_saved_baseline_model,
    main,
    predict_with_saved_model,
    split_baseline_dataset,
    train_baseline_model,
)
from kubera.utils.paths import PathManager
from kubera.utils.serialization import write_json_file


BASE_FEATURE_COLUMNS = (
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_5d",
    "volatility_10d",
    "volume_change_1d",
    "volume_ma_ratio",
    "macd",
    "macd_signal",
    "price_vs_52w_high",
    "price_vs_52w_low",
    "rsi_14",
    "day_of_week",
)

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)
for lag in (1, 2):
    for col in BASE_FEATURE_COLUMNS:
        FEATURE_COLUMNS.append(f"{col}_lag{lag}")
FEATURE_COLUMNS = tuple(FEATURE_COLUMNS)


def test_persisted_baseline_model_uses_canonical_module_name() -> None:
    assert PersistedBaselineModel.__module__ == "kubera.models.train_baseline"


def make_mock_feature_table(row_count: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=row_count)
    rows: list[dict[str, object]] = []
    for index, current_date in enumerate(dates):
        target = 1 if index % 4 in {1, 2} else 0
        direction = 1.0 if target == 1 else -1.0
        base_close = 100.0 + index
        row_dict = {
            "date": current_date.strftime("%Y-%m-%d"),
            "target_date": dates[min(index + 1, len(dates) - 1)].strftime("%Y-%m-%d"),
            "ticker": "INFY",
            "exchange": "NSE",
            "close": base_close,
            "volume": 1000.0 + (index * 20.0),
            "ret_1d": 0.01 * direction,
            "ret_3d": 0.02 * direction,
            "ret_5d": 0.03 * direction,
            "ma_5": base_close + (2.0 * direction),
            "ma_10": base_close + (4.0 * direction),
            "ma_20": base_close + (6.0 * direction),
            "volatility_5d": 0.01 + (0.002 * (index % 3)),
            "volatility_10d": 0.02 + (0.002 * (index % 4)),
            "volume_change_1d": 0.05 * direction,
            "volume_ma_ratio": 1.1 + (0.1 * direction),
            "macd": 1.4 * direction,
            "macd_signal": 1.1 * direction,
            "price_vs_52w_high": 0.98 if target == 1 else 0.9,
            "price_vs_52w_low": 1.18 if target == 1 else 1.08,
            "rsi_14": 65.0 if target == 1 else 35.0,
            "day_of_week": current_date.weekday(),
            "target_next_day_direction": target,
        }
        for lag in (1, 2):
            for feat in BASE_FEATURE_COLUMNS:
                row_dict[f"{feat}_lag{lag}"] = float(row_dict[feat]) * (1.0 - (0.1 * lag))
        rows.append(row_dict)
    return pd.DataFrame(rows)


def write_mock_feature_artifacts(
    repo_root: Path,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    feature_table_path = path_manager.build_historical_feature_table_path("INFY", "NSE")
    feature_metadata_path = path_manager.build_historical_feature_metadata_path("INFY", "NSE")
    feature_table_path.parent.mkdir(parents=True, exist_ok=True)
    working_frame = feature_frame.copy() if feature_frame is not None else make_mock_feature_table()
    working_frame.to_csv(feature_table_path, index=False)
    write_json_file(
        feature_metadata_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
            "feature_columns": list(FEATURE_COLUMNS),
            "target_column": "target_next_day_direction",
            "formula_version": "5",
            "run_id": "feature_run",
        },
    )
    return feature_table_path, feature_metadata_path


def make_default_cleaned_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=320)
    close_values = [
        100.0 + (index * 0.35) + ((index % 5) - 2) * 0.4
        for index in range(len(dates))
    ]
    volume_values = [1000 + (index * 25) + ((index % 3) * 10) for index in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ticker": ["INFY"] * len(dates),
            "exchange": ["NSE"] * len(dates),
            "provider": ["yfinance"] * len(dates),
            "provider_symbol": ["INFY.NS"] * len(dates),
            "open": close_values,
            "high": [value + 1 for value in close_values],
            "low": [value - 1 for value in close_values],
            "close": close_values,
            "adj_close": close_values,
            "volume": volume_values,
            "fetched_at_utc": ["2026-03-10T00:00:00+00:00"] * len(dates),
            "raw_snapshot_path": ["data/raw/source.json"] * len(dates),
        }
    )


def write_stage_three_inputs(repo_root: Path) -> Path:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    cleaned_path = path_manager.build_processed_market_data_path("INFY", "NSE")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_frame = make_default_cleaned_market_data()
    cleaned_frame.to_csv(cleaned_path, index=False)
    write_json_file(
        path_manager.build_processed_market_data_metadata_path("INFY", "NSE"),
        {
            "ticker": "INFY",
            "exchange": "NSE",
            "provider": "yfinance",
            "coverage_start": str(cleaned_frame["date"].min()),
            "coverage_end": str(cleaned_frame["date"].max()),
        },
    )
    return cleaned_path


def test_load_baseline_dataset_uses_feature_metadata_order(isolated_repo) -> None:
    feature_frame = make_mock_feature_table().loc[
        :,
        [
            "date",
            "target_date",
            "ticker",
            "exchange",
            "close",
            "volume",
            *list(FEATURE_COLUMNS),
            "target_next_day_direction",
        ],
    ].sort_values("date", ascending=False)
    feature_table_path, feature_metadata_path = write_mock_feature_artifacts(
        isolated_repo,
        feature_frame=feature_frame,
    )

    dataset = load_baseline_dataset(
        feature_table_path=feature_table_path,
        feature_metadata_path=feature_metadata_path,
        ticker="INFY",
        exchange="NSE",
    )

    assert dataset.feature_columns == FEATURE_COLUMNS
    assert dataset.dataset_frame["date"].tolist()[0] == "2026-01-05"
    assert dataset.target_column == "target_next_day_direction"


def test_load_baseline_dataset_rejects_stale_formula_version(isolated_repo) -> None:
    feature_table_path, feature_metadata_path = write_mock_feature_artifacts(isolated_repo)
    metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    metadata["formula_version"] = "3"
    write_json_file(feature_metadata_path, metadata)

    with pytest.raises(BaselineModelError, match="Historical feature artifact is stale") as exc_info:
        load_baseline_dataset(
            feature_table_path=feature_table_path,
            feature_metadata_path=feature_metadata_path,
            ticker="INFY",
            exchange="NSE",
        )

    assert "Expected formula_version 5, found 3" in str(exc_info.value)
    assert str(feature_metadata_path) in str(exc_info.value)


def test_split_baseline_dataset_uses_strict_temporal_order(isolated_repo) -> None:
    feature_table_path, feature_metadata_path = write_mock_feature_artifacts(isolated_repo)
    dataset = load_baseline_dataset(
        feature_table_path=feature_table_path,
        feature_metadata_path=feature_metadata_path,
        ticker="INFY",
        exchange="NSE",
    )

    split = split_baseline_dataset(dataset.dataset_frame, load_settings().baseline_model)

    assert len(split.train_frame) == 14
    assert len(split.validation_frame) == 3
    assert len(split.test_frame) == 3
    assert split.train_frame.iloc[-1]["date"] < split.validation_frame.iloc[0]["date"]
    assert split.validation_frame.iloc[-1]["date"] < split.test_frame.iloc[0]["date"]


def test_predict_with_saved_model_rejects_feature_order_mismatch(isolated_repo) -> None:
    feature_table_path, feature_metadata_path = write_mock_feature_artifacts(isolated_repo)
    settings = load_settings()
    dataset = load_baseline_dataset(
        feature_table_path=feature_table_path,
        feature_metadata_path=feature_metadata_path,
        ticker="INFY",
        exchange="NSE",
    )
    split = split_baseline_dataset(dataset.dataset_frame, settings.baseline_model)
    saved_model = fit_baseline_model(
        train_frame=split.train_frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        baseline_settings=settings.baseline_model,
        random_seed=settings.run.random_seed,
    )
    shuffled_columns = tuple(reversed(dataset.feature_columns))

    with pytest.raises(BaselineModelError, match="Feature order mismatch"):
        predict_with_saved_model(
            saved_model,
            split.test_frame.loc[:, shuffled_columns],
        )


def test_train_baseline_model_persists_artifacts_and_reloads_predictions(
    isolated_repo,
) -> None:
    write_stage_three_inputs(isolated_repo)
    settings = load_settings()
    build_historical_features(settings)

    result = train_baseline_model(settings)

    assert result.model_path.exists()
    assert result.metadata_path.exists()
    assert result.predictions_path.exists()
    assert result.metrics_path.exists()

    saved_model = load_saved_baseline_model(result.model_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(result.predictions_path)
    dataset = load_baseline_dataset(
        feature_table_path=Path(metadata["source_feature_table_path"]),
        feature_metadata_path=Path(metadata["source_feature_metadata_path"]),
        ticker="INFY",
        exchange="NSE",
    )
    split = split_baseline_dataset(dataset.dataset_frame, settings.baseline_model)
    expected_test_labels, expected_test_probabilities = predict_with_saved_model(
        saved_model,
        split.test_frame.loc[:, saved_model.feature_columns],
    )
    test_predictions = predictions.loc[predictions["split"] == "test"].reset_index(drop=True)

    assert result.train_row_count == len(split.train_frame)
    assert result.validation_row_count == len(split.validation_frame)
    assert result.test_row_count == len(split.test_frame)
    assert metadata["split_summary"]["train"]["row_count"] == len(split.train_frame)
    assert metrics["validation"]["row_count"] == len(split.validation_frame)
    assert metrics["validation"]["has_probability_scores"] is True
    assert metrics["validation"]["roc_auc"] is not None
    assert metrics["validation"]["log_loss"] is not None
    assert metrics["validation"]["brier_score"] is not None
    assert test_predictions["predicted_next_day_direction"].tolist() == expected_test_labels.tolist()
    assert test_predictions["predicted_probability_up"].tolist() == pytest.approx(
        expected_test_probabilities.tolist()
    )


def test_train_baseline_model_supports_gradient_boosting(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_stage_three_inputs(isolated_repo)
    monkeypatch.setenv("KUBERA_BASELINE_MODEL_TYPE", "gradient_boosting")
    settings = load_settings()
    build_historical_features(settings)

    result = train_baseline_model(settings)

    saved_model = load_saved_baseline_model(result.model_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert saved_model.model_type == "gradient_boosting"
    assert tuple(saved_model.pipeline.named_steps) == ("classifier",)
    assert metadata["model_type"] == "gradient_boosting"
    assert metadata["model_params"] == {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "min_samples_leaf": 10,
        "random_seed": settings.run.random_seed,
        "enable_calibration": True,
        "enable_class_weight": True,
        "class_weight_strategy": "balanced",
    }

def test_baseline_random_forest_model_type(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_stage_three_inputs(isolated_repo)
    monkeypatch.setenv("KUBERA_BASELINE_MODEL_TYPE", "random_forest")
    settings = load_settings()
    build_historical_features(settings)

    result = train_baseline_model(settings)

    saved_model = load_saved_baseline_model(result.model_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert saved_model.model_type == "random_forest"
    assert tuple(saved_model.pipeline.named_steps) == ("classifier",)
    assert metadata["model_type"] == "random_forest"
    assert metadata["model_params"] == {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 10,
        "random_seed": settings.run.random_seed,
        "enable_calibration": True,
        "enable_class_weight": True,
        "class_weight_strategy": "balanced",
    }

def test_baseline_command_smoke_builds_expected_artifacts(isolated_repo) -> None:
    write_stage_three_inputs(isolated_repo)
    build_historical_features(load_settings())

    exit_code = main([])

    assert exit_code == 0
    assert (
        isolated_repo
        / "artifacts"
        / "models"
        / "baseline"
        / "INFY_NSE_baseline_model.pkl"
    ).exists()
    assert (
        isolated_repo
        / "artifacts"
        / "reports"
        / "baseline"
        / "INFY_NSE_baseline_metrics.json"
    ).exists()
