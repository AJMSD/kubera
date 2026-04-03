from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.news_features import NEWS_FEATURE_COLUMNS
from kubera.models.train_baseline import (
    BaselineModelError,
    load_baseline_dataset,
    split_baseline_dataset,
    train_baseline_model,
)
from kubera.models.train_enhanced import (
    EnhancedModelError,
    PersistedEnhancedModel,
    build_enhanced_feature_spec,
    build_live_enhanced_feature_row,
    build_merged_enhanced_dataset,
    infer_news_feature_metadata_path,
    load_news_feature_dataset,
    load_saved_enhanced_model,
    main,
    train_enhanced_models,
)
from kubera.utils.paths import PathManager
from kubera.utils.serialization import write_json_file


BASE_HISTORICAL_FEATURE_COLUMNS = (
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

HISTORICAL_FEATURE_COLUMNS = list(BASE_HISTORICAL_FEATURE_COLUMNS)
for lag in (1, 2):
    for col in BASE_HISTORICAL_FEATURE_COLUMNS:
        HISTORICAL_FEATURE_COLUMNS.append(f"{col}_lag{lag}")
HISTORICAL_FEATURE_COLUMNS = tuple(HISTORICAL_FEATURE_COLUMNS)


def test_persisted_enhanced_model_uses_canonical_module_name() -> None:
    assert PersistedEnhancedModel.__module__ == "kubera.models.train_enhanced"


def make_historical_feature_frame(row_count: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=row_count + 1)
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        target = 1 if index % 3 != 0 else 0
        direction = 1.0 if target == 1 else -1.0
        base_close = 100.0 + index
        row_dict = {
            "date": dates[index].strftime("%Y-%m-%d"),
            "target_date": dates[index + 1].strftime("%Y-%m-%d"),
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
            "day_of_week": dates[index].weekday(),
            "target_next_day_direction": target,
        }
        for lag in (1, 2):
            for feat in BASE_HISTORICAL_FEATURE_COLUMNS:
                row_dict[f"{feat}_lag{lag}"] = float(row_dict[feat]) * (1.0 - (0.1 * lag))
        rows.append(row_dict)
    return pd.DataFrame(rows)


def make_zero_news_feature_row(
    *,
    prediction_date: str,
    prediction_mode: str,
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> dict[str, object]:
    row: dict[str, object] = {
        "date": prediction_date,
        "ticker": ticker,
        "exchange": exchange,
        "prediction_mode": prediction_mode,
    }
    for column in NEWS_FEATURE_COLUMNS:
        row[column] = 0.0
    return row


def make_news_feature_frame(
    historical_frame: pd.DataFrame,
    *,
    missing_rows: set[tuple[str, str]] | None = None,
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> pd.DataFrame:
    missing = missing_rows or set()
    rows: list[dict[str, object]] = []
    for source_row in historical_frame.to_dict(orient="records"):
        prediction_date = str(source_row["target_date"])
        target = int(source_row["target_next_day_direction"])
        for prediction_mode in ("pre_market", "after_close"):
            if (prediction_date, prediction_mode) in missing:
                continue
            mode_multiplier = 1.0 if prediction_mode == "pre_market" else 1.2
            row = make_zero_news_feature_row(
                prediction_date=prediction_date,
                prediction_mode=prediction_mode,
                ticker=ticker,
                exchange=exchange,
            )
            row["news_article_count"] = 1.0
            row["news_avg_sentiment"] = 0.8 * mode_multiplier if target == 1 else -0.8 * mode_multiplier
            row["news_max_severity"] = 0.7
            row["news_avg_relevance"] = 0.9
            row["news_avg_confidence"] = 0.85
            row["news_bullish_article_count"] = 1.0 if target == 1 else 0.0
            row["news_bearish_article_count"] = 0.0 if target == 1 else 1.0
            row["news_neutral_article_count"] = 0.0
            row["news_full_article_count"] = 1.0
            row["news_headline_plus_snippet_count"] = 0.0
            row["news_headline_only_count"] = 0.0
            row["news_warning_article_count"] = 0.0
            row["news_fallback_article_ratio"] = 0.0
            row["news_avg_content_quality_score"] = 1.0
            row["news_weighted_sentiment_score"] = row["news_avg_sentiment"]
            row["news_weighted_relevance_score"] = 0.9
            row["news_weighted_confidence_score"] = 0.85
            row["news_weighted_bullish_score"] = 1.0 if target == 1 else 0.0
            row["news_weighted_bearish_score"] = 0.0 if target == 1 else 1.0
            if "news_event_count_earnings" in row:
                row["news_event_count_earnings"] = 1.0 if target == 1 else 0.0
            if "news_event_count_lawsuit" in row:
                row["news_event_count_lawsuit"] = 0.0 if target == 1 else 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def write_stage_eight_artifacts(
    *,
    historical_frame: pd.DataFrame | None = None,
    news_frame: pd.DataFrame | None = None,
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> tuple[Path, Path]:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    historical_feature_frame = (
        historical_frame.copy() if historical_frame is not None else make_historical_feature_frame()
    )
    news_feature_frame = (
        news_frame.copy()
        if news_frame is not None
        else make_news_feature_frame(historical_feature_frame, ticker=ticker, exchange=exchange)
    )

    historical_path = path_manager.build_historical_feature_table_path(ticker, exchange)
    historical_metadata_path = path_manager.build_historical_feature_metadata_path(ticker, exchange)
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    historical_feature_frame.to_csv(historical_path, index=False)
    write_json_file(
        historical_metadata_path,
        {
            "ticker": ticker,
            "exchange": exchange,
            "feature_columns": list(HISTORICAL_FEATURE_COLUMNS),
            "target_column": "target_next_day_direction",
            "formula_version": "4",
            "run_id": "historical_feature_fixture",
        },
    )

    news_path = path_manager.build_news_feature_table_path(ticker, exchange)
    news_metadata_path = path_manager.build_news_feature_metadata_path(ticker, exchange)
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news_feature_frame.to_csv(news_path, index=False)
    write_json_file(
        news_metadata_path,
        {
            "ticker": ticker,
            "exchange": exchange,
            "feature_columns": list(NEWS_FEATURE_COLUMNS),
            "formula_version": "3",
            "supported_prediction_modes": ["pre_market", "after_close"],
            "coverage_start": str(news_feature_frame["date"].min()),
            "coverage_end": str(news_feature_frame["date"].max()),
        },
    )

    return historical_path, news_path


def test_build_merged_enhanced_dataset_zero_fills_missing_news_rows(isolated_repo) -> None:
    historical_frame = make_historical_feature_frame(row_count=8)
    missing_key = (str(historical_frame.iloc[2]["target_date"]), "after_close")
    write_stage_eight_artifacts(
        historical_frame=historical_frame,
        news_frame=make_news_feature_frame(historical_frame, missing_rows={missing_key}),
    )
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    historical_dataset = load_baseline_dataset(
        feature_table_path=path_manager.build_historical_feature_table_path("INFY", "NSE"),
        feature_metadata_path=path_manager.build_historical_feature_metadata_path("INFY", "NSE"),
        ticker="INFY",
        exchange="NSE",
    )
    news_dataset = load_news_feature_dataset(
        news_feature_table_path=path_manager.build_news_feature_table_path("INFY", "NSE"),
        news_feature_metadata_path=infer_news_feature_metadata_path(
            path_manager.build_news_feature_table_path("INFY", "NSE")
        ),
        ticker="INFY",
        exchange="NSE",
        supported_prediction_modes=("pre_market", "after_close"),
    )

    merged_dataset = build_merged_enhanced_dataset(
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        lag_windows=settings.historical_features.lag_windows,
    )

    assert len(merged_dataset.dataset_frame) == len(historical_frame) * 2
    missing_row = merged_dataset.dataset_frame.loc[
        (merged_dataset.dataset_frame["prediction_date"] == missing_key[0])
        & (merged_dataset.dataset_frame["prediction_mode"] == missing_key[1])
    ].iloc[0]
    assert missing_row["prediction_date"] == historical_frame.iloc[2]["target_date"]
    assert (merged_dataset.dataset_frame["historical_date"] < merged_dataset.dataset_frame["prediction_date"]).all()
    assert missing_row["news_article_count"] == 0.0
    assert missing_row["news_weighted_sentiment_score"] == 0.0
    assert merged_dataset.missing_news_row_count == 1


def test_build_live_enhanced_feature_row_matches_stage8_feature_values(isolated_repo) -> None:
    historical_frame = make_historical_feature_frame(row_count=10)
    news_frame = make_news_feature_frame(historical_frame)
    write_stage_eight_artifacts(
        historical_frame=historical_frame,
        news_frame=news_frame,
    )
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    historical_dataset = load_baseline_dataset(
        feature_table_path=path_manager.build_historical_feature_table_path("INFY", "NSE"),
        feature_metadata_path=path_manager.build_historical_feature_metadata_path("INFY", "NSE"),
        ticker="INFY",
        exchange="NSE",
    )
    news_dataset = load_news_feature_dataset(
        news_feature_table_path=path_manager.build_news_feature_table_path("INFY", "NSE"),
        news_feature_metadata_path=infer_news_feature_metadata_path(
            path_manager.build_news_feature_table_path("INFY", "NSE")
        ),
        ticker="INFY",
        exchange="NSE",
        supported_prediction_modes=("pre_market", "after_close"),
    )
    merged_dataset = build_merged_enhanced_dataset(
        historical_dataset=historical_dataset,
        news_dataset=news_dataset,
        lag_windows=settings.historical_features.lag_windows,
    )
    feature_spec = build_enhanced_feature_spec(
        historical_feature_columns=historical_dataset.feature_columns,
        base_news_feature_columns=news_dataset.feature_columns,
        lag_windows=settings.historical_features.lag_windows,
    )

    expected_row = merged_dataset.dataset_frame.loc[
        (merged_dataset.dataset_frame["prediction_mode"] == "after_close")
        & (merged_dataset.dataset_frame["prediction_date"] == str(historical_frame.iloc[4]["target_date"]))
    ].iloc[0]
    historical_row = historical_frame.loc[
        historical_frame["target_date"] == str(expected_row["prediction_date"])
    ].iloc[0]
    live_news_row = news_frame.loc[
        (news_frame["date"] == str(expected_row["prediction_date"]))
        & (news_frame["prediction_mode"] == "after_close")
    ].iloc[0]
    lag_history = news_frame.loc[
        (news_frame["prediction_mode"] == "after_close")
        & (news_frame["date"] < str(expected_row["prediction_date"]))
    ].copy()
    lag_history["date"] = pd.to_datetime(lag_history["date"])
    lag_history = lag_history.sort_values("date").reset_index(drop=True)

    live_row = build_live_enhanced_feature_row(
        historical_row_mapping=historical_row.to_dict(),
        news_feature_row_mapping=live_news_row.to_dict(),
        feature_spec=feature_spec,
        news_history_frame=lag_history,
    )

    for column in feature_spec.extended_news_feature_columns:
        assert live_row[column] == pytest.approx(float(expected_row[column]))


def test_train_enhanced_models_rejects_stale_historical_formula_version(isolated_repo) -> None:
    historical_path, _ = write_stage_eight_artifacts()
    historical_metadata_path = Path(str(historical_path).replace(".csv", ".metadata.json"))
    metadata = json.loads(historical_metadata_path.read_text(encoding="utf-8"))
    metadata["formula_version"] = "3"
    write_json_file(historical_metadata_path, metadata)

    with pytest.raises(BaselineModelError, match="Historical feature artifact is stale"):
        train_enhanced_models(load_settings())


def test_load_news_feature_dataset_rejects_stale_formula_version(isolated_repo) -> None:
    _, news_path = write_stage_eight_artifacts()
    news_metadata_path = infer_news_feature_metadata_path(news_path)
    metadata = json.loads(news_metadata_path.read_text(encoding="utf-8"))
    metadata["formula_version"] = "0"
    write_json_file(news_metadata_path, metadata)

    with pytest.raises(EnhancedModelError, match="News feature artifact is stale") as exc_info:
        load_news_feature_dataset(
            news_feature_table_path=news_path,
            news_feature_metadata_path=news_metadata_path,
            ticker="INFY",
            exchange="NSE",
            supported_prediction_modes=("pre_market", "after_close"),
        )

    assert "Expected formula_version 3, found 0" in str(exc_info.value)
    assert str(news_metadata_path) in str(exc_info.value)


def test_train_enhanced_models_builds_mode_artifacts_and_feature_importance(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_GBM_MIN_SAMPLES_LEAF", "1")
    monkeypatch.setenv("KUBERA_ENHANCED_GBM_MIN_SAMPLES_LEAF", "1")
    write_stage_eight_artifacts()
    settings = load_settings()

    result = train_enhanced_models(settings)

    assert result.baseline_artifact_status == "refreshed"
    assert set(result.mode_results) == {"pre_market", "after_close"}
    for prediction_mode, mode_result in result.mode_results.items():
        assert mode_result.model_path.exists()
        assert mode_result.predictions_path.exists()
        assert mode_result.metrics_path.exists()
        assert mode_result.comparison_path.exists()
        assert mode_result.comparison_summary_path.exists()

        saved_model = load_saved_enhanced_model(mode_result.model_path)
        metrics_payload = json.loads(mode_result.metrics_path.read_text(encoding="utf-8"))

        assert saved_model.prediction_mode == prediction_mode
        assert saved_model.model_type == "gradient_boosting"
        assert metrics_payload["feature_importance"]["news_features_contributed"] is True
        assert metrics_payload["feature_importance"]["importance_metric"] == "feature_importances"
        assert metrics_payload["feature_importance"]["top_news_features"]
        assert (
            metrics_payload["feature_importance"]["group_summaries"]["historical_features"][
                "importance_sum"
            ]
            >= 0.0
        )


def test_enhanced_and_baseline_share_the_same_evaluation_rows(isolated_repo) -> None:
    historical_path, _ = write_stage_eight_artifacts()
    settings = load_settings()

    result = train_enhanced_models(settings)

    historical_dataset = load_baseline_dataset(
        feature_table_path=historical_path,
        feature_metadata_path=Path(str(historical_path).replace(".csv", ".metadata.json")),
        ticker="INFY",
        exchange="NSE",
    )
    split = split_baseline_dataset(historical_dataset.dataset_frame, settings.baseline_model)
    expected_prediction_dates = (
        split.validation_frame["target_date"].tolist()
        + split.test_frame["target_date"].tolist()
    )

    for mode_result in result.mode_results.values():
        comparison_frame = pd.read_csv(mode_result.comparison_path)
        assert comparison_frame["prediction_date"].tolist() == expected_prediction_dates
        assert comparison_frame["target_next_day_direction"].tolist() == (
            split.validation_frame["target_next_day_direction"].tolist()
            + split.test_frame["target_next_day_direction"].tolist()
        )


def test_train_enhanced_models_refreshes_stale_baseline_artifacts(isolated_repo) -> None:
    historical_path, _ = write_stage_eight_artifacts()
    settings = load_settings()
    baseline_result = train_baseline_model(settings, feature_table_path=historical_path)
    baseline_metadata = json.loads(baseline_result.metadata_path.read_text(encoding="utf-8"))
    baseline_metadata["source_feature_table_hash"] = "stale-hash"
    write_json_file(baseline_result.metadata_path, baseline_metadata)

    result = train_enhanced_models(settings)

    assert result.baseline_artifact_status == "refreshed"


def test_train_enhanced_models_reuses_cached_merged_dataset(
    isolated_repo,
    monkeypatch,
) -> None:
    write_stage_eight_artifacts()
    settings = load_settings()
    first_result = train_enhanced_models(settings)

    def fail_if_recomputed(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Expected the cached merged dataset to be reused.")

    monkeypatch.setattr(
        "kubera.models.train_enhanced.build_merged_enhanced_dataset",
        fail_if_recomputed,
    )

    second_result = train_enhanced_models(settings)

    assert first_result.merged_dataset_path == second_result.merged_dataset_path
    cached_model = load_saved_enhanced_model(
        second_result.mode_results["pre_market"].model_path
    )
    assert any(column.endswith("_lag1") for column in cached_model.feature_columns)
    assert any(column.startswith("cross_") for column in cached_model.feature_columns)


def test_enhanced_command_smoke_builds_expected_artifacts(isolated_repo) -> None:
    write_stage_eight_artifacts()

    exit_code = main(["--ticker", "INFY", "--exchange", "NSE"])

    assert exit_code == 0
    assert (
        isolated_repo
        / "artifacts"
        / "models"
        / "enhanced"
        / "INFY_NSE_pre_market_enhanced_model.pkl"
    ).exists()
    assert (
        isolated_repo
        / "artifacts"
        / "reports"
        / "enhanced"
        / "INFY_NSE_after_close_baseline_comparison.csv"
    ).exists()


def test_train_enhanced_models_supports_runtime_ticker_override(isolated_repo) -> None:
    historical_frame = make_historical_feature_frame()
    historical_frame["ticker"] = "TCS"
    historical_frame["exchange"] = "NSE"
    news_frame = make_news_feature_frame(
        historical_frame,
        ticker="TCS",
        exchange="NSE",
    )
    write_stage_eight_artifacts(
        historical_frame=historical_frame,
        news_frame=news_frame,
        ticker="TCS",
        exchange="NSE",
    )
    settings = load_settings()

    result = train_enhanced_models(settings, ticker="TCS", exchange="NSE")

    assert result.merged_dataset_path.name == "TCS_NSE_enhanced_dataset.csv"
    assert set(result.mode_results) == {"pre_market", "after_close"}
    for mode_result in result.mode_results.values():
        metadata = json.loads(mode_result.metadata_path.read_text(encoding="utf-8"))
        assert metadata["ticker"] == "TCS"


def test_train_enhanced_models_support_gradient_boosting(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_MODEL_TYPE", "gradient_boosting")
    monkeypatch.setenv("KUBERA_ENHANCED_MODEL_TYPE", "gradient_boosting")
    monkeypatch.setenv("KUBERA_BASELINE_GBM_MIN_SAMPLES_LEAF", "1")
    monkeypatch.setenv("KUBERA_ENHANCED_GBM_MIN_SAMPLES_LEAF", "1")
    write_stage_eight_artifacts()
    settings = load_settings()

    result = train_enhanced_models(settings)

    assert result.baseline_artifact_status == "refreshed"
    for mode_result in result.mode_results.values():
        saved_model = load_saved_enhanced_model(mode_result.model_path)
        metadata = json.loads(mode_result.metadata_path.read_text(encoding="utf-8"))
        metrics_payload = json.loads(mode_result.metrics_path.read_text(encoding="utf-8"))

        assert saved_model.model_type == "gradient_boosting"
        assert metadata["model_type"] == "gradient_boosting"
        assert metadata["model_params"] == {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "min_samples_leaf": 1,
            "random_seed": settings.run.random_seed,
            "enable_calibration": False,
        }
        assert metrics_payload["feature_importance"]["importance_metric"] == "feature_importances"


def test_train_enhanced_models_support_random_forest(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_MODEL_TYPE", "random_forest")
    monkeypatch.setenv("KUBERA_ENHANCED_MODEL_TYPE", "random_forest")
    monkeypatch.setenv("KUBERA_BASELINE_RF_MIN_SAMPLES_LEAF", "1")
    monkeypatch.setenv("KUBERA_ENHANCED_RF_MIN_SAMPLES_LEAF", "1")
    write_stage_eight_artifacts()
    settings = load_settings()

    result = train_enhanced_models(settings)

    assert result.baseline_artifact_status == "refreshed"
    for mode_result in result.mode_results.values():
        saved_model = load_saved_enhanced_model(mode_result.model_path)
        metadata = json.loads(mode_result.metadata_path.read_text(encoding="utf-8"))
        metrics_payload = json.loads(mode_result.metrics_path.read_text(encoding="utf-8"))

        assert saved_model.model_type == "random_forest"
        assert tuple(saved_model.pipeline.named_steps) == ("classifier",)
        assert metadata["model_type"] == "random_forest"
        assert metadata["model_params"] == {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "random_seed": settings.run.random_seed,
            "enable_calibration": False,
        }
        assert metrics_payload["feature_importance"]["importance_metric"] == "feature_importances"
