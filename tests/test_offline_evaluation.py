from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.news_features import NEWS_FEATURE_COLUMNS
from kubera.models.common import compute_prediction_metrics
from kubera.models.train_baseline import BaselineModelError
from kubera.models.train_enhanced import EnhancedModelError
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
    build_mode_diagnostics,
    build_mode_evidence_summary,
    evaluate_offline,
    main,
)
from kubera.utils.paths import PathManager
from kubera.utils.serialization import write_json_file


HISTORICAL_FEATURE_COLUMNS = (
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
CONTENT_QUALITY_BY_MODE = {
    "full_article": 1.0,
    "headline_plus_snippet": 0.75,
    "headline_only": 0.5,
}


def make_historical_feature_frame(row_count: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=row_count + 1)
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        target = 1 if index % 3 != 0 else 0
        direction = 1.0 if target == 1 else -1.0
        base_close = 100.0 + index
        rows.append(
            {
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
        )
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
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_row in historical_frame.to_dict(orient="records"):
        prediction_date = str(source_row["target_date"])
        target = int(source_row["target_next_day_direction"])
        for prediction_mode in ("pre_market", "after_close"):
            mode_multiplier = 1.0 if prediction_mode == "pre_market" else 1.25
            row = make_zero_news_feature_row(
                prediction_date=prediction_date,
                prediction_mode=prediction_mode,
                ticker=ticker,
                exchange=exchange,
            )
            row["news_article_count"] = 2.0
            row["news_avg_sentiment"] = 0.6 * mode_multiplier if target == 1 else -0.6 * mode_multiplier
            row["news_max_severity"] = 0.7
            row["news_avg_relevance"] = 0.85
            row["news_avg_confidence"] = 0.6 if target == 1 else 0.4
            row["news_bullish_article_count"] = 1.0 if target == 1 else 0.0
            row["news_bearish_article_count"] = 0.0 if target == 1 else 1.0
            row["news_neutral_article_count"] = 1.0
            row["news_full_article_count"] = 1.0
            row["news_headline_plus_snippet_count"] = 1.0
            row["news_headline_only_count"] = 0.0
            row["news_warning_article_count"] = 0.0
            row["news_fallback_article_ratio"] = 0.5
            row["news_avg_content_quality_score"] = 0.875
            row["news_weighted_sentiment_score"] = row["news_avg_sentiment"] * row["news_avg_confidence"]
            row["news_weighted_relevance_score"] = 0.9
            row["news_weighted_confidence_score"] = row["news_avg_confidence"]
            row["news_weighted_bullish_score"] = 0.8 if target == 1 else 0.2
            row["news_weighted_bearish_score"] = 0.2 if target == 1 else 0.8
            if "news_event_count_earnings" in row:
                row["news_event_count_earnings"] = 1.0 if target == 1 else 0.0
            if "news_event_count_lawsuit" in row:
                row["news_event_count_lawsuit"] = 0.0 if target == 1 else 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def make_extraction_row(
    *,
    article_id: str,
    published_at: str,
    ticker: str = "INFY",
    exchange: str = "NSE",
    company_name: str = "Infosys Limited",
    extraction_mode: str,
    relevance_score: float,
    confidence_score: float,
    sentiment_score: float,
    directional_bias: str,
    event_type: str,
) -> dict[str, object]:
    timestamp = pd.Timestamp(published_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Kolkata")
    published_at_utc = timestamp.tz_convert("UTC")
    published_at_ist = timestamp.tz_convert("Asia/Kolkata")
    sentiment_label = "positive" if sentiment_score > 0 else "negative"
    return {
        "article_id": article_id,
        "ticker": ticker,
        "exchange": exchange,
        "company_name": company_name,
        "article_title": f"Article {article_id}",
        "article_url": f"https://example.com/{article_id}",
        "canonical_url": f"https://example.com/{article_id}",
        "source_domain": "example.com",
        "provider": "marketaux",
        "provider_source": "Example News",
        "published_at_utc": published_at_utc.isoformat(),
        "published_at_ist": published_at_ist.isoformat(),
        "published_date_ist": published_at_ist.strftime("%Y-%m-%d"),
        "extraction_mode": extraction_mode,
        "content_quality_score": CONTENT_QUALITY_BY_MODE[extraction_mode],
        "warning_flag": extraction_mode != "full_article",
        "source_fetch_warning_flag": extraction_mode != "full_article",
        "prompt_truncated": False,
        "article_input_hash": f"hash-{article_id}",
        "llm_provider": "gemini_api",
        "llm_model": "gemma-3-27b-it",
        "prompt_version": "stage6_v1",
        "schema_version": "1",
        "relevance_score": relevance_score,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "event_type": event_type,
        "event_severity": 0.7,
        "expected_horizon": "short_term",
        "directional_bias": directional_bias,
        "confidence_score": confidence_score,
        "rationale_short": "Fixture rationale",
    }


def make_extraction_frame(
    historical_frame: pd.DataFrame,
    *,
    ticker: str = "INFY",
    exchange: str = "NSE",
    company_name: str = "Infosys Limited",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, source_row in enumerate(historical_frame.to_dict(orient="records")):
        target = int(source_row["target_next_day_direction"])
        historical_date = str(source_row["date"])
        prediction_date = str(source_row["target_date"])
        bullish = target == 1
        rows.append(
            make_extraction_row(
                article_id=f"pre_{index}",
                published_at=f"{prediction_date}T08:05:00+05:30",
                ticker=ticker,
                exchange=exchange,
                company_name=company_name,
                extraction_mode="headline_plus_snippet" if bullish else "headline_only",
                relevance_score=0.9,
                confidence_score=0.35 if bullish else 0.85,
                sentiment_score=0.7 if bullish else -0.7,
                directional_bias="bullish" if bullish else "bearish",
                event_type="earnings" if bullish else "lawsuit",
            )
        )
        rows.append(
            make_extraction_row(
                article_id=f"after_{index}",
                published_at=f"{historical_date}T11:15:00+05:30",
                ticker=ticker,
                exchange=exchange,
                company_name=company_name,
                extraction_mode="full_article" if bullish else "headline_plus_snippet",
                relevance_score=0.85,
                confidence_score=0.55 if bullish else 0.45,
                sentiment_score=0.5 if bullish else -0.5,
                directional_bias="bullish" if bullish else "bearish",
                event_type="earnings" if bullish else "lawsuit",
            )
        )
    return pd.DataFrame(rows)


def write_stage_nine_inputs(
    *,
    ticker: str = "INFY",
    exchange: str = "NSE",
    company_name: str = "Infosys Limited",
) -> tuple[PathManager, pd.DataFrame, pd.DataFrame]:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    historical_frame = make_historical_feature_frame()
    historical_frame["ticker"] = ticker
    historical_frame["exchange"] = exchange
    news_feature_frame = make_news_feature_frame(
        historical_frame,
        ticker=ticker,
        exchange=exchange,
    )
    extraction_frame = make_extraction_frame(
        historical_frame,
        ticker=ticker,
        exchange=exchange,
        company_name=company_name,
    )

    historical_path = path_manager.build_historical_feature_table_path(ticker, exchange)
    historical_metadata_path = path_manager.build_historical_feature_metadata_path(ticker, exchange)
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    historical_frame.to_csv(historical_path, index=False)
    write_json_file(
        historical_metadata_path,
        {
            "ticker": ticker,
            "exchange": exchange,
            "feature_columns": list(HISTORICAL_FEATURE_COLUMNS),
            "target_column": "target_next_day_direction",
            "formula_version": "3",
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
            "formula_version": "1",
            "supported_prediction_modes": ["pre_market", "after_close"],
            "coverage_start": str(news_feature_frame["date"].min()),
            "coverage_end": str(news_feature_frame["date"].max()),
            "output_row_count": int(len(news_feature_frame)),
            "zero_news_row_count": int((news_feature_frame["news_article_count"] == 0).sum()),
            "nonzero_news_row_count": int((news_feature_frame["news_article_count"] > 0).sum()),
            "prediction_mode_row_counts": {
                "pre_market": int((news_feature_frame["prediction_mode"] == "pre_market").sum()),
                "after_close": int((news_feature_frame["prediction_mode"] == "after_close").sum()),
            },
        },
    )

    extraction_path = path_manager.build_processed_llm_extractions_path(ticker, exchange)
    extraction_metadata_path = path_manager.build_processed_llm_extractions_metadata_path(
        ticker,
        exchange,
    )
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_frame.to_csv(extraction_path, index=False)
    extraction_mode_counts = extraction_frame["extraction_mode"].value_counts().to_dict()
    write_json_file(
        extraction_metadata_path,
        {
            "ticker": ticker,
            "exchange": exchange,
            "source_row_count": int(len(extraction_frame)),
            "success_count": int(len(extraction_frame)),
            "failure_count": 0,
            "coverage_start": str(extraction_frame["published_date_ist"].min()),
            "coverage_end": str(extraction_frame["published_date_ist"].max()),
            "extraction_mode_counts": {
                str(key): int(value) for key, value in extraction_mode_counts.items()
            },
        },
    )

    news_metadata_path = path_manager.build_processed_news_metadata_path(ticker, exchange)
    write_json_file(
        news_metadata_path,
        {
            "row_count": int(len(extraction_frame)),
            "coverage_start": str(extraction_frame["published_date_ist"].min()),
            "coverage_end": str(extraction_frame["published_date_ist"].max()),
            "cache_hit_count": 0,
            "fresh_fetch_count": int(len(extraction_frame)),
            "content_origin_counts": {"direct_publisher_text": int(len(extraction_frame))},
            "source_name_counts": {"Example News": int(len(extraction_frame))},
        },
    )

    return path_manager, historical_frame, extraction_frame


def test_compute_prediction_metrics_handles_probabilities_and_missing_probabilities(
    isolated_repo,
) -> None:
    prediction_frame = pd.DataFrame(
        {
            "prediction_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "target_next_day_direction": [0, 1, 1, 0],
            "predicted_next_day_direction": [0, 1, 0, 0],
            "predicted_probability_up": [0.1, 0.9, 0.4, 0.2],
        }
    )
    logger = type("Logger", (), {"warning": lambda *args, **kwargs: None})()

    metrics = compute_prediction_metrics(
        split_name="test",
        prediction_frame=prediction_frame,
        target_column="target_next_day_direction",
        logger=logger,
        date_column="prediction_date",
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["has_probability_scores"] is True
    assert metrics["roc_auc"] == pytest.approx(1.0)

    classification_only = compute_prediction_metrics(
        split_name="test",
        prediction_frame=prediction_frame.drop(columns=["predicted_probability_up"]),
        target_column="target_next_day_direction",
        logger=logger,
        date_column="prediction_date",
        probability_column=None,
    )
    assert classification_only["has_probability_scores"] is False
    assert classification_only["roc_auc"] is None
    assert classification_only["log_loss"] is None
    assert classification_only["brier_score"] is None


def test_build_mode_evidence_summary_treats_small_deltas_as_ties(isolated_repo) -> None:
    summary = build_mode_evidence_summary(
        prediction_mode="pre_market",
        metrics_by_subset={
            ALL_ROWS_SUBSET_NAME: {
                ENHANCED_VARIANT_NAME: {"row_count": 5, "accuracy": 0.51, "f1": 0.49, "roc_auc": 0.60, "log_loss": 0.69, "brier_score": 0.24},
                BASELINE_VARIANT_NAME: {"row_count": 5, "accuracy": 0.50, "f1": 0.48, "roc_auc": 0.59, "log_loss": 0.70, "brier_score": 0.25},
            },
            NEWS_HEAVY_SUBSET_NAME: {
                ENHANCED_VARIANT_NAME: {"row_count": 0, "accuracy": None, "f1": None, "roc_auc": None, "log_loss": None, "brier_score": None},
                BASELINE_VARIANT_NAME: {"row_count": 0, "accuracy": None, "f1": None, "roc_auc": None, "log_loss": None, "brier_score": None},
            },
            ZERO_NEWS_SUBSET_NAME: {
                ENHANCED_VARIANT_NAME: {"row_count": 0, "accuracy": None, "f1": None, "roc_auc": None, "log_loss": None, "brier_score": None},
                BASELINE_VARIANT_NAME: {"row_count": 0, "accuracy": None, "f1": None, "roc_auc": None, "log_loss": None, "brier_score": None},
            },
        },
        materiality_threshold=0.02,
    )

    assert "effectively tied" in summary["subsets"][ALL_ROWS_SUBSET_NAME]["note"]


def test_build_mode_diagnostics_flags_zero_news_contribution_on_tied_results(
    isolated_repo,
) -> None:
    diagnostics = build_mode_diagnostics(
        prediction_mode="pre_market",
        metrics_by_subset={
            ALL_ROWS_SUBSET_NAME: {
                ENHANCED_VARIANT_NAME: {
                    "row_count": 5,
                    "accuracy": 0.50,
                    "precision": 0.50,
                    "recall": 0.50,
                    "f1": 0.50,
                    "roc_auc": 0.50,
                    "log_loss": 0.69,
                    "brier_score": 0.25,
                },
                BASELINE_VARIANT_NAME: {
                    "row_count": 5,
                    "accuracy": 0.50,
                    "precision": 0.50,
                    "recall": 0.50,
                    "f1": 0.50,
                    "roc_auc": 0.50,
                    "log_loss": 0.69,
                    "brier_score": 0.25,
                },
            }
        },
        feature_importance_summary={"news_features_contributed": False},
        materiality_threshold=0.02,
    )

    assert diagnostics
    assert "no news-feature contribution" in diagnostics[0]


def test_evaluate_offline_rejects_stale_historical_formula_version(isolated_repo) -> None:
    path_manager, _, _ = write_stage_nine_inputs()
    historical_metadata_path = path_manager.build_historical_feature_metadata_path("INFY", "NSE")
    metadata = json.loads(historical_metadata_path.read_text(encoding="utf-8"))
    metadata["formula_version"] = "2"
    write_json_file(historical_metadata_path, metadata)

    with pytest.raises(BaselineModelError, match="Historical feature artifact is stale"):
        evaluate_offline(load_settings())


def test_evaluate_offline_rejects_stale_news_formula_version(isolated_repo) -> None:
    path_manager, _, _ = write_stage_nine_inputs()
    news_metadata_path = path_manager.build_news_feature_metadata_path("INFY", "NSE")
    metadata = json.loads(news_metadata_path.read_text(encoding="utf-8"))
    metadata["formula_version"] = "0"
    write_json_file(news_metadata_path, metadata)

    with pytest.raises(EnhancedModelError, match="News feature artifact is stale"):
        evaluate_offline(load_settings())


def test_evaluate_offline_builds_reports_and_aligned_predictions(isolated_repo) -> None:
    path_manager, historical_frame, _ = write_stage_nine_inputs()
    settings = load_settings()

    result = evaluate_offline(settings)

    assert result.metrics_path.exists()
    assert result.summary_json_path.exists()
    assert result.summary_markdown_path.exists()
    assert set(result.mode_results) == {"pre_market", "after_close"}

    metrics_frame = pd.read_csv(result.metrics_path)
    expected_variants = {
        BASELINE_VARIANT_NAME,
        ENHANCED_VARIANT_NAME,
        MAJORITY_VARIANT_NAME,
        PREVIOUS_DAY_VARIANT_NAME,
        SENTIMENT_ABLATION_VARIANT_NAME,
        EVENT_ABLATION_VARIANT_NAME,
        NO_CONFIDENCE_VARIANT_NAME,
        NO_FALLBACK_VARIANT_NAME,
    }
    assert set(metrics_frame["model_variant"]) == expected_variants
    assert set(metrics_frame["subset_name"]) == {
        ALL_ROWS_SUBSET_NAME,
        NEWS_HEAVY_SUBSET_NAME,
        ZERO_NEWS_SUBSET_NAME,
    }

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["headline_split"] == "test"
    assert "mode_summaries" in summary_payload
    assert "pre_market" in summary_payload["mode_summaries"]
    assert "after_close" in summary_payload["mode_summaries"]
    assert "source_historical_formula_version" in summary_payload
    assert "source_news_formula_version" in summary_payload
    assert "baseline_model_metadata_hash" in summary_payload
    assert "enhanced_model_metadata_hashes" in summary_payload

    for prediction_mode, mode_result in result.mode_results.items():
        prediction_frame = pd.read_csv(mode_result.predictions_path)
        assert prediction_frame["prediction_mode"].nunique() == 1
        assert prediction_frame["prediction_mode"].iloc[0] == prediction_mode
        assert prediction_frame["news_heavy_flag"].dtype == bool
        assert prediction_frame["zero_news_flag"].dtype == bool
        for variant_name in expected_variants - {PREVIOUS_DAY_VARIANT_NAME}:
            assert f"{variant_name}_predicted_probability_up" in prediction_frame.columns
        assert f"{PREVIOUS_DAY_VARIANT_NAME}_predicted_probability_up" not in prediction_frame.columns

        all_rows = metrics_frame.loc[
            (metrics_frame["prediction_mode"] == prediction_mode)
            & (metrics_frame["subset_name"] == ALL_ROWS_SUBSET_NAME)
        ]
        assert set(all_rows["row_count"]) == {len(prediction_frame)}

    train_end = int(len(historical_frame) * settings.enhanced_model.train_ratio)
    validation_end = int(
        len(historical_frame)
        * (settings.enhanced_model.train_ratio + settings.enhanced_model.validation_ratio)
    )
    default_test_rows = len(historical_frame) - validation_end
    assert result.mode_results["pre_market"].headline_row_count == default_test_rows

    no_confidence_metadata = json.loads(
        path_manager.build_news_feature_metadata_path(
            "INFY",
            "NSE",
            artifact_variant="no_confidence",
        ).read_text(encoding="utf-8")
    )
    no_fallback_metadata = json.loads(
        path_manager.build_news_feature_metadata_path(
            "INFY",
            "NSE",
            artifact_variant="no_fallback_penalties",
        ).read_text(encoding="utf-8")
    )
    assert no_confidence_metadata["feature_config"]["use_confidence_in_article_weight"] is False
    assert no_fallback_metadata["feature_config"]["full_article_weight"] == pytest.approx(1.0)
    assert no_fallback_metadata["feature_config"]["headline_plus_snippet_weight"] == pytest.approx(1.0)
    assert no_fallback_metadata["feature_config"]["headline_only_weight"] == pytest.approx(1.0)


def test_offline_evaluation_cli_smoke_builds_expected_outputs(isolated_repo) -> None:
    write_stage_nine_inputs()

    exit_code = main([])

    settings = load_settings()
    path_manager = PathManager(settings.paths)

    assert exit_code == 0
    assert path_manager.build_offline_metrics_path("INFY", "NSE").exists()
    assert path_manager.build_offline_evaluation_summary_json_path("INFY", "NSE").exists()
    assert path_manager.build_offline_evaluation_summary_markdown_path("INFY", "NSE").exists()


def test_evaluate_offline_supports_runtime_ticker_override(isolated_repo) -> None:
    path_manager, _, _ = write_stage_nine_inputs(
        ticker="TCS",
        exchange="NSE",
        company_name="Tata Consultancy Services",
    )
    settings = load_settings()

    result = evaluate_offline(settings, ticker="TCS", exchange="NSE")
    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert result.metrics_path.name == "TCS_NSE_offline_metrics.csv"
    assert summary_payload["ticker"] == "TCS"
    assert path_manager.build_offline_metrics_path("TCS", "NSE").exists()
