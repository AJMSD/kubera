from __future__ import annotations

import json

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.news_features import NEWS_FEATURE_COLUMNS
from kubera.reporting.final_review import (
    FinalReviewError,
    resolve_offline_evaluation_artifacts,
)
from kubera.reporting.offline_evaluation import evaluate_offline
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
    "rsi_14",
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
                "rsi_14": 65.0 if target == 1 else 35.0,
                "target_next_day_direction": target,
            }
        )
    return pd.DataFrame(rows)


def make_zero_news_feature_row(
    *,
    prediction_date: str,
    prediction_mode: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "date": prediction_date,
        "ticker": "INFY",
        "exchange": "NSE",
        "prediction_mode": prediction_mode,
    }
    for column in NEWS_FEATURE_COLUMNS:
        row[column] = 0.0
    return row


def make_news_feature_frame(historical_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_row in historical_frame.to_dict(orient="records"):
        prediction_date = str(source_row["target_date"])
        target = int(source_row["target_next_day_direction"])
        for prediction_mode in ("pre_market", "after_close"):
            mode_multiplier = 1.0 if prediction_mode == "pre_market" else 1.25
            row = make_zero_news_feature_row(
                prediction_date=prediction_date,
                prediction_mode=prediction_mode,
            )
            row["news_article_count"] = 2.0
            row["news_avg_sentiment"] = (
                0.6 * mode_multiplier if target == 1 else -0.6 * mode_multiplier
            )
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
            row["news_weighted_sentiment_score"] = (
                row["news_avg_sentiment"] * row["news_avg_confidence"]
            )
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
        "ticker": "INFY",
        "exchange": "NSE",
        "company_name": "Infosys Limited",
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


def make_extraction_frame(historical_frame: pd.DataFrame) -> pd.DataFrame:
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
                extraction_mode="full_article" if bullish else "headline_plus_snippet",
                relevance_score=0.85,
                confidence_score=0.55 if bullish else 0.45,
                sentiment_score=0.5 if bullish else -0.5,
                directional_bias="bullish" if bullish else "bearish",
                event_type="earnings" if bullish else "lawsuit",
            )
        )
    return pd.DataFrame(rows)


def write_stage_nine_inputs() -> tuple[PathManager, pd.DataFrame, pd.DataFrame]:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    historical_frame = make_historical_feature_frame()
    news_feature_frame = make_news_feature_frame(historical_frame)
    extraction_frame = make_extraction_frame(historical_frame)

    historical_path = path_manager.build_historical_feature_table_path("INFY", "NSE")
    historical_metadata_path = path_manager.build_historical_feature_metadata_path("INFY", "NSE")
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    historical_frame.to_csv(historical_path, index=False)
    write_json_file(
        historical_metadata_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
            "feature_columns": list(HISTORICAL_FEATURE_COLUMNS),
            "target_column": "target_next_day_direction",
            "formula_version": "2",
            "run_id": "historical_feature_fixture",
        },
    )

    news_path = path_manager.build_news_feature_table_path("INFY", "NSE")
    news_metadata_path = path_manager.build_news_feature_metadata_path("INFY", "NSE")
    news_path.parent.mkdir(parents=True, exist_ok=True)
    news_feature_frame.to_csv(news_path, index=False)
    write_json_file(
        news_metadata_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
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

    extraction_path = path_manager.build_processed_llm_extractions_path("INFY", "NSE")
    extraction_metadata_path = path_manager.build_processed_llm_extractions_metadata_path(
        "INFY",
        "NSE",
    )
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_frame.to_csv(extraction_path, index=False)
    extraction_mode_counts = extraction_frame["extraction_mode"].value_counts().to_dict()
    write_json_file(
        extraction_metadata_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
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

    processed_news_metadata_path = path_manager.build_processed_news_metadata_path("INFY", "NSE")
    write_json_file(
        processed_news_metadata_path,
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


def test_resolve_offline_evaluation_artifacts_reuses_saved_outputs(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()
    evaluate_offline(settings)

    artifacts = resolve_offline_evaluation_artifacts(settings)

    assert artifacts.refreshed is False
    assert artifacts.metrics_path.exists()
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_markdown_path.exists()
    assert not artifacts.metrics_frame.empty
    assert "mode_summaries" in artifacts.summary_payload


def test_resolve_offline_evaluation_artifacts_refreshes_when_missing(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()

    artifacts = resolve_offline_evaluation_artifacts(settings)

    assert artifacts.refreshed is True
    assert artifacts.metrics_path.exists()
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_markdown_path.exists()


def test_resolve_offline_evaluation_artifacts_fails_when_refresh_does_not_produce_outputs(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.reporting.final_review.evaluate_offline",
        lambda runtime_settings: runtime_settings,
    )

    with pytest.raises(FinalReviewError, match="after refresh"):
        resolve_offline_evaluation_artifacts(settings)
