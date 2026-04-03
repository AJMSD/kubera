from __future__ import annotations

from datetime import date
import json

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.news_features import NEWS_FEATURE_COLUMNS
from kubera.pilot.live_pilot import (
    ACTUAL_STATUS_BACKFILLED,
    ACTUAL_STATUS_PENDING,
    PILOT_PREDICTION_MODES,
    PILOT_STATUS_PARTIAL_FAILURE,
    PILOT_STATUS_SUCCESS,
)
from kubera.reporting.final_review import (
    FinalReviewError,
    generate_final_review,
    main as final_review_main,
    resolve_offline_evaluation_artifacts,
)
from kubera.reporting.offline_evaluation import evaluate_offline
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
_lag_cols = [f"{col}_lag{lag}" for lag in (1, 2) for col in BASE_HISTORICAL_FEATURE_COLUMNS]
HISTORICAL_FEATURE_COLUMNS = BASE_HISTORICAL_FEATURE_COLUMNS + tuple(_lag_cols)
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
                row_dict[f"{feat}_lag{lag}"] = float(row_dict[feat]) * (1.0 - 0.1 * lag)
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
        "request_mode": "plain_text",
        "recovery_reason": None,
        "recovery_status": "not_needed",
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

    processed_news_metadata_path = path_manager.build_processed_news_metadata_path(ticker, exchange)
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


def write_model_metadata(
    path_manager: PathManager,
    *,
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> None:
    settings = load_settings()
    baseline_metadata_path = path_manager.build_baseline_model_metadata_path(ticker, exchange)
    write_json_file(
        baseline_metadata_path,
        {
            "run_id": "baseline_model_fixture",
            "ticker": ticker,
            "exchange": exchange,
        },
    )
    for prediction_mode in PILOT_PREDICTION_MODES:
        enhanced_metadata_path = path_manager.build_enhanced_model_metadata_path(
            ticker,
            exchange,
            prediction_mode,
        )
        write_json_file(
            enhanced_metadata_path,
            {
                "run_id": f"enhanced_model_fixture_{prediction_mode}",
                "ticker": ticker,
                "exchange": exchange,
                "prediction_mode": prediction_mode,
            },
        )


def make_pilot_log_row(
    *,
    prediction_mode: str,
    market_session_date: str,
    prediction_date: str,
    pilot_entry_id: str,
    pilot_timestamp_utc: str,
    prediction_attempt_number: int = 1,
    ticker: str = "INFY",
    exchange: str = "NSE",
    status: str = PILOT_STATUS_SUCCESS,
    baseline_direction: int = 1,
    baseline_probability_up: float = 0.5,
    enhanced_direction: int = 1,
    enhanced_probability_up: float = 0.5,
    actual_direction: int | None = None,
    actual_status: str = ACTUAL_STATUS_PENDING,
    disagreement_flag: bool | None = None,
    fallback_heavy_flag: bool = False,
    news_article_count: int = 1,
    baseline_correct: bool | None = None,
    enhanced_correct: bool | None = None,
    failure_stage: str | None = None,
    news_quality_note: str | None = None,
    market_shock_note: str | None = None,
    source_outage_note: str | None = None,
    warning_codes: list[str] | None = None,
    linked_article_ids: list[str] | None = None,
    top_event_counts: dict[str, int] | None = None,
    total_duration_seconds: float | None = None,
    stage5_provider_request_retry_count: int = 0,
    stage5_article_fetch_retry_count: int = 0,
    stage6_retry_count: int = 0,
    runtime_warning_flag: bool = False,
    runtime_warning_message: str | None = None,
) -> dict[str, object]:
    return {
        "pilot_entry_id": pilot_entry_id,
        "prediction_key": f"{ticker}_{exchange}_{prediction_mode}_{prediction_date}",
        "prediction_attempt_number": prediction_attempt_number,
        "ticker": ticker,
        "exchange": exchange,
        "prediction_mode": prediction_mode,
        "pilot_run_id": f"pilot_run_{pilot_entry_id}",
        "pilot_timestamp_utc": pilot_timestamp_utc,
        "pilot_timestamp_market": pilot_timestamp_utc,
        "market_session_date": market_session_date,
        "historical_cutoff_date": market_session_date,
        "news_cutoff_timestamp_utc": pilot_timestamp_utc,
        "historical_date": market_session_date,
        "prediction_date": prediction_date,
        "baseline_predicted_next_day_direction": baseline_direction,
        "baseline_predicted_probability_up": baseline_probability_up,
        "enhanced_predicted_next_day_direction": enhanced_direction,
        "enhanced_predicted_probability_up": enhanced_probability_up,
        "disagreement_flag": disagreement_flag,
        "news_article_count": news_article_count,
        "news_warning_article_count": 1 if news_article_count else 0,
        "news_fallback_article_ratio": 0.75 if fallback_heavy_flag else 0.25,
        "news_avg_confidence": 0.4 if fallback_heavy_flag else 0.8,
        "fallback_heavy_flag": fallback_heavy_flag,
        "news_feature_synthetic_flag": news_article_count == 0,
        "linked_article_ids_json": json.dumps(linked_article_ids or [], sort_keys=True),
        "top_event_counts_json": json.dumps(top_event_counts or {}, sort_keys=True),
        "warning_codes_json": json.dumps(warning_codes or [], sort_keys=True),
        "status": status,
        "failure_stage": failure_stage,
        "failure_message": None,
        "total_duration_seconds": total_duration_seconds,
        "runtime_warning_flag": runtime_warning_flag,
        "runtime_warning_message": runtime_warning_message,
        "pilot_snapshot_path": f"artifacts/pilot/{pilot_entry_id}.json",
        "stage2_cleaned_path": "artifacts/market.csv",
        "stage2_metadata_path": "artifacts/market.json",
        "stage2_run_id": "stage2_fixture",
        "stage2_duration_seconds": 0.25,
        "stage5_processed_news_path": "artifacts/news.csv",
        "stage5_metadata_path": "artifacts/news.json",
        "stage5_run_id": "stage5_fixture",
        "stage5_duration_seconds": 1.2,
        "stage5_provider_request_count": 2,
        "stage5_provider_request_retry_count": stage5_provider_request_retry_count,
        "stage5_article_fetch_attempt_count": 3,
        "stage5_article_fetch_retry_count": stage5_article_fetch_retry_count,
        "stage6_extraction_path": "artifacts/extractions.csv",
        "stage6_metadata_path": "artifacts/extractions.json",
        "stage6_failure_log_path": "artifacts/extractions_failures.csv",
        "stage6_run_id": "stage6_fixture",
        "stage6_duration_seconds": 0.8,
        "stage6_provider_request_count": 1,
        "stage6_retry_count": stage6_retry_count,
        "stage7_feature_path": "artifacts/news_features.csv",
        "stage7_metadata_path": "artifacts/news_features.json",
        "stage7_raw_snapshot_path": "artifacts/news_snapshot.json",
        "stage7_run_id": "stage7_fixture",
        "stage7_duration_seconds": 0.3,
        "baseline_model_path": "artifacts/baseline_model.joblib",
        "baseline_model_metadata_path": "artifacts/baseline_model.json",
        "baseline_model_run_id": "baseline_model_fixture",
        "baseline_duration_seconds": 0.05,
        "enhanced_model_path": "artifacts/enhanced_model.joblib",
        "enhanced_model_metadata_path": "artifacts/enhanced_model.json",
        "enhanced_model_run_id": f"enhanced_model_fixture_{prediction_mode}",
        "enhanced_duration_seconds": 0.06,
        "actual_historical_close": 100.0,
        "actual_prediction_close": 101.0 if actual_direction == 1 else 99.0 if actual_direction == 0 else None,
        "actual_next_day_direction": actual_direction,
        "actual_outcome_status": actual_status,
        "actual_outcome_backfilled_at_utc": (
            "2026-01-07T00:00:00+00:00" if actual_status == ACTUAL_STATUS_BACKFILLED else None
        ),
        "actual_backfill_error": None,
        "baseline_correct": baseline_correct,
        "enhanced_correct": enhanced_correct,
        "news_quality_note": news_quality_note,
        "market_shock_note": market_shock_note,
        "source_outage_note": source_outage_note,
        "manual_notes_updated_at_utc": "2026-01-07T01:00:00+00:00"
        if any(note is not None for note in (news_quality_note, market_shock_note, source_outage_note))
        else None,
    }


def write_pilot_log(
    path_manager: PathManager,
    prediction_mode: str,
    rows: list[dict[str, object]],
    *,
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> None:
    log_path = path_manager.build_pilot_log_path(ticker, exchange, prediction_mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(log_path, index=False)


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


def test_resolve_offline_evaluation_artifacts_requires_refresh_when_missing(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()

    with pytest.raises(FinalReviewError, match="--refresh-offline-evaluation"):
        resolve_offline_evaluation_artifacts(settings)


def test_resolve_offline_evaluation_artifacts_refreshes_when_requested(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()

    artifacts = resolve_offline_evaluation_artifacts(
        settings,
        refresh_offline_evaluation=True,
    )

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
        resolve_offline_evaluation_artifacts(
            settings,
            refresh_offline_evaluation=True,
        )


def test_resolve_offline_evaluation_artifacts_rejects_stale_saved_summary(
    isolated_repo,
) -> None:
    write_stage_nine_inputs()
    settings = load_settings()
    artifacts = evaluate_offline(settings)
    summary_payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    summary_payload["source_historical_formula_version"] = "2"
    write_json_file(artifacts.summary_json_path, summary_payload)

    with pytest.raises(FinalReviewError, match="stale relative to the current Stage 3"):
        resolve_offline_evaluation_artifacts(settings)


def test_resolve_offline_evaluation_artifacts_refreshes_stale_saved_summary_when_requested(
    isolated_repo,
) -> None:
    write_stage_nine_inputs()
    settings = load_settings()
    artifacts = evaluate_offline(settings)
    summary_payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    summary_payload["source_historical_formula_version"] = "2"
    write_json_file(artifacts.summary_json_path, summary_payload)

    refreshed_artifacts = resolve_offline_evaluation_artifacts(
        settings,
        refresh_offline_evaluation=True,
    )

    assert refreshed_artifacts.refreshed is True
    refreshed_payload = json.loads(
        refreshed_artifacts.summary_json_path.read_text(encoding="utf-8")
    )
    assert refreshed_payload["source_historical_formula_version"] == "4"


def test_generate_final_review_with_saved_outputs_and_full_pilot_logs(isolated_repo) -> None:
    path_manager, _, _ = write_stage_nine_inputs()
    write_model_metadata(path_manager)
    settings = load_settings()
    evaluate_offline(settings)

    write_pilot_log(
        path_manager,
        "pre_market",
        [
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-05",
                prediction_date="2026-01-05",
                pilot_entry_id="pre_older",
                pilot_timestamp_utc="2026-01-05T11:00:00+00:00",
                prediction_attempt_number=1,
                disagreement_flag=False,
                baseline_correct=False,
                enhanced_correct=False,
            ),
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-05",
                prediction_date="2026-01-05",
                pilot_entry_id="pre_latest",
                pilot_timestamp_utc="2026-01-05T12:00:00+00:00",
                prediction_attempt_number=2,
                disagreement_flag=True,
                fallback_heavy_flag=True,
                news_article_count=3,
                actual_direction=1,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=True,
                enhanced_correct=True,
                news_quality_note="Sparse but usable coverage",
                warning_codes=["fallback_heavy"],
                linked_article_ids=["pre_a", "pre_b"],
                top_event_counts={"earnings": 2},
            ),
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-06",
                prediction_date="2026-01-06",
                pilot_entry_id="pre_day_two",
                pilot_timestamp_utc="2026-01-06T12:00:00+00:00",
                disagreement_flag=False,
                fallback_heavy_flag=False,
                news_article_count=1,
                actual_direction=1,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=False,
                enhanced_correct=True,
            ),
        ],
    )
    write_pilot_log(
        path_manager,
        "after_close",
        [
            make_pilot_log_row(
                prediction_mode="after_close",
                market_session_date="2026-01-05",
                prediction_date="2026-01-06",
                pilot_entry_id="after_day_one",
                pilot_timestamp_utc="2026-01-05T13:00:00+00:00",
                status=PILOT_STATUS_PARTIAL_FAILURE,
                disagreement_flag=True,
                news_article_count=0,
                baseline_direction=0,
                enhanced_direction=1,
                actual_status=ACTUAL_STATUS_PENDING,
                failure_stage="stage6",
                source_outage_note="Provider delay",
                warning_codes=["zero_news_available"],
            ),
            make_pilot_log_row(
                prediction_mode="after_close",
                market_session_date="2026-01-06",
                prediction_date="2026-01-07",
                pilot_entry_id="after_day_two",
                pilot_timestamp_utc="2026-01-06T13:00:00+00:00",
                disagreement_flag=False,
                news_article_count=2,
                baseline_direction=0,
                enhanced_direction=0,
                actual_direction=0,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=True,
                enhanced_correct=True,
            ),
        ],
    )

    result = generate_final_review(
        settings,
        pilot_start_date=date(2026, 1, 5),
        pilot_end_date=date(2026, 1, 6),
    )

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    markdown = result.summary_markdown_path.read_text(encoding="utf-8")

    assert result.offline_artifacts_refreshed is False
    assert result.pilot_coverage_status == "complete"
    assert summary_payload["pilot_summary"]["overall"]["disagreement_count"] == 2
    assert summary_payload["pilot_summary"]["overall"]["fallback_heavy_count"] == 1
    assert summary_payload["pilot_summary"]["overall"]["degraded_news_row_count"] == 2
    assert summary_payload["pilot_summary"]["overall"]["zero_news_count"] == 1
    assert summary_payload["pilot_summary"]["overall"]["partial_failure_count"] == 1
    assert summary_payload["pilot_summary"]["per_mode"]["pre_market"]["rerun_row_count"] == 1
    assert summary_payload["pilot_summary"]["per_mode"]["pre_market"]["baseline_accuracy"] == pytest.approx(0.5)
    assert summary_payload["pilot_summary"]["per_mode"]["pre_market"]["enhanced_accuracy"] == pytest.approx(1.0)
    baseline_metadata = json.loads(
        path_manager.build_baseline_model_metadata_path("INFY", "NSE").read_text(encoding="utf-8")
    )
    pre_market_metadata = json.loads(
        path_manager.build_enhanced_model_metadata_path("INFY", "NSE", "pre_market").read_text(
            encoding="utf-8"
        )
    )
    assert (
        summary_payload["traceability"]["models"]["baseline"]["run_id"]
        == baseline_metadata["run_id"]
    )
    assert (
        summary_payload["traceability"]["models"]["enhanced"]["pre_market"]["run_id"]
        == pre_market_metadata["run_id"]
    )
    assert any(
        "partial failures" in issue
        for issue in summary_payload["pilot_summary"]["operational_issues"]
    )
    assert any(
        "degraded-news conditions" in issue
        for issue in summary_payload["pilot_summary"]["operational_issues"]
    )
    assert "degraded_news" in summary_payload["pilot_summary"]["daily_prediction_rows"][0]["notes"]
    assert "rerun_attempt_2" in summary_payload["pilot_summary"]["daily_prediction_rows"][0]["notes"]
    assert "| 2026-01-05 | pre_market | 2026-01-05 | success |" in markdown
    enhanced_accuracy = summary_payload["offline_evaluation"]["per_mode"]["pre_market"]["variants"][
        "enhanced_full"
    ]["subsets"]["all_rows"]["accuracy"]
    assert f"accuracy {enhanced_accuracy:.3f}" in markdown


def test_generate_final_review_requires_refresh_when_offline_outputs_are_missing(
    isolated_repo,
) -> None:
    write_stage_nine_inputs()
    settings = load_settings()

    with pytest.raises(FinalReviewError, match="--refresh-offline-evaluation"):
        generate_final_review(
            settings,
            pilot_start_date=date(2026, 1, 5),
            pilot_end_date=date(2026, 1, 6),
        )


def test_generate_final_review_refreshes_offline_outputs_when_requested(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()

    result = generate_final_review(
        settings,
        pilot_start_date=date(2026, 1, 5),
        pilot_end_date=date(2026, 1, 6),
        refresh_offline_evaluation=True,
    )

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert result.offline_artifacts_refreshed is True
    assert summary_payload["pilot_summary"]["coverage_status"] == "unavailable"


def test_generate_final_review_reports_missing_pilot_logs_honestly(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()
    evaluate_offline(settings)

    result = generate_final_review(
        settings,
        pilot_start_date=date(2026, 1, 5),
        pilot_end_date=date(2026, 1, 6),
    )

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    markdown = result.summary_markdown_path.read_text(encoding="utf-8")

    assert summary_payload["pilot_summary"]["coverage_status"] == "unavailable"
    assert summary_payload["claim_checks"]["operational_reliability_supported"] is False
    assert summary_payload["claim_checks"]["pilot_claims_are_artifact_backed"] is False
    assert len(summary_payload["pilot_summary"]["missing_expected_pairs"]) == 4
    assert "No pilot log rows were found for the requested market-session window." in markdown
    assert "not trading advice" in markdown


def test_generate_final_review_reports_partial_pilot_coverage(isolated_repo) -> None:
    path_manager, _, _ = write_stage_nine_inputs()
    settings = load_settings()
    evaluate_offline(settings)

    write_pilot_log(
        path_manager,
        "pre_market",
        [
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-05",
                prediction_date="2026-01-05",
                pilot_entry_id="pre_only_one",
                pilot_timestamp_utc="2026-01-05T12:00:00+00:00",
                actual_direction=1,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=True,
                enhanced_correct=True,
            ),
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-06",
                prediction_date="2026-01-06",
                pilot_entry_id="pre_only_two",
                pilot_timestamp_utc="2026-01-06T12:00:00+00:00",
                actual_direction=0,
                baseline_direction=0,
                enhanced_direction=0,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=True,
                enhanced_correct=True,
            ),
        ],
    )

    result = generate_final_review(
        settings,
        pilot_start_date=date(2026, 1, 5),
        pilot_end_date=date(2026, 1, 6),
    )

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert result.pilot_coverage_status == "partial"
    assert summary_payload["pilot_summary"]["available_pair_count"] == 2
    assert len(summary_payload["pilot_summary"]["missing_expected_pairs"]) == 2
    assert summary_payload["pilot_summary"]["per_mode"]["after_close"]["log_exists"] is False
    assert summary_payload["pilot_summary"]["per_mode"]["after_close"]["missing_market_session_dates"] == [
        "2026-01-05",
        "2026-01-06",
    ]
    assert summary_payload["claim_checks"]["operational_reliability_supported"] is False


def test_final_review_cli_writes_outputs(isolated_repo) -> None:
    write_stage_nine_inputs()
    settings = load_settings()
    evaluate_offline(settings)
    path_manager = PathManager(settings.paths)

    assert (
        final_review_main(
            [
                "--pilot-start-date",
                "2026-01-05",
                "--pilot-end-date",
                "2026-01-06",
            ]
        )
        == 0
    )
    assert path_manager.build_final_review_json_path("INFY", "NSE").exists()
    assert path_manager.build_final_review_markdown_path("INFY", "NSE").exists()


def test_generate_final_review_supports_runtime_ticker_override_and_observability(
    isolated_repo,
) -> None:
    path_manager, _, _ = write_stage_nine_inputs(
        ticker="TCS",
        exchange="NSE",
        company_name="Tata Consultancy Services",
    )
    write_model_metadata(path_manager, ticker="TCS", exchange="NSE")
    settings = load_settings()
    evaluate_offline(settings, ticker="TCS", exchange="NSE")

    write_pilot_log(
        path_manager,
        "pre_market",
        [
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-05",
                prediction_date="2026-01-05",
                pilot_entry_id="tcs_pre",
                pilot_timestamp_utc="2026-01-05T12:00:00+00:00",
                ticker="TCS",
                exchange="NSE",
                actual_direction=1,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=True,
                enhanced_correct=True,
                total_duration_seconds=4.2,
                stage5_provider_request_retry_count=1,
                stage5_article_fetch_retry_count=2,
                stage6_retry_count=1,
            )
        ],
        ticker="TCS",
        exchange="NSE",
    )
    write_pilot_log(
        path_manager,
        "after_close",
        [
            make_pilot_log_row(
                prediction_mode="after_close",
                market_session_date="2026-01-05",
                prediction_date="2026-01-06",
                pilot_entry_id="tcs_after",
                pilot_timestamp_utc="2026-01-05T13:00:00+00:00",
                ticker="TCS",
                exchange="NSE",
                actual_direction=0,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_direction=0,
                enhanced_direction=0,
                baseline_correct=True,
                enhanced_correct=True,
                total_duration_seconds=3.8,
            )
        ],
        ticker="TCS",
        exchange="NSE",
    )

    result = generate_final_review(
        settings,
        pilot_start_date=date(2026, 1, 5),
        pilot_end_date=date(2026, 1, 5),
        ticker="TCS",
        exchange="NSE",
    )

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    markdown = result.summary_markdown_path.read_text(encoding="utf-8")

    assert summary_payload["ticker"] == "TCS"
    assert summary_payload["pilot_summary"]["overall"]["stage5_article_fetch_retry_count_sum"] == 2
    assert any(
        "Stage 5 recorded 1 provider-request retries and 2 article-fetch retries."
        in issue
        for issue in summary_payload["pilot_summary"]["operational_issues"]
    )
    assert "Average total runtime:" in markdown
    assert "Stage 6 retries: 1" in markdown


def test_generate_final_review_reports_runtime_warnings_and_week_status_summary(
    isolated_repo,
) -> None:
    path_manager, _, _ = write_stage_nine_inputs()
    settings = load_settings()
    evaluate_offline(settings)

    write_pilot_log(
        path_manager,
        "pre_market",
        [
            make_pilot_log_row(
                prediction_mode="pre_market",
                market_session_date="2026-01-05",
                prediction_date="2026-01-05",
                pilot_entry_id="pre_runtime_warning",
                pilot_timestamp_utc="2026-01-05T12:00:00+00:00",
                actual_direction=1,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_correct=True,
                enhanced_correct=True,
                total_duration_seconds=9.5,
                runtime_warning_flag=True,
                runtime_warning_message="Pilot runtime exceeded the configured threshold.",
            )
        ],
    )
    write_pilot_log(
        path_manager,
        "after_close",
        [
            make_pilot_log_row(
                prediction_mode="after_close",
                market_session_date="2026-01-05",
                prediction_date="2026-01-06",
                pilot_entry_id="after_clean",
                pilot_timestamp_utc="2026-01-05T13:00:00+00:00",
                actual_direction=0,
                actual_status=ACTUAL_STATUS_BACKFILLED,
                baseline_direction=0,
                enhanced_direction=0,
                baseline_correct=True,
                enhanced_correct=True,
                total_duration_seconds=3.2,
            )
        ],
    )
    week_status_summary_path = path_manager.build_pilot_week_status_summary_path(
        "INFY",
        "NSE",
        date(2026, 1, 5),
        date(2026, 1, 5),
    )
    write_json_file(
        week_status_summary_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
            "slot_count": 2,
            "completed_slot_count": 0,
            "partial_failure_count": 1,
            "failure_count": 1,
            "pending_slot_count": 1,
            "slot_statuses": [],
        },
    )

    result = generate_final_review(
        settings,
        pilot_start_date=date(2026, 1, 5),
        pilot_end_date=date(2026, 1, 5),
    )

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    daily_rows = summary_payload["pilot_summary"]["daily_prediction_rows"]

    assert summary_payload["pilot_summary"]["overall"]["runtime_warning_count"] == 1
    assert summary_payload["pilot_summary"]["week_status_summary"]["pending_slot_count"] == 1
    assert any(
        "1 pilot rows exceeded the configured runtime warning threshold."
        in issue
        for issue in summary_payload["pilot_summary"]["operational_issues"]
    )
    assert any(
        "Pilot week plan still has 1 pending slots." in issue
        for issue in summary_payload["pilot_summary"]["operational_issues"]
    )
    assert any(
        "Pilot week plan recorded 1 partial-failure slots and 1 failed slots." in issue
        for issue in summary_payload["pilot_summary"]["operational_issues"]
    )
    assert daily_rows[0]["notes"] == ["runtime_warning"]


def test_final_review_cli_accepts_ticker_override(isolated_repo) -> None:
    write_stage_nine_inputs(
        ticker="TCS",
        exchange="NSE",
        company_name="Tata Consultancy Services",
    )
    settings = load_settings()
    evaluate_offline(settings, ticker="TCS", exchange="NSE")
    path_manager = PathManager(settings.paths)

    assert (
        final_review_main(
            [
                "--ticker",
                "TCS",
                "--exchange",
                "NSE",
                "--pilot-start-date",
                "2026-01-05",
                "--pilot-end-date",
                "2026-01-06",
            ]
        )
        == 0
    )
    assert path_manager.build_final_review_json_path("TCS", "NSE").exists()
    assert path_manager.build_final_review_markdown_path("TCS", "NSE").exists()
