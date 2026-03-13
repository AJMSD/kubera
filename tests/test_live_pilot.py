from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.historical_features import (
    build_live_historical_feature_row,
    validate_cleaned_market_data,
)
from kubera.features.news_features import NEWS_FEATURE_COLUMNS
from kubera.models.train_enhanced import train_enhanced_models
from kubera.pilot.live_pilot import (
    ACTUAL_STATUS_BACKFILLED,
    NewsFeatureResolution,
    PILOT_LOG_COLUMNS,
    annotate_pilot_entry,
    backfill_pilot_actuals,
    backfill_due_pilot_week,
    main as live_pilot_main,
    plan_pilot_week,
    predict_live_baseline,
    resolve_prediction_window,
    run_due_pilot_week,
    run_live_pilot,
)
from kubera.utils.calendar import build_market_calendar
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


def make_training_news_feature_frame(historical_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_row in historical_frame.to_dict(orient="records"):
        prediction_date = str(source_row["target_date"])
        target = int(source_row["target_next_day_direction"])
        for prediction_mode in ("pre_market", "after_close"):
            row = make_zero_news_feature_row(
                prediction_date=prediction_date,
                prediction_mode=prediction_mode,
            )
            row["news_article_count"] = 1.0
            row["news_avg_sentiment"] = 0.8 if target == 1 else -0.8
            row["news_max_severity"] = 0.7
            row["news_avg_relevance"] = 0.9
            row["news_avg_confidence"] = 0.85
            row["news_bullish_article_count"] = 1.0 if target == 1 else 0.0
            row["news_bearish_article_count"] = 0.0 if target == 1 else 1.0
            row["news_neutral_article_count"] = 0.0
            row["news_full_article_count"] = 1.0
            row["news_warning_article_count"] = 0.0
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


def write_model_training_inputs() -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    historical_frame = make_historical_feature_frame()
    news_feature_frame = make_training_news_feature_frame(historical_frame)

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
            "formula_version": "3",
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
            "run_id": "stage7_fixture",
        },
    )


def make_cleaned_market_frame(*, end_date: str, row_count: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range(end=end_date, periods=row_count)
    close_values = [100.0 + index + ((index % 4) * 0.5) for index in range(len(dates))]
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


def write_stage2_artifacts(
    *,
    settings,
    end_date: date,
    run_id: str,
) -> SimpleNamespace:
    path_manager = PathManager(settings.paths)
    cleaned_path = path_manager.build_processed_market_data_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_processed_market_data_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_cleaned_market_frame(end_date=end_date.isoformat())
    frame["ticker"] = settings.ticker.symbol
    frame["exchange"] = settings.ticker.exchange
    frame["provider_symbol"] = settings.ticker.provider_symbol_map["yahoo_finance"]
    frame.to_csv(cleaned_path, index=False)
    write_json_file(
        metadata_path,
        {
            "ticker": settings.ticker.symbol,
            "exchange": settings.ticker.exchange,
            "provider": "yfinance",
            "coverage_start": str(frame["date"].min()),
            "coverage_end": str(frame["date"].max()),
            "run_id": run_id,
        },
    )
    return SimpleNamespace(
        cleaned_table_path=cleaned_path,
        metadata_path=metadata_path,
    )


def write_stage5_artifacts(
    *,
    settings,
    run_id: str = "stage5_run",
    provider_request_count: int = 0,
    provider_request_retry_count: int = 0,
    article_fetch_attempt_count: int = 0,
    article_fetch_retry_count: int = 0,
) -> SimpleNamespace:
    path_manager = PathManager(settings.paths)
    cleaned_path = path_manager.build_processed_news_data_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_processed_news_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    raw_snapshot_path = path_manager.build_raw_news_data_path(settings.ticker.symbol, run_id)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "article_id": "news-1",
                "ticker": settings.ticker.symbol,
                "exchange": settings.ticker.exchange,
                "provider": "marketaux",
                "discovery_mode": "entity_symbols",
            }
        ]
    ).to_csv(cleaned_path, index=False)
    write_json_file(raw_snapshot_path, {"run_id": run_id})
    write_json_file(
        metadata_path,
        {
            "ticker": settings.ticker.symbol,
            "exchange": settings.ticker.exchange,
            "row_count": 1,
            "warnings": [],
            "provider_request_count": provider_request_count,
            "provider_request_retry_count": provider_request_retry_count,
            "article_fetch_attempt_count": article_fetch_attempt_count,
            "article_fetch_retry_count": article_fetch_retry_count,
            "run_id": run_id,
        },
    )
    return SimpleNamespace(
        cleaned_table_path=cleaned_path,
        metadata_path=metadata_path,
        raw_snapshot_path=raw_snapshot_path,
    )


def write_stage6_artifacts(
    *,
    settings,
    run_id: str = "stage6_run",
    provider_request_count: int = 0,
    retry_count: int = 0,
) -> SimpleNamespace:
    path_manager = PathManager(settings.paths)
    extraction_path = path_manager.build_processed_llm_extractions_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_processed_llm_extractions_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    failure_log_path = path_manager.build_processed_llm_extraction_failures_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"article_id": "news-1"}]).to_csv(extraction_path, index=False)
    write_json_file(failure_log_path, {"failure_count": 0, "failures": []})
    write_json_file(
        metadata_path,
        {
            "ticker": settings.ticker.symbol,
            "exchange": settings.ticker.exchange,
            "warnings": [],
            "provider_request_count": provider_request_count,
            "retry_count": retry_count,
            "run_id": run_id,
        },
    )
    return SimpleNamespace(
        extraction_table_path=extraction_path,
        metadata_path=metadata_path,
        failure_log_path=failure_log_path,
    )


def make_live_news_feature_row(
    *,
    prediction_date: str,
    prediction_mode: str,
    ticker: str = "INFY",
    exchange: str = "NSE",
    news_article_count: float = 2.0,
    fallback_ratio: float = 0.75,
    avg_confidence: float = 0.6,
    warning_article_count: float = 1.0,
) -> dict[str, object]:
    row = make_zero_news_feature_row(
        prediction_date=prediction_date,
        prediction_mode=prediction_mode,
        ticker=ticker,
        exchange=exchange,
    )
    row["news_article_count"] = news_article_count
    row["news_warning_article_count"] = warning_article_count
    row["news_fallback_article_ratio"] = fallback_ratio
    row["news_avg_confidence"] = avg_confidence
    row["news_avg_sentiment"] = 0.4
    row["news_avg_relevance"] = 0.8
    row["news_max_severity"] = 0.7
    row["news_avg_content_quality_score"] = 0.75
    row["news_weighted_sentiment_score"] = 0.3
    row["news_weighted_relevance_score"] = 0.75
    row["news_weighted_confidence_score"] = avg_confidence
    row["news_bullish_article_count"] = 1.0
    row["news_bearish_article_count"] = 1.0 if news_article_count > 1 else 0.0
    row["news_full_article_count"] = 0.0 if fallback_ratio == 1.0 else 1.0
    row["news_headline_plus_snippet_count"] = news_article_count
    if "news_event_count_earnings" in row:
        row["news_event_count_earnings"] = news_article_count
    if "news_event_count_lawsuit" in row:
        row["news_event_count_lawsuit"] = 1.0 if news_article_count > 1 else 0.0
    return row


def write_stage7_artifacts(
    *,
    settings,
    prediction_date: date,
    prediction_mode: str,
    include_target_row: bool,
    run_id: str = "stage7_run",
) -> SimpleNamespace:
    path_manager = PathManager(settings.paths)
    feature_path = path_manager.build_news_feature_table_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_news_feature_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    raw_snapshot_path = path_manager.build_raw_news_feature_data_path(
        settings.ticker.symbol,
        run_id,
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if include_target_row:
        feature_frame = pd.DataFrame(
            [
                make_live_news_feature_row(
                    prediction_date=prediction_date.isoformat(),
                    prediction_mode=prediction_mode,
                    ticker=settings.ticker.symbol,
                    exchange=settings.ticker.exchange,
                )
            ]
        )
        row_lineage = [
            {
                "date": prediction_date.isoformat(),
                "prediction_mode": prediction_mode,
                "article_ids": ["news-1", "news-2"],
            }
        ]
    else:
        feature_frame = pd.DataFrame(
            [
                make_live_news_feature_row(
                    prediction_date=(prediction_date - pd.offsets.BDay(1)).date().isoformat(),
                    prediction_mode=prediction_mode,
                    ticker=settings.ticker.symbol,
                    exchange=settings.ticker.exchange,
                    news_article_count=1.0,
                    fallback_ratio=0.0,
                )
            ]
        )
        row_lineage = []
    feature_frame.to_csv(feature_path, index=False)
    write_json_file(
        raw_snapshot_path,
        {
            "run_id": run_id,
            "row_lineage": row_lineage,
        },
    )
    write_json_file(
        metadata_path,
        {
            "ticker": settings.ticker.symbol,
            "exchange": settings.ticker.exchange,
            "raw_snapshot_path": str(raw_snapshot_path),
            "warnings": [],
            "run_id": run_id,
        },
    )
    return SimpleNamespace(
        feature_table_path=feature_path,
        metadata_path=metadata_path,
        raw_snapshot_path=raw_snapshot_path,
    )


def prepare_saved_models() -> None:
    write_model_training_inputs()
    train_enhanced_models(load_settings())


def install_live_pilot_happy_path_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_week",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
            run_id="stage7_week",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.resolve_live_news_feature_row",
        lambda settings, path_manager, *, prediction_mode, prediction_date: NewsFeatureResolution(
            feature_row=pd.DataFrame(
                [
                    make_live_news_feature_row(
                        prediction_date=prediction_date.isoformat(),
                        prediction_mode=prediction_mode,
                        ticker=settings.ticker.symbol,
                        exchange=settings.ticker.exchange,
                    )
                ]
            ),
            metadata_path=path_manager.build_news_feature_metadata_path(
                settings.ticker.symbol,
                settings.ticker.exchange,
            ),
            metadata={},
            raw_snapshot_path=path_manager.build_raw_news_feature_data_path(
                settings.ticker.symbol,
                "stage7_week",
            ),
            linked_article_ids=["news-1", "news-2"],
            top_event_counts={"news_event_count_earnings": 2},
            synthetic=False,
        ),
    )


def test_build_live_historical_feature_row_uses_the_unlabeled_tail(isolated_repo) -> None:
    settings = load_settings()
    cleaned_frame = validate_cleaned_market_data(
        make_cleaned_market_frame(end_date="2026-03-10"),
        ticker="INFY",
        exchange="NSE",
        feature_settings=settings.historical_features,
    )

    live_row = build_live_historical_feature_row(
        cleaned_frame,
        settings.historical_features,
        prediction_date=date(2026, 3, 11),
    )

    assert live_row.iloc[0]["date"] == "2026-03-10"
    assert live_row.iloc[0]["target_date"] == "2026-03-11"
    assert pd.isna(live_row.iloc[0]["target_next_day_direction"])


def test_resolve_prediction_window_respects_market_phase_and_calendar(isolated_repo) -> None:
    settings = load_settings()
    holiday_path = settings.market.local_holiday_override_path
    write_json_file(holiday_path, {"holidays": ["2026-03-11"]})
    calendar = build_market_calendar(settings.market)

    pre_market = resolve_prediction_window(
        settings=settings,
        prediction_mode="pre_market",
        timestamp=pd.Timestamp("2026-03-10T08:05:00+05:30").to_pydatetime(),
        calendar=calendar,
    )
    assert pre_market.historical_cutoff_date == date(2026, 3, 9)
    assert pre_market.prediction_date == date(2026, 3, 10)

    after_close = resolve_prediction_window(
        settings=settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
        calendar=calendar,
    )
    assert after_close.historical_cutoff_date == date(2026, 3, 10)
    assert after_close.prediction_date == date(2026, 3, 12)

    friday_after_close = resolve_prediction_window(
        settings=settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-13T16:15:00+05:30").to_pydatetime(),
        calendar=calendar,
    )
    assert friday_after_close.prediction_date == date(2026, 3, 16)

    with pytest.raises(Exception, match="trading-day"):
        resolve_prediction_window(
            settings=settings,
            prediction_mode="pre_market",
            timestamp=pd.Timestamp("2026-03-11T08:05:00+05:30").to_pydatetime(),
            calendar=calendar,
        )


def test_run_live_pilot_appends_rows_and_records_metadata(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()
    stage2_calls: list[date] = []
    stage5_calls: list[datetime] = []

    def fake_stage2(runtime_settings, *, end_date=None, **kwargs):
        del kwargs
        assert end_date is not None
        stage2_calls.append(end_date)
        return write_stage2_artifacts(settings=runtime_settings, end_date=end_date, run_id="stage2_run")

    def fake_stage5(runtime_settings, *, published_before=None, **kwargs):
        del kwargs
        assert published_before is not None
        stage5_calls.append(published_before)
        return write_stage5_artifacts(settings=runtime_settings)

    def fake_stage6(runtime_settings, **kwargs):
        del kwargs
        return write_stage6_artifacts(settings=runtime_settings)

    def fake_stage7(runtime_settings, **kwargs):
        del kwargs
        return write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
        )

    monkeypatch.setattr("kubera.pilot.live_pilot.fetch_historical_market_data", fake_stage2)
    monkeypatch.setattr("kubera.pilot.live_pilot.fetch_company_news", fake_stage5)
    monkeypatch.setattr("kubera.pilot.live_pilot.extract_news", fake_stage6)
    monkeypatch.setattr("kubera.pilot.live_pilot.build_news_features", fake_stage7)

    timestamp = pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime()
    first_result = run_live_pilot(settings, prediction_mode="after_close", timestamp=timestamp)
    second_result = run_live_pilot(settings, prediction_mode="after_close", timestamp=timestamp)

    log_frame = pd.read_csv(first_result.log_path)
    assert len(log_frame) == 2
    assert log_frame["pilot_entry_id"].nunique() == 2
    assert log_frame["prediction_key"].nunique() == 1
    assert log_frame["prediction_attempt_number"].tolist() == [1, 2]
    assert log_frame["status"].tolist() == ["success", "success"]
    assert stage2_calls == [date(2026, 3, 10), date(2026, 3, 10)]
    assert stage5_calls[0].isoformat() == "2026-03-10T10:45:00+00:00"
    assert log_frame["stage2_run_id"].tolist() == ["stage2_run", "stage2_run"]
    assert log_frame["stage5_run_id"].tolist() == ["stage5_run", "stage5_run"]
    assert log_frame["stage6_run_id"].tolist() == ["stage6_run", "stage6_run"]
    assert log_frame["stage7_run_id"].tolist() == ["stage7_run", "stage7_run"]
    assert log_frame["news_feature_synthetic_flag"].tolist() == [False, False]
    assert log_frame["linked_article_ids_json"].tolist() == [
        json.dumps(["news-1", "news-2"], separators=(",", ":"), sort_keys=True),
        json.dumps(["news-1", "news-2"], separators=(",", ":"), sort_keys=True),
    ]
    assert all("fallback_heavy" in value for value in log_frame["warning_codes_json"].tolist())
    assert first_result.snapshot_path.exists()
    assert second_result.snapshot_path.exists()


def test_run_live_pilot_uses_configured_news_lookback(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_LOOKBACK_DAYS", "33")
    prepare_saved_models()
    settings = load_settings()
    install_live_pilot_happy_path_mocks(monkeypatch)
    observed_stage5_call: dict[str, object] = {}

    def fake_stage5(runtime_settings, *, published_before=None, **kwargs):
        observed_stage5_call["lookback_days"] = runtime_settings.news_ingestion.lookback_days
        observed_stage5_call["published_before"] = published_before
        observed_stage5_call["kwargs"] = kwargs
        return write_stage5_artifacts(settings=runtime_settings)

    monkeypatch.setattr("kubera.pilot.live_pilot.fetch_company_news", fake_stage5)

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )

    assert result.snapshot_path.exists()
    assert observed_stage5_call["lookback_days"] == 33
    assert observed_stage5_call["published_before"] == pd.Timestamp(
        "2026-03-10T10:45:00+00:00"
    ).to_pydatetime()
    assert observed_stage5_call["kwargs"] == {}


def test_run_live_pilot_synthesizes_zero_news_rows(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_zero_news",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 10),
            prediction_mode="pre_market",
            include_target_row=False,
        ),
    )

    result = run_live_pilot(
        settings,
        prediction_mode="pre_market",
        timestamp=pd.Timestamp("2026-03-10T08:05:00+05:30").to_pydatetime(),
    )

    log_frame = pd.read_csv(result.log_path)
    assert log_frame.iloc[0]["status"] == "success"
    assert log_frame.iloc[0]["news_article_count"] == 0
    assert bool(log_frame.iloc[0]["news_feature_synthetic_flag"]) is True
    assert "zero_news_row_synthesized" in log_frame.iloc[0]["warning_codes_json"]
    assert "zero_news_available" in log_frame.iloc[0]["warning_codes_json"]


def test_run_live_pilot_records_disagreement_flags(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_disagreement",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
        ),
    )

    def fake_predict_live_enhanced(
        runtime_settings,
        path_manager,
        *,
        prediction_mode,
        historical_row,
        news_feature_row,
    ):
        del news_feature_row
        baseline_prediction = predict_live_baseline(
            runtime_settings,
            path_manager,
            historical_row,
        )
        return {
            "enhanced_predicted_next_day_direction": 1
            - int(baseline_prediction["baseline_predicted_next_day_direction"]),
            "enhanced_predicted_probability_up": 0.2,
            "enhanced_model_path": str(
                path_manager.build_enhanced_model_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                    prediction_mode,
                )
            ),
            "enhanced_model_metadata_path": str(
                path_manager.build_enhanced_model_metadata_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                    prediction_mode,
                )
            ),
            "enhanced_model_run_id": "enhanced_disagreement_fixture",
        }

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.predict_live_enhanced",
        fake_predict_live_enhanced,
    )

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )

    log_frame = pd.read_csv(result.log_path)
    assert log_frame.iloc[0]["status"] == "success"
    assert bool(log_frame.iloc[0]["disagreement_flag"]) is True


def test_run_live_pilot_slices_reused_market_history_to_live_cutoff(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=date(2026, 3, 10),
            run_id="stage2_reuse_future_rows",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 10),
            prediction_mode="pre_market",
            include_target_row=True,
            run_id="stage7_reuse_future_rows",
        ),
    )

    result = run_live_pilot(
        settings,
        prediction_mode="pre_market",
        timestamp=pd.Timestamp("2026-03-10T08:05:00+05:30").to_pydatetime(),
    )

    log_frame = pd.read_csv(result.log_path)
    assert log_frame.iloc[0]["status"] == "success"
    assert log_frame.iloc[0]["historical_date"] == "2026-03-09"


def test_run_live_pilot_logs_partial_failures(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_partial",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )

    def failing_stage6(runtime_settings, **kwargs):
        del runtime_settings, kwargs
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("kubera.pilot.live_pilot.extract_news", failing_stage6)

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )

    log_frame = pd.read_csv(result.log_path)
    assert log_frame.iloc[0]["status"] == "partial_failure"
    assert log_frame.iloc[0]["failure_stage"] == "stage6"
    assert pd.isna(log_frame.iloc[0]["enhanced_predicted_next_day_direction"])
    assert not pd.isna(log_frame.iloc[0]["baseline_predicted_next_day_direction"])


def test_backfill_pilot_actuals_updates_matching_rows(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()

    def fake_stage2_for_run(runtime_settings, *, end_date=None, **kwargs):
        del kwargs
        return write_stage2_artifacts(settings=runtime_settings, end_date=end_date, run_id="stage2_run")

    monkeypatch.setattr("kubera.pilot.live_pilot.fetch_historical_market_data", fake_stage2_for_run)
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
        ),
    )

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )
    baseline_before = pd.read_csv(result.log_path).iloc[0]["baseline_predicted_next_day_direction"]

    def fake_stage2_for_backfill(runtime_settings, *, end_date=None, **kwargs):
        del kwargs
        path_manager = PathManager(runtime_settings.paths)
        cleaned_path = path_manager.build_processed_market_data_path("INFY", "NSE")
        metadata_path = path_manager.build_processed_market_data_metadata_path("INFY", "NSE")
        frame = make_cleaned_market_frame(end_date=end_date.isoformat())
        frame.loc[frame["date"] == "2026-03-10", "close"] = 120.0
        frame.loc[frame["date"] == "2026-03-11", "close"] = 125.0
        frame.to_csv(cleaned_path, index=False)
        write_json_file(metadata_path, {"run_id": "stage2_backfill"})
        return SimpleNamespace(cleaned_table_path=cleaned_path, metadata_path=metadata_path)

    monkeypatch.setattr("kubera.pilot.live_pilot.fetch_historical_market_data", fake_stage2_for_backfill)

    backfill_result = backfill_pilot_actuals(
        settings,
        prediction_date=date(2026, 3, 11),
        prediction_mode="after_close",
    )

    log_frame = pd.read_csv(result.log_path)
    assert backfill_result.updated_row_count == 1
    assert log_frame.iloc[0]["actual_outcome_status"] == ACTUAL_STATUS_BACKFILLED
    assert log_frame.iloc[0]["actual_next_day_direction"] == 1
    assert log_frame.iloc[0]["baseline_predicted_next_day_direction"] == baseline_before
    assert not pd.isna(log_frame.iloc[0]["baseline_correct"])


def test_annotate_pilot_entry_updates_only_the_latest_matching_row(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_saved_models()
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_notes",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
        ),
    )

    run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )
    run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:20:00+05:30").to_pydatetime(),
    )

    annotation_result = annotate_pilot_entry(
        settings,
        prediction_mode="after_close",
        prediction_date=date(2026, 3, 11),
        news_quality_note="Sparse coverage",
    )

    log_frame = pd.read_csv(annotation_result.log_path)
    assert pd.isna(log_frame.iloc[0]["news_quality_note"])
    assert log_frame.iloc[1]["news_quality_note"] == "Sparse coverage"
    assert not pd.isna(log_frame.iloc[1]["manual_notes_updated_at_utc"])


def test_run_live_pilot_supports_runtime_ticker_override_and_observability(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_tcs",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(
            settings=runtime_settings,
            run_id="stage5_tcs",
            provider_request_count=4,
            provider_request_retry_count=1,
            article_fetch_attempt_count=3,
            article_fetch_retry_count=2,
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(
            settings=runtime_settings,
            run_id="stage6_tcs",
            provider_request_count=2,
            retry_count=1,
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
            run_id="stage7_tcs",
        ),
    )

    def fake_predict_live_baseline(runtime_settings, path_manager, historical_row):
        del historical_row
        return {
            "baseline_predicted_next_day_direction": 1,
            "baseline_predicted_probability_up": 0.7,
            "baseline_model_path": str(
                path_manager.build_baseline_model_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                )
            ),
            "baseline_model_metadata_path": str(
                path_manager.build_baseline_model_metadata_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                )
            ),
            "baseline_model_run_id": "baseline_tcs_fixture",
        }

    def fake_predict_live_enhanced(
        runtime_settings,
        path_manager,
        *,
        prediction_mode,
        historical_row,
        news_feature_row,
    ):
        del historical_row, news_feature_row
        return {
            "enhanced_predicted_next_day_direction": 1,
            "enhanced_predicted_probability_up": 0.8,
            "enhanced_model_path": str(
                path_manager.build_enhanced_model_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                    prediction_mode,
                )
            ),
            "enhanced_model_metadata_path": str(
                path_manager.build_enhanced_model_metadata_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                    prediction_mode,
                )
            ),
            "enhanced_model_run_id": "enhanced_tcs_fixture",
        }

    monkeypatch.setattr("kubera.pilot.live_pilot.predict_live_baseline", fake_predict_live_baseline)
    monkeypatch.setattr("kubera.pilot.live_pilot.predict_live_enhanced", fake_predict_live_enhanced)

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
        ticker="TCS",
        exchange="NSE",
    )

    log_frame = pd.read_csv(result.log_path)
    snapshot_payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))

    assert result.log_path.name == "TCS_NSE_after_close_pilot_log.csv"
    assert log_frame.iloc[0]["ticker"] == "TCS"
    assert log_frame.iloc[0]["stage5_provider_request_count"] == 4
    assert log_frame.iloc[0]["stage5_provider_request_retry_count"] == 1
    assert log_frame.iloc[0]["stage5_article_fetch_retry_count"] == 2
    assert log_frame.iloc[0]["stage6_retry_count"] == 1
    assert float(log_frame.iloc[0]["total_duration_seconds"]) >= 0.0
    assert float(log_frame.iloc[0]["stage5_duration_seconds"]) >= 0.0
    assert snapshot_payload["retry_summary"]["stage5"]["article_fetch_retry_count"] == 2
    assert snapshot_payload["retry_summary"]["stage6"]["retry_count"] == 1
    assert snapshot_payload["timing"]["total_duration_seconds"] is not None


def test_run_live_pilot_records_runtime_warning_in_log_and_snapshot(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_PILOT_RUNTIME_WARNING_SECONDS", "0.000001")
    prepare_saved_models()
    settings = load_settings()
    install_live_pilot_happy_path_mocks(monkeypatch)

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )

    log_frame = pd.read_csv(result.log_path)
    snapshot_payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))

    assert bool(log_frame.iloc[0]["runtime_warning_flag"]) is True
    assert "configured threshold" in str(log_frame.iloc[0]["runtime_warning_message"])
    assert "runtime_warning" in json.loads(log_frame.iloc[0]["warning_codes_json"])
    assert snapshot_payload["runtime_warning"]["flag"] is True
    assert "configured threshold" in str(snapshot_payload["runtime_warning"]["message"])


def test_run_live_pilot_prints_summary_and_snapshot_context(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_saved_models()
    settings = load_settings()
    install_live_pilot_happy_path_mocks(monkeypatch)
    path_manager = PathManager(settings.paths)
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    prior_row = {column_name: pd.NA for column_name in PILOT_LOG_COLUMNS}
    prior_row.update(
        {
            "pilot_entry_id": "prior_after_close",
            "prediction_key": "INFY|NSE|after_close|2026-03-10",
            "ticker": "INFY",
            "exchange": "NSE",
            "prediction_mode": "after_close",
            "pilot_timestamp_utc": "2026-03-09T10:45:00+00:00",
            "pilot_timestamp_market": "2026-03-09T16:15:00+05:30",
            "market_session_date": "2026-03-09",
            "historical_date": "2026-03-09",
            "prediction_date": "2026-03-10",
            "actual_outcome_status": ACTUAL_STATUS_BACKFILLED,
            "actual_historical_close": 118.0,
            "actual_prediction_close": 121.0,
            "actual_next_day_direction": 1,
            "baseline_correct": True,
            "enhanced_correct": False,
        }
    )
    pd.DataFrame([prior_row], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    result = run_live_pilot(
        settings,
        prediction_mode="after_close",
        timestamp=pd.Timestamp("2026-03-10T16:15:00+05:30").to_pydatetime(),
    )

    captured = capsys.readouterr()
    snapshot_payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))

    assert "Kubera Live Pilot Summary" in captured.out
    assert "Ticker: INFY | Exchange: NSE | Mode: after_close" in captured.out
    assert "Warnings fired: yes" in captured.out
    assert "top_events=earnings (2)" in captured.out
    assert "Prior day outcome: 2026-03-10 | baseline_correct=yes | enhanced_correct=no" in captured.out
    assert snapshot_payload["summary_context"]["news_context"]["article_count"] == 2
    assert snapshot_payload["summary_context"]["warnings"]["fired"] is True
    assert snapshot_payload["summary_context"]["model_agreement"] in {"agree", "disagree"}
    assert snapshot_payload["prior_prediction_outcome"]["backfilled"] is True
    assert snapshot_payload["prior_prediction_outcome"]["historical_close"] == 118.0


def test_live_pilot_main_with_explain_prints_generated_explanation(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KUBERA_LLM_API_KEY", "test-key")
    prepare_saved_models()
    install_live_pilot_happy_path_mocks(monkeypatch)
    captured_prompt: dict[str, str] = {}
    captured_options: dict[str, object | None] = {}

    class FakeGeminiClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def generate(self, prompt: str, *, options=None) -> SimpleNamespace:  # type: ignore[no-untyped-def]
            captured_prompt["value"] = prompt
            captured_options["value"] = options
            return SimpleNamespace(response_text="Both models lean up on supportive company news.")

    monkeypatch.setattr("kubera.pilot.live_pilot.GeminiApiExtractionClient", FakeGeminiClient)

    exit_code = live_pilot_main(
        [
            "run",
            "--prediction-mode",
            "after_close",
            "--timestamp",
            "2026-03-10T16:15:00+05:30",
            "--explain",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pilot explanation:" in captured.out
    assert "Both models lean up on supportive company news." in captured.out
    assert "\"summary_context\"" in captured_prompt["value"]
    assert captured_options["value"] is None


def test_live_pilot_main_with_explain_skips_when_llm_key_is_missing(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_saved_models()
    install_live_pilot_happy_path_mocks(monkeypatch)

    exit_code = live_pilot_main(
        [
            "run",
            "--prediction-mode",
            "after_close",
            "--timestamp",
            "2026-03-10T16:15:00+05:30",
            "--explain",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pilot explanation skipped: KUBERA_LLM_API_KEY is not set." in captured.out


def test_live_pilot_main_with_explain_degrades_when_gemini_fails(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KUBERA_LLM_API_KEY", "test-key")
    prepare_saved_models()
    install_live_pilot_happy_path_mocks(monkeypatch)

    class FailingGeminiClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def generate(self, prompt: str) -> SimpleNamespace:
            del prompt
            raise RuntimeError("gemini offline")

    monkeypatch.setattr("kubera.pilot.live_pilot.GeminiApiExtractionClient", FailingGeminiClient)

    exit_code = live_pilot_main(
        [
            "run",
            "--prediction-mode",
            "after_close",
            "--timestamp",
            "2026-03-10T16:15:00+05:30",
            "--explain",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pilot explanation unavailable: gemini offline" in captured.out


def test_plan_pilot_week_writes_trading_day_manifest_and_pending_summary(
    isolated_repo,
) -> None:
    settings = load_settings()

    result = plan_pilot_week(
        settings,
        pilot_start_date=date(2026, 3, 7),
        pilot_end_date=date(2026, 3, 10),
    )

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    status_summary = json.loads(result.status_summary_path.read_text(encoding="utf-8"))

    assert result.slot_count == 4
    assert manifest_payload["pilot_window"]["expected_market_session_dates"] == [
        "2026-03-09",
        "2026-03-10",
    ]
    assert manifest_payload["slots"][0]["scheduled_timestamp_market"].endswith("08:05:00+05:30")
    assert manifest_payload["slots"][1]["scheduled_timestamp_market"].endswith("16:15:00+05:30")
    assert manifest_payload["slots"][0]["prediction_date"] == "2026-03-09"
    assert manifest_payload["slots"][1]["prediction_date"] == "2026-03-10"
    assert status_summary["slot_count"] == 4
    assert status_summary["pending_slot_count"] == 4
    assert status_summary["completed_slot_count"] == 0


def test_run_due_pilot_week_executes_due_slots_once_and_records_statuses(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    plan_result = plan_pilot_week(
        settings,
        pilot_start_date=date(2026, 3, 9),
        pilot_end_date=date(2026, 3, 10),
    )
    manifest_payload = json.loads(plan_result.manifest_path.read_text(encoding="utf-8"))
    executed_slot_ids: list[str] = []
    scripted_outcomes = {
        "2026-03-09_pre_market": "success",
        "2026-03-09_after_close": "partial_failure",
        "2026-03-10_pre_market": "failure",
        "2026-03-10_after_close": "success",
    }

    def fake_run_live_pilot(
        runtime_settings,
        *,
        prediction_mode: str,
        timestamp: datetime | None = None,
        ticker: str | None = None,
        exchange: str | None = None,
    ):
        del runtime_settings, ticker, exchange
        assert timestamp is not None
        market_session_date = (
            pd.Timestamp(timestamp)
            .tz_convert("Asia/Kolkata")
            .date()
            .isoformat()
        )
        slot_id = f"{market_session_date}_{prediction_mode}"
        executed_slot_ids.append(slot_id)
        if scripted_outcomes[slot_id] == "failure":
            raise RuntimeError("Authorization: Bearer secret-run-due-token")
        matching_slot = next(
            slot for slot in manifest_payload["slots"] if slot["slot_id"] == slot_id
        )
        return SimpleNamespace(
            log_path=Path(matching_slot["pilot_log_path"]),
            snapshot_path=Path(matching_slot["slot_status_path"]).with_suffix(".snapshot.json"),
            pilot_entry_id=f"entry_{slot_id}",
            status=scripted_outcomes[slot_id],
            prediction_date=date.fromisoformat(matching_slot["prediction_date"]),
            prediction_mode=prediction_mode,
        )

    monkeypatch.setattr("kubera.pilot.live_pilot.run_live_pilot", fake_run_live_pilot)

    result = run_due_pilot_week(
        settings,
        plan_path=plan_result.manifest_path,
        now=pd.Timestamp("2026-03-10T23:59:00Z").to_pydatetime(),
    )

    summary_payload = json.loads(result.status_summary_path.read_text(encoding="utf-8"))
    failure_slot_payload = json.loads(
        Path(manifest_payload["slots"][2]["slot_status_path"]).read_text(encoding="utf-8")
    )

    assert result.due_slot_count == 4
    assert result.executed_slot_count == 4
    assert executed_slot_ids == [slot["slot_id"] for slot in manifest_payload["slots"]]
    assert summary_payload["completed_slot_count"] == 2
    assert summary_payload["partial_failure_count"] == 1
    assert summary_payload["failure_count"] == 1
    assert summary_payload["pending_slot_count"] == 0
    assert failure_slot_payload["slot_status"] == "failure"
    assert "[redacted]" in failure_slot_payload["error_message"]
    assert "secret-run-due-token" not in failure_slot_payload["error_message"]

    second_result = run_due_pilot_week(
        settings,
        plan_path=plan_result.manifest_path,
        now=pd.Timestamp("2026-03-10T23:59:00Z").to_pydatetime(),
    )

    assert second_result.due_slot_count == 0
    assert second_result.executed_slot_count == 0


def test_live_pilot_cli_run_due_prints_summary(
    isolated_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = load_settings()
    plan_result = plan_pilot_week(
        settings,
        pilot_start_date=date(2026, 3, 9),
        pilot_end_date=date(2026, 3, 10),
    )

    exit_code = live_pilot_main(
        [
            "run-due",
            "--plan-path",
            str(plan_result.manifest_path),
            "--now",
            "2026-03-10T23:59:00Z",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pilot week due-run summary" in captured.out
    assert "Due=4" in captured.out
    assert "dry_run=yes" in captured.out


def test_backfill_due_pilot_week_targets_only_pending_eligible_rows(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    pre_market_log_path = path_manager.build_pilot_log_path("INFY", "NSE", "pre_market")
    after_close_log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pre_market_log_path.parent.mkdir(parents=True, exist_ok=True)
    after_close_log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "pilot_entry_id": "pre_pending",
                "pilot_timestamp_utc": "2026-03-09T02:35:00+00:00",
                "market_session_date": "2026-03-09",
                "prediction_date": "2026-03-09",
                "actual_outcome_status": "pending",
            },
            {
                "pilot_entry_id": "pre_backfilled",
                "pilot_timestamp_utc": "2026-03-10T02:35:00+00:00",
                "market_session_date": "2026-03-10",
                "prediction_date": "2026-03-10",
                "actual_outcome_status": ACTUAL_STATUS_BACKFILLED,
            },
        ]
    ).to_csv(pre_market_log_path, index=False)
    pd.DataFrame(
        [
            {
                "pilot_entry_id": "after_pending",
                "pilot_timestamp_utc": "2026-03-09T10:45:00+00:00",
                "market_session_date": "2026-03-09",
                "prediction_date": "2026-03-10",
                "actual_outcome_status": "pending",
            },
            {
                "pilot_entry_id": "after_future",
                "pilot_timestamp_utc": "2026-03-10T10:45:00+00:00",
                "market_session_date": "2026-03-10",
                "prediction_date": "2026-03-11",
                "actual_outcome_status": "pending",
            },
        ]
    ).to_csv(after_close_log_path, index=False)
    calls: list[tuple[str, date]] = []

    def fake_backfill_pilot_actuals(
        runtime_settings,
        *,
        prediction_date: date,
        prediction_mode: str | None = None,
        ticker: str | None = None,
        exchange: str | None = None,
    ):
        del runtime_settings, ticker, exchange
        assert prediction_mode is not None
        calls.append((prediction_mode, prediction_date))
        log_path = pre_market_log_path if prediction_mode == "pre_market" else after_close_log_path
        updated = 1 if prediction_mode == "pre_market" else 2
        return SimpleNamespace(
            updated_row_count=updated,
            unresolved_row_count=0,
            log_paths=(log_path,),
        )

    monkeypatch.setattr("kubera.pilot.live_pilot.backfill_pilot_actuals", fake_backfill_pilot_actuals)

    result = backfill_due_pilot_week(
        settings,
        pilot_start_date=date(2026, 3, 9),
        pilot_end_date=date(2026, 3, 10),
        as_of=date(2026, 3, 10),
    )

    assert sorted(calls) == [
        ("after_close", date(2026, 3, 10)),
        ("pre_market", date(2026, 3, 9)),
    ]
    assert result.updated_row_count == 3
    assert result.unresolved_row_count == 0
    assert set(result.log_paths) == {pre_market_log_path, after_close_log_path}


def test_live_pilot_cli_backfill_due_prints_summary(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_backfill_due_pilot_week(
        runtime_settings,
        *,
        pilot_start_date,
        pilot_end_date,
        as_of=None,
        ticker=None,
        exchange=None,
    ):
        del runtime_settings, pilot_start_date, pilot_end_date, as_of, ticker, exchange
        return SimpleNamespace(
            updated_row_count=2,
            unresolved_row_count=1,
            log_paths=(Path("pilot_a.csv"), Path("pilot_b.csv")),
        )

    monkeypatch.setattr(
        "kubera.pilot.live_pilot.backfill_due_pilot_week",
        fake_backfill_due_pilot_week,
    )

    exit_code = live_pilot_main(
        [
            "backfill-due",
            "--pilot-start-date",
            "2026-03-09",
            "--pilot-end-date",
            "2026-03-10",
            "--as-of",
            "2026-03-10",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pilot week backfill summary" in captured.out
    assert "updated=2 | unresolved=1" in captured.out
    assert "Logs touched: 2" in captured.out


def test_live_pilot_cli_operate_week_orchestrates_manifest_runs_and_backfills(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    expected_manifest_path = path_manager.build_pilot_week_manifest_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        date(2026, 3, 9),
        date(2026, 3, 10),
    )
    expected_status_summary_path = path_manager.build_pilot_week_status_summary_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        date(2026, 3, 9),
        date(2026, 3, 10),
    )
    observed: dict[str, object] = {}

    def fake_run_due_pilot_week(
        runtime_settings,
        *,
        plan_path,
        now=None,
        dry_run=False,
    ):
        observed["plan_path"] = Path(plan_path)
        observed["run_now"] = now
        observed["dry_run"] = dry_run
        return SimpleNamespace(
            manifest_path=Path(plan_path),
            status_summary_path=expected_status_summary_path,
            due_slot_count=4,
            executed_slot_count=2,
            dry_run=dry_run,
        )

    def fake_backfill_due_pilot_week(
        runtime_settings,
        *,
        pilot_start_date,
        pilot_end_date,
        as_of=None,
        ticker=None,
        exchange=None,
    ):
        del runtime_settings, ticker, exchange
        observed["backfill_window"] = (pilot_start_date, pilot_end_date, as_of)
        return SimpleNamespace(
            updated_row_count=3,
            unresolved_row_count=1,
            log_paths=(),
        )

    monkeypatch.setattr("kubera.pilot.live_pilot.run_due_pilot_week", fake_run_due_pilot_week)
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.backfill_due_pilot_week",
        fake_backfill_due_pilot_week,
    )

    exit_code = live_pilot_main(
        [
            "operate-week",
            "--pilot-start-date",
            "2026-03-09",
            "--pilot-end-date",
            "2026-03-10",
            "--now",
            "2026-03-10T23:59:00Z",
            "--as-of",
            "2026-03-10",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert expected_manifest_path.exists()
    assert observed["plan_path"] == expected_manifest_path
    assert observed["backfill_window"] == (
        date(2026, 3, 9),
        date(2026, 3, 10),
        date(2026, 3, 10),
    )
    assert "Pilot week operator summary" in captured.out
    assert "due=4" in captured.out
    assert "Backfill updated=3 | unresolved=1" in captured.out


def test_live_pilot_cli_run_accepts_ticker_override(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_historical_market_data",
        lambda runtime_settings, *, end_date=None, **kwargs: write_stage2_artifacts(
            settings=runtime_settings,
            end_date=end_date,
            run_id="stage2_cli_tcs",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.fetch_company_news",
        lambda runtime_settings, **kwargs: write_stage5_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.extract_news",
        lambda runtime_settings, **kwargs: write_stage6_artifacts(settings=runtime_settings),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.build_news_features",
        lambda runtime_settings, **kwargs: write_stage7_artifacts(
            settings=runtime_settings,
            prediction_date=date(2026, 3, 11),
            prediction_mode="after_close",
            include_target_row=True,
            run_id="stage7_cli_tcs",
        ),
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.predict_live_baseline",
        lambda runtime_settings, path_manager, historical_row: {
            "baseline_predicted_next_day_direction": 1,
            "baseline_predicted_probability_up": 0.7,
            "baseline_model_path": str(
                path_manager.build_baseline_model_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                )
            ),
            "baseline_model_metadata_path": str(
                path_manager.build_baseline_model_metadata_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                )
            ),
            "baseline_model_run_id": "baseline_cli_tcs",
        },
    )
    monkeypatch.setattr(
        "kubera.pilot.live_pilot.predict_live_enhanced",
        lambda runtime_settings, path_manager, *, prediction_mode, historical_row, news_feature_row: {
            "enhanced_predicted_next_day_direction": 1,
            "enhanced_predicted_probability_up": 0.8,
            "enhanced_model_path": str(
                path_manager.build_enhanced_model_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                    prediction_mode,
                )
            ),
            "enhanced_model_metadata_path": str(
                path_manager.build_enhanced_model_metadata_path(
                    runtime_settings.ticker.symbol,
                    runtime_settings.ticker.exchange,
                    prediction_mode,
                )
            ),
            "enhanced_model_run_id": "enhanced_cli_tcs",
        },
    )

    exit_code = live_pilot_main(
        [
            "run",
            "--prediction-mode",
            "after_close",
            "--timestamp",
            "2026-03-10T16:15:00+05:30",
            "--ticker",
            "TCS",
            "--exchange",
            "NSE",
        ]
    )

    path_manager = PathManager(settings.paths)
    assert exit_code == 0
    assert path_manager.build_pilot_log_path("TCS", "NSE", "after_close").exists()


def test_live_pilot_cli_plan_week_writes_manifest(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    exit_code = live_pilot_main(
        [
            "plan-week",
            "--pilot-start-date",
            "2026-03-09",
            "--pilot-end-date",
            "2026-03-10",
        ]
    )

    assert exit_code == 0
    assert path_manager.build_pilot_week_manifest_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
        date(2026, 3, 9),
        date(2026, 3, 10),
    ).exists()


def test_live_pilot_cli_plan_week_prints_summary(
    isolated_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = live_pilot_main(
        [
            "plan-week",
            "--pilot-start-date",
            "2026-03-09",
            "--pilot-end-date",
            "2026-03-10",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Pilot week plan ready" in captured.out
    assert "slots=4" in captured.out
    assert "Manifest:" in captured.out
