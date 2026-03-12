from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.features.news_features import (
    NewsFeatureError,
    build_news_features,
    build_quality_weight_map,
    enrich_extraction_frame,
    main,
    validate_extraction_frame,
)
from kubera.utils.calendar import build_market_calendar
from kubera.utils.paths import PathManager
from kubera.utils.serialization import write_json_file


CONTENT_QUALITY_BY_MODE = {
    "full_article": 1.0,
    "headline_plus_snippet": 0.75,
    "headline_only": 0.5,
}


def make_extraction_row(
    *,
    article_id: str,
    published_at: str,
    ticker: str = "INFY",
    exchange: str = "NSE",
    company_name: str = "Infosys Limited",
    extraction_mode: str = "full_article",
    warning_flag: bool = False,
    relevance_score: float = 0.8,
    sentiment_score: float = 0.2,
    directional_bias: str = "bullish",
    event_type: str = "earnings",
    event_severity: float = 0.6,
    confidence_score: float = 0.7,
) -> dict[str, object]:
    timestamp = pd.Timestamp(published_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    published_at_utc = timestamp.tz_convert("UTC")
    published_at_ist = timestamp.tz_convert("Asia/Kolkata")
    sentiment_label = "neutral"
    if sentiment_score > 0:
        sentiment_label = "positive"
    elif sentiment_score < 0:
        sentiment_label = "negative"

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
        "warning_flag": warning_flag,
        "source_fetch_warning_flag": warning_flag,
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
        "event_severity": event_severity,
        "expected_horizon": "short_term",
        "directional_bias": directional_bias,
        "confidence_score": confidence_score,
        "rationale_short": "Fixture rationale",
    }


def make_stage_seven_fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            make_extraction_row(
                article_id="a",
                published_at="2026-03-09T08:45:00+05:30",
                extraction_mode="full_article",
                warning_flag=False,
                relevance_score=1.0,
                sentiment_score=0.5,
                directional_bias="bullish",
                event_type="earnings",
                event_severity=0.7,
                confidence_score=0.8,
            ),
            make_extraction_row(
                article_id="b",
                published_at="2026-03-09T11:00:00+05:30",
                extraction_mode="headline_plus_snippet",
                warning_flag=True,
                relevance_score=0.8,
                sentiment_score=-0.2,
                directional_bias="bearish",
                event_type="lawsuit",
                event_severity=0.9,
                confidence_score=0.5,
            ),
            make_extraction_row(
                article_id="c",
                published_at="2026-03-10T08:00:00+05:30",
                extraction_mode="headline_only",
                warning_flag=True,
                relevance_score=0.4,
                sentiment_score=0.1,
                directional_bias="neutral",
                event_type="other",
                event_severity=0.3,
                confidence_score=0.6,
            ),
        ]
    )


def write_extraction_artifacts(
    frame: pd.DataFrame,
    *,
    ticker: str = "INFY",
    exchange: str = "NSE",
) -> tuple[Path, object, PathManager]:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    extraction_path = path_manager.build_processed_llm_extractions_path(ticker, exchange)
    metadata_path = path_manager.build_processed_llm_extractions_metadata_path(
        ticker,
        exchange,
    )
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(extraction_path, index=False)
    write_json_file(
        metadata_path,
        {
            "ticker": ticker,
            "exchange": exchange,
            "source_row_count": int(len(frame)),
            "coverage_start": (
                str(frame["published_date_ist"].min()) if not frame.empty else None
            ),
            "coverage_end": (
                str(frame["published_date_ist"].max()) if not frame.empty else None
            ),
        },
    )
    return extraction_path, settings, path_manager


def test_enrich_extraction_frame_assigns_expected_market_dates_and_targets(
    isolated_repo,
) -> None:
    settings = load_settings()
    holiday_path = settings.market.local_holiday_override_path
    write_json_file(holiday_path, {"holidays": ["2026-03-11"]})
    frame = pd.DataFrame(
        [
            make_extraction_row(article_id="pre", published_at="2026-03-10T08:30:00+05:30"),
            make_extraction_row(article_id="intra", published_at="2026-03-10T10:30:00+05:30"),
            make_extraction_row(article_id="after", published_at="2026-03-10T16:15:00+05:30"),
            make_extraction_row(article_id="weekend", published_at="2026-03-14T10:00:00+05:30"),
            make_extraction_row(article_id="holiday", published_at="2026-03-11T11:00:00+05:30"),
            make_extraction_row(article_id="mixed", published_at="2026-03-10T03:30:00+00:00"),
        ]
    )
    validated = validate_extraction_frame(frame, ticker="INFY", exchange="NSE")
    calendar = build_market_calendar(settings.market)

    enriched_frame, _ = enrich_extraction_frame(
        validated,
        settings=settings,
        quality_weight_map=build_quality_weight_map(settings.news_features),
        calendar=calendar,
    )
    by_id = enriched_frame.set_index("article_id")

    assert by_id.loc["pre", "market_phase"] == "pre_market"
    assert by_id.loc["pre", "market_date"].isoformat() == "2026-03-10"
    assert by_id.loc["pre", "pre_market_target_date"].isoformat() == "2026-03-10"
    assert by_id.loc["pre", "after_close_target_date"].isoformat() == "2026-03-12"

    assert by_id.loc["intra", "market_phase"] == "intraday"
    assert by_id.loc["intra", "market_date"].isoformat() == "2026-03-10"
    assert by_id.loc["intra", "pre_market_target_date"].isoformat() == "2026-03-12"
    assert by_id.loc["intra", "after_close_target_date"].isoformat() == "2026-03-12"

    assert by_id.loc["after", "market_phase"] == "after_close"
    assert by_id.loc["after", "market_date"].isoformat() == "2026-03-12"
    assert by_id.loc["after", "pre_market_target_date"].isoformat() == "2026-03-12"
    assert by_id.loc["after", "after_close_target_date"].isoformat() == "2026-03-12"

    assert by_id.loc["weekend", "market_phase"] == "non_trading"
    assert by_id.loc["weekend", "market_date"].isoformat() == "2026-03-16"
    assert by_id.loc["weekend", "pre_market_target_date"].isoformat() == "2026-03-16"
    assert by_id.loc["weekend", "after_close_target_date"].isoformat() == "2026-03-17"

    assert by_id.loc["holiday", "market_phase"] == "non_trading"
    assert by_id.loc["holiday", "market_date"].isoformat() == "2026-03-12"
    assert by_id.loc["holiday", "pre_market_target_date"].isoformat() == "2026-03-12"
    assert by_id.loc["holiday", "after_close_target_date"].isoformat() == "2026-03-13"

    assert by_id.loc["mixed", "market_phase"] == "pre_market"
    assert by_id.loc["mixed", "published_at_market"].endswith("+05:30")
    assert by_id.loc["mixed", "pre_market_target_date"].isoformat() == "2026-03-10"


def test_enrich_extraction_frame_can_disable_confidence_weighting(isolated_repo) -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        [
            make_extraction_row(
                article_id="confidence_test",
                published_at="2026-03-10T08:30:00+05:30",
                extraction_mode="headline_plus_snippet",
                relevance_score=0.8,
                confidence_score=0.25,
                sentiment_score=0.4,
                directional_bias="bullish",
                event_type="earnings",
                event_severity=0.5,
            )
        ]
    )
    validated = validate_extraction_frame(frame, ticker="INFY", exchange="NSE")
    calendar = build_market_calendar(settings.market)

    default_enriched, _ = enrich_extraction_frame(
        validated,
        settings=settings,
        quality_weight_map=build_quality_weight_map(settings.news_features),
        calendar=calendar,
    )
    no_confidence_settings = replace(
        settings,
        news_features=replace(
            settings.news_features,
            use_confidence_in_article_weight=False,
        ),
    )
    no_confidence_enriched, _ = enrich_extraction_frame(
        validated,
        settings=no_confidence_settings,
        quality_weight_map=build_quality_weight_map(no_confidence_settings.news_features),
        calendar=calendar,
    )

    assert default_enriched.iloc[0]["confidence_weight"] == pytest.approx(0.25)
    assert no_confidence_enriched.iloc[0]["confidence_weight"] == pytest.approx(1.0)
    assert default_enriched.iloc[0]["article_weight"] == pytest.approx(0.75 * 0.8 * 0.25)
    assert no_confidence_enriched.iloc[0]["article_weight"] == pytest.approx(0.75 * 0.8)


def test_build_news_features_persists_mode_aware_aggregates_and_zero_fills(
    isolated_repo,
) -> None:
    frame = make_stage_seven_fixture_frame()
    _, settings, _ = write_extraction_artifacts(frame)

    result = build_news_features(settings)

    feature_frame = pd.read_csv(result.feature_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    raw_snapshot = json.loads(result.raw_snapshot_path.read_text(encoding="utf-8"))

    assert result.row_count == 6
    assert metadata["zero_news_row_count"] == 2
    assert metadata["supported_prediction_modes"] == ["pre_market", "after_close"]
    assert metadata["timing"]["elapsed_seconds"] >= 0.0
    assert metadata["workload"]["source_row_count"] == len(frame)

    monday_after_close = feature_frame.loc[
        (feature_frame["date"] == "2026-03-09")
        & (feature_frame["prediction_mode"] == "after_close")
    ].iloc[0]
    assert monday_after_close["news_article_count"] == 0
    assert monday_after_close["news_weighted_sentiment_score"] == pytest.approx(0.0)

    tuesday_pre_market = feature_frame.loc[
        (feature_frame["date"] == "2026-03-10")
        & (feature_frame["prediction_mode"] == "pre_market")
    ].iloc[0]
    assert tuesday_pre_market["news_article_count"] == 2
    assert tuesday_pre_market["news_avg_sentiment"] == pytest.approx(-0.05)
    assert tuesday_pre_market["news_max_severity"] == pytest.approx(0.9)
    assert tuesday_pre_market["news_avg_relevance"] == pytest.approx(0.6)
    assert tuesday_pre_market["news_avg_confidence"] == pytest.approx(0.55)
    assert tuesday_pre_market["news_bullish_article_count"] == 0
    assert tuesday_pre_market["news_bearish_article_count"] == 1
    assert tuesday_pre_market["news_neutral_article_count"] == 1
    assert tuesday_pre_market["news_headline_plus_snippet_count"] == 1
    assert tuesday_pre_market["news_headline_only_count"] == 1
    assert tuesday_pre_market["news_warning_article_count"] == 2
    assert tuesday_pre_market["news_fallback_article_ratio"] == pytest.approx(1.0)
    assert tuesday_pre_market["news_avg_content_quality_score"] == pytest.approx(0.625)
    assert tuesday_pre_market["news_weighted_relevance_score"] == pytest.approx(0.64)
    assert tuesday_pre_market["news_weighted_confidence_score"] == pytest.approx(0.54)
    assert tuesday_pre_market["news_weighted_sentiment_score"] == pytest.approx(
        -0.1142857142857143
    )
    assert tuesday_pre_market["news_weighted_bearish_score"] == pytest.approx(
        0.7142857142857143
    )
    assert tuesday_pre_market["news_weighted_bullish_score"] == pytest.approx(0.0)
    assert tuesday_pre_market["news_event_count_lawsuit"] == 1
    assert tuesday_pre_market["news_event_count_other"] == 1

    tuesday_after_close = feature_frame.loc[
        (feature_frame["date"] == "2026-03-10")
        & (feature_frame["prediction_mode"] == "after_close")
    ].iloc[0]
    assert tuesday_after_close["news_article_count"] == 2
    assert tuesday_after_close["news_full_article_count"] == 1
    assert tuesday_after_close["news_headline_plus_snippet_count"] == 1
    assert tuesday_after_close["news_warning_article_count"] == 1
    assert tuesday_after_close["news_fallback_article_ratio"] == pytest.approx(0.5)
    assert tuesday_after_close["news_avg_content_quality_score"] == pytest.approx(0.875)
    assert tuesday_after_close["news_weighted_relevance_score"] == pytest.approx(
        0.9142857142857143
    )
    assert tuesday_after_close["news_weighted_confidence_score"] == pytest.approx(
        0.6714285714285715
    )
    assert tuesday_after_close["news_weighted_sentiment_score"] == pytest.approx(
        0.3090909090909091
    )
    assert tuesday_after_close["news_weighted_bullish_score"] == pytest.approx(
        0.7272727272727273
    )
    assert tuesday_after_close["news_weighted_bearish_score"] == pytest.approx(
        0.2727272727272727
    )
    assert tuesday_after_close["news_event_count_earnings"] == 1
    assert tuesday_after_close["news_event_count_lawsuit"] == 1

    row_lineage = {
        (entry["date"], entry["prediction_mode"]): entry["article_ids"]
        for entry in raw_snapshot["row_lineage"]
    }
    assert row_lineage[("2026-03-10", "pre_market")] == ["b", "c"]
    assert row_lineage[("2026-03-10", "after_close")] == ["a", "b"]


def test_build_news_features_uses_cache_when_source_is_unchanged(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_extraction_artifacts(make_stage_seven_fixture_frame())
    settings = load_settings()
    first_result = build_news_features(settings)

    def fail_if_recomputed(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Expected cached Stage 7 news features to be reused.")

    monkeypatch.setattr(
        "kubera.features.news_features.compute_news_feature_frame",
        fail_if_recomputed,
    )

    second_result = build_news_features(settings)

    assert first_result.row_count == second_result.row_count
    assert second_result.cache_hit is True


def test_force_rebuild_bypasses_news_feature_cache(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_extraction_artifacts(make_stage_seven_fixture_frame())
    settings = load_settings()
    build_news_features(settings)

    def fail_if_recomputed(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise NewsFeatureError("Recompute path hit")

    monkeypatch.setattr(
        "kubera.features.news_features.compute_news_feature_frame",
        fail_if_recomputed,
    )

    with pytest.raises(NewsFeatureError, match="Recompute path hit"):
        build_news_features(settings, force=True)


def test_news_feature_command_smoke_builds_expected_artifacts(isolated_repo) -> None:
    write_extraction_artifacts(make_stage_seven_fixture_frame())

    exit_code = main(["--ticker", "INFY", "--exchange", "NSE"])

    assert exit_code == 0
    assert (
        isolated_repo
        / "data"
        / "features"
        / "news"
        / "INFY_NSE_news_features.csv"
    ).exists()
    assert (
        isolated_repo
        / "data"
        / "features"
        / "news"
        / "INFY_NSE_news_features.metadata.json"
    ).exists()


def test_build_news_features_supports_runtime_ticker_override(isolated_repo) -> None:
    frame = pd.DataFrame(
        [
            make_extraction_row(
                article_id="tcs-a",
                ticker="TCS",
                exchange="NSE",
                company_name="Tata Consultancy Services",
                published_at="2026-03-10T08:45:00+05:30",
                extraction_mode="full_article",
                warning_flag=False,
                relevance_score=0.9,
                sentiment_score=0.4,
                directional_bias="bullish",
                event_type="earnings",
                confidence_score=0.8,
            )
        ]
    )
    _, settings, path_manager = write_extraction_artifacts(frame, ticker="TCS", exchange="NSE")

    result = build_news_features(settings, ticker="TCS", exchange="NSE")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.feature_table_path.name == "TCS_NSE_news_features.csv"
    assert metadata["ticker"] == "TCS"
    assert path_manager.build_news_feature_table_path("TCS", "NSE").exists()
