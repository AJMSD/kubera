from __future__ import annotations

import json

import pytest

from kubera.config import (
    SettingsError,
    load_settings,
    resolve_runtime_settings,
    settings_to_dict,
)


def test_load_settings_uses_stage_one_defaults(isolated_repo) -> None:
    settings = load_settings()

    assert settings.project.name == "Kubera"
    assert settings.ticker.symbol == "INFY"
    assert settings.ticker.exchange == "NSE"
    assert settings.ticker.sector_name == "Information Technology"
    assert settings.ticker.industry_name == "IT Services and Consulting"
    assert settings.ticker.sector_query_terms == (
        "information technology",
        "IT services",
        "digital transformation",
    )
    assert settings.ticker.macro_query_terms == (
        "NSE IT index",
        "India technology exports",
    )
    assert settings.providers.historical_data_provider == "yfinance"
    assert settings.providers.historical_parallel_providers == ()
    assert settings.historical_data.default_lookback_months == 60
    assert settings.historical_data.minimum_lookback_months == 12
    assert settings.historical_features.price_basis == "close"
    assert settings.historical_features.return_windows == (1, 3, 5)
    assert settings.historical_features.moving_average_windows == (5, 10, 20)
    assert settings.historical_features.volatility_windows == (5, 10)
    assert settings.historical_features.rsi_window == 14
    assert settings.historical_features.volume_ratio_window == 20
    assert settings.historical_features.macd_fast_span == 12
    assert settings.historical_features.macd_slow_span == 26
    assert settings.historical_features.macd_signal_span == 9
    assert settings.historical_features.rolling_year_window == 252
    assert settings.historical_features.lag_windows == (1, 2)
    assert settings.historical_features.include_day_of_week is True
    assert settings.historical_features.drop_warmup_rows is True
    assert settings.historical_features.bollinger_window == 20
    assert settings.historical_features.bollinger_std_dev == pytest.approx(2.0)
    assert settings.historical_features.stochastic_period == 14
    assert settings.historical_features.atr_window == 14
    assert settings.baseline_model.model_type == "gradient_boosting"
    assert settings.baseline_model.train_ratio == pytest.approx(0.70)
    assert settings.baseline_model.validation_ratio == pytest.approx(0.15)
    assert settings.baseline_model.test_ratio == pytest.approx(0.15)
    assert settings.baseline_model.logistic_c == pytest.approx(1.0)
    assert settings.baseline_model.classification_threshold == pytest.approx(0.5)
    assert settings.baseline_model.gbm_n_estimators == 300
    assert settings.baseline_model.gbm_max_depth == 4
    assert settings.baseline_model.gbm_learning_rate == pytest.approx(0.02)
    assert settings.baseline_model.gbm_subsample == pytest.approx(0.8)
    assert settings.baseline_model.gbm_min_samples_leaf == 10
    assert settings.baseline_model.rf_n_estimators == 300
    assert settings.baseline_model.rf_max_depth is None
    assert settings.baseline_model.rf_min_samples_leaf == 10
    assert settings.baseline_model.enable_calibration is True
    assert settings.baseline_model.enable_class_weight is True
    assert settings.baseline_model.class_weight_strategy == "balanced"
    assert settings.enhanced_model.model_type == "gradient_boosting"
    assert settings.enhanced_model.train_ratio == pytest.approx(0.70)
    assert settings.enhanced_model.validation_ratio == pytest.approx(0.15)
    assert settings.enhanced_model.test_ratio == pytest.approx(0.15)
    assert settings.enhanced_model.logistic_c == pytest.approx(1.0)
    assert settings.enhanced_model.classification_threshold == pytest.approx(0.5)
    assert settings.enhanced_model.gbm_n_estimators == 300
    assert settings.enhanced_model.gbm_max_depth == 4
    assert settings.enhanced_model.gbm_learning_rate == pytest.approx(0.02)
    assert settings.enhanced_model.gbm_subsample == pytest.approx(0.8)
    assert settings.enhanced_model.gbm_min_samples_leaf == 10
    assert settings.enhanced_model.rf_n_estimators == 300
    assert settings.enhanced_model.rf_max_depth is None
    assert settings.enhanced_model.rf_min_samples_leaf == 10
    assert settings.enhanced_model.enable_calibration is True
    assert settings.enhanced_model.enable_class_weight is True
    assert settings.enhanced_model.class_weight_strategy == "balanced"
    assert settings.offline_evaluation.headline_split == "test"
    assert settings.offline_evaluation.news_heavy_min_article_count == 1
    assert settings.offline_evaluation.metric_materiality_threshold == pytest.approx(0.02)
    assert settings.news_ingestion.lookback_days == 90
    assert settings.news_ingestion.marketaux_limit_per_request == 3
    assert settings.news_ingestion.marketaux_max_news_requests == 0
    assert settings.news_ingestion.marketaux_entity_cache_ttl_hours == 168
    assert settings.news_ingestion.max_articles_per_run == 50
    assert settings.news_ingestion.request_timeout_seconds == 15
    assert settings.news_ingestion.marketaux_connect_timeout_seconds == pytest.approx(10.0)
    assert settings.news_ingestion.marketaux_read_timeout_seconds == 15
    assert settings.news_ingestion.article_fetch_timeout_seconds == 15
    assert settings.news_ingestion.article_retry_attempts == 3
    assert settings.news_ingestion.article_cache_ttl_hours == 24
    assert settings.news_ingestion.provider_request_pause_seconds == pytest.approx(0.5)
    assert settings.news_ingestion.article_request_pause_seconds == pytest.approx(0.5)
    assert settings.news_ingestion.language == "en"
    assert settings.news_ingestion.country == "in"
    assert settings.news_ingestion.user_agent == "KuberaNewsFetcher/1.0"
    assert settings.news_ingestion.full_text_min_chars == 250
    assert settings.news_ingestion.enable_google_news_rss is True
    assert settings.news_ingestion.enable_nse_announcements is True
    assert settings.llm_extraction.model == "gemma-3-27b-it"
    assert settings.llm_extraction.request_timeout_seconds == 30
    assert settings.llm_extraction.retry_attempts == 3
    assert settings.llm_extraction.retry_base_delay_seconds == pytest.approx(1.0)
    assert settings.llm_extraction.max_input_chars == 12000
    assert settings.llm_extraction.max_input_chars_pilot is None
    assert settings.llm_extraction.prompt_version == "stage6_v1"
    assert settings.llm_extraction.recovery_url_context_enabled is True
    assert settings.llm_extraction.recovery_google_search_enabled is False
    assert settings.llm_extraction.recovery_max_articles_per_run == 3
    assert len(settings.llm_extraction.recovery_model_pool) == 1
    assert settings.llm_extraction.recovery_model_pool[0].model == "gemini-2.5-flash"
    assert settings.llm_extraction.recovery_model_pool[0].supports_url_context is True
    assert settings.llm_extraction.recovery_model_pool[0].supports_google_search is False
    assert settings.news_features.full_article_weight == pytest.approx(1.0)
    assert settings.news_features.headline_plus_snippet_weight == pytest.approx(0.75)
    assert settings.news_features.headline_only_weight == pytest.approx(0.5)
    assert settings.news_features.use_confidence_in_article_weight is True
    assert settings.news_features.carry_forward_days == 2
    assert settings.news_features.carry_decay_factor == pytest.approx(0.7)
    assert settings.pilot.fallback_heavy_ratio_threshold == pytest.approx(0.5)
    assert settings.pilot.default_pre_market_run_time.isoformat(timespec="minutes") == "08:05"
    assert settings.pilot.default_after_close_run_time.isoformat(timespec="minutes") == "16:15"
    assert settings.pilot.runtime_warning_seconds == pytest.approx(120.0)
    assert settings.pilot.historical_incremental_overlap_days == 5
    assert settings.pilot.abstain_low_conviction_threshold == pytest.approx(0.05)
    assert settings.pilot.abstain_data_quality_floor == pytest.approx(55.0)
    assert settings.pilot.abstain_carried_forward_margin_penalty == pytest.approx(0.02)
    assert settings.pilot.abstain_degraded_margin_penalty == pytest.approx(0.05)
    assert settings.market.timezone_name == "Asia/Kolkata"
    assert settings.market.market_open.isoformat(timespec="minutes") == "09:15"
    assert settings.market.market_close.isoformat(timespec="minutes") == "15:30"
    assert settings.market.exchange_closures_path == (
        isolated_repo / "config" / "exchange_closures" / "india.json"
    )
    assert settings.paths.data_dir == isolated_repo / "data"
    assert settings.paths.artifacts_dir == isolated_repo / "artifacts"
    assert settings.paths.final_review_reports_dir == (
        isolated_repo / "artifacts" / "reports" / "final_review"
    )


def test_env_overrides_are_applied(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_TICKER", "TCS")
    monkeypatch.setenv("KUBERA_COMPANY_NAME", "Tata Consultancy Services Limited Override")
    monkeypatch.setenv(
        "KUBERA_NEWS_ALIASES",
        "TCS,TCS Override,Tata Consultancy Services Limited Override",
    )
    monkeypatch.setenv("KUBERA_DEFAULT_PREDICTION_MODE", "after_close")

    settings = load_settings()

    assert settings.ticker.symbol == "TCS"
    assert settings.ticker.company_name == "Tata Consultancy Services Limited Override"
    assert settings.ticker.search_aliases == (
        "TCS",
        "TCS Override",
        "Tata Consultancy Services Limited Override",
    )
    assert settings.run.default_prediction_mode == "after_close"
    assert settings.paths.repo_root == isolated_repo


def test_catalog_backed_ticker_resolution_uses_builtin_metadata(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_TICKER", "TCS")

    settings = load_settings()

    assert settings.ticker.symbol == "TCS"
    assert settings.ticker.company_name == "Tata Consultancy Services"
    assert settings.ticker.search_aliases == ("TCS", "Tata Consultancy Services")
    assert settings.ticker.sector_name == "Information Technology"
    assert settings.ticker.provider_symbol_map["yahoo_finance"] == "TCS.NS"


def test_catalog_path_override_adds_custom_ticker(monkeypatch, isolated_repo) -> None:
    catalog_path = isolated_repo / "config" / "ticker_catalog.local.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "tickers": [
                    {
                        "symbol": "WIPRO",
                        "exchange": "NSE",
                        "company_name": "Wipro Limited",
                        "search_aliases": ["WIPRO", "Wipro Limited"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KUBERA_TICKER_CATALOG_PATH", str(catalog_path))
    monkeypatch.setenv("KUBERA_TICKER", "WIPRO")

    settings = load_settings()

    assert settings.ticker.symbol == "WIPRO"
    assert settings.ticker.company_name == "Wipro Limited"
    assert settings.ticker.sector_name is None
    assert settings.ticker.macro_query_terms == ()
    assert settings.ticker.provider_symbol_map["yahoo_finance"] == "WIPRO.NS"


def test_exchange_override_resolves_market_calendar_and_provider_symbol(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_TICKER", "TCS")
    monkeypatch.setenv("KUBERA_EXCHANGE", "BSE")

    settings = load_settings()

    assert settings.market.exchange_code == "BSE"
    assert settings.market.calendar_name == "BSE"
    assert settings.ticker.exchange == "BSE"
    assert settings.ticker.provider_symbol_map["yahoo_finance"] == "TCS.BO"


def test_invalid_ticker_symbol_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_TICKER", "TCS$")

    with pytest.raises(SettingsError, match="Ticker symbol contains unsupported characters"):
        load_settings()


def test_invalid_prediction_mode_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_DEFAULT_PREDICTION_MODE", "intraday")

    with pytest.raises(SettingsError, match="prediction mode"):
        load_settings()


def test_gradient_boosting_model_type_overrides_are_applied(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_MODEL_TYPE", "gradient_boosting")
    monkeypatch.setenv("KUBERA_ENHANCED_MODEL_TYPE", "gradient_boosting")

    settings = load_settings()

    assert settings.baseline_model.model_type == "gradient_boosting"
    assert settings.enhanced_model.model_type == "gradient_boosting"


def test_invalid_baseline_model_type_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_MODEL_TYPE", "support_vector_machine")

    with pytest.raises(SettingsError, match="baseline model type"):
        load_settings()


def test_invalid_enhanced_model_type_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_ENHANCED_MODEL_TYPE", "support_vector_machine")

    with pytest.raises(SettingsError, match="enhanced model type"):
        load_settings()


def test_missing_api_key_fails_when_provider_is_enabled(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", "marketaux")

    with pytest.raises(SettingsError, match="requires an API key"):
        load_settings()


def test_missing_alphavantage_key_fails_when_provider_is_enabled(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", "alphavantage")

    with pytest.raises(SettingsError, match="KUBERA_ALPHAVANTAGE_API_KEY"):
        load_settings()


def test_unsafe_managed_path_is_rejected(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_DATA_DIR", "../outside")

    with pytest.raises(SettingsError, match="repo root"):
        load_settings()


def test_unsorted_historical_feature_windows_fail_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_MOVING_AVERAGE_WINDOWS", "10,5,20")

    with pytest.raises(SettingsError, match="Moving-average windows"):
        load_settings()


def test_historical_parallel_providers_rejects_duplicates(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_PARALLEL_PROVIDERS", "upstox,upstox")
    monkeypatch.setenv("KUBERA_UPSTOX_ACCESS_TOKEN", "x")

    with pytest.raises(SettingsError, match="duplicates"):
        load_settings()


def test_historical_parallel_must_not_repeat_canonical_provider(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_PARALLEL_PROVIDERS", "yfinance")

    with pytest.raises(SettingsError, match="must not repeat"):
        load_settings()


def test_parallel_upstox_requires_access_token(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_PARALLEL_PROVIDERS", "upstox")

    with pytest.raises(SettingsError, match="KUBERA_UPSTOX_ACCESS_TOKEN"):
        load_settings()


def test_secret_values_are_redacted_from_settings_dict(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_API_KEY", "super-secret-value")
    monkeypatch.setenv("KUBERA_ALPHAVANTAGE_API_KEY", "another-secret-value")
    monkeypatch.setenv("KUBERA_UPSTOX_ACCESS_TOKEN", "upstox-token")
    settings = load_settings()

    payload = settings_to_dict(settings, redact_secrets=True)

    assert payload["providers"]["news_api_key"] == "[redacted]"
    assert payload["providers"]["alphavantage_api_key"] == "[redacted]"
    assert payload["providers"]["upstox_access_token"] == "[redacted]"
    assert "super-secret-value" not in json.dumps(payload)
    assert "another-secret-value" not in json.dumps(payload)
    assert "upstox-token" not in json.dumps(payload)


def test_alphavantage_provider_env_is_loaded(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", "alphavantage")
    monkeypatch.setenv("KUBERA_ALPHAVANTAGE_API_KEY", "alphavantage-secret")

    settings = load_settings()

    assert settings.providers.news_provider == "alphavantage"
    assert settings.providers.alphavantage_api_key == "alphavantage-secret"


def test_llm_recovery_model_pool_json_override_is_loaded(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv(
        "KUBERA_LLM_RECOVERY_MODEL_POOL_JSON",
        json.dumps(
            [
                {
                    "model": "gemini-2.5-pro",
                    "supports_url_context": True,
                    "supports_google_search": True,
                    "requests_per_minute_limit": 7,
                    "requests_per_day_limit": 70,
                }
            ]
        ),
    )

    settings = load_settings()

    assert settings.llm_extraction.recovery_model_pool[0].model == "gemini-2.5-pro"
    assert settings.llm_extraction.recovery_model_pool[0].supports_google_search is True
    assert settings.llm_extraction.recovery_model_pool[0].requests_per_minute_limit == 7


def test_llm_recovery_stage6_plan_four_model_pool_json_loads(
    monkeypatch,
    isolated_repo,
) -> None:
    """Stage 6 plan: four-model recovery pool parses with expected order and limits."""
    monkeypatch.setenv(
        "KUBERA_LLM_RECOVERY_MODEL_POOL_JSON",
        json.dumps(
            [
                {
                    "model": "gemini-3-flash-preview",
                    "supports_url_context": True,
                    "supports_google_search": True,
                    "requests_per_minute_limit": 5,
                    "requests_per_day_limit": 20,
                },
                {
                    "model": "gemini-2.5-flash",
                    "supports_url_context": True,
                    "supports_google_search": True,
                    "requests_per_minute_limit": 5,
                    "requests_per_day_limit": 20,
                },
                {
                    "model": "gemini-2.5-flash-lite",
                    "supports_url_context": True,
                    "supports_google_search": True,
                    "requests_per_minute_limit": 10,
                    "requests_per_day_limit": 20,
                },
                {
                    "model": "gemini-3.1-flash-lite-preview",
                    "supports_url_context": True,
                    "supports_google_search": True,
                    "requests_per_minute_limit": 15,
                    "requests_per_day_limit": 500,
                },
            ]
        ),
    )

    settings = load_settings()

    pool = settings.llm_extraction.recovery_model_pool
    assert len(pool) == 4
    assert [m.model for m in pool] == [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
    ]
    assert pool[0].requests_per_minute_limit == 5
    assert pool[0].requests_per_day_limit == 20
    assert pool[3].requests_per_minute_limit == 15
    assert pool[3].requests_per_day_limit == 500
    assert all(m.supports_url_context for m in pool)
    assert all(m.supports_google_search for m in pool)


def test_invalid_baseline_split_ratios_fail_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_TRAIN_RATIO", "0.70")
    monkeypatch.setenv("KUBERA_BASELINE_VALIDATION_RATIO", "0.20")
    monkeypatch.setenv("KUBERA_BASELINE_TEST_RATIO", "0.20")

    with pytest.raises(SettingsError, match="sum to 1.0"):
        load_settings()


def test_invalid_enhanced_split_ratios_fail_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_ENHANCED_TRAIN_RATIO", "0.70")
    monkeypatch.setenv("KUBERA_ENHANCED_VALIDATION_RATIO", "0.20")
    monkeypatch.setenv("KUBERA_ENHANCED_TEST_RATIO", "0.20")

    with pytest.raises(SettingsError, match="Enhanced split ratios must sum to 1.0"):
        load_settings()


def test_invalid_offline_evaluation_headline_split_fails_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_OFFLINE_EVALUATION_HEADLINE_SPLIT", "train")

    with pytest.raises(SettingsError, match="headline split"):
        load_settings()


def test_invalid_offline_evaluation_materiality_threshold_fails_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv(
        "KUBERA_OFFLINE_EVALUATION_METRIC_MATERIALITY_THRESHOLD",
        "-0.01",
    )

    with pytest.raises(SettingsError, match="materiality threshold"):
        load_settings()


def test_invalid_news_lookback_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_LOOKBACK_DAYS", "0")

    with pytest.raises(SettingsError, match="lookback days"):
        load_settings()


def test_marketaux_read_timeout_defaults_to_request_timeout(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS", "22")

    settings = load_settings()

    assert settings.news_ingestion.request_timeout_seconds == 22
    assert settings.news_ingestion.marketaux_read_timeout_seconds == 22


def test_marketaux_connect_exceeds_read_timeout_fails_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_MARKETAUX_CONNECT_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("KUBERA_NEWS_MARKETAUX_READ_TIMEOUT_SECONDS", "10")

    with pytest.raises(SettingsError, match="Marketaux connect timeout"):
        load_settings()


def test_marketaux_connect_timeout_below_one_fails_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_MARKETAUX_CONNECT_TIMEOUT_SECONDS", "0.5")

    with pytest.raises(SettingsError, match="Marketaux connect timeout"):
        load_settings()


def test_resolve_runtime_settings_respects_company_name_for_uncatalogued_ticker(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_COMPANY_NAME", "Unknown Co Limited")

    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker="UNKNX", exchange="NSE")

    assert runtime.ticker.symbol == "UNKNX"
    assert runtime.ticker.company_name == "Unknown Co Limited"


def test_negative_article_request_pause_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS", "-0.1")

    with pytest.raises(SettingsError, match="pause seconds"):
        load_settings()


def test_negative_provider_request_pause_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS", "-0.1")

    with pytest.raises(SettingsError, match="Provider request pause"):
        load_settings()


def test_invalid_llm_retry_delay_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_LLM_RETRY_BASE_DELAY_SECONDS", "0")

    with pytest.raises(SettingsError, match="retry base delay"):
        load_settings()


def test_invalid_news_feature_weight_fails_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_NEWS_FEATURE_HEADLINE_ONLY_WEIGHT", "0")

    with pytest.raises(SettingsError, match="Headline-only weight"):
        load_settings()


def test_invalid_pilot_fallback_threshold_fails_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_PILOT_FALLBACK_HEAVY_RATIO_THRESHOLD", "1.1")

    with pytest.raises(SettingsError, match="fallback-heavy ratio threshold"):
        load_settings()


def test_invalid_pilot_runtime_warning_seconds_fail_cleanly(
    monkeypatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_PILOT_RUNTIME_WARNING_SECONDS", "0")

    with pytest.raises(SettingsError, match="runtime warning seconds"):
        load_settings()
