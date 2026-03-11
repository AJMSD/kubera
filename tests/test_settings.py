from __future__ import annotations

import json

import pytest

from kubera.config import SettingsError, load_settings, settings_to_dict


def test_load_settings_uses_stage_one_defaults(isolated_repo) -> None:
    settings = load_settings()

    assert settings.project.name == "Kubera"
    assert settings.ticker.symbol == "INFY"
    assert settings.ticker.exchange == "NSE"
    assert settings.providers.historical_data_provider == "yfinance"
    assert settings.historical_data.default_lookback_months == 24
    assert settings.historical_data.minimum_lookback_months == 12
    assert settings.historical_features.price_basis == "close"
    assert settings.historical_features.return_windows == (1, 3, 5)
    assert settings.historical_features.moving_average_windows == (5, 10, 20)
    assert settings.historical_features.volatility_windows == (5, 10)
    assert settings.historical_features.rsi_window == 14
    assert settings.historical_features.volume_ratio_window == 20
    assert settings.historical_features.drop_warmup_rows is True
    assert settings.baseline_model.model_type == "logistic_regression"
    assert settings.baseline_model.train_ratio == pytest.approx(0.70)
    assert settings.baseline_model.validation_ratio == pytest.approx(0.15)
    assert settings.baseline_model.test_ratio == pytest.approx(0.15)
    assert settings.baseline_model.logistic_c == pytest.approx(1.0)
    assert settings.baseline_model.logistic_max_iter == 1000
    assert settings.baseline_model.classification_threshold == pytest.approx(0.5)
    assert settings.news_ingestion.lookback_days == 14
    assert settings.news_ingestion.marketaux_limit_per_request == 3
    assert settings.news_ingestion.max_articles_per_run == 15
    assert settings.news_ingestion.request_timeout_seconds == 15
    assert settings.news_ingestion.article_fetch_timeout_seconds == 15
    assert settings.news_ingestion.article_retry_attempts == 3
    assert settings.news_ingestion.language == "en"
    assert settings.news_ingestion.country == "in"
    assert settings.news_ingestion.user_agent == "KuberaNewsFetcher/1.0"
    assert settings.news_ingestion.full_text_min_chars == 250
    assert settings.market.timezone_name == "Asia/Kolkata"
    assert settings.market.market_open.isoformat(timespec="minutes") == "09:15"
    assert settings.market.market_close.isoformat(timespec="minutes") == "15:30"
    assert settings.paths.data_dir == isolated_repo / "data"
    assert settings.paths.artifacts_dir == isolated_repo / "artifacts"


def test_env_overrides_are_applied(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_TICKER", "TCS")
    monkeypatch.setenv("KUBERA_COMPANY_NAME", "Tata Consultancy Services")
    monkeypatch.setenv("KUBERA_NEWS_ALIASES", "TCS,Tata Consultancy Services")
    monkeypatch.setenv("KUBERA_DEFAULT_PREDICTION_MODE", "after_close")

    settings = load_settings()

    assert settings.ticker.symbol == "TCS"
    assert settings.ticker.company_name == "Tata Consultancy Services"
    assert settings.ticker.search_aliases == ("TCS", "Tata Consultancy Services")
    assert settings.run.default_prediction_mode == "after_close"
    assert settings.paths.repo_root == isolated_repo


def test_invalid_prediction_mode_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_DEFAULT_PREDICTION_MODE", "intraday")

    with pytest.raises(SettingsError, match="prediction mode"):
        load_settings()


def test_missing_api_key_fails_when_provider_is_enabled(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_PROVIDER", "marketaux")

    with pytest.raises(SettingsError, match="requires an API key"):
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


def test_secret_values_are_redacted_from_settings_dict(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_API_KEY", "super-secret-value")
    settings = load_settings()

    payload = settings_to_dict(settings, redact_secrets=True)

    assert payload["providers"]["news_api_key"] == "[redacted]"
    assert "super-secret-value" not in json.dumps(payload)


def test_invalid_baseline_split_ratios_fail_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_BASELINE_TRAIN_RATIO", "0.70")
    monkeypatch.setenv("KUBERA_BASELINE_VALIDATION_RATIO", "0.20")
    monkeypatch.setenv("KUBERA_BASELINE_TEST_RATIO", "0.20")

    with pytest.raises(SettingsError, match="sum to 1.0"):
        load_settings()


def test_invalid_news_lookback_fails_cleanly(monkeypatch, isolated_repo) -> None:
    monkeypatch.setenv("KUBERA_NEWS_LOOKBACK_DAYS", "0")

    with pytest.raises(SettingsError, match="lookback days"):
        load_settings()
