"""Typed settings for Kubera."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import math
from pathlib import Path
import os
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


ALLOWED_PREDICTION_MODES = frozenset({"pre_market", "after_close", "both"})
ALLOWED_EVALUATION_HEADLINE_SPLITS = frozenset({"test"})
ALLOWED_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
REDACTED_VALUE = "[redacted]"


class SettingsError(ValueError):
    """Raised when Kubera settings are invalid."""


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SettingsError("Project name must not be empty.")


@dataclass(frozen=True)
class PathSettings:
    repo_root: Path
    package_root: Path
    runtime_config_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    features_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    runs_dir: Path
    models_dir: Path
    reports_dir: Path
    baseline_models_dir: Path
    baseline_reports_dir: Path
    enhanced_models_dir: Path
    enhanced_reports_dir: Path
    evaluation_reports_dir: Path
    pilot_reports_dir: Path
    pilot_snapshots_dir: Path
    merged_features_dir: Path

    def managed_directories(self) -> tuple[Path, ...]:
        return (
            self.runtime_config_dir,
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.features_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.runs_dir,
            self.models_dir,
            self.reports_dir,
            self.baseline_models_dir,
            self.baseline_reports_dir,
            self.enhanced_models_dir,
            self.enhanced_reports_dir,
            self.evaluation_reports_dir,
            self.pilot_reports_dir,
            self.pilot_snapshots_dir,
            self.merged_features_dir,
        )


@dataclass(frozen=True)
class MarketSettings:
    timezone_name: str
    market_open: time
    market_close: time
    supported_prediction_modes: tuple[str, ...]
    local_holiday_override_path: Path

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise SettingsError(f"Unknown market timezone: {self.timezone_name}") from exc

        invalid_modes = set(self.supported_prediction_modes) - ALLOWED_PREDICTION_MODES
        invalid_modes.discard("both")
        if invalid_modes:
            raise SettingsError(
                f"Unsupported prediction modes: {sorted(invalid_modes)}"
            )

        if self.market_open >= self.market_close:
            raise SettingsError("Market open time must be earlier than market close.")


@dataclass(frozen=True)
class TickerSettings:
    symbol: str
    exchange: str
    company_name: str
    search_aliases: tuple[str, ...]
    provider_symbol_map: dict[str, str]

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise SettingsError("Ticker symbol must not be empty.")
        if not self.exchange.strip():
            raise SettingsError("Exchange must not be empty.")
        if not self.company_name.strip():
            raise SettingsError("Company name must not be empty.")
        if not self.search_aliases:
            raise SettingsError("At least one search alias is required.")


@dataclass(frozen=True)
class ProviderSettings:
    historical_data_provider: str
    news_provider: str
    llm_provider: str
    historical_data_api_key: str | None
    news_api_key: str | None
    llm_api_key: str | None


@dataclass(frozen=True)
class HistoricalDataSettings:
    default_lookback_months: int
    minimum_lookback_months: int

    def __post_init__(self) -> None:
        if self.default_lookback_months < 1:
            raise SettingsError("Default historical lookback must be at least one month.")
        if self.minimum_lookback_months < 1:
            raise SettingsError("Minimum historical lookback must be at least one month.")
        if self.default_lookback_months < self.minimum_lookback_months:
            raise SettingsError(
                "Default historical lookback must be greater than or equal to the minimum lookback."
            )


@dataclass(frozen=True)
class HistoricalFeatureSettings:
    price_basis: str
    return_windows: tuple[int, ...]
    moving_average_windows: tuple[int, ...]
    volatility_windows: tuple[int, ...]
    rsi_window: int
    volume_ratio_window: int
    drop_warmup_rows: bool

    def __post_init__(self) -> None:
        if self.price_basis != "close":
            raise SettingsError("Historical feature price basis must stay 'close' in v1.")

        _validate_positive_sorted_windows(self.return_windows, "Return windows")
        _validate_positive_sorted_windows(
            self.moving_average_windows,
            "Moving-average windows",
        )
        _validate_positive_sorted_windows(
            self.volatility_windows,
            "Volatility windows",
        )
        if self.rsi_window < 1:
            raise SettingsError("RSI window must be at least one.")
        if self.volume_ratio_window < 1:
            raise SettingsError("Volume ratio window must be at least one.")


@dataclass(frozen=True)
class RunSettings:
    random_seed: int
    default_prediction_mode: str
    log_level: str
    run_id_time_format: str
    config_snapshot_filename: str

    def __post_init__(self) -> None:
        if self.default_prediction_mode not in ALLOWED_PREDICTION_MODES:
            raise SettingsError(
                f"Unsupported default prediction mode: {self.default_prediction_mode}"
            )
        if self.log_level not in ALLOWED_LOG_LEVELS:
            raise SettingsError(f"Unsupported log level: {self.log_level}")
        if not self.config_snapshot_filename.strip():
            raise SettingsError("Config snapshot filename must not be empty.")


@dataclass(frozen=True)
class BaselineModelSettings:
    model_type: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    logistic_c: float
    logistic_max_iter: int
    classification_threshold: float

    def __post_init__(self) -> None:
        if self.model_type != "logistic_regression":
            raise SettingsError(
                "Baseline model type must stay 'logistic_regression' in Stage 4."
            )
        _validate_model_split_ratios(
            train_ratio=self.train_ratio,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            label="Baseline",
        )
        if self.logistic_c <= 0:
            raise SettingsError("Baseline logistic C must be greater than 0.")
        if self.logistic_max_iter < 1:
            raise SettingsError("Baseline logistic max_iter must be at least 1.")
        if not 0.0 <= self.classification_threshold <= 1.0:
            raise SettingsError("Baseline classification threshold must be between 0 and 1.")


@dataclass(frozen=True)
class NewsIngestionSettings:
    lookback_days: int
    marketaux_limit_per_request: int
    max_articles_per_run: int
    request_timeout_seconds: int
    article_fetch_timeout_seconds: int
    article_retry_attempts: int
    article_cache_ttl_hours: int
    provider_request_pause_seconds: float
    article_request_pause_seconds: float
    language: str
    country: str
    user_agent: str
    full_text_min_chars: int

    def __post_init__(self) -> None:
        integer_fields = (
            ("News lookback days", self.lookback_days),
            ("Marketaux limit per request", self.marketaux_limit_per_request),
            ("Max articles per run", self.max_articles_per_run),
            ("Request timeout seconds", self.request_timeout_seconds),
            ("Article fetch timeout seconds", self.article_fetch_timeout_seconds),
            ("Article retry attempts", self.article_retry_attempts),
            ("Article cache TTL hours", self.article_cache_ttl_hours),
            ("Full text minimum characters", self.full_text_min_chars),
        )
        for label, value in integer_fields:
            if value < 0:
                raise SettingsError(f"{label} must not be negative.")
        required_positive_labels = {
            "News lookback days",
            "Marketaux limit per request",
            "Max articles per run",
            "Request timeout seconds",
            "Article fetch timeout seconds",
            "Article retry attempts",
            "Full text minimum characters",
        }
        for label, value in integer_fields:
            if label in required_positive_labels and value < 1:
                raise SettingsError(f"{label} must be at least 1.")
        if self.provider_request_pause_seconds < 0:
            raise SettingsError("Provider request pause seconds must not be negative.")
        if self.article_request_pause_seconds < 0:
            raise SettingsError("Article request pause seconds must not be negative.")
        if not self.language.strip():
            raise SettingsError("News ingestion language must not be empty.")
        if not self.country.strip():
            raise SettingsError("News ingestion country must not be empty.")
        if not self.user_agent.strip():
            raise SettingsError("News ingestion user agent must not be empty.")


@dataclass(frozen=True)
class PilotSettings:
    fallback_heavy_ratio_threshold: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.fallback_heavy_ratio_threshold):
            raise SettingsError("Pilot fallback-heavy ratio threshold must be finite.")
        if not 0.0 <= self.fallback_heavy_ratio_threshold <= 1.0:
            raise SettingsError(
                "Pilot fallback-heavy ratio threshold must be between 0 and 1."
            )


@dataclass(frozen=True)
class EnhancedModelSettings:
    model_type: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    logistic_c: float
    logistic_max_iter: int
    classification_threshold: float

    def __post_init__(self) -> None:
        if self.model_type != "logistic_regression":
            raise SettingsError(
                "Enhanced model type must stay 'logistic_regression' in Stage 8."
            )
        _validate_model_split_ratios(
            train_ratio=self.train_ratio,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            label="Enhanced",
        )
        if self.logistic_c <= 0:
            raise SettingsError("Enhanced logistic C must be greater than 0.")
        if self.logistic_max_iter < 1:
            raise SettingsError("Enhanced logistic max_iter must be at least 1.")
        if not 0.0 <= self.classification_threshold <= 1.0:
            raise SettingsError("Enhanced classification threshold must be between 0 and 1.")


@dataclass(frozen=True)
class OfflineEvaluationSettings:
    headline_split: str
    news_heavy_min_article_count: int
    metric_materiality_threshold: float

    def __post_init__(self) -> None:
        if self.headline_split not in ALLOWED_EVALUATION_HEADLINE_SPLITS:
            raise SettingsError(
                f"Unsupported offline evaluation headline split: {self.headline_split}"
            )
        if self.news_heavy_min_article_count < 1:
            raise SettingsError(
                "Offline evaluation news-heavy minimum article count must be at least 1."
            )
        if not math.isfinite(self.metric_materiality_threshold):
            raise SettingsError(
                "Offline evaluation metric materiality threshold must be finite."
            )
        if self.metric_materiality_threshold < 0 or self.metric_materiality_threshold > 1:
            raise SettingsError(
                "Offline evaluation metric materiality threshold must be between 0 and 1."
            )


@dataclass(frozen=True)
class LlmExtractionSettings:
    model: str
    request_timeout_seconds: int
    retry_attempts: int
    retry_base_delay_seconds: float
    max_input_chars: int
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise SettingsError("LLM extraction model must not be empty.")
        if self.request_timeout_seconds < 1:
            raise SettingsError("LLM extraction request timeout must be at least 1 second.")
        if self.retry_attempts < 1:
            raise SettingsError("LLM extraction retry attempts must be at least 1.")
        if self.retry_base_delay_seconds <= 0:
            raise SettingsError("LLM extraction retry base delay must be greater than 0.")
        if self.max_input_chars < 1:
            raise SettingsError("LLM extraction max input chars must be at least 1.")
        if not self.prompt_version.strip():
            raise SettingsError("LLM extraction prompt version must not be empty.")


@dataclass(frozen=True)
class NewsFeatureSettings:
    full_article_weight: float
    headline_plus_snippet_weight: float
    headline_only_weight: float
    use_confidence_in_article_weight: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("Full-article weight", self.full_article_weight),
            ("Headline-plus-snippet weight", self.headline_plus_snippet_weight),
            ("Headline-only weight", self.headline_only_weight),
        ):
            if not math.isfinite(value):
                raise SettingsError(f"{label} must be finite.")
            if value <= 0 or value > 1:
                raise SettingsError(f"{label} must be greater than 0 and at most 1.")


@dataclass(frozen=True)
class AppSettings:
    project: ProjectSettings
    paths: PathSettings
    market: MarketSettings
    ticker: TickerSettings
    providers: ProviderSettings
    historical_data: HistoricalDataSettings
    historical_features: HistoricalFeatureSettings
    run: RunSettings
    baseline_model: BaselineModelSettings
    enhanced_model: EnhancedModelSettings
    offline_evaluation: OfflineEvaluationSettings
    news_ingestion: NewsIngestionSettings
    llm_extraction: LlmExtractionSettings
    news_features: NewsFeatureSettings
    pilot: PilotSettings


def load_settings(repo_root: str | Path | None = None) -> AppSettings:
    """Load, validate, and return the active Kubera settings."""

    resolved_repo_root = _resolve_repo_root(repo_root)
    load_dotenv(dotenv_path=resolved_repo_root / ".env", override=False)

    runtime_config_dir = _resolve_subpath(
        resolved_repo_root,
        os.getenv("KUBERA_RUNTIME_CONFIG_DIR", "config"),
    )
    data_dir = _resolve_subpath(
        resolved_repo_root,
        os.getenv("KUBERA_DATA_DIR", "data"),
    )
    artifacts_dir = _resolve_subpath(
        resolved_repo_root,
        os.getenv("KUBERA_ARTIFACTS_DIR", "artifacts"),
    )
    holiday_override_path = _resolve_subpath(
        resolved_repo_root,
        os.getenv(
            "KUBERA_HOLIDAY_OVERRIDE_PATH",
            "config/market_holidays.local.json",
        ),
    )

    paths = PathSettings(
        repo_root=resolved_repo_root,
        package_root=resolved_repo_root / "src" / "kubera",
        runtime_config_dir=runtime_config_dir,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        features_dir=data_dir / "features",
        artifacts_dir=artifacts_dir,
        logs_dir=artifacts_dir / "logs",
        runs_dir=artifacts_dir / "runs",
        models_dir=artifacts_dir / "models",
        reports_dir=artifacts_dir / "reports",
        baseline_models_dir=artifacts_dir / "models" / "baseline",
        baseline_reports_dir=artifacts_dir / "reports" / "baseline",
        enhanced_models_dir=artifacts_dir / "models" / "enhanced",
        enhanced_reports_dir=artifacts_dir / "reports" / "enhanced",
        evaluation_reports_dir=artifacts_dir / "reports" / "evaluation",
        pilot_reports_dir=artifacts_dir / "reports" / "pilot",
        pilot_snapshots_dir=artifacts_dir / "reports" / "pilot" / "snapshots",
        merged_features_dir=data_dir / "features" / "merged",
    )
    _validate_path_settings(paths)

    project = ProjectSettings(
        name=os.getenv("KUBERA_PROJECT_NAME", "Kubera"),
        description=os.getenv(
            "KUBERA_PROJECT_DESCRIPTION",
            "LLM-enhanced Indian stock movement prediction system",
        ),
    )

    market = MarketSettings(
        timezone_name=os.getenv("KUBERA_MARKET_TIMEZONE", "Asia/Kolkata"),
        market_open=_parse_time(os.getenv("KUBERA_MARKET_OPEN", "09:15")),
        market_close=_parse_time(os.getenv("KUBERA_MARKET_CLOSE", "15:30")),
        supported_prediction_modes=_parse_csv(
            os.getenv(
                "KUBERA_SUPPORTED_PREDICTION_MODES",
                "pre_market,after_close",
            )
        ),
        local_holiday_override_path=holiday_override_path,
    )

    ticker_symbol = os.getenv("KUBERA_TICKER", "INFY").strip()
    ticker = TickerSettings(
        symbol=ticker_symbol,
        exchange=os.getenv("KUBERA_EXCHANGE", "NSE").strip(),
        company_name=os.getenv("KUBERA_COMPANY_NAME", "Infosys Limited").strip(),
        search_aliases=_parse_csv(
            os.getenv(
                "KUBERA_NEWS_ALIASES",
                f"{ticker_symbol},Infosys,Infosys Limited",
            )
        ),
        provider_symbol_map={
            "yahoo_finance": os.getenv(
                "KUBERA_YAHOO_TICKER",
                f"{ticker_symbol}.NS",
            ).strip()
        },
    )

    providers = ProviderSettings(
        historical_data_provider=os.getenv(
            "KUBERA_HISTORICAL_DATA_PROVIDER",
            "yfinance",
        ).strip(),
        news_provider=os.getenv("KUBERA_NEWS_PROVIDER", "not_configured").strip(),
        llm_provider=os.getenv("KUBERA_LLM_PROVIDER", "not_configured").strip(),
        historical_data_api_key=_clean_optional(
            os.getenv("KUBERA_HISTORICAL_DATA_API_KEY")
        ),
        news_api_key=_clean_optional(os.getenv("KUBERA_NEWS_API_KEY")),
        llm_api_key=_clean_optional(os.getenv("KUBERA_LLM_API_KEY")),
    )
    _validate_provider_settings(providers)

    historical_data = HistoricalDataSettings(
        default_lookback_months=_parse_int(
            os.getenv("KUBERA_HISTORICAL_LOOKBACK_MONTHS", "24")
        ),
        minimum_lookback_months=_parse_int(
            os.getenv("KUBERA_MINIMUM_HISTORICAL_LOOKBACK_MONTHS", "12")
        ),
    )

    historical_features = HistoricalFeatureSettings(
        price_basis=os.getenv(
            "KUBERA_HISTORICAL_FEATURE_PRICE_BASIS",
            "close",
        ).strip().lower(),
        return_windows=_parse_int_csv(
            os.getenv("KUBERA_HISTORICAL_RETURN_WINDOWS", "1,3,5")
        ),
        moving_average_windows=_parse_int_csv(
            os.getenv("KUBERA_HISTORICAL_MOVING_AVERAGE_WINDOWS", "5,10,20")
        ),
        volatility_windows=_parse_int_csv(
            os.getenv("KUBERA_HISTORICAL_VOLATILITY_WINDOWS", "5,10")
        ),
        rsi_window=_parse_int(os.getenv("KUBERA_HISTORICAL_RSI_WINDOW", "14")),
        volume_ratio_window=_parse_int(
            os.getenv("KUBERA_HISTORICAL_VOLUME_RATIO_WINDOW", "20")
        ),
        drop_warmup_rows=_parse_bool(
            os.getenv("KUBERA_HISTORICAL_DROP_WARMUP_ROWS", "true")
        ),
    )

    run = RunSettings(
        random_seed=_parse_int(os.getenv("KUBERA_RANDOM_SEED", "42")),
        default_prediction_mode=os.getenv(
            "KUBERA_DEFAULT_PREDICTION_MODE",
            "both",
        ).strip(),
        log_level=os.getenv("KUBERA_LOG_LEVEL", "INFO").strip().upper(),
        run_id_time_format=os.getenv("KUBERA_RUN_ID_TIME_FORMAT", "%Y%m%d_%H%M%S"),
        config_snapshot_filename=os.getenv(
            "KUBERA_CONFIG_SNAPSHOT_FILENAME",
            "config.json",
        ).strip(),
    )

    baseline_model = BaselineModelSettings(
        model_type=os.getenv(
            "KUBERA_BASELINE_MODEL_TYPE",
            "logistic_regression",
        ).strip().lower(),
        train_ratio=_parse_float(os.getenv("KUBERA_BASELINE_TRAIN_RATIO", "0.70")),
        validation_ratio=_parse_float(
            os.getenv("KUBERA_BASELINE_VALIDATION_RATIO", "0.15")
        ),
        test_ratio=_parse_float(os.getenv("KUBERA_BASELINE_TEST_RATIO", "0.15")),
        logistic_c=_parse_float(os.getenv("KUBERA_BASELINE_LOGISTIC_C", "1.0")),
        logistic_max_iter=_parse_int(
            os.getenv("KUBERA_BASELINE_LOGISTIC_MAX_ITER", "1000")
        ),
        classification_threshold=_parse_float(
            os.getenv("KUBERA_BASELINE_CLASSIFICATION_THRESHOLD", "0.5")
        ),
    )

    news_ingestion = NewsIngestionSettings(
        lookback_days=_parse_int(os.getenv("KUBERA_NEWS_LOOKBACK_DAYS", "14")),
        marketaux_limit_per_request=_parse_int(
            os.getenv("KUBERA_NEWS_MARKETAUX_LIMIT_PER_REQUEST", "3")
        ),
        max_articles_per_run=_parse_int(
            os.getenv("KUBERA_NEWS_MAX_ARTICLES_PER_RUN", "15")
        ),
        request_timeout_seconds=_parse_int(
            os.getenv("KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS", "15")
        ),
        article_fetch_timeout_seconds=_parse_int(
            os.getenv("KUBERA_NEWS_ARTICLE_FETCH_TIMEOUT_SECONDS", "15")
        ),
        article_retry_attempts=_parse_int(
            os.getenv("KUBERA_NEWS_ARTICLE_RETRY_ATTEMPTS", "3")
        ),
        article_cache_ttl_hours=_parse_int(
            os.getenv("KUBERA_NEWS_ARTICLE_CACHE_TTL_HOURS", "24")
        ),
        provider_request_pause_seconds=_parse_float(
            os.getenv("KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS", "0.5")
        ),
        article_request_pause_seconds=_parse_float(
            os.getenv("KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS", "0.5")
        ),
        language=os.getenv("KUBERA_NEWS_LANGUAGE", "en").strip().lower(),
        country=os.getenv("KUBERA_NEWS_COUNTRY", "in").strip().lower(),
        user_agent=os.getenv(
            "KUBERA_NEWS_USER_AGENT",
            "KuberaNewsFetcher/1.0",
        ).strip(),
        full_text_min_chars=_parse_int(
            os.getenv("KUBERA_NEWS_FULL_TEXT_MIN_CHARS", "250")
        ),
    )

    enhanced_model = EnhancedModelSettings(
        model_type=os.getenv(
            "KUBERA_ENHANCED_MODEL_TYPE",
            "logistic_regression",
        ).strip().lower(),
        train_ratio=_parse_float(os.getenv("KUBERA_ENHANCED_TRAIN_RATIO", "0.70")),
        validation_ratio=_parse_float(
            os.getenv("KUBERA_ENHANCED_VALIDATION_RATIO", "0.15")
        ),
        test_ratio=_parse_float(os.getenv("KUBERA_ENHANCED_TEST_RATIO", "0.15")),
        logistic_c=_parse_float(os.getenv("KUBERA_ENHANCED_LOGISTIC_C", "1.0")),
        logistic_max_iter=_parse_int(
            os.getenv("KUBERA_ENHANCED_LOGISTIC_MAX_ITER", "1000")
        ),
        classification_threshold=_parse_float(
            os.getenv("KUBERA_ENHANCED_CLASSIFICATION_THRESHOLD", "0.5")
        ),
    )

    offline_evaluation = OfflineEvaluationSettings(
        headline_split=os.getenv(
            "KUBERA_OFFLINE_EVALUATION_HEADLINE_SPLIT",
            "test",
        ).strip().lower(),
        news_heavy_min_article_count=_parse_int(
            os.getenv("KUBERA_OFFLINE_EVALUATION_NEWS_HEAVY_MIN_ARTICLE_COUNT", "1")
        ),
        metric_materiality_threshold=_parse_float(
            os.getenv("KUBERA_OFFLINE_EVALUATION_METRIC_MATERIALITY_THRESHOLD", "0.02")
        ),
    )

    llm_extraction = LlmExtractionSettings(
        model=os.getenv("KUBERA_LLM_MODEL", "gemma-3-27b-it").strip(),
        request_timeout_seconds=_parse_int(
            os.getenv("KUBERA_LLM_REQUEST_TIMEOUT_SECONDS", "30")
        ),
        retry_attempts=_parse_int(
            os.getenv("KUBERA_LLM_RETRY_ATTEMPTS", "3")
        ),
        retry_base_delay_seconds=_parse_float(
            os.getenv("KUBERA_LLM_RETRY_BASE_DELAY_SECONDS", "1.0")
        ),
        max_input_chars=_parse_int(
            os.getenv("KUBERA_LLM_MAX_INPUT_CHARS", "12000")
        ),
        prompt_version=os.getenv("KUBERA_LLM_PROMPT_VERSION", "stage6_v1").strip(),
    )

    news_features = NewsFeatureSettings(
        full_article_weight=_parse_float(
            os.getenv("KUBERA_NEWS_FEATURE_FULL_ARTICLE_WEIGHT", "1.0")
        ),
        headline_plus_snippet_weight=_parse_float(
            os.getenv("KUBERA_NEWS_FEATURE_HEADLINE_PLUS_SNIPPET_WEIGHT", "0.75")
        ),
        headline_only_weight=_parse_float(
            os.getenv("KUBERA_NEWS_FEATURE_HEADLINE_ONLY_WEIGHT", "0.5")
        ),
        use_confidence_in_article_weight=_parse_bool(
            os.getenv("KUBERA_NEWS_FEATURE_USE_CONFIDENCE_IN_ARTICLE_WEIGHT", "true")
        ),
    )

    pilot = PilotSettings(
        fallback_heavy_ratio_threshold=_parse_float(
            os.getenv("KUBERA_PILOT_FALLBACK_HEAVY_RATIO_THRESHOLD", "0.5")
        ),
    )

    return AppSettings(
        project=project,
        paths=paths,
        market=market,
        ticker=ticker,
        providers=providers,
        historical_data=historical_data,
        historical_features=historical_features,
        run=run,
        baseline_model=baseline_model,
        enhanced_model=enhanced_model,
        offline_evaluation=offline_evaluation,
        news_ingestion=news_ingestion,
        llm_extraction=llm_extraction,
        news_features=news_features,
        pilot=pilot,
    )


def settings_to_dict(
    settings: AppSettings,
    *,
    redact_secrets: bool = True,
) -> dict[str, Any]:
    """Convert settings into a JSON-safe dictionary."""

    return {
        "project": _serialize_dataclass(settings.project, redact_secrets=redact_secrets),
        "paths": _serialize_dataclass(settings.paths, redact_secrets=redact_secrets),
        "market": _serialize_dataclass(settings.market, redact_secrets=redact_secrets),
        "ticker": _serialize_dataclass(settings.ticker, redact_secrets=redact_secrets),
        "providers": _serialize_dataclass(
            settings.providers,
            redact_secrets=redact_secrets,
        ),
        "historical_data": _serialize_dataclass(
            settings.historical_data,
            redact_secrets=redact_secrets,
        ),
        "historical_features": _serialize_dataclass(
            settings.historical_features,
            redact_secrets=redact_secrets,
        ),
        "run": _serialize_dataclass(settings.run, redact_secrets=redact_secrets),
        "baseline_model": _serialize_dataclass(
            settings.baseline_model,
            redact_secrets=redact_secrets,
        ),
        "enhanced_model": _serialize_dataclass(
            settings.enhanced_model,
            redact_secrets=redact_secrets,
        ),
        "offline_evaluation": _serialize_dataclass(
            settings.offline_evaluation,
            redact_secrets=redact_secrets,
        ),
        "news_ingestion": _serialize_dataclass(
            settings.news_ingestion,
            redact_secrets=redact_secrets,
        ),
        "llm_extraction": _serialize_dataclass(
            settings.llm_extraction,
            redact_secrets=redact_secrets,
        ),
        "news_features": _serialize_dataclass(
            settings.news_features,
            redact_secrets=redact_secrets,
        ),
        "pilot": _serialize_dataclass(settings.pilot, redact_secrets=redact_secrets),
    }


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()

    env_repo_root = os.getenv("KUBERA_REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root).expanduser().resolve()

    return Path(__file__).resolve().parents[3]


def _resolve_subpath(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate

    resolved = candidate.resolve()
    if resolved == repo_root or resolved.is_relative_to(repo_root):
        return resolved

    raise SettingsError(
        f"Managed paths must stay inside the repo root: {resolved}"
    )


def _validate_path_settings(paths: PathSettings) -> None:
    for path_value in paths.managed_directories():
        if not path_value.is_relative_to(paths.repo_root):
            raise SettingsError(f"Managed path escapes repo root: {path_value}")


def _validate_provider_settings(providers: ProviderSettings) -> None:
    provider_pairs = (
        ("historical data", providers.historical_data_provider, providers.historical_data_api_key),
        ("news", providers.news_provider, providers.news_api_key),
        ("llm", providers.llm_provider, providers.llm_api_key),
    )

    for label, provider_name, api_key in provider_pairs:
        normalized_provider = provider_name.strip().lower()
        if normalized_provider in {
            "",
            "not_configured",
            "none",
            "disabled",
            "public",
            "yfinance",
        }:
            continue
        if not api_key:
            raise SettingsError(
                f"{label.capitalize()} provider '{provider_name}' requires an API key."
            )


def _parse_time(raw_value: str) -> time:
    parts = raw_value.split(":")
    if len(parts) != 2:
        raise SettingsError(f"Invalid time value: {raw_value}")

    try:
        hour, minute = (int(part) for part in parts)
        return time(hour=hour, minute=minute)
    except ValueError as exc:
        raise SettingsError(f"Invalid time value: {raw_value}") from exc


def _parse_csv(raw_value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not values:
        raise SettingsError("Expected at least one comma-separated value.")
    return values


def _parse_int(raw_value: str) -> int:
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Expected an integer value, got: {raw_value}") from exc


def _parse_float(raw_value: str) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Expected a float value, got: {raw_value}") from exc


def _parse_int_csv(raw_value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    except ValueError as exc:
        raise SettingsError(f"Expected comma-separated integers, got: {raw_value}") from exc


def _parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"Expected a boolean value, got: {raw_value}")


def _clean_optional(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    cleaned = raw_value.strip()
    return cleaned or None


def _serialize_dataclass(value: Any, *, redact_secrets: bool) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field_name in value.__dataclass_fields__:
        output[field_name] = _serialize_value(
            getattr(value, field_name),
            field_name=field_name,
            redact_secrets=redact_secrets,
        )
    return output


def _serialize_value(
    value: Any,
    *,
    field_name: str,
    redact_secrets: bool,
) -> Any:
    if redact_secrets and ("secret" in field_name or field_name.endswith("_api_key")):
        return REDACTED_VALUE if value else None

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, time):
        return value.isoformat(timespec="minutes")
    if isinstance(value, tuple):
        return [
            _serialize_value(item, field_name=field_name, redact_secrets=redact_secrets)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _serialize_value(
                item,
                field_name=key,
                redact_secrets=redact_secrets,
            )
            for key, item in value.items()
        }
    return value


def _validate_positive_sorted_windows(values: tuple[int, ...], label: str) -> None:
    if not values:
        raise SettingsError(f"{label} must not be empty.")
    if any(value < 1 for value in values):
        raise SettingsError(f"{label} must contain only positive integers.")
    if tuple(sorted(values)) != values:
        raise SettingsError(f"{label} must be sorted in ascending order.")
    if len(set(values)) != len(values):
        raise SettingsError(f"{label} must not contain duplicates.")


def _validate_model_split_ratios(
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    label: str,
) -> None:
    for ratio_name, ratio_value in (
        ("Train ratio", train_ratio),
        ("Validation ratio", validation_ratio),
        ("Test ratio", test_ratio),
    ):
        if ratio_value <= 0 or ratio_value >= 1:
            raise SettingsError(
                f"{label} {ratio_name.lower()} must be greater than 0 and less than 1."
            )
    if not math.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SettingsError(f"{label} split ratios must sum to 1.0.")
