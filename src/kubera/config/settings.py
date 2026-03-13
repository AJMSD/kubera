"""Typed settings for Kubera."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import time
import json
import math
from pathlib import Path
import os
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


ALLOWED_PREDICTION_MODES = frozenset({"pre_market", "after_close", "both"})
ALLOWED_EVALUATION_HEADLINE_SPLITS = frozenset({"test"})
ALLOWED_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
ALLOWED_MODEL_TYPES = frozenset({"logistic_regression", "gradient_boosting"})
REDACTED_VALUE = "[redacted]"
EXCHANGE_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
TICKER_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.&_-]{0,24}$")
DEFAULT_EXCHANGE_CONFIGS = {
    "NSE": {
        "timezone_name": "Asia/Kolkata",
        "market_open": "09:15",
        "market_close": "15:30",
        "supported_prediction_modes": ("pre_market", "after_close"),
        "calendar_name": "NSE",
        "provider_symbol_suffixes": {
            "yahoo_finance": ".NS",
        },
    },
    "BSE": {
        "timezone_name": "Asia/Kolkata",
        "market_open": "09:15",
        "market_close": "15:30",
        "supported_prediction_modes": ("pre_market", "after_close"),
        "calendar_name": "BSE",
        "provider_symbol_suffixes": {
            "yahoo_finance": ".BO",
        },
    },
}
DEFAULT_TICKER_CATALOG = (
    {
        "symbol": "INFY",
        "exchange": "NSE",
        "company_name": "Infosys Limited",
        "search_aliases": ("INFY", "Infosys", "Infosys Limited"),
    },
    {
        "symbol": "INFY",
        "exchange": "BSE",
        "company_name": "Infosys Limited",
        "search_aliases": ("INFY", "Infosys", "Infosys Limited"),
    },
    {
        "symbol": "TCS",
        "exchange": "NSE",
        "company_name": "Tata Consultancy Services",
        "search_aliases": ("TCS", "Tata Consultancy Services"),
    },
    {
        "symbol": "TCS",
        "exchange": "BSE",
        "company_name": "Tata Consultancy Services",
        "search_aliases": ("TCS", "Tata Consultancy Services"),
    },
)
DEFAULT_GEMINI_RECOVERY_MODEL_POOL = (
    {
        "model": "gemini-2.5-flash",
        "supports_url_context": True,
        "supports_google_search": False,
        "requests_per_minute_limit": 0,
        "requests_per_day_limit": 0,
    },
)


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
    final_review_reports_dir: Path
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
            self.final_review_reports_dir,
            self.merged_features_dir,
        )


@dataclass(frozen=True)
class MarketSettings:
    exchange_code: str
    calendar_name: str
    timezone_name: str
    market_open: time
    market_close: time
    supported_prediction_modes: tuple[str, ...]
    local_holiday_override_path: Path

    def __post_init__(self) -> None:
        if not EXCHANGE_CODE_PATTERN.fullmatch(self.exchange_code.strip().upper()):
            raise SettingsError(
                f"Unsupported exchange code format: {self.exchange_code}"
            )
        if not self.calendar_name.strip():
            raise SettingsError("Market calendar name must not be empty.")
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
        if not TICKER_SYMBOL_PATTERN.fullmatch(self.symbol.strip().upper()):
            raise SettingsError(
                f"Ticker symbol contains unsupported characters: {self.symbol}"
            )
        if not self.exchange.strip():
            raise SettingsError("Exchange must not be empty.")
        if not EXCHANGE_CODE_PATTERN.fullmatch(self.exchange.strip().upper()):
            raise SettingsError(
                f"Exchange code contains unsupported characters: {self.exchange}"
            )
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
    alphavantage_api_key: str | None
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
    macd_fast_span: int
    macd_slow_span: int
    macd_signal_span: int
    rolling_year_window: int
    include_day_of_week: bool
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
        if self.macd_fast_span < 1:
            raise SettingsError("MACD fast span must be at least one.")
        if self.macd_slow_span <= self.macd_fast_span:
            raise SettingsError("MACD slow span must be greater than the fast span.")
        if self.macd_signal_span < 1:
            raise SettingsError("MACD signal span must be at least one.")
        if self.rolling_year_window < 1:
            raise SettingsError("Rolling year window must be at least one.")


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
        if self.model_type not in ALLOWED_MODEL_TYPES:
            raise SettingsError(
                f"Unsupported baseline model type: {self.model_type}"
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
    enable_google_news_rss: bool
    enable_nse_announcements: bool

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
    default_pre_market_run_time: time
    default_after_close_run_time: time
    runtime_warning_seconds: float
    historical_incremental_overlap_days: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.fallback_heavy_ratio_threshold):
            raise SettingsError("Pilot fallback-heavy ratio threshold must be finite.")
        if not 0.0 <= self.fallback_heavy_ratio_threshold <= 1.0:
            raise SettingsError(
                "Pilot fallback-heavy ratio threshold must be between 0 and 1."
            )
        if self.default_pre_market_run_time >= self.default_after_close_run_time:
            raise SettingsError(
                "Pilot pre-market run time must be earlier than the after-close run time."
            )
        if not math.isfinite(self.runtime_warning_seconds):
            raise SettingsError("Pilot runtime warning seconds must be finite.")
        if self.runtime_warning_seconds <= 0:
            raise SettingsError("Pilot runtime warning seconds must be greater than 0.")
        if self.historical_incremental_overlap_days < 1:
            raise SettingsError(
                "Pilot historical incremental overlap days must be at least 1."
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
        if self.model_type not in ALLOWED_MODEL_TYPES:
            raise SettingsError(
                f"Unsupported enhanced model type: {self.model_type}"
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
    @dataclass(frozen=True)
    class GeminiRecoveryModelSettings:
        model: str
        supports_url_context: bool
        supports_google_search: bool
        requests_per_minute_limit: int
        requests_per_day_limit: int

        def __post_init__(self) -> None:
            if not self.model.strip():
                raise SettingsError("Recovery model names must not be empty.")
            if self.requests_per_minute_limit < 0:
                raise SettingsError("Recovery model RPM limits must not be negative.")
            if self.requests_per_day_limit < 0:
                raise SettingsError("Recovery model RPD limits must not be negative.")

    model: str
    request_timeout_seconds: int
    retry_attempts: int
    retry_base_delay_seconds: float
    max_input_chars: int
    prompt_version: str
    recovery_url_context_enabled: bool
    recovery_google_search_enabled: bool
    recovery_max_articles_per_run: int
    recovery_model_pool: tuple[GeminiRecoveryModelSettings, ...]

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
        if self.recovery_max_articles_per_run < 0:
            raise SettingsError(
                "LLM recovery max articles per run must not be negative."
            )


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
    ticker_catalog_path = _resolve_ticker_catalog_path(
        repo_root=resolved_repo_root,
        runtime_config_dir=runtime_config_dir,
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
        final_review_reports_dir=artifacts_dir / "reports" / "final_review",
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

    exchange_code = normalize_exchange_code(os.getenv("KUBERA_EXCHANGE", "NSE"))
    market_defaults = get_exchange_defaults(exchange_code)
    market = MarketSettings(
        exchange_code=exchange_code,
        calendar_name=str(market_defaults["calendar_name"]),
        timezone_name=os.getenv(
            "KUBERA_MARKET_TIMEZONE",
            str(market_defaults["timezone_name"]),
        ),
        market_open=_parse_time(
            os.getenv("KUBERA_MARKET_OPEN", str(market_defaults["market_open"]))
        ),
        market_close=_parse_time(
            os.getenv("KUBERA_MARKET_CLOSE", str(market_defaults["market_close"]))
        ),
        supported_prediction_modes=_parse_csv(
            os.getenv(
                "KUBERA_SUPPORTED_PREDICTION_MODES",
                ",".join(market_defaults["supported_prediction_modes"]),
            )
        ),
        local_holiday_override_path=holiday_override_path,
    )

    ticker_catalog = load_ticker_catalog(catalog_path=ticker_catalog_path)
    ticker = resolve_ticker_settings(
        symbol=os.getenv("KUBERA_TICKER", "INFY"),
        exchange=exchange_code,
        ticker_catalog=ticker_catalog,
        company_name_override=_clean_optional(os.getenv("KUBERA_COMPANY_NAME")),
        search_aliases_override=_clean_optional(os.getenv("KUBERA_NEWS_ALIASES")),
        yahoo_symbol_override=_clean_optional(os.getenv("KUBERA_YAHOO_TICKER")),
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
        alphavantage_api_key=_clean_optional(os.getenv("KUBERA_ALPHAVANTAGE_API_KEY")),
        llm_api_key=_clean_optional(os.getenv("KUBERA_LLM_API_KEY")),
    )
    _validate_provider_settings(providers)

    historical_data = HistoricalDataSettings(
        default_lookback_months=_parse_int(
            os.getenv("KUBERA_HISTORICAL_LOOKBACK_MONTHS", "36")
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
        macd_fast_span=_parse_int(os.getenv("KUBERA_HISTORICAL_MACD_FAST_SPAN", "12")),
        macd_slow_span=_parse_int(os.getenv("KUBERA_HISTORICAL_MACD_SLOW_SPAN", "26")),
        macd_signal_span=_parse_int(os.getenv("KUBERA_HISTORICAL_MACD_SIGNAL_SPAN", "9")),
        rolling_year_window=_parse_int(
            os.getenv("KUBERA_HISTORICAL_ROLLING_YEAR_WINDOW", "252")
        ),
        include_day_of_week=_parse_bool(
            os.getenv("KUBERA_HISTORICAL_INCLUDE_DAY_OF_WEEK", "true")
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
        lookback_days=_parse_int(os.getenv("KUBERA_NEWS_LOOKBACK_DAYS", "90")),
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
        enable_google_news_rss=_parse_bool(
            os.getenv("KUBERA_NEWS_ENABLE_GOOGLE_NEWS_RSS", "true")
        ),
        enable_nse_announcements=_parse_bool(
            os.getenv("KUBERA_NEWS_ENABLE_NSE_ANNOUNCEMENTS", "true")
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
        recovery_url_context_enabled=_parse_bool(
            os.getenv("KUBERA_LLM_RECOVERY_URL_CONTEXT_ENABLED", "true")
        ),
        recovery_google_search_enabled=_parse_bool(
            os.getenv("KUBERA_LLM_RECOVERY_GOOGLE_SEARCH_ENABLED", "false")
        ),
        recovery_max_articles_per_run=_parse_int(
            os.getenv("KUBERA_LLM_RECOVERY_MAX_ARTICLES_PER_RUN", "3")
        ),
        recovery_model_pool=_parse_gemini_recovery_model_pool(
            os.getenv("KUBERA_LLM_RECOVERY_MODEL_POOL_JSON")
        ),
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
        default_pre_market_run_time=_parse_time(
            os.getenv("KUBERA_PILOT_DEFAULT_PRE_MARKET_RUN_TIME", "08:05")
        ),
        default_after_close_run_time=_parse_time(
            os.getenv("KUBERA_PILOT_DEFAULT_AFTER_CLOSE_RUN_TIME", "16:15")
        ),
        runtime_warning_seconds=_parse_float(
            os.getenv("KUBERA_PILOT_RUNTIME_WARNING_SECONDS", "120.0")
        ),
        historical_incremental_overlap_days=_parse_int(
            os.getenv("KUBERA_PILOT_HISTORICAL_INCREMENTAL_OVERLAP_DAYS", "5")
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


def resolve_runtime_settings(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
) -> AppSettings:
    """Resolve a runtime ticker or exchange override through the shared catalog."""

    if ticker is None and exchange is None:
        return settings

    resolved_symbol = normalize_ticker_symbol(ticker or settings.ticker.symbol)
    resolved_exchange = normalize_exchange_code(exchange or settings.ticker.exchange)
    ticker_catalog = load_ticker_catalog(
        catalog_path=_resolve_ticker_catalog_path(
            repo_root=settings.paths.repo_root,
            runtime_config_dir=settings.paths.runtime_config_dir,
        )
    )
    updated_ticker = resolve_ticker_settings(
        symbol=resolved_symbol,
        exchange=resolved_exchange,
        ticker_catalog=ticker_catalog,
        fallback_ticker=settings.ticker,
    )
    updated_market = replace(
        settings.market,
        exchange_code=resolved_exchange,
        calendar_name=resolve_exchange_calendar_name(resolved_exchange),
    )
    return replace(
        settings,
        market=updated_market,
        ticker=updated_ticker,
    )


def normalize_ticker_symbol(raw_value: str) -> str:
    """Normalize one ticker symbol and reject unsupported characters."""

    normalized = raw_value.strip().upper()
    if not normalized:
        raise SettingsError("Ticker symbol must not be empty.")
    if not TICKER_SYMBOL_PATTERN.fullmatch(normalized):
        raise SettingsError(
            f"Ticker symbol contains unsupported characters: {raw_value}"
        )
    return normalized


def normalize_exchange_code(raw_value: str) -> str:
    """Normalize one exchange code and require a supported exchange."""

    normalized = raw_value.strip().upper()
    if not normalized:
        raise SettingsError("Exchange must not be empty.")
    if not EXCHANGE_CODE_PATTERN.fullmatch(normalized):
        raise SettingsError(
            f"Exchange code contains unsupported characters: {raw_value}"
        )
    if normalized not in DEFAULT_EXCHANGE_CONFIGS:
        raise SettingsError(f"Unsupported exchange code: {normalized}")
    return normalized


def get_exchange_defaults(exchange_code: str) -> dict[str, Any]:
    """Return the built-in defaults for one supported exchange."""

    normalized_exchange = normalize_exchange_code(exchange_code)
    return DEFAULT_EXCHANGE_CONFIGS[normalized_exchange]


def resolve_exchange_calendar_name(exchange_code: str) -> str:
    """Return the exchange calendar name used by trading-day helpers."""

    return str(get_exchange_defaults(exchange_code)["calendar_name"])


def build_provider_symbol(
    ticker: str,
    exchange: str,
    *,
    provider_name: str = "yahoo_finance",
) -> str:
    """Build one provider symbol from the canonical ticker and exchange."""

    normalized_symbol = normalize_ticker_symbol(ticker)
    exchange_defaults = get_exchange_defaults(exchange)
    suffix = str(
        exchange_defaults.get("provider_symbol_suffixes", {}).get(provider_name, "")
    )
    return f"{normalized_symbol}{suffix}"


def resolve_ticker_settings(
    *,
    symbol: str,
    exchange: str,
    ticker_catalog: dict[tuple[str, str], dict[str, Any]],
    company_name_override: str | None = None,
    search_aliases_override: str | tuple[str, ...] | None = None,
    yahoo_symbol_override: str | None = None,
    fallback_ticker: TickerSettings | None = None,
) -> TickerSettings:
    """Resolve one full ticker configuration from the catalog and overrides."""

    normalized_symbol = normalize_ticker_symbol(symbol)
    normalized_exchange = normalize_exchange_code(exchange)
    catalog_entry = resolve_catalog_entry(
        ticker_catalog,
        symbol=normalized_symbol,
        exchange=normalized_exchange,
        fallback_ticker=fallback_ticker,
    )

    company_name = (company_name_override or catalog_entry.get("company_name") or "").strip()
    if not company_name:
        raise SettingsError(
            "Ticker metadata is missing a company name. Add the ticker to the catalog or set KUBERA_COMPANY_NAME."
        )

    search_aliases = resolve_search_aliases(
        normalized_symbol,
        company_name,
        search_aliases_override=search_aliases_override,
        catalog_entry=catalog_entry,
        fallback_ticker=fallback_ticker,
    )
    provider_symbol_map = resolve_provider_symbol_map(
        symbol=normalized_symbol,
        exchange=normalized_exchange,
        catalog_entry=catalog_entry,
        yahoo_symbol_override=yahoo_symbol_override,
        fallback_ticker=fallback_ticker,
    )
    return TickerSettings(
        symbol=normalized_symbol,
        exchange=normalized_exchange,
        company_name=company_name,
        search_aliases=search_aliases,
        provider_symbol_map=provider_symbol_map,
    )


def resolve_catalog_entry(
    ticker_catalog: dict[tuple[str, str], dict[str, Any]],
    *,
    symbol: str,
    exchange: str,
    fallback_ticker: TickerSettings | None,
) -> dict[str, Any]:
    """Resolve the most relevant catalog entry for a symbol and exchange."""

    catalog_entry = ticker_catalog.get((symbol, exchange))
    if catalog_entry is not None:
        return dict(catalog_entry)

    if fallback_ticker is not None and fallback_ticker.symbol == symbol:
        return {
            "company_name": fallback_ticker.company_name,
            "search_aliases": tuple(fallback_ticker.search_aliases),
            "provider_symbol_map": dict(fallback_ticker.provider_symbol_map),
        }

    same_symbol_entries = [
        value
        for (entry_symbol, _entry_exchange), value in ticker_catalog.items()
        if entry_symbol == symbol
    ]
    if same_symbol_entries:
        return dict(same_symbol_entries[0])

    return {}


def resolve_search_aliases(
    symbol: str,
    company_name: str,
    *,
    search_aliases_override: str | tuple[str, ...] | None,
    catalog_entry: dict[str, Any],
    fallback_ticker: TickerSettings | None,
) -> tuple[str, ...]:
    """Resolve the search alias list for Stage 5 discovery."""

    if isinstance(search_aliases_override, tuple):
        return search_aliases_override
    if isinstance(search_aliases_override, str):
        return _parse_csv(search_aliases_override)

    raw_catalog_aliases = catalog_entry.get("search_aliases")
    if isinstance(raw_catalog_aliases, tuple) and raw_catalog_aliases:
        return raw_catalog_aliases
    if isinstance(raw_catalog_aliases, list):
        aliases = tuple(str(value).strip() for value in raw_catalog_aliases if str(value).strip())
        if aliases:
            return aliases

    if fallback_ticker is not None and fallback_ticker.symbol == symbol:
        return tuple(fallback_ticker.search_aliases)

    return tuple(dict.fromkeys((symbol, company_name)))


def resolve_provider_symbol_map(
    *,
    symbol: str,
    exchange: str,
    catalog_entry: dict[str, Any],
    yahoo_symbol_override: str | None,
    fallback_ticker: TickerSettings | None,
) -> dict[str, str]:
    """Resolve provider-specific symbol mappings for the active ticker."""

    provider_symbol_map: dict[str, str] = {}
    if fallback_ticker is not None and fallback_ticker.symbol == symbol:
        for key, value in fallback_ticker.provider_symbol_map.items():
            cleaned_key = str(key)
            cleaned_value = str(value).strip()
            if not cleaned_value:
                continue
            if cleaned_key == "yahoo_finance" and fallback_ticker.exchange != exchange:
                continue
            provider_symbol_map[cleaned_key] = cleaned_value

    raw_catalog_provider_map = catalog_entry.get("provider_symbol_map", {})
    if isinstance(raw_catalog_provider_map, dict):
        provider_symbol_map.update(
            {
                str(key): str(value).strip()
                for key, value in raw_catalog_provider_map.items()
                if str(value).strip()
            }
        )

    provider_symbol_map["yahoo_finance"] = (
        yahoo_symbol_override.strip()
        if yahoo_symbol_override is not None and yahoo_symbol_override.strip()
        else provider_symbol_map.get("yahoo_finance")
        or build_provider_symbol(symbol, exchange, provider_name="yahoo_finance")
    )
    return provider_symbol_map


def load_ticker_catalog(
    *,
    catalog_path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the ticker catalog from built-in defaults plus an optional JSON file."""

    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_entry in DEFAULT_TICKER_CATALOG:
        _upsert_catalog_entry(entries, raw_entry)

    if catalog_path is None:
        return entries

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Ticker catalog is not valid JSON: {catalog_path}") from exc

    raw_entries: Any
    if isinstance(payload, dict):
        raw_entries = payload.get("tickers")
    else:
        raw_entries = payload
    if not isinstance(raw_entries, list):
        raise SettingsError(
            "Ticker catalog must be a JSON list or an object with a 'tickers' list."
        )

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise SettingsError("Ticker catalog entries must be JSON objects.")
        _upsert_catalog_entry(entries, raw_entry)

    return entries


def _upsert_catalog_entry(
    entries: dict[tuple[str, str], dict[str, Any]],
    raw_entry: dict[str, Any],
) -> None:
    symbol = normalize_ticker_symbol(str(raw_entry.get("symbol", "")))
    exchange = normalize_exchange_code(str(raw_entry.get("exchange", "")))
    company_name = str(raw_entry.get("company_name", "")).strip()
    if not company_name:
        raise SettingsError(
            f"Ticker catalog entry is missing company_name for {symbol} on {exchange}."
        )

    raw_aliases = raw_entry.get("search_aliases")
    if raw_aliases is None:
        search_aliases = tuple(dict.fromkeys((symbol, company_name)))
    elif isinstance(raw_aliases, (list, tuple)):
        search_aliases = tuple(
            str(value).strip()
            for value in raw_aliases
            if str(value).strip()
        )
    else:
        raise SettingsError(
            f"Ticker catalog search_aliases must be a list for {symbol} on {exchange}."
        )
    if not search_aliases:
        raise SettingsError(
            f"Ticker catalog entry needs at least one alias for {symbol} on {exchange}."
        )

    provider_symbol_map: dict[str, str] = {}
    raw_provider_symbol_map = raw_entry.get("provider_symbol_map", {})
    if raw_provider_symbol_map not in ({}, None):
        if not isinstance(raw_provider_symbol_map, dict):
            raise SettingsError(
                f"Ticker catalog provider_symbol_map must be an object for {symbol} on {exchange}."
            )
        provider_symbol_map.update(
            {
                str(key): str(value).strip()
                for key, value in raw_provider_symbol_map.items()
                if str(value).strip()
            }
        )
    provider_symbol_map["yahoo_finance"] = (
        provider_symbol_map.get("yahoo_finance")
        or build_provider_symbol(symbol, exchange, provider_name="yahoo_finance")
    )

    entries[(symbol, exchange)] = {
        "company_name": company_name,
        "search_aliases": search_aliases,
        "provider_symbol_map": provider_symbol_map,
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


def _resolve_ticker_catalog_path(*, repo_root: Path, runtime_config_dir: Path) -> Path | None:
    raw_path = _clean_optional(os.getenv("KUBERA_TICKER_CATALOG_PATH"))
    if raw_path is None:
        default_path = runtime_config_dir / "ticker_catalog.json"
        return default_path if default_path.exists() else None

    resolved_path = _resolve_subpath(repo_root, raw_path)
    if not resolved_path.exists():
        raise SettingsError(f"Ticker catalog path does not exist: {resolved_path}")
    return resolved_path


def _validate_path_settings(paths: PathSettings) -> None:
    for path_value in paths.managed_directories():
        if not path_value.is_relative_to(paths.repo_root):
            raise SettingsError(f"Managed path escapes repo root: {path_value}")


def _validate_provider_settings(providers: ProviderSettings) -> None:
    provider_pairs = (
        ("historical data", providers.historical_data_provider, providers.historical_data_api_key),
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

    normalized_news_provider = providers.news_provider.strip().lower()
    if normalized_news_provider == "marketaux" and not providers.news_api_key:
        raise SettingsError("News provider 'marketaux' requires an API key.")
    if normalized_news_provider == "alphavantage" and not providers.alphavantage_api_key:
        raise SettingsError("News provider 'alphavantage' requires KUBERA_ALPHAVANTAGE_API_KEY.")


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


def _parse_gemini_recovery_model_pool(
    raw_value: str | None,
) -> tuple[LlmExtractionSettings.GeminiRecoveryModelSettings, ...]:
    if raw_value is None or not raw_value.strip():
        raw_models = DEFAULT_GEMINI_RECOVERY_MODEL_POOL
    else:
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise SettingsError(
                "KUBERA_LLM_RECOVERY_MODEL_POOL_JSON must be valid JSON."
            ) from exc
        if not isinstance(parsed, list):
            raise SettingsError(
                "KUBERA_LLM_RECOVERY_MODEL_POOL_JSON must decode to a list."
            )
        raw_models = tuple(parsed)

    recovery_models: list[LlmExtractionSettings.GeminiRecoveryModelSettings] = []
    for entry in raw_models:
        if not isinstance(entry, dict):
            raise SettingsError(
                "Every recovery model configuration must be a JSON object."
            )
        recovery_models.append(
            LlmExtractionSettings.GeminiRecoveryModelSettings(
                model=str(entry.get("model", "")).strip(),
                supports_url_context=_coerce_bool_value(
                    entry.get("supports_url_context", entry.get("enable_url_context", False)),
                    label="Recovery model supports_url_context",
                ),
                supports_google_search=_coerce_bool_value(
                    entry.get(
                        "supports_google_search",
                        entry.get("enable_google_search", False),
                    ),
                    label="Recovery model supports_google_search",
                ),
                requests_per_minute_limit=_coerce_int_value(
                    entry.get("requests_per_minute_limit", entry.get("rpm_limit", 0)),
                    label="Recovery model requests_per_minute_limit",
                ),
                requests_per_day_limit=_coerce_int_value(
                    entry.get("requests_per_day_limit", entry.get("rpd_limit", 0)),
                    label="Recovery model requests_per_day_limit",
                ),
            )
        )
    return tuple(recovery_models)


def _coerce_bool_value(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{label} must be a boolean.")


def _coerce_int_value(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{label} must be an integer.") from exc


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

    if hasattr(value, "__dataclass_fields__"):
        return _serialize_dataclass(value, redact_secrets=redact_secrets)
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
