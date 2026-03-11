"""Path management for Kubera runtime directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from kubera.config import PathSettings


@dataclass(frozen=True)
class PathManager:
    """Resolve and create managed runtime directories."""

    settings: PathSettings

    def managed_directories(self) -> tuple[Path, ...]:
        return self.settings.managed_directories()

    def ensure_managed_directories(self) -> None:
        for directory in self.managed_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_run_directory(self, run_id: str) -> Path:
        run_directory = self.settings.runs_dir / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory

    def build_config_snapshot_path(self, run_id: str, filename: str) -> Path:
        return self.settings.runs_dir / run_id / filename

    def build_log_file_path(self, run_id: str) -> Path:
        return self.settings.logs_dir / f"{run_id}.log"

    def build_raw_market_data_path(self, ticker: str, run_id: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        return self.settings.raw_dir / "market_data" / safe_ticker / f"{run_id}.json"

    def build_processed_market_data_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "market_data"
            / f"{safe_ticker}_{safe_exchange}_daily.csv"
        )

    def build_processed_market_data_metadata_path(
        self,
        ticker: str,
        exchange: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "market_data"
            / f"{safe_ticker}_{safe_exchange}_daily.metadata.json"
        )

    def build_historical_feature_table_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.features_dir
            / "historical"
            / f"{safe_ticker}_{safe_exchange}_historical_features.csv"
        )

    def build_historical_feature_metadata_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.features_dir
            / "historical"
            / f"{safe_ticker}_{safe_exchange}_historical_features.metadata.json"
        )

    def build_baseline_model_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.baseline_models_dir
            / f"{safe_ticker}_{safe_exchange}_baseline_model.pkl"
        )

    def build_baseline_model_metadata_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.baseline_models_dir
            / f"{safe_ticker}_{safe_exchange}_baseline_model.metadata.json"
        )

    def build_baseline_predictions_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.baseline_reports_dir
            / f"{safe_ticker}_{safe_exchange}_baseline_predictions.csv"
        )

    def build_baseline_metrics_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.baseline_reports_dir
            / f"{safe_ticker}_{safe_exchange}_baseline_metrics.json"
        )

    def build_raw_news_data_path(self, ticker: str, run_id: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        return self.settings.raw_dir / "news" / safe_ticker / f"{run_id}.json"

    def build_processed_news_data_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_news.csv"
        )

    def build_processed_news_metadata_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_news.metadata.json"
        )

    def build_raw_llm_data_path(self, ticker: str, run_id: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        return self.settings.raw_dir / "llm" / safe_ticker / f"{run_id}.json"

    def build_processed_llm_extractions_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_llm_extractions.csv"
        )

    def build_processed_llm_extractions_metadata_path(
        self,
        ticker: str,
        exchange: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_llm_extractions.metadata.json"
        )

    def build_processed_llm_extraction_failures_path(
        self,
        ticker: str,
        exchange: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_llm_extraction_failures.json"
        )


def safe_path_token(value: str) -> str:
    """Sanitize a dynamic token before using it in a file or directory name."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not cleaned:
        raise ValueError("Expected a non-empty safe path token.")
    return cleaned
