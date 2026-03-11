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

    def build_article_fetch_cache_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.processed_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_article_fetch_cache.json"
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

    def build_raw_news_feature_data_path(
        self,
        ticker: str,
        run_id: str,
        artifact_variant: str | None = None,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        variant_suffix = build_artifact_variant_suffix(artifact_variant)
        return (
            self.settings.raw_dir
            / "news_features"
            / safe_ticker
            / f"{run_id}{variant_suffix}.json"
        )

    def build_news_feature_table_path(
        self,
        ticker: str,
        exchange: str,
        artifact_variant: str | None = None,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        variant_suffix = build_artifact_variant_suffix(artifact_variant)
        return (
            self.settings.features_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_news_features{variant_suffix}.csv"
        )

    def build_news_feature_metadata_path(
        self,
        ticker: str,
        exchange: str,
        artifact_variant: str | None = None,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        variant_suffix = build_artifact_variant_suffix(artifact_variant)
        return (
            self.settings.features_dir
            / "news"
            / f"{safe_ticker}_{safe_exchange}_news_features{variant_suffix}.metadata.json"
        )

    def build_merged_enhanced_dataset_path(
        self,
        ticker: str,
        exchange: str,
        artifact_variant: str | None = None,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        variant_suffix = build_artifact_variant_suffix(artifact_variant)
        return (
            self.settings.merged_features_dir
            / f"{safe_ticker}_{safe_exchange}_enhanced_dataset{variant_suffix}.csv"
        )

    def build_merged_enhanced_dataset_metadata_path(
        self,
        ticker: str,
        exchange: str,
        artifact_variant: str | None = None,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        variant_suffix = build_artifact_variant_suffix(artifact_variant)
        return (
            self.settings.merged_features_dir
            / f"{safe_ticker}_{safe_exchange}_enhanced_dataset{variant_suffix}.metadata.json"
        )

    def build_enhanced_model_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.enhanced_models_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_enhanced_model.pkl"
        )

    def build_enhanced_model_metadata_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.enhanced_models_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_enhanced_model.metadata.json"
        )

    def build_enhanced_predictions_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.enhanced_reports_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_enhanced_predictions.csv"
        )

    def build_enhanced_metrics_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.enhanced_reports_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_enhanced_metrics.json"
        )

    def build_enhanced_comparison_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.enhanced_reports_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_baseline_comparison.csv"
        )

    def build_enhanced_comparison_summary_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.enhanced_reports_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_baseline_comparison.json"
        )

    def build_offline_evaluation_predictions_path(
        self,
        ticker: str,
        exchange: str,
        prediction_mode: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        safe_mode = safe_path_token(prediction_mode)
        return (
            self.settings.evaluation_reports_dir
            / f"{safe_ticker}_{safe_exchange}_{safe_mode}_offline_evaluation_predictions.csv"
        )

    def build_offline_metrics_path(self, ticker: str, exchange: str) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.evaluation_reports_dir
            / f"{safe_ticker}_{safe_exchange}_offline_metrics.csv"
        )

    def build_offline_evaluation_summary_json_path(
        self,
        ticker: str,
        exchange: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.evaluation_reports_dir
            / f"{safe_ticker}_{safe_exchange}_offline_evaluation_summary.json"
        )

    def build_offline_evaluation_summary_markdown_path(
        self,
        ticker: str,
        exchange: str,
    ) -> Path:
        safe_ticker = safe_path_token(ticker)
        safe_exchange = safe_path_token(exchange)
        return (
            self.settings.evaluation_reports_dir
            / f"{safe_ticker}_{safe_exchange}_offline_evaluation_summary.md"
        )


def safe_path_token(value: str) -> str:
    """Sanitize a dynamic token before using it in a file or directory name."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not cleaned:
        raise ValueError("Expected a non-empty safe path token.")
    return cleaned


def build_artifact_variant_suffix(artifact_variant: str | None) -> str:
    """Format an optional artifact variant for stable derived filenames."""

    if artifact_variant is None:
        return ""
    return f"_{safe_path_token(artifact_variant)}"
