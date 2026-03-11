from __future__ import annotations

import json
from datetime import datetime, timezone

from kubera.config import load_settings, settings_to_dict
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.serialization import write_json_file


def test_path_manager_creates_managed_directories(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    path_manager.ensure_managed_directories()

    for directory in path_manager.managed_directories():
        assert directory.exists()
        assert directory.is_dir()
        assert directory.is_relative_to(isolated_repo)


def test_run_ids_are_timestamp_first_and_unique(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    started_at = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    first_context = create_run_context(settings, path_manager, started_at=started_at)
    second_context = create_run_context(settings, path_manager, started_at=started_at)

    assert first_context.run_id == "20260310_120000"
    assert second_context.run_id == "20260310_120000_01"
    assert first_context.config_snapshot_path.parent == first_context.run_directory


def test_config_snapshot_path_is_writable(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    context = create_run_context(settings, path_manager)

    write_json_file(context.config_snapshot_path, settings_to_dict(settings))

    payload = json.loads(context.config_snapshot_path.read_text(encoding="utf-8"))
    assert payload["project"]["name"] == "Kubera"


def test_path_manager_builds_baseline_artifact_paths(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    assert path_manager.build_baseline_model_path("INFY", "NSE") == (
        isolated_repo
        / "artifacts"
        / "models"
        / "baseline"
        / "INFY_NSE_baseline_model.pkl"
    )
    assert path_manager.build_baseline_predictions_path("INFY", "NSE") == (
        isolated_repo
        / "artifacts"
        / "reports"
        / "baseline"
        / "INFY_NSE_baseline_predictions.csv"
    )


def test_path_manager_builds_news_artifact_paths(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    assert path_manager.build_raw_news_data_path("INFY", "20260311_120000") == (
        isolated_repo
        / "data"
        / "raw"
        / "news"
        / "INFY"
        / "20260311_120000.json"
    )
    assert path_manager.build_processed_news_data_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_news.csv"
    )
    assert path_manager.build_processed_news_metadata_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_news.metadata.json"
    )


def test_path_manager_builds_llm_artifact_paths(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    assert path_manager.build_raw_llm_data_path("INFY", "20260311_120000") == (
        isolated_repo
        / "data"
        / "raw"
        / "llm"
        / "INFY"
        / "20260311_120000.json"
    )
    assert path_manager.build_processed_llm_extractions_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_llm_extractions.csv"
    )
    assert path_manager.build_processed_llm_extractions_metadata_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_llm_extractions.metadata.json"
    )
    assert path_manager.build_processed_llm_extraction_failures_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "processed"
        / "news"
        / "INFY_NSE_llm_extraction_failures.json"
    )


def test_path_manager_builds_news_feature_artifact_paths(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)

    assert path_manager.build_raw_news_feature_data_path("INFY", "20260311_120000") == (
        isolated_repo
        / "data"
        / "raw"
        / "news_features"
        / "INFY"
        / "20260311_120000.json"
    )
    assert path_manager.build_news_feature_table_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "features"
        / "news"
        / "INFY_NSE_news_features.csv"
    )
    assert path_manager.build_news_feature_metadata_path("INFY", "NSE") == (
        isolated_repo
        / "data"
        / "features"
        / "news"
        / "INFY_NSE_news_features.metadata.json"
    )
