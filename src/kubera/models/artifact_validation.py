"""Shared artifact freshness validation for Kubera model and reporting stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from kubera.features.historical_features import FEATURE_FORMULA_VERSION as HISTORICAL_FORMULA_VERSION
from kubera.features.news_features import FEATURE_FORMULA_VERSION as NEWS_FORMULA_VERSION


ErrorFactory = Callable[[str], Exception]


def validate_historical_feature_artifact_metadata(
    metadata: Mapping[str, Any],
    *,
    metadata_path: Path,
    error_factory: ErrorFactory,
) -> None:
    """Require the active Stage 3 formula version before downstream reuse."""

    require_formula_version(
        metadata=metadata,
        metadata_path=metadata_path,
        artifact_label="Historical feature",
        expected_formula_version=HISTORICAL_FORMULA_VERSION,
        rerun_command="python -m kubera.features.historical_features --force",
        error_factory=error_factory,
    )


def validate_news_feature_artifact_metadata(
    metadata: Mapping[str, Any],
    *,
    metadata_path: Path,
    error_factory: ErrorFactory,
) -> None:
    """Require the active Stage 7 formula version before downstream reuse."""

    require_formula_version(
        metadata=metadata,
        metadata_path=metadata_path,
        artifact_label="News feature",
        expected_formula_version=NEWS_FORMULA_VERSION,
        rerun_command="python -m kubera.features.news_features --force",
        error_factory=error_factory,
    )


def require_formula_version(
    *,
    metadata: Mapping[str, Any],
    metadata_path: Path,
    artifact_label: str,
    expected_formula_version: str,
    rerun_command: str,
    error_factory: ErrorFactory,
) -> None:
    """Fail fast when a persisted artifact was built with an older formula version."""

    actual_formula_version = normalize_formula_version(metadata.get("formula_version"))
    if actual_formula_version == expected_formula_version:
        return

    actual_text = actual_formula_version if actual_formula_version is not None else "missing"
    raise error_factory(
        f"{artifact_label} artifact is stale. Expected formula_version "
        f"{expected_formula_version}, found {actual_text}, metadata={metadata_path}. "
        f"Rebuild with: {rerun_command}"
    )


def normalize_formula_version(value: Any) -> str | None:
    """Normalize one metadata formula version to a simple string value."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
