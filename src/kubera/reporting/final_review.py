"""Stage 11 final review helpers and workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kubera.config import AppSettings, load_settings
from kubera.reporting.offline_evaluation import evaluate_offline
from kubera.utils.paths import PathManager


class FinalReviewError(RuntimeError):
    """Raised when the final review workflow cannot continue."""


@dataclass(frozen=True)
class OfflineEvaluationArtifacts:
    """Loaded Stage 9 artifacts reused by the final review workflow."""

    metrics_path: Path
    summary_json_path: Path
    summary_markdown_path: Path
    metrics_frame: pd.DataFrame
    summary_payload: dict[str, Any]
    refreshed: bool


def resolve_offline_evaluation_artifacts(
    settings: AppSettings,
    *,
    refresh_offline_evaluation: bool = False,
) -> OfflineEvaluationArtifacts:
    """Load the Stage 9 artifacts, refreshing them once when needed."""

    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    metrics_path = path_manager.build_offline_metrics_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    summary_json_path = path_manager.build_offline_evaluation_summary_json_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    summary_markdown_path = path_manager.build_offline_evaluation_summary_markdown_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )

    refreshed = False
    if refresh_offline_evaluation or not offline_evaluation_artifacts_exist(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    ):
        evaluate_offline(settings)
        refreshed = True

    if not offline_evaluation_artifacts_exist(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    ):
        raise FinalReviewError(
            "Stage 9 offline evaluation artifacts are unavailable after refresh."
        )

    metrics_frame = load_required_csv(
        metrics_path,
        artifact_label="Stage 9 offline metrics",
    )
    summary_payload = load_required_json(
        summary_json_path,
        artifact_label="Stage 9 offline summary JSON",
    )
    return OfflineEvaluationArtifacts(
        metrics_path=metrics_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        metrics_frame=metrics_frame,
        summary_payload=summary_payload,
        refreshed=refreshed,
    )


def offline_evaluation_artifacts_exist(
    *,
    metrics_path: Path,
    summary_json_path: Path,
    summary_markdown_path: Path,
) -> bool:
    """Return True when the required Stage 9 outputs exist."""

    return metrics_path.exists() and summary_json_path.exists() and summary_markdown_path.exists()


def load_required_csv(path: Path, *, artifact_label: str) -> pd.DataFrame:
    """Load one required CSV artifact or fail clearly."""

    if not path.exists():
        raise FinalReviewError(f"{artifact_label} does not exist: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise FinalReviewError(f"{artifact_label} is empty: {path}") from exc


def load_required_json(path: Path, *, artifact_label: str) -> dict[str, Any]:
    """Load one required JSON artifact or fail clearly."""

    if not path.exists():
        raise FinalReviewError(f"{artifact_label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalReviewError(f"{artifact_label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalReviewError(f"{artifact_label} must contain a JSON object: {path}")
    return payload


def parse_review_date(raw_value: str) -> date:
    """Parse a final-review date value."""

    return date.fromisoformat(raw_value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse final-review CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Kubera Stage 11 final review.")
    parser.add_argument(
        "--pilot-start-date",
        required=True,
        help="Pilot market-session start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--pilot-end-date",
        required=True,
        help="Pilot market-session end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--refresh-offline-evaluation",
        action="store_true",
        help="Rebuild Stage 9 outputs before writing the final review.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 11 CLI."""

    args = parse_args(argv)
    settings = load_settings()
    raise FinalReviewError(
        "Stage 11 final review workflow is not implemented yet. "
        "Use the loader helpers directly until the workflow lands."
    )
