"""Run metadata for Kubera pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kubera.config import AppSettings
from kubera.utils.git_utils import read_git_state
from kubera.utils.paths import PathManager


@dataclass(frozen=True)
class RunContext:
    """Metadata for a single local Kubera run."""

    run_id: str
    started_at_utc: datetime
    repo_root: Path
    run_directory: Path
    config_snapshot_path: Path
    log_file_path: Path
    git_commit: str | None
    git_is_dirty: bool | None


def create_run_context(
    settings: AppSettings,
    path_manager: PathManager,
    *,
    started_at: datetime | None = None,
) -> RunContext:
    """Create a run context and reserve its output paths."""

    started_at_utc = (started_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = build_run_id(
        path_manager,
        started_at_utc=started_at_utc,
        run_id_time_format=settings.run.run_id_time_format,
    )
    run_directory = path_manager.ensure_run_directory(run_id)
    git_commit, git_is_dirty = read_git_state(settings.paths.repo_root)
    return RunContext(
        run_id=run_id,
        started_at_utc=started_at_utc,
        repo_root=settings.paths.repo_root,
        run_directory=run_directory,
        config_snapshot_path=path_manager.build_config_snapshot_path(
            run_id,
            settings.run.config_snapshot_filename,
        ),
        log_file_path=path_manager.build_log_file_path(run_id),
        git_commit=git_commit,
        git_is_dirty=git_is_dirty,
    )


def build_run_id(
    path_manager: PathManager,
    *,
    started_at_utc: datetime,
    run_id_time_format: str,
) -> str:
    """Build a timestamp-first run id and avoid collisions inside local outputs."""

    base_run_id = started_at_utc.astimezone(timezone.utc).strftime(run_id_time_format)
    run_id = base_run_id
    suffix = 0

    while _run_id_exists(path_manager, run_id):
        suffix += 1
        run_id = f"{base_run_id}_{suffix:02d}"

    return run_id


def _run_id_exists(path_manager: PathManager, run_id: str) -> bool:
    return (
        (path_manager.settings.runs_dir / run_id).exists()
        or path_manager.build_log_file_path(run_id).exists()
    )
