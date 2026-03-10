"""Path management for Kubera runtime directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
