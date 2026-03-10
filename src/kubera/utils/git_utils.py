"""Small Git metadata helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess


def read_git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    """Return the local commit and dirty state when Git metadata is available."""

    commit = _run_git_command(repo_root, "rev-parse", "--short", "HEAD")
    if commit is None:
        return None, None

    dirty_output = _run_git_command(repo_root, "status", "--porcelain")
    return commit, bool(dirty_output)


def _run_git_command(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip()
