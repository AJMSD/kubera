"""Hashing helpers for persisted artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_sha256(path: Path) -> str:
    """Hash a file so persisted artifacts can record their exact source inputs."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
