"""JSON serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from kubera.config import AppSettings, settings_to_dict


def write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_settings_snapshot(settings: AppSettings, path: Path) -> Path:
    """Serialize settings into a redacted config snapshot."""

    return write_json_file(path, settings_to_dict(settings, redact_secrets=True))
