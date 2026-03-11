"""Modeling modules live here."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaselineModelError",
    "BaselineTrainingResult",
    "PersistedBaselineModel",
    "load_saved_baseline_model",
    "predict_with_saved_model",
    "train_baseline_model",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from kubera.models import train_baseline

        return getattr(train_baseline, name)
    raise AttributeError(f"module 'kubera.models' has no attribute {name!r}")
