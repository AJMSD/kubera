"""Pilot workflows for Kubera."""

from __future__ import annotations

from typing import Any

__all__ = [
    "annotate_pilot_entry",
    "backfill_pilot_actuals",
    "run_live_pilot",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from kubera.pilot import live_pilot

        return getattr(live_pilot, name)
    raise AttributeError(f"module 'kubera.pilot' has no attribute {name!r}")
