"""Feature engineering helpers for Kubera."""

__all__ = [
    "HistoricalFeatureBuildResult",
    "HistoricalFeatureError",
    "build_historical_features",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module 'kubera.features' has no attribute {name!r}")

    from kubera.features.historical_features import (
        HistoricalFeatureBuildResult,
        HistoricalFeatureError,
        build_historical_features,
    )

    exports = {
        "HistoricalFeatureBuildResult": HistoricalFeatureBuildResult,
        "HistoricalFeatureError": HistoricalFeatureError,
        "build_historical_features": build_historical_features,
    }
    return exports[name]
