"""Unit tests for Kubera v3 focused improvements."""

from __future__ import annotations

from datetime import date
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from kubera.utils.calendar import PandasMarketCalendar
from kubera.models.common import (
    compute_news_context_weight,
    blend_probabilities,
    compute_calibration_bins,
    explain_prediction_shap
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def test_compute_calibration_bins():
    """Verify calibration binning logic."""
    actual = pd.Series([1, 1, 0, 0, 1])
    probs = pd.Series([0.9, 0.8, 0.1, 0.2, 0.85])

    bins = compute_calibration_bins(actual, probs, n_bins=5)
    # Bin 0: [0.0, 0.2] -> probs [0.1, 0.2], actual [0, 0]
    # Bin 4: [0.8, 1.0] -> probs [0.9, 0.8, 0.85], actual [1, 1, 1]

    bin0 = next(b for b in bins if b["bin_index"] == 0)
    assert bin0["row_count"] == 2
    assert bin0["actual_frequency"] == 0.0

    bin4 = next(b for b in bins if b["bin_index"] == 4)
    assert bin4["row_count"] == 2
    assert bin4["actual_frequency"] == 1.0

def test_explain_prediction_shap_fallback():
    """Verify that SHAP explanation falls back gracefully to coefficients."""
    # Create a simple linear pipeline
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([0, 1, 1])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(X, y)

    row = pd.DataFrame([{"a": 1, "b": 4}])
    cols = ("a", "b")

    # This should use fallback if shap is not available or just work
    expl = explain_prediction_shap(pipeline, row, cols)
    assert "a" in expl
    assert "b" in expl
    assert isinstance(expl["a"], float)
