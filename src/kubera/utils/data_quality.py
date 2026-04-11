"""Shared operator-facing data quality scoring helpers."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from kubera.features.news_features import (
    NEWS_SIGNAL_STATE_CARRIED_FORWARD,
    NEWS_SIGNAL_STATE_FALLBACK_HEAVY,
    NEWS_SIGNAL_STATE_ZERO,
    determine_news_signal_state,
)


def build_data_quality_payload(
    *,
    row_mapping: Mapping[str, Any],
    news_feature_row: Mapping[str, Any] | None = None,
    stage2_metadata: Mapping[str, Any] | None = None,
    stage5_metadata: Mapping[str, Any] | None = None,
    stage6_metadata: Mapping[str, Any] | None = None,
    stage7_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one row for operator-facing data quality review."""

    del stage2_metadata

    component_scores = {
        "market_data": 100.0,
        "source_coverage": 100.0,
        "extraction_quality": 100.0,
        "signal_state": 100.0,
    }
    reasons: list[str] = []
    news_reference = dict(news_feature_row or row_mapping)

    recent_gap_count = _resolve_first_int(
        row_mapping,
        "historical_market_gap_count_5d",
        "market_data_gap_count_5d",
    ) or 0
    if recent_gap_count > 0:
        component_scores["market_data"] -= min(35.0, 10.0 + (recent_gap_count * 8.0))
        reasons.append(f"recent market-data gap fill count {recent_gap_count}")

    article_count = _resolve_first_int(row_mapping, "news_article_count") or 0
    provider_count = len(
        [
            provider
            for provider in ((stage5_metadata or {}).get("providers_used") or [])
            if str(provider).strip()
        ]
    )
    source_domain_count = len((stage5_metadata or {}).get("source_domain_counts") or {})
    if article_count == 0:
        component_scores["source_coverage"] -= 60.0
        reasons.append("no fresh source articles")
    elif article_count == 1:
        component_scores["source_coverage"] -= 20.0
        reasons.append("only one source article")
    if article_count > 0 and provider_count <= 1 and provider_count > 0:
        component_scores["source_coverage"] -= 10.0
        reasons.append("single news provider")
    if article_count > 0 and source_domain_count <= 1 and source_domain_count > 0:
        component_scores["source_coverage"] -= 10.0
        reasons.append("single source domain")

    stage5_warnings = [str(value) for value in ((stage5_metadata or {}).get("warnings") or [])]
    if any(code.startswith("degraded_source_") for code in stage5_warnings):
        component_scores["source_coverage"] -= 10.0
        reasons.append("degraded stage5 source mix")
    if (_resolve_first_int(row_mapping, "stage5_provider_request_retry_count") or 0) > 0:
        component_scores["source_coverage"] -= 5.0
        reasons.append("stage5 provider retries")
    if (_resolve_first_int(row_mapping, "stage5_article_fetch_retry_count") or 0) > 0:
        component_scores["source_coverage"] -= 5.0
        reasons.append("stage5 article fetch retries")

    avg_confidence = _resolve_first_float(news_reference, "news_avg_confidence") or 0.0
    avg_content_quality = (
        _resolve_first_float(news_reference, "news_avg_content_quality_score") or 0.0
    )
    if avg_confidence < 0.55:
        component_scores["extraction_quality"] -= 15.0
        reasons.append(f"low average extraction confidence {avg_confidence:.2f}")
    if avg_content_quality < 0.75:
        component_scores["extraction_quality"] -= 15.0
        reasons.append(f"low content quality score {avg_content_quality:.2f}")
    if (_resolve_first_int(row_mapping, "stage6_retry_count") or 0) > 0:
        component_scores["extraction_quality"] -= 5.0
        reasons.append("stage6 retries")
    if ((stage6_metadata or {}).get("warnings") or []):
        component_scores["extraction_quality"] -= 5.0
        reasons.append("stage6 extraction warnings")

    signal_state = _clean_string(row_mapping.get("news_signal_state")) or determine_news_signal_state(
        news_reference
    )
    if signal_state == NEWS_SIGNAL_STATE_FALLBACK_HEAVY:
        component_scores["signal_state"] -= 55.0
        reasons.append("fallback-heavy news state")
    elif signal_state == NEWS_SIGNAL_STATE_CARRIED_FORWARD:
        component_scores["signal_state"] -= 25.0
        reasons.append("carried-forward-only news state")
    elif signal_state == NEWS_SIGNAL_STATE_ZERO:
        component_scores["signal_state"] -= 40.0
        reasons.append("zero-news state")
    if bool(_coerce_optional_bool(row_mapping.get("news_feature_synthetic_flag"))):
        component_scores["signal_state"] -= 10.0
        reasons.append("synthetic zero-news Stage 7 row")

    if ((stage7_metadata or {}).get("warnings") or []):
        component_scores["signal_state"] -= 5.0
        reasons.append("stage7 feature warnings")

    for key, value in component_scores.items():
        component_scores[key] = round(max(0.0, min(100.0, value)), 1)

    score = round(
        (component_scores["market_data"] * 0.25)
        + (component_scores["source_coverage"] * 0.30)
        + (component_scores["extraction_quality"] * 0.20)
        + (component_scores["signal_state"] * 0.25),
        1,
    )
    return {
        "score": score,
        "grade": grade_data_quality_score(score),
        "reasons": dedupe_quality_reasons(reasons),
        "components": component_scores,
    }


def grade_data_quality_score(score: float) -> str:
    """Convert one numeric quality score into an operator-facing grade band."""

    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    if score >= 50.0:
        return "E"
    return "F"


def dedupe_quality_reasons(reasons: list[str]) -> list[str]:
    """Keep the quality-reason list stable and compact."""

    ordered: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        cleaned_reason = reason.strip()
        if not cleaned_reason or cleaned_reason in seen:
            continue
        seen.add(cleaned_reason)
        ordered.append(cleaned_reason)
    return ordered[:6]


def _resolve_first_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _coerce_optional_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _resolve_first_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_optional_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str | None:
    if value is None or _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False
