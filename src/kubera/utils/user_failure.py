"""User-facing failure explanations: calm, truthful summaries and next steps."""

from __future__ import annotations

from kubera.utils.logging import sanitize_log_text

# Pilot pipeline stages (aligned with live_pilot failure_stage labels).
_STAGE_DEFAULTS: dict[str, tuple[str, str]] = {
    "stage2": (
        "Market data for this run could not be prepared as expected.",
        "Confirm network access, then retry. After the trading session ends, same-day bars are usually available.",
    ),
    "baseline": (
        "The historical-only (baseline) model step did not complete.",
        "Run `kubera train` to rebuild baseline artifacts, or check that Stage 3 feature files exist for this ticker.",
    ),
    "stage5": (
        "Company news could not be fetched or processed for this cutoff.",
        "Check connectivity and provider settings, then retry. Some feeds rate-limit or fail intermittently.",
    ),
    "stage6": (
        "Structured news extraction (LLM) did not complete for this run.",
        "Verify `KUBERA_LLM_API_KEY` and quota, then retry.",
    ),
    "stage7": (
        "Daily news features could not be built from the extracted articles.",
        "Run `kubera train` after a successful news pipeline, or inspect Stage 6/7 logs for inconsistent inputs.",
    ),
    "enhanced": (
        "The news-augmented (enhanced) model step did not complete.",
        "Run `kubera train` to refresh enhanced models, or confirm Stage 7 features align with training.",
    ),
}


def _norm(msg: str | None) -> str:
    if not msg:
        return ""
    return sanitize_log_text(str(msg)).strip().lower()


def _match_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def describe_pilot_stage_failure(
    stage: str | None,
    raw_message: str | None,
) -> tuple[str, str]:
    """Return (public reason, suggested next step) for a pilot failure_stage and sanitized message."""

    st = (stage or "unknown").strip().lower() or "unknown"
    msg = _norm(raw_message)

    if st == "unknown" or not stage:
        reason = "The run did not complete successfully."
        step = "Check the message above if present, then review logs or retry after data providers stabilize."
        if msg:
            reason = _refine_unknown_reason(msg)
        return reason, step

    default_reason, default_step = _STAGE_DEFAULTS.get(
        st,
        (
            "A pipeline step did not complete.",
            "Review recent logs and retry; if artifacts are missing, run `kubera train`.",
        ),
    )

    if st == "stage2":
        if _match_any(msg, ("cutoff_provider_lag:",)):
            return (
                "The market data provider has not published the bar for the next expected trading session yet.",
                "Wait a few minutes and retry (especially right after the NSE close).",
            )
        if _match_any(msg, ("cutoff_calendar_mismatch:",)):
            return (
                "The configured exchange calendar and the downloaded OHLCV history disagree on trading sessions.",
                "Run `kubera sync-holidays`, update `config/exchange_closures/india.json`, or add an emergency closure in `config/market_holidays.local.json`.",
            )
        if _match_any(msg, ("cutoff_stale_holiday_cache:",)):
            return (
                "The checked-in holiday list may be missing closures for the year you are trading.",
                "Run `kubera sync-holidays` or refresh `config/exchange_closures/india.json` from the latest NSE/BSE circular.",
            )
        if _match_any(msg, ("does not cover", "cutoff", "expected live cutoff", "historical market data")):
            return (
                "The downloaded market history does not yet include the session needed for this prediction window.",
                "For same-day after-close runs, try again once the close is available; otherwise confirm `kubera train` / fetch has data through the required date.",
            )
        if _match_any(msg, ("network", "timeout", "connection", "unreachable", "resolve")):
            return (
                "Market data could not be retrieved (network or provider issue).",
                "Check internet access and retry; transient provider outages usually clear within minutes.",
            )
        if _match_any(msg, ("not exist", "no such file", "missing", "empty")):
            return (
                "Expected market data files were missing or empty.",
                "Run training or fetch so Stage 2 outputs exist for this ticker and exchange.",
            )
        return default_reason, default_step

    if st == "baseline":
        if _match_any(msg, ("does not exist", "not found", "missing", "metadata")):
            return (
                "Baseline model files or metadata were missing or unreadable.",
                "Run `kubera train` to produce baseline artifacts under `artifacts/models/baseline`.",
            )
        return default_reason, default_step

    if st == "stage5":
        if _match_any(msg, ("rate", "429", "quota", "limit")):
            return (
                "A news provider declined or throttled the request.",
                "Wait briefly and retry, or reduce how often you refresh news during development.",
            )
        return default_reason, default_step

    if st == "stage6":
        if _match_any(msg, ("api", "key", "auth", "401", "403", "permission")):
            return (
                "The LLM provider rejected the request (often key, quota, or permissions).",
                "Confirm `KUBERA_LLM_API_KEY` and account limits, then retry.",
            )
        if _match_any(msg, ("schema", "validation", "json")):
            return (
                "The model returned a response that did not match the expected format after retries.",
                "Retry later; if it persists, check for provider or prompt changes.",
            )
        return default_reason, default_step

    if st == "stage7":
        if _match_any(msg, ("does not exist", "empty", "no rows")):
            return (
                "News feature inputs were missing or empty for the target window.",
                "Complete Stage 5–6 successfully, then rebuild news features with `kubera train` or the feature build step.",
            )
        return default_reason, default_step

    if st == "enhanced":
        if _match_any(msg, ("does not exist", "not found", "missing", "metadata")):
            return (
                "Enhanced model files or metadata were missing or unreadable.",
                "Run `kubera train` to produce enhanced artifacts under `artifacts/models/enhanced`.",
            )
        if _match_any(msg, ("column", "feature", "missing")):
            return (
                "Live features did not match what the enhanced model was trained on.",
                "Run `kubera train` so models and feature columns stay in sync.",
            )
        return default_reason, default_step

    return default_reason, default_step


def _refine_unknown_reason(msg: str) -> str:
    if _match_any(msg, ("pilot", "timestamp", "market open", "market close")):
        return "The run stopped because the requested time does not match the rules for this prediction mode."
    if _match_any(msg, ("does not exist", "not found")):
        return "A required file or artifact was not found."
    return "The run did not complete successfully."


def describe_partial_failure_paths(failure_stage: str | None) -> str:
    """One line clarifying which parts of the pipeline completed (partial_failure only)."""

    fs = (failure_stage or "").strip().lower()
    if fs == "enhanced":
        return (
            "Path status: baseline prediction completed; the enhanced model step did not complete."
        )
    if fs in ("stage5", "stage6", "stage7"):
        return (
            "Path status: baseline prediction completed; the news pipeline did not finish, "
            "so a full enhanced prediction was not produced."
        )
    return "Path status: one stage completed; another stage in the pipeline did not."


def describe_domain_error(exc: BaseException) -> str:
    """Single-paragraph message for CLI users (training, features, ingest)."""

    name = type(exc).__name__
    msg = _norm(str(exc))

    if name == "BaselineModelError":
        if "does not exist" in msg or "not exist" in msg:
            return (
                "Baseline training stopped because a required historical feature file or metadata is missing. "
                "Build Stage 3 features first, then run `kubera train` again."
            )
        if "empty" in msg:
            return (
                "Baseline training stopped because the historical feature table has no usable rows. "
                "Check market ingest and feature build, then retry."
            )
        if "duplicate" in msg or "invalid date" in msg:
            return (
                "Baseline training stopped because the historical feature table failed validation (dates or duplicates). "
                "Rebuild historical features from clean market data."
            )
        return (
            "Baseline training could not finish. "
            "See the message above for details; often rebuilding historical features and re-running `kubera train` resolves it."
        )

    if name == "EnhancedModelError":
        if "does not exist" in msg or "not exist" in msg:
            return (
                "Enhanced training stopped because a required news feature file or metadata is missing. "
                "Complete the news pipeline through Stage 7, then run `kubera train` again."
            )
        if "duplicate" in msg or "invalid date" in msg:
            return (
                "Enhanced training stopped because merged or news data failed validation. "
                "Rebuild news and historical features, then retry."
            )
        return (
            "Enhanced training could not finish. "
            "Ensure news features and merged datasets exist for this ticker; then run `kubera train` again."
        )

    if name == "HistoricalFeatureError":
        return (
            "Historical features could not be built from the current market data. "
            "Confirm OHLCV data covers the needed range and is valid, then retry."
        )

    if name == "HistoricalMarketDataProviderError":
        return (
            "Market data ingest failed (provider or network). "
            "Check connectivity and ticker/exchange settings, then retry."
        )

    if name == "LivePilotError":
        return sanitize_log_text(str(exc))

    # Generic: first line only, sanitized, truncated
    text = sanitize_log_text(str(exc)).strip()
    if len(text) > 280:
        text = text[:277] + "..."
    return f"{name}: {text}"


def pilot_failure_note_for_review(
    status: str,
    failure_stage: str | None,
    failure_message: str | None,
) -> str | None:
    """Compact note for final-review daily rows."""

    st = (status or "").strip().lower()
    if st not in ("failure", "partial_failure"):
        return None
    reason, step = describe_pilot_stage_failure(failure_stage, failure_message)
    stage = (failure_stage or "unknown").strip()
    short_step = step.split(".")[0] if step else ""
    if short_step and not short_step.endswith((".", "?", "!")):
        short_step += "."
    return f"{status}: {reason} (stage {stage}). {short_step}".strip()
