"""Tests for user-facing failure copy."""

from kubera.models.train_baseline import BaselineModelError
from kubera.models.train_enhanced import EnhancedModelError
from kubera.features.historical_features import HistoricalFeatureError
from kubera.ingest.market_data import HistoricalMarketDataProviderError
from kubera.pilot.live_pilot import LivePilotError
from kubera.utils.user_failure import (
    describe_domain_error,
    describe_partial_failure_paths,
    describe_pilot_stage_failure,
    pilot_failure_note_for_review,
)


def test_describe_pilot_stage_failure_stage2_cutoff():
    reason, step = describe_pilot_stage_failure(
        "stage2",
        "Historical market data does not cover the expected live cutoff date.",
    )
    assert "history" in reason.lower() or "session" in reason.lower() or "market" in reason.lower()
    assert "retry" in step.lower() or "close" in step.lower() or "fetch" in step.lower()


def test_describe_pilot_stage_failure_stage2_cutoff_provider_lag():
    reason, step = describe_pilot_stage_failure(
        "stage2",
        "cutoff_provider_lag: Latest historical bar is 2026-03-09; ...",
    )
    assert "published" in reason.lower() or "provider" in reason.lower() or "bar" in reason.lower()
    assert "retry" in step.lower() or "close" in step.lower() or "minutes" in step.lower()


def test_describe_pilot_stage_failure_stage2_cutoff_calendar_mismatch():
    reason, step = describe_pilot_stage_failure(
        "stage2",
        "cutoff_calendar_mismatch: Latest historical bar is 2026-04-02 ...",
    )
    assert "calendar" in reason.lower() or "exchange" in reason.lower()
    assert "sync-holidays" in step.lower() or "holiday" in step.lower() or "india.json" in step


def test_describe_pilot_stage_failure_stage6_auth():
    reason, step = describe_pilot_stage_failure("stage6", "401 Unauthorized")
    assert "llm" in reason.lower() or "provider" in reason.lower()
    assert "KUBERA_LLM_API_KEY" in step or "key" in step.lower()


def test_describe_pilot_stage_failure_unknown_stage():
    reason, _ = describe_pilot_stage_failure(None, "Something broke")
    assert "did not complete" in reason.lower()


def test_describe_partial_failure_paths_enhanced():
    text = describe_partial_failure_paths("enhanced")
    assert "baseline" in text.lower()
    assert "enhanced" in text.lower()


def test_describe_partial_failure_paths_news():
    text = describe_partial_failure_paths("stage6")
    assert "baseline" in text.lower()
    assert "news" in text.lower()


def test_describe_domain_error_baseline_missing_file():
    msg = describe_domain_error(
        BaselineModelError("Historical feature table does not exist: /tmp/x.csv")
    )
    assert "baseline" in msg.lower() or "historical" in msg.lower()
    assert "train" in msg.lower()


def test_describe_domain_error_enhanced():
    msg = describe_domain_error(EnhancedModelError("News feature table does not contain any rows for Stage 8."))
    assert "enhanced" in msg.lower() or "news" in msg.lower()


def test_describe_domain_error_historical_feature():
    msg = describe_domain_error(HistoricalFeatureError("No data"))
    assert "historical" in msg.lower() or "feature" in msg.lower()


def test_describe_domain_error_market_provider():
    msg = describe_domain_error(HistoricalMarketDataProviderError("timeout"))
    assert "market" in msg.lower() or "network" in msg.lower() or "ingest" in msg.lower()


def test_describe_domain_error_live_pilot():
    msg = describe_domain_error(LivePilotError("Unsupported pilot prediction mode: x"))
    assert "Unsupported" in msg or "mode" in msg.lower()


def test_pilot_failure_note_for_review_none_when_success():
    assert pilot_failure_note_for_review("success", "stage2", None) is None


def test_pilot_failure_note_for_review_partial():
    note = pilot_failure_note_for_review("partial_failure", "stage5", "timeout")
    assert note is not None
    assert "partial_failure" in note
    assert "stage5" in note


def _minimal_snapshot_payload(status: str) -> dict:
    return {
        "summary_context": {
            "ticker": "T",
            "exchange": "NSE",
            "prediction_mode": "after_close",
            "market_session_date": "2025-12-31",
            "historical_cutoff_date": "2025-12-31",
            "window_resolution_kind": "override",
            "window_resolution_reason": "Used an explicit mode or timestamp override.",
            "prediction_date": "2026-01-01",
            "run_timestamp_ist": "2026-01-01T00:00:00+05:30",
            "status": status,
            "baseline_prediction": {
                "direction": "up",
                "raw_probability_up": 0.5,
                "calibrated_probability_up": 0.5,
                "model_run_id": "r1",
            },
            "enhanced_prediction": {
                "direction": "up",
                "raw_probability_up": 0.5,
                "calibrated_probability_up": 0.5,
                "model_run_id": "r2",
            },
            "blended_prediction": {
                "direction": "up",
                "action": "abstain",
                "abstain_flag": True,
                "raw_probability_up": 0.5,
                "calibrated_probability_up": 0.5,
                "news_weight": 0.5,
                "probability_margin": 0.01,
                "required_margin": 0.1,
                "abstain_reasons": ["low_conviction"],
            },
            "model_agreement": "agree",
            "news_context": {
                "article_count": 0,
                "avg_confidence": None,
                "fallback_ratio": None,
                "signal_state": None,
                "historical_market_gap_count_5d": 0,
                "top_event_types": [],
            },
            "warnings": {"fired": False, "codes": []},
            "data_quality": {"score": 50.0, "grade": "C", "reasons": [], "components": {}},
            "prior_prediction_outcome": None,
            "total_run_duration_seconds": 1.0,
            "failure_stage": None,
            "failure_message": None,
        }
    }


def test_format_pilot_summary_abstain_no_failure_block():
    from kubera.pilot.live_pilot import format_pilot_summary

    text = format_pilot_summary(_minimal_snapshot_payload("abstain"))
    assert "Failure stage" not in text
    assert "What happened" not in text


def test_format_pilot_summary_failure_includes_public_reason():
    from kubera.pilot.live_pilot import format_pilot_summary

    payload = _minimal_snapshot_payload("failure")
    payload["summary_context"]["failure_stage"] = "stage2"
    payload["summary_context"]["failure_message"] = "network timeout"
    payload["summary_context"]["failure_reason_public"] = "test reason"
    payload["summary_context"]["failure_next_step"] = "test step"
    text = format_pilot_summary(payload)
    assert "Failure stage: stage2" in text
    assert "What happened: test reason" in text
    assert "Suggestion: test step" in text
