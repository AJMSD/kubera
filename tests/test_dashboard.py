from __future__ import annotations

import json

import pandas as pd
from rich.console import Console

from kubera.config import load_settings
from kubera.pilot.live_pilot import PILOT_LOG_COLUMNS
from kubera.reporting.dashboard import export_dashboard_html, launch_dashboard
from kubera.utils.paths import PathManager


def _make_pilot_row(**overrides: object) -> dict[str, object]:
    row = {column_name: pd.NA for column_name in PILOT_LOG_COLUMNS}
    row.update(
        {
            "pilot_entry_id": "pilot_entry_1",
            "prediction_key": "INFY|NSE|after_close|2026-03-10",
            "prediction_attempt_number": 1,
            "ticker": "INFY",
            "exchange": "NSE",
            "prediction_mode": "after_close",
            "pilot_timestamp_utc": "2026-03-10T10:45:00+00:00",
            "pilot_timestamp_market": "2026-03-10T16:15:00+05:30",
            "market_session_date": "2026-03-10",
            "historical_cutoff_date": "2026-03-10",
            "historical_date": "2026-03-10",
            "prediction_date": "2026-03-11",
            "status": "success",
            "selected_action": "up",
            "news_signal_state": "fresh_news",
            "data_quality_score": 82.5,
            "data_quality_grade": "B",
            "baseline_predicted_next_day_direction": 1,
            "baseline_raw_predicted_probability_up": 0.61,
            "baseline_calibrated_predicted_probability_up": 0.64,
            "baseline_predicted_probability_up": 0.64,
            "enhanced_predicted_next_day_direction": 1,
            "enhanced_raw_predicted_probability_up": 0.67,
            "enhanced_calibrated_predicted_probability_up": 0.72,
            "enhanced_predicted_probability_up": 0.72,
            "blended_predicted_next_day_direction": 1,
            "blended_raw_predicted_probability_up": 0.65,
            "blended_calibrated_predicted_probability_up": 0.70,
            "blended_predicted_probability_up": 0.70,
            "selective_probability_margin": 0.20,
            "selective_required_margin": 0.05,
            "actual_outcome_status": "pending",
            "warning_codes_json": json.dumps(["runtime_warning"]),
            "abstain_reason_codes_json": json.dumps([]),
        }
    )
    row.update(overrides)
    return row


def test_export_dashboard_html_writes_file(isolated_repo, tmp_path) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pd.DataFrame([_make_pilot_row()], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    out = tmp_path / "dash.html"
    export_dashboard_html(settings, out, view="latest")
    text = out.read_text(encoding="utf-8")
    assert "Latest Stored Runs" in text or "Latest" in text
    assert len(text) > 100


def test_launch_dashboard_latest_view_renders_run_summary(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pd.DataFrame([_make_pilot_row()], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    console = Console(record=True, width=160)
    launch_dashboard(settings, view="latest", console_obj=console)

    output = console.export_text()
    assert "Latest Stored Runs" in output
    assert "Offline Calibration" in output
    assert "B (82.5)" in output
    assert "fresh_news" in output
    assert "pending" in output


def test_launch_dashboard_run_view_renders_top_drivers(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_1", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "summary_context": {
                    "market_session_date": "2026-03-10",
                    "historical_cutoff_date": "2026-03-10",
                    "window_resolution_kind": "snapped",
                    "window_resolution_reason": "Snapped to the same-day pre-market window because it is the latest completed scheduled window during market hours.",
                    "enhanced_prediction": {
                        "feature_contributions": {
                            "shap_values": {
                                "news_weighted_sentiment_score": 0.23,
                                "ret_1d": -0.18,
                            }
                        }
                    },
                    "news_context": {
                        "recent_news_summaries": [
                            {
                                "article_title": "Strong quarter guidance",
                                "sentiment_label": "positive",
                                "relevance_score": 0.91,
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    row = _make_pilot_row(pilot_snapshot_path=str(snapshot_path))
    pd.DataFrame([row], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    console = Console(record=True, width=160)
    launch_dashboard(
        settings,
        view="run",
        prediction_key="INFY|NSE|after_close|2026-03-10",
        console_obj=console,
    )

    output = console.export_text()
    assert "Run Detail" in output
    assert "Market session date" in output
    assert "Historical cutoff date" in output
    assert "Window resolution" in output
    assert "Resolution reason" in output
    assert "snapped" in output
    assert "Top Drivers" in output
    assert "news_weighted_sentiment_score" in output
    assert "ret_1d" in output
    assert "Strong quarter guidance" in output


def test_launch_dashboard_run_view_shows_failure_guidance(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_fail", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text("{}", encoding="utf-8")
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    row = _make_pilot_row(
        prediction_key="INFY|NSE|after_close|2026-03-11",
        prediction_date="2026-03-11",
        pilot_snapshot_path=str(snapshot_path),
        status="failure",
        failure_stage="stage2",
        failure_message="network timeout",
    )
    pd.DataFrame([row], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    console = Console(record=True, width=160)
    launch_dashboard(
        settings,
        view="run",
        prediction_key="INFY|NSE|after_close|2026-03-11",
        console_obj=console,
    )

    output = console.export_text()
    assert "What happened" in output
    assert "Failure stage" in output
    assert "Suggested next step" in output
