from __future__ import annotations

import json

import pandas as pd
from rich.console import Console

from kubera.config import load_settings
from kubera.pilot.live_pilot import PILOT_LOG_COLUMNS
from kubera.reporting.dashboard import (
    _build_unique_day_history_frame,
    export_dashboard_html,
    launch_dashboard,
    load_dashboard_run_frame,
)
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


def _render_run_view_with_recent_news(recent_news: list[dict[str, object]]) -> str:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_news", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "summary_context": {
                    "market_session_date": "2026-03-10",
                    "historical_cutoff_date": "2026-03-10",
                    "window_resolution_kind": "natural",
                    "window_resolution_reason": "Resolved to the after-close window.",
                    "news_context": {
                        "recent_news_summaries": recent_news,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    row = _make_pilot_row(pilot_snapshot_path=str(snapshot_path))
    pd.DataFrame([row], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    console = Console(record=True, width=180)
    launch_dashboard(
        settings,
        view="run",
        prediction_key="INFY|NSE|after_close|2026-03-10",
        console_obj=console,
    )
    return console.export_text()


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
    assert "Prediction vs Actual History" in text
    assert 'name="viewport"' in text
    assert "kubera-dashboard-responsive" in text
    assert "html, body" in text
    assert "width: 100%" in text
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
    assert "Prediction vs Actual History" in output
    assert "Mode & Date Semantics" in output
    assert "Decision session = when the forecast was made" in output
    assert "Offline Calibration" in output
    assert "Blend" in output
    assert "Next-day forecast (made after prior close)" in output
    assert "Latest Run Detail" in output
    assert "Selected action" in output
    assert "Decision session date" in output
    assert "Model Probabilities" in output
    assert "Raw up probability" in output
    assert "Calibrated up probability" in output
    assert "10 Mar 2026, 16:15 IST" in output
    assert "B (82.5)" in output
    assert "fresh_news" in output
    assert "pending" in output


def test_launch_dashboard_latest_view_keeps_probabilities_for_abstain_and_degraded_news(
    isolated_repo,
) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    row = _make_pilot_row(
        status="abstain",
        selected_action=pd.NA,
        news_signal_state="zero_news",
        blended_predicted_next_day_direction=0,
    )
    pd.DataFrame([row], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    console = Console(record=True, width=160)
    launch_dashboard(settings, view="latest", console_obj=console)

    output = console.export_text()
    assert "abstain" in output
    assert "zero_news" in output
    assert "Model Probabilities" in output
    assert "Baseline" in output
    assert "Enhanced" in output
    assert "Blended" in output
    assert "0.640" in output
    assert "0.720" in output
    assert "0.700" in output


def test_launch_dashboard_run_view_renders_top_drivers(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_1", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "explanation": {
                    "enabled": True,
                    "status": "generated",
                    "headline": "Pilot explanation (model=gemini-2.5-flash)",
                    "text": "Pilot explanation (model=gemini-2.5-flash):\nThe enhanced model leans up on positive sentiment.",
                    "model": "gemini-2.5-flash",
                    "source": "gemini",
                },
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
    assert "Decision session date" in output
    assert "Decision timestamp (market)" in output
    assert "Mode & Date Semantics" in output
    assert "Historical cutoff date" in output
    assert "Window resolution" in output
    assert "Resolution reason" in output
    assert "Model Probabilities" in output
    assert "Raw up probability" in output
    assert "Calibrated up probability" in output
    assert "Pilot explanation (model=gemini-2.5-flash)" in output
    assert "enhanced model leans up on positive sentiment" in output.lower()
    assert "snapped" in output
    assert "Enhanced Model Top Drivers" in output
    assert "news_weighted_sentiment_score" in output
    assert "ret_1d" in output
    assert "Driver Impact Legend" in output
    assert "Positive impact = pushes model toward UP probability." in output
    assert "Top Driver Explanations" in output
    assert "Likely effect in this" in output
    assert "Heuristic explanation for operator interpretation; not a causal statement." in output
    assert "Strong quarter guidance" in output


def test_launch_dashboard_run_view_uses_unknown_driver_fallback_text(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_unknown_driver", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "summary_context": {
                    "market_session_date": "2026-03-10",
                    "historical_cutoff_date": "2026-03-10",
                    "window_resolution_kind": "natural",
                    "window_resolution_reason": "Resolved to the after-close window.",
                    "enhanced_prediction": {
                        "feature_contributions": {
                            "shap_values": {
                                "mystery_alpha_signal_7d": 0.11,
                            }
                        }
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
    assert "Pilot explanation (model=" not in output
    assert "mystery_alpha_signal_7d" in output
    assert "Derived feature from" in output
    assert "mystery alpha signal" in output
    assert "UP-leaning SHAP" in output
    assert "this feature" in output


def test_launch_dashboard_run_view_explains_stoch_k_meaning(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_stoch", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "summary_context": {
                    "market_session_date": "2026-03-10",
                    "historical_cutoff_date": "2026-03-10",
                    "window_resolution_kind": "natural",
                    "window_resolution_reason": "Resolved to the after-close window.",
                    "enhanced_prediction": {
                        "feature_contributions": {
                            "shap_values": {
                                "stoch_k": 0.0387,
                            }
                        }
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
    assert "stoch_k" in output
    assert "Stochastic %K (0-100)" in output
    assert "price sits within its" in output
    assert "recent high-low range" in output
    assert "short-term momentum" in output
    assert "stochastic reading aligns" in output
    assert "direction." in output


def test_export_dashboard_html_run_view_includes_snapshot_explanation(
    isolated_repo,
    tmp_path,
) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    snapshot_path = path_manager.build_pilot_snapshot_path("INFY", "run_html_explain", "after_close")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "explanation": {
                    "enabled": True,
                    "status": "generated",
                    "headline": "Pilot explanation (model=gemini-2.5-flash)",
                    "text": "Pilot explanation (model=gemini-2.5-flash):\nCanonical explanation text for dashboard parity.",
                    "model": "gemini-2.5-flash",
                    "source": "gemini",
                },
                "summary_context": {
                    "market_session_date": "2026-03-10",
                    "historical_cutoff_date": "2026-03-10",
                    "window_resolution_kind": "natural",
                    "window_resolution_reason": "Resolved to the after-close window.",
                },
            }
        ),
        encoding="utf-8",
    )
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    row = _make_pilot_row(
        prediction_key="INFY|NSE|after_close|2026-03-11",
        pilot_snapshot_path=str(snapshot_path),
    )
    pd.DataFrame([row], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    out = tmp_path / "dash_run.html"
    export_dashboard_html(
        settings,
        out,
        view="run",
        prediction_key="INFY|NSE|after_close|2026-03-11",
    )
    text = out.read_text(encoding="utf-8")
    assert "Pilot explanation (model=gemini-2.5-flash)" in text
    assert "Canonical explanation text for dashboard parity." in text


def test_launch_dashboard_run_view_limits_linked_news_to_top_three(isolated_repo) -> None:
    output = _render_run_view_with_recent_news(
        [
            {
                "article_title": "First ranked linked article",
                "provider_source": "Source A",
                "summary_snippet": "First linked article snippet.",
                "sentiment_label": "positive",
                "relevance_score": 0.95,
            },
            {
                "article_title": "Second ranked linked article",
                "provider_source": "Source B",
                "summary_snippet": "Second linked article snippet.",
                "sentiment_label": "neutral",
                "relevance_score": 0.88,
            },
            {
                "article_title": "Third ranked linked article",
                "provider_source": "Source C",
                "summary_snippet": "Third linked article snippet.",
                "sentiment_label": "negative",
                "relevance_score": 0.77,
            },
            {
                "article_title": "Fourth ranked linked article",
                "provider_source": "Source D",
                "summary_snippet": "Fourth linked article snippet.",
                "sentiment_label": "positive",
                "relevance_score": 0.70,
            },
        ]
    )

    assert "Top linked news for this prediction window" in output
    assert "First ranked linked" in output
    assert "Second ranked" in output
    assert "Third ranked linked" in output
    assert "Fourth ranked linked" not in output
    assert "Source A" in output
    assert "positive" in output
    assert "0.950" in output
    assert "article snippet." in output


def test_launch_dashboard_run_view_handles_one_and_two_linked_news_items(
    isolated_repo,
) -> None:
    one_item_output = _render_run_view_with_recent_news(
        [
            {
                "article_title": "Only linked article",
                "provider_source": "Solo Source",
                "summary_snippet": "Only linked article snippet.",
                "sentiment_label": "neutral",
                "relevance_score": 0.62,
            }
        ]
    )
    assert "Only linked" in one_item_output
    assert "article snippet." in one_item_output
    assert "No linked news exists for this prediction window." not in one_item_output

    two_item_output = _render_run_view_with_recent_news(
        [
            {
                "article_title": "First linked article",
                "provider_source": "First Source",
                "summary_snippet": "First snippet.",
                "sentiment_label": "positive",
                "relevance_score": 0.81,
            },
            {
                "article_title": "Second linked article",
                "provider_source": "Second Source",
                "summary_snippet": "Second snippet.",
                "sentiment_label": "negative",
                "relevance_score": 0.74,
            },
        ]
    )
    assert "First linked" in two_item_output
    assert "Second linked" in two_item_output
    assert "No linked news exists for this prediction window." not in two_item_output


def test_launch_dashboard_run_view_shows_zero_linked_news_state(isolated_repo) -> None:
    output = _render_run_view_with_recent_news([])

    assert "Top linked news for this prediction window" in output
    assert "No linked news exists for this prediction window." in output


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


def test_launch_dashboard_run_view_hides_explanation_block_when_no_drivers(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    log_path = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pd.DataFrame([_make_pilot_row()], columns=PILOT_LOG_COLUMNS).to_csv(log_path, index=False)

    console = Console(record=True, width=160)
    launch_dashboard(
        settings,
        view="run",
        prediction_key="INFY|NSE|after_close|2026-03-10",
        console_obj=console,
    )

    output = console.export_text()
    assert "Enhanced Model Top Drivers" in output
    assert "No explanation payload saved" in output
    assert "Top Driver Explanations" not in output


def test_load_dashboard_run_frame_parses_mixed_iso_timestamp_precision(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    after_close_log = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    rows = [
        _make_pilot_row(
            pilot_entry_id="after_close_old",
            prediction_key="INFY|NSE|after_close|2026-04-10",
            pilot_timestamp_utc="2026-04-10T02:34:58.517825+00:00",
            prediction_date="2026-04-11",
        ),
        _make_pilot_row(
            pilot_entry_id="after_close_new",
            prediction_key="INFY|NSE|after_close|2026-04-13",
            pilot_timestamp_utc="2026-04-13T02:35:00+00:00",
            prediction_date="2026-04-14",
        ),
    ]
    pd.DataFrame(rows, columns=PILOT_LOG_COLUMNS).to_csv(after_close_log, index=False)

    run_frame = load_dashboard_run_frame(path_manager=path_manager, ticker="INFY", exchange="NSE")
    assert run_frame["pilot_timestamp_sort_key"].notna().all()
    assert run_frame.iloc[0]["pilot_entry_id"] == "after_close_new"


def test_load_dashboard_run_frame_latest_per_mode_uses_newest_rows(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    after_close_log = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pre_market_log = path_manager.build_pilot_log_path("INFY", "NSE", "pre_market")
    after_close_rows = [
        _make_pilot_row(
            pilot_entry_id="ac_old",
            prediction_key="INFY|NSE|after_close|2026-04-10",
            prediction_mode="after_close",
            pilot_timestamp_utc="2026-04-10T02:34:58.517825+00:00",
            prediction_date="2026-04-11",
        ),
        _make_pilot_row(
            pilot_entry_id="ac_new",
            prediction_key="INFY|NSE|after_close|2026-04-13",
            prediction_mode="after_close",
            pilot_timestamp_utc="2026-04-13T02:35:00+00:00",
            prediction_date="2026-04-14",
        ),
    ]
    pre_market_rows = [
        _make_pilot_row(
            pilot_entry_id="pm_old",
            prediction_key="INFY|NSE|pre_market|2026-04-10",
            prediction_mode="pre_market",
            pilot_timestamp_utc="2026-04-10T02:30:00+00:00",
            prediction_date="2026-04-10",
        ),
        _make_pilot_row(
            pilot_entry_id="pm_new",
            prediction_key="INFY|NSE|pre_market|2026-04-13",
            prediction_mode="pre_market",
            pilot_timestamp_utc="2026-04-13T02:36:00.111111+00:00",
            prediction_date="2026-04-13",
        ),
    ]
    pd.DataFrame(after_close_rows, columns=PILOT_LOG_COLUMNS).to_csv(after_close_log, index=False)
    pd.DataFrame(pre_market_rows, columns=PILOT_LOG_COLUMNS).to_csv(pre_market_log, index=False)

    run_frame = load_dashboard_run_frame(path_manager=path_manager, ticker="INFY", exchange="NSE")
    latest_rows = (
        run_frame.sort_values("pilot_timestamp_sort_key", ascending=False)
        .drop_duplicates(subset=["prediction_mode"], keep="first")
        .sort_values("prediction_mode")
    )
    selected_keys = set(latest_rows["prediction_key"].astype(str))
    assert selected_keys == {
        "INFY|NSE|after_close|2026-04-13",
        "INFY|NSE|pre_market|2026-04-13",
    }


def test_launch_dashboard_latest_view_history_table_includes_correctness_and_limit(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    after_close_log = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pre_market_log = path_manager.build_pilot_log_path("INFY", "NSE", "pre_market")

    ac_rows = [
        _make_pilot_row(
            pilot_entry_id="ac_latest",
            prediction_key="INFY|NSE|after_close|2026-04-13",
            prediction_mode="after_close",
            pilot_timestamp_market="2026-04-13T16:15:00+05:30",
            pilot_timestamp_utc="2026-04-13T10:45:00+00:00",
            prediction_date="2026-04-13",
            blended_correct=True,
            actual_outcome_status="backfilled",
            actual_next_day_direction=1,
        ),
        _make_pilot_row(
            pilot_entry_id="ac_next_day",
            prediction_key="INFY|NSE|after_close|2026-04-14",
            prediction_mode="after_close",
            pilot_timestamp_market="2026-04-14T16:15:00+05:30",
            pilot_timestamp_utc="2026-04-14T10:45:00+00:00",
            prediction_date="2026-04-14",
            blended_correct=False,
            actual_outcome_status="backfilled",
            actual_next_day_direction=0,
        ),
        _make_pilot_row(
            pilot_entry_id="ac_older",
            prediction_key="INFY|NSE|after_close|2026-04-10",
            prediction_mode="after_close",
            pilot_timestamp_market="2026-04-10T16:15:00+05:30",
            pilot_timestamp_utc="2026-04-10T10:45:00+00:00",
            prediction_date="2026-04-11",
            blended_correct=False,
            actual_outcome_status="backfilled",
            actual_next_day_direction=0,
        ),
    ]
    pm_rows = [
        _make_pilot_row(
            pilot_entry_id="pm_latest",
            prediction_key="INFY|NSE|pre_market|2026-04-13",
            prediction_mode="pre_market",
            pilot_timestamp_market="2026-04-13T08:05:00+05:30",
            pilot_timestamp_utc="2026-04-13T02:35:00+00:00",
            prediction_date="2026-04-13",
            blended_correct=pd.NA,
            enhanced_correct=pd.NA,
            baseline_correct=pd.NA,
            actual_outcome_status="pending",
        ),
    ]
    pd.DataFrame(ac_rows, columns=PILOT_LOG_COLUMNS).to_csv(after_close_log, index=False)
    pd.DataFrame(pm_rows, columns=PILOT_LOG_COLUMNS).to_csv(pre_market_log, index=False)

    console = Console(record=True, width=200)
    launch_dashboard(settings, view="latest", limit=2, console_obj=console)

    output = console.export_text()
    assert "Prediction vs Actual History" in output
    assert "Correctness" in output
    assert "correct" in output
    assert "pending" in output
    assert "Next-day forecast (made after prior close)" in output
    assert "before" in output
    assert "open)" in output
    assert "2026-04-13T16:15:00+05:30" not in output
    assert "2026-04-10T16:15:00+05:30" not in output


def test_build_unique_day_history_frame_prefers_pre_market_for_same_target_day(isolated_repo) -> None:
    rows = pd.DataFrame(
        [
            _make_pilot_row(
                pilot_entry_id="ac_same_day",
                prediction_mode="after_close",
                prediction_date="2026-04-13",
                pilot_timestamp_utc="2026-04-13T10:45:00+00:00",
            ),
            _make_pilot_row(
                pilot_entry_id="pm_same_day",
                prediction_mode="pre_market",
                prediction_date="2026-04-13",
                pilot_timestamp_utc="2026-04-13T02:35:00+00:00",
            ),
            _make_pilot_row(
                pilot_entry_id="ac_next_day",
                prediction_mode="after_close",
                prediction_date="2026-04-14",
                pilot_timestamp_utc="2026-04-14T10:45:00+00:00",
            ),
        ]
    )
    rows["pilot_timestamp_sort_key"] = pd.to_datetime(rows["pilot_timestamp_utc"], utc=True, errors="coerce")
    rows["prediction_date_sort_key"] = pd.to_datetime(rows["prediction_date"], errors="coerce")

    selected = _build_unique_day_history_frame(rows, limit=2)

    assert selected.shape[0] == 2
    picked_same_day = selected.loc[selected["prediction_date"].astype(str) == "2026-04-13"]
    assert picked_same_day.iloc[0]["prediction_mode"] == "pre_market"


def test_launch_dashboard_latest_view_detail_uses_newest_selected_run(isolated_repo) -> None:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    after_close_log = path_manager.build_pilot_log_path("INFY", "NSE", "after_close")
    pre_market_log = path_manager.build_pilot_log_path("INFY", "NSE", "pre_market")
    pd.DataFrame(
        [
            _make_pilot_row(
                pilot_entry_id="ac_latest",
                prediction_key="INFY|NSE|after_close|2026-04-13",
                prediction_mode="after_close",
                pilot_timestamp_utc="2026-04-13T02:35:00+00:00",
                prediction_date="2026-04-13",
            )
        ],
        columns=PILOT_LOG_COLUMNS,
    ).to_csv(after_close_log, index=False)
    pd.DataFrame(
        [
            _make_pilot_row(
                pilot_entry_id="pm_latest",
                prediction_key="INFY|NSE|pre_market|2026-04-13",
                prediction_mode="pre_market",
                pilot_timestamp_utc="2026-04-13T02:36:00.111111+00:00",
                prediction_date="2026-04-13",
            )
        ],
        columns=PILOT_LOG_COLUMNS,
    ).to_csv(pre_market_log, index=False)

    console = Console(record=True, width=160)
    launch_dashboard(settings, view="latest", console_obj=console)

    output = console.export_text()
    assert "Latest Run Detail" in output
    assert "INFY|NSE|pre_market|2026-04-13" in output
