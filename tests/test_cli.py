"""Smoke tests for the Kubera CLI."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kubera.cli import (
    _execute_live_predict,
    _resolve_live_window_request,
    cmd_dash,
    cmd_doctor,
    cmd_evaluate,
    cmd_setup,
    main,
)
from kubera.config import load_settings
from kubera.pilot.live_pilot import LiveWindowResolution, PilotPendingBackfillResult

@pytest.fixture
def mock_settings():
    with patch("kubera.cli.load_settings") as mock:
        settings = MagicMock()
        settings.paths.data_dir = Path("tmp_kubera_test")
        settings.ticker.symbol = "INFY"
        settings.ticker.exchange = "NSE"
        mock.return_value = settings
        yield mock

@pytest.fixture
def mock_runtime():
    with patch("kubera.cli.resolve_runtime_settings") as mock:
        runtime = MagicMock()
        runtime.ticker.symbol = "INFY"
        runtime.ticker.exchange = "NSE"
        mock.return_value = runtime
        yield mock

def test_cmd_setup_invokes_bootstrap(mock_settings):
    """Verify kubera setup calls bootstrap."""
    args = argparse.Namespace(command="setup")
    with patch("kubera.bootstrap.bootstrap") as mock_init:
        exit_code = cmd_setup(args)
        assert exit_code == 0
        mock_init.assert_called_once()

def test_cmd_dash_invokes_dashboard(mock_settings):
    """Verify kubera dash calls launch_dashboard."""
    args = argparse.Namespace(
        ticker="INFY",
        exchange="NSE",
        view="latest",
        prediction_key=None,
        date=None,
        mode=None,
        limit=20,
        command="dash"
    )
    with patch("kubera.reporting.dashboard.launch_dashboard") as mock_launch:
        exit_code = cmd_dash(args)
        assert exit_code == 0
        mock_launch.assert_called_once()

def test_main_doctor(capsys, mock_settings):
    """Verify kubera doctor runs without crashing."""
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = MagicMock(empty=False)
        with patch("importlib.util.find_spec", return_value=True):
            exit_code = main(["doctor"])
            assert exit_code == 0
            out, err = capsys.readouterr()
            assert "--- Kubera Health Check ---" in out
            assert "Python Version" in out


def test_main_help_lists_run_before_predict(capsys):
    """Top-level help keeps ``kubera run`` ahead of ``predict`` as the default path."""

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out, _err = capsys.readouterr()
    run_index = out.find("\n    run")
    predict_index = out.find("\n    predict")
    assert run_index != -1
    assert predict_index != -1
    assert run_index < predict_index
    normalized = " ".join(out.split())
    assert "Use 'kubera run' for the default end-to-end flow." in normalized


def test_main_run_help_marks_mode_and_timestamp_as_advanced(capsys):
    """``kubera run --help`` keeps mode and timestamp as advanced overrides only."""

    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0
    out, _err = capsys.readouterr()
    normalized = " ".join(out.split())
    assert "Advanced override for auto-detected mode." in normalized
    assert "Advanced ISO-8601 timestamp override." in normalized
    assert "News discovery uses free RSS and NSE sources" in normalized


def test_resolve_live_window_request_uses_timestamp_override_phase(isolated_repo) -> None:
    runtime = load_settings()
    args = argparse.Namespace(mode=None, timestamp="2026-03-10T08:05:00+05:30")

    resolved = _resolve_live_window_request(args, runtime)

    assert resolved.prediction_window.prediction_mode == "pre_market"
    assert resolved.prediction_window.market_session_date == date(2026, 3, 10)
    assert resolved.prediction_window.prediction_date == date(2026, 3, 10)
    assert resolved.resolution_kind == "override"
    assert resolved.resolution_reason == "Used the window implied by the explicit timestamp override."


def test_execute_live_predict_uses_resolved_window_for_refresh_and_run(monkeypatch: pytest.MonkeyPatch):
    settings = MagicMock()
    runtime = MagicMock()
    runtime.market = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    resolved_timestamp = datetime(2026, 3, 10, 2, 35, tzinfo=timezone.utc)
    resolved_window = LiveWindowResolution(
        prediction_window=SimpleNamespace(
            prediction_mode="pre_market",
            timestamp_utc=resolved_timestamp,
            market_session_date=date(2026, 3, 10),
            historical_cutoff_date=date(2026, 3, 9),
            prediction_date=date(2026, 3, 10),
        ),
        resolution_kind="snapped",
        resolution_reason="Snapped to the same-day pre-market window because it is the latest completed scheduled window during market hours.",
    )
    args = argparse.Namespace(
        mode=None,
        timestamp=None,
        no_refresh=False,
        interactive=False,
        ticker="INFY",
        exchange="NSE",
        explain=False,
    )
    pilot_result = SimpleNamespace(
        status="success",
        prediction_date=date(2026, 3, 10),
        log_path=Path("pilot.log"),
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr("kubera.cli._resolve_live_window_request", lambda _args, _runtime: resolved_window)

    def fake_check_market_data_freshness(
        _runtime,
        *,
        ticker,
        exchange,
        required_end_date,
    ):
        observed["required_end_date"] = required_end_date
        observed["freshness_ticker"] = ticker
        observed["freshness_exchange"] = exchange
        return True, required_end_date, "fresh"

    def fake_fetch_company_news(
        _runtime,
        *,
        ticker,
        exchange,
        ensure_fresh_until,
    ):
        observed["news_ticker"] = ticker
        observed["news_exchange"] = exchange
        observed["news_cutoff"] = ensure_fresh_until
        return None

    def fake_run_live_pilot(
        _settings,
        *,
        prediction_mode,
        timestamp,
        ticker,
        exchange,
        explain,
        window_resolution_kind,
        window_resolution_reason,
    ):
        observed["prediction_mode"] = prediction_mode
        observed["timestamp"] = timestamp
        observed["run_ticker"] = ticker
        observed["run_exchange"] = exchange
        observed["explain"] = explain
        observed["resolution_kind"] = window_resolution_kind
        observed["resolution_reason"] = window_resolution_reason
        return pilot_result

    monkeypatch.setattr("kubera.ingest.market_data.check_market_data_freshness", fake_check_market_data_freshness)
    monkeypatch.setattr("kubera.ingest.news_data.fetch_company_news", fake_fetch_company_news)
    monkeypatch.setattr("kubera.pilot.live_pilot.run_live_pilot", fake_run_live_pilot)
    monkeypatch.setattr("kubera.cli.run_live_pilot", fake_run_live_pilot)

    code, result = _execute_live_predict(args, settings, runtime)

    assert code == 0
    assert result is pilot_result
    assert observed["required_end_date"] == date(2026, 3, 9)
    assert observed["news_cutoff"] == resolved_timestamp
    assert observed["prediction_mode"] == "pre_market"
    assert observed["timestamp"] == resolved_timestamp
    assert observed["resolution_kind"] == "snapped"


def test_main_run_skips_training_when_aligned(tmp_path):
    """``kubera run`` skips training when feature tables match saved models."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    html_path = tmp_path / "INFY_NSE_latest.html"
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.bootstrap.bootstrap"):
                with patch(
                    "kubera.cli.should_run_training_for_current_features",
                    return_value=(False, "stage 4/8 artifacts match current feature tables"),
                ):
                    with patch("kubera.cli._execute_training_pipeline") as mock_train:
                        with patch("kubera.cli._execute_live_predict", return_value=(0, None)):
                            with patch("kubera.cli.launch_dashboard"):
                                with patch("kubera.cli.export_dashboard_html") as mock_export:
                                    with patch("kubera.cli.PathManager") as mock_pm_cls:
                                        with patch("kubera.cli.webbrowser.open") as mock_browser:
                                            inst = MagicMock()
                                            inst.build_operator_dashboard_html_path.return_value = html_path
                                            mock_pm_cls.return_value = inst
                                            exit_code = main(["run", "--no-browser"])
    assert exit_code == 0
    mock_train.assert_not_called()
    mock_export.assert_called_once()
    mock_browser.assert_not_called()


def test_main_predict_without_dashboard_skips_launch():
    """``kubera predict`` does not run the dashboard unless ``--dashboard``."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.cli._execute_live_predict", return_value=(0, None)) as mock_predict:
                with patch("kubera.cli.launch_dashboard") as mock_launch:
                    with patch("kubera.cli.export_dashboard_html") as mock_export:
                        exit_code = main(["predict", "--no-browser"])
    assert exit_code == 0
    mock_predict.assert_called_once()
    mock_launch.assert_not_called()
    mock_export.assert_not_called()


def test_main_predict_with_dashboard_matches_run_dashboard(tmp_path):
    """``kubera predict --dashboard`` invokes the same dashboard exports as ``run``."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    html_path = tmp_path / "INFY_NSE_latest.html"
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.cli._execute_live_predict", return_value=(0, None)):
                with patch("kubera.cli.launch_dashboard") as mock_launch:
                    with patch("kubera.cli.export_dashboard_html") as mock_export:
                        with patch("kubera.cli.PathManager") as mock_pm_cls:
                            with patch("kubera.cli.webbrowser.open") as mock_browser:
                                inst = MagicMock()
                                inst.build_operator_dashboard_html_path.return_value = html_path
                                mock_pm_cls.return_value = inst
                                exit_code = main(["predict", "--dashboard", "--no-browser"])
    assert exit_code == 0
    mock_launch.assert_called_once()
    mock_export.assert_called_once()
    mock_browser.assert_not_called()


def test_main_run_invokes_training_when_forced(tmp_path):
    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    html_path = tmp_path / "INFY_NSE_latest.html"
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.bootstrap.bootstrap"):
                with patch("kubera.cli.should_run_training_for_current_features") as mock_policy:
                    with patch("kubera.cli._execute_training_pipeline", return_value=0) as mock_train:
                        with patch("kubera.cli._execute_live_predict", return_value=(0, None)):
                            with patch("kubera.cli.launch_dashboard"):
                                with patch("kubera.cli.export_dashboard_html"):
                                    with patch("kubera.cli.PathManager") as mock_pm_cls:
                                        with patch("kubera.cli.webbrowser.open"):
                                            inst = MagicMock()
                                            inst.build_operator_dashboard_html_path.return_value = html_path
                                            mock_pm_cls.return_value = inst
                                            exit_code = main(["run", "--no-browser", "--retrain"])
    assert exit_code == 0
    mock_train.assert_called_once()
    mock_policy.assert_not_called()


def test_cmd_evaluate_calls_evaluate_offline(mock_settings):
    """Verify kubera evaluate invokes the offline evaluation function."""
    args = argparse.Namespace(
        ticker="INFY",
        exchange="NSE",
        force_refresh=True,
        command="evaluate"
    )
    with patch("kubera.cli.evaluate_offline") as mock_eval:
        exit_code = cmd_evaluate(args)
        assert exit_code == 0
        mock_eval.assert_called_once()
        _, kwargs = mock_eval.call_args
        assert kwargs["force_stage8_refresh"] is True


def test_main_run_invokes_integrated_backfill(tmp_path):
    """``kubera run`` calls pending backfill after predict when a pilot result exists."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    html_path = tmp_path / "INFY_NSE_latest.html"
    pilot = MagicMock()
    pilot.prediction_mode = "after_close"
    pilot.prediction_date = date(2026, 4, 8)
    pilot.historical_cutoff_date = date(2026, 4, 7)
    pilot.window_resolution_kind = "natural"
    pilot.window_resolution_reason = "Used the same-day after-close window because the market session is complete."
    bf_result = PilotPendingBackfillResult(
        updated_row_count=1,
        unresolved_row_count=0,
        error_count=0,
        effective_as_of=date(2026, 4, 7),
        prediction_dates_attempted=(date(2026, 4, 4),),
    )
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.bootstrap.bootstrap"):
                with patch(
                    "kubera.cli.should_run_training_for_current_features",
                    return_value=(False, "aligned"),
                ):
                    with patch("kubera.cli._execute_training_pipeline"):
                        with patch(
                            "kubera.cli._execute_live_predict",
                            return_value=(0, pilot),
                        ):
                            with patch(
                                "kubera.cli.backfill_pending_pilot_actuals_for_cli",
                                return_value=bf_result,
                            ) as mock_bf:
                                with patch("kubera.cli.launch_dashboard"):
                                    with patch("kubera.cli.export_dashboard_html"):
                                        with patch("kubera.cli.PathManager") as mock_pm_cls:
                                            with patch("kubera.cli.webbrowser.open"):
                                                inst = MagicMock()
                                                inst.build_operator_dashboard_html_path.return_value = (
                                                    html_path
                                                )
                                                mock_pm_cls.return_value = inst
                                                exit_code = main(["run", "--no-browser"])
    assert exit_code == 0
    mock_bf.assert_called_once()


def test_main_run_no_backfill_skips_integrated_backfill(tmp_path):
    """``kubera run --no-backfill`` does not call integrated backfill."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    html_path = tmp_path / "INFY_NSE_latest.html"
    pilot = MagicMock()
    pilot.prediction_mode = "after_close"
    pilot.prediction_date = date(2026, 4, 8)
    pilot.historical_cutoff_date = date(2026, 4, 7)
    pilot.window_resolution_kind = "natural"
    pilot.window_resolution_reason = "Used the same-day after-close window because the market session is complete."
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.bootstrap.bootstrap"):
                with patch(
                    "kubera.cli.should_run_training_for_current_features",
                    return_value=(False, "aligned"),
                ):
                    with patch("kubera.cli._execute_training_pipeline"):
                        with patch(
                            "kubera.cli._execute_live_predict",
                            return_value=(0, pilot),
                        ):
                            with patch(
                                "kubera.cli.backfill_pending_pilot_actuals_for_cli",
                            ) as mock_bf:
                                with patch("kubera.cli.launch_dashboard"):
                                    with patch("kubera.cli.export_dashboard_html"):
                                        with patch("kubera.cli.PathManager") as mock_pm_cls:
                                            with patch("kubera.cli.webbrowser.open"):
                                                inst = MagicMock()
                                                inst.build_operator_dashboard_html_path.return_value = (
                                                    html_path
                                                )
                                                mock_pm_cls.return_value = inst
                                                exit_code = main(
                                                    ["run", "--no-browser", "--no-backfill"]
                                                )
    assert exit_code == 0
    mock_bf.assert_not_called()


def test_main_run_prints_unified_complete_summary(capsys, tmp_path):
    """``kubera run`` ends with a single Phase-7 recap block (paths, backfill, total time)."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    html_path = tmp_path / "INFY_NSE_latest.html"
    pilot = MagicMock()
    pilot.prediction_mode = "after_close"
    pilot.market_session_date = date(2026, 4, 7)
    pilot.prediction_date = date(2026, 4, 8)
    pilot.historical_cutoff_date = date(2026, 4, 7)
    pilot.status = "success"
    pilot.log_path = tmp_path / "pilot.log"
    pilot.window_resolution_kind = "natural"
    pilot.window_resolution_reason = "Used the same-day after-close window because the market session is complete."
    bf_result = PilotPendingBackfillResult(
        updated_row_count=1,
        unresolved_row_count=0,
        error_count=0,
        effective_as_of=date(2026, 4, 7),
        prediction_dates_attempted=(date(2026, 4, 4),),
    )
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch("kubera.bootstrap.bootstrap"):
                with patch(
                    "kubera.cli.should_run_training_for_current_features",
                    return_value=(False, "stage 4/8 aligned"),
                ):
                    with patch("kubera.cli._execute_training_pipeline"):
                        with patch(
                            "kubera.cli._execute_live_predict",
                            return_value=(0, pilot),
                        ):
                            with patch(
                                "kubera.cli.backfill_pending_pilot_actuals_for_cli",
                                return_value=bf_result,
                            ):
                                with patch("kubera.cli.launch_dashboard"):
                                    with patch("kubera.cli.export_dashboard_html"):
                                        with patch("kubera.cli.PathManager") as mock_pm_cls:
                                            with patch("kubera.cli.webbrowser.open"):
                                                inst = MagicMock()
                                                inst.build_operator_dashboard_html_path.return_value = (
                                                    html_path
                                                )
                                                mock_pm_cls.return_value = inst
                                                exit_code = main(["run", "--no-browser"])
    assert exit_code == 0
    out, _err = capsys.readouterr()
    assert "Kubera run — complete" in out
    assert "Total time:" in out
    assert "Training:" in out and "skipped (stage 4/8 aligned)" in out
    assert "Data refresh:  completed (market + news)" in out
    assert "Resolved window:" in out
    assert "mode=after_close" in out
    assert "market_session_date=2026-04-07" in out
    assert "historical_cutoff_date=2026-04-07" in out
    assert "prediction_date=2026-04-08" in out
    assert "resolution=natural" in out
    assert "Resolution reason: Used the same-day after-close window because the market session is complete." in out
    assert "Pilot:" in out and "status=success" in out
    assert "Backfill:" in out and "updated=1" in out
    assert "Dashboard:" in out and str(html_path) in out


def test_main_predict_backfill_invokes_integrated_backfill():
    """``kubera predict --backfill`` runs integrated backfill."""

    from unittest.mock import MagicMock

    settings = MagicMock()
    runtime = MagicMock()
    runtime.ticker.symbol = "INFY"
    runtime.ticker.exchange = "NSE"
    runtime.paths = settings.paths
    pilot = MagicMock()
    pilot.prediction_mode = "pre_market"
    pilot.prediction_date = date(2026, 4, 8)
    pilot.historical_cutoff_date = date(2026, 4, 7)
    pilot.window_resolution_kind = "override"
    pilot.window_resolution_reason = "Used the explicit mode override with the current time."
    bf_result = PilotPendingBackfillResult(
        updated_row_count=0,
        unresolved_row_count=0,
        error_count=0,
        effective_as_of=date(2026, 4, 7),
        prediction_dates_attempted=(),
    )
    with patch("kubera.cli.load_settings", return_value=settings):
        with patch("kubera.cli.resolve_runtime_settings", return_value=runtime):
            with patch(
                "kubera.cli._execute_live_predict",
                return_value=(0, pilot),
            ):
                with patch(
                    "kubera.cli.backfill_pending_pilot_actuals_for_cli",
                    return_value=bf_result,
                ) as mock_bf:
                    exit_code = main(["predict", "--backfill"])
    assert exit_code == 0
    mock_bf.assert_called_once()
