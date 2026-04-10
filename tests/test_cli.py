"""Smoke tests for the Kubera CLI."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from kubera.cli import (
    cmd_dash,
    cmd_doctor,
    cmd_evaluate,
    cmd_setup,
    main,
)
from kubera.pilot.live_pilot import PilotPendingBackfillResult

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
