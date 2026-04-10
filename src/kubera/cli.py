"""Unified Kubera command-line interface."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
import webbrowser
from typing import Any

import pandas as pd

from kubera.config import load_settings, resolve_runtime_settings
from kubera.pilot.live_pilot import (
    LivePilotError,
    PilotPendingBackfillResult,
    PilotRunResult,
    backfill_due_pilot_week,
    backfill_pending_pilot_actuals_for_cli,
    backfill_pilot_actuals,
    format_pilot_summary,
    load_pilot_log_frame,
    operate_pilot_week,
    plan_pilot_week,
    resolve_pilot_explanation_output,
    run_due_pilot_week,
    run_live_pilot,
)
from kubera.reporting.final_review import generate_final_review
from kubera.reporting.dashboard import export_dashboard_html, launch_dashboard
from kubera.reporting.offline_evaluation import evaluate_offline, should_run_training_for_current_features
from kubera.utils.calendar import build_market_calendar
from kubera.utils.paths import PathManager
from kubera.utils.time_utils import is_after_close, is_pre_market, utc_to_market_time


DEFAULT_DASH_LIMIT = 20


def _auto_detect_prediction_mode(settings: Any) -> str:
    """Return the active pilot mode based on the current market time."""

    now_utc = datetime.now(timezone.utc)
    market_now = utc_to_market_time(now_utc, settings.market)
    calendar = build_market_calendar(settings.market)
    if not calendar.is_trading_day(market_now.date()):
        return "pre_market"
    if is_pre_market(market_now, settings.market):
        return "pre_market"
    if is_after_close(market_now, settings.market):
        return "after_close"
    return "after_close"


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def _parse_optional_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid timestamp: {value}") from exc
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _add_ticker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", help="Ticker symbol override.")
    parser.add_argument("--exchange", help="Exchange override, for example NSE or BSE.")


def _execute_training_pipeline(args: argparse.Namespace, settings: Any, runtime: Any) -> int:
    """Shared body for ``kubera train`` and ``kubera run``."""

    from datetime import timedelta

    from kubera.features.historical_features import HistoricalFeatureError, build_historical_features
    from kubera.features.news_features import build_news_features
    from kubera.ingest.market_data import (
        HistoricalMarketDataProviderError,
        check_market_data_freshness,
        fetch_historical_market_data,
    )
    from kubera.ingest.news_data import fetch_company_news
    from kubera.llm.extract_news import extract_news
    from kubera.models.train_baseline import BaselineModelError, train_baseline_model
    from kubera.models.train_enhanced import EnhancedModelError, train_enhanced_models
    from kubera.utils.user_failure import describe_domain_error

    try:
        kwargs = {"ticker": args.ticker, "exchange": args.exchange}

        if not getattr(args, "skip_fetch", False):
            target_date = getattr(args, "date", None) or (
                datetime.now(timezone.utc).date() - timedelta(days=1)
            )

            print(f"Checking market data freshness (target: {target_date})...")
            is_fresh, actual_end, reason = check_market_data_freshness(
                runtime,
                ticker=args.ticker,
                exchange=args.exchange,
                required_end_date=target_date,
            )

            if not is_fresh:
                print(f"Market data stale ({reason}). Fetching...")
                fetch_historical_market_data(
                    settings,
                    **kwargs,
                    ensure_fresh_until=target_date,
                )
            else:
                print(f"Market data is fresh ({reason}).")

            print("Fetching latest company news...")
            fetch_company_news(settings, **kwargs)
        else:
            print("Skipping data fetch (--skip-fetch enabled).")

        tune = getattr(args, "tune", False)
        if tune:
            print("Hyperparameter tuning enabled (--tune). This may take several minutes.")
        build_historical_features(settings, **kwargs)
        train_baseline_model(settings, tune=tune, **kwargs)
        extract_news(settings, **kwargs)
        build_news_features(settings, **kwargs)
        train_enhanced_models(settings, tune=tune, **kwargs)
        print("Training pipeline complete.")
        return 0
    except (
        BaselineModelError,
        EnhancedModelError,
        HistoricalFeatureError,
        HistoricalMarketDataProviderError,
    ) as exc:
        print(f"Training stopped: {describe_domain_error(exc)}", file=sys.stderr)
        if os.environ.get("KUBERA_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            traceback.print_exc()
        return 1


def _execute_live_predict(
    args: argparse.Namespace, settings: Any, runtime: Any
) -> tuple[int, PilotRunResult | None]:
    """Shared body for ``kubera predict`` and ``kubera run``."""

    from datetime import timedelta

    from kubera.ingest.market_data import check_market_data_freshness, fetch_historical_market_data
    from kubera.ingest.news_data import fetch_company_news

    prediction_mode = args.mode or _auto_detect_prediction_mode(runtime)

    now_utc = datetime.now(timezone.utc)
    market_now = utc_to_market_time(now_utc, runtime.market)
    calendar = build_market_calendar(runtime.market)

    if (
        prediction_mode == "after_close"
        and calendar.is_trading_day(market_now.date())
        and is_after_close(market_now, runtime.market)
    ):
        required_market_date = market_now.date()
    else:
        required_market_date = market_now.date() - timedelta(days=1)
        while not calendar.is_trading_day(required_market_date):
            required_market_date = required_market_date - timedelta(days=1)

    if not getattr(args, "no_refresh", False):
        is_market_fresh, actual_end_date, market_reason = check_market_data_freshness(
            runtime,
            ticker=args.ticker,
            exchange=args.exchange,
            required_end_date=required_market_date,
        )

        if not is_market_fresh:
            if getattr(args, "interactive", False):
                response = input(f"Market data stale ({market_reason}). Refetch? [y/N]: ")
                if response.lower() not in ("y", "yes"):
                    print("Warning: Proceeding with stale market data")
                else:
                    print(f"Fetching market data up to {required_market_date}...")
                    fetch_historical_market_data(
                        runtime,
                        ticker=args.ticker,
                        exchange=args.exchange,
                        ensure_fresh_until=required_market_date,
                    )
            else:
                print(f"Auto-fetching market data ({market_reason})...")
                fetch_historical_market_data(
                    runtime,
                    ticker=args.ticker,
                    exchange=args.exchange,
                    ensure_fresh_until=required_market_date,
                )

        required_news_cutoff = now_utc
        print("Refreshing news data...")
        fetch_company_news(
            runtime,
            ticker=args.ticker,
            exchange=args.exchange,
            ensure_fresh_until=required_news_cutoff,
        )

    timestamp = _parse_optional_timestamp(args.timestamp)
    result = run_live_pilot(
        settings,
        prediction_mode=prediction_mode,
        timestamp=timestamp,
        ticker=args.ticker,
        exchange=args.exchange,
        explain=bool(getattr(args, "explain", False)),
    )
    print(
        f"Prediction recorded: status={result.status} prediction_date={result.prediction_date} "
        f"log={result.log_path}"
    )
    return 0, result


def _run_integrated_pilot_backfill(
    args: argparse.Namespace,
    settings: Any,
    pilot_result: PilotRunResult,
    *,
    log_prefix: str,
) -> PilotPendingBackfillResult | None:
    """Optional post-predict backfill; failures are logged and do not change exit code."""

    try:
        bf = backfill_pending_pilot_actuals_for_cli(
            settings,
            prediction_mode=pilot_result.prediction_mode,
            current_prediction_date=pilot_result.prediction_date,
            historical_cutoff_date=pilot_result.historical_cutoff_date,
            as_of=getattr(args, "backfill_as_of", None),
            limit=getattr(args, "backfill_limit", None),
            ticker=args.ticker,
            exchange=args.exchange,
        )
    except Exception as exc:
        print(f"{log_prefix} backfill: warning: {exc}", file=sys.stderr)
        return None
    print(
        f"{log_prefix} backfill: updated={bf.updated_row_count} "
        f"unresolved={bf.unresolved_row_count} errors={bf.error_count} "
        f"as_of={bf.effective_as_of.isoformat()}"
    )
    return bf


def _launch_operator_dashboard(
    settings: Any,
    runtime: Any,
    args: argparse.Namespace,
    log_prefix: str,
) -> tuple[Path | None, bool]:
    """Terminal Rich dashboard + optional HTML export (same artifacts as ``kubera run``).

    Returns ``(html_path, browser_opened)``. ``html_path`` is None when ``--no-html``.
    """

    limit = getattr(args, "limit", DEFAULT_DASH_LIMIT)
    print(f"{log_prefix} dashboard (terminal)...")
    launch_dashboard(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        view="latest",
        limit=limit,
    )

    if getattr(args, "no_html", False):
        return None, False

    path_manager = PathManager(runtime.paths)
    html_path = path_manager.build_operator_dashboard_html_path(
        runtime.ticker.symbol,
        runtime.ticker.exchange,
    )
    export_dashboard_html(
        settings,
        html_path,
        ticker=args.ticker,
        exchange=args.exchange,
        view="latest",
        limit=limit,
    )
    print(f"{log_prefix} dashboard HTML: {html_path}")
    browser_opened = not getattr(args, "no_browser", False)
    if browser_opened:
        webbrowser.open(html_path.as_uri())
    return html_path, browser_opened


def _print_kubera_run_complete_summary(
    *,
    training_desc: str,
    data_refresh_desc: str,
    pilot_result: PilotRunResult | None,
    backfill_desc: str,
    html_path: Path | None,
    browser_opened: bool,
    no_html: bool,
    total_seconds: float,
) -> None:
    """Single final stdout block for ``kubera run`` (Phase 7)."""

    width = 72
    resolved_window_line = (
        f"mode={pilot_result.prediction_mode} | market_session_date={pilot_result.market_session_date} | "
        f"historical_cutoff_date={pilot_result.historical_cutoff_date} | "
        f"prediction_date={pilot_result.prediction_date}"
        if pilot_result is not None
        else "n/a"
    )
    pilot_line = (
        f"status={pilot_result.status} | log={pilot_result.log_path}"
        if pilot_result is not None
        else "n/a"
    )
    if no_html:
        dash_line = "terminal=yes | HTML skipped (--no-html) | browser=n/a"
    elif html_path is not None:
        dash_line = (
            f"terminal=yes | HTML={html_path} | browser={'opened' if browser_opened else 'skipped (--no-browser)'}"
        )
    else:
        dash_line = "terminal=yes | HTML=n/a | browser=n/a"

    print()
    print("=" * width)
    print("Kubera run — complete")
    print("=" * width)
    print("Bootstrap:     ok")
    print(f"Training:      {training_desc}")
    print(f"Data refresh:  {data_refresh_desc}")
    print(f"Resolved window: {resolved_window_line}")
    print(f"Pilot:         {pilot_line}")
    print(f"Backfill:      {backfill_desc}")
    print(f"Dashboard:     {dash_line}")
    print(f"Total time:    {total_seconds:.1f}s")
    print("=" * width)


def cmd_sync_holidays(args: argparse.Namespace) -> int:
    """Validate or refresh exchange closure JSON from pinned PDF URLs."""

    from kubera.sync_holidays import run_sync_holidays

    return run_sync_holidays(dry_run=args.dry_run, check_only=args.check_only)


def cmd_setup(args: argparse.Namespace) -> int:
    """Initialize the Kubera runtime directories."""

    del args
    from kubera.bootstrap import bootstrap

    bootstrap()
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """[DEPRECATED] Data fetching is now automatic in 'train' and 'predict' commands."""

    print("Warning: 'kubera fetch' is deprecated.")
    print("Data is now automatically fetched by 'kubera train' and 'kubera predict'.")
    print("If you need manual fetching, use the Python API directly.")

    from kubera.ingest.market_data import fetch_historical_market_data
    from kubera.ingest.news_data import fetch_company_news

    settings = load_settings()
    fetch_historical_market_data(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
    )
    result = fetch_company_news(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        lookback_days=args.lookback_days,
    )
    print(
        f"Fetched news rows={result.row_count} dropped={result.dropped_row_count} "
        f"duplicates={result.duplicate_count}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run the end-to-end training pipeline with auto-fetch."""

    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
    return _execute_training_pipeline(args, settings, runtime)


def cmd_predict(args: argparse.Namespace) -> int:
    """Run one live prediction with smart data refresh."""

    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
    code, pilot_result = _execute_live_predict(args, settings, runtime)
    if code != 0:
        return code
    if getattr(args, "backfill", False) and pilot_result is not None:
        _run_integrated_pilot_backfill(args, settings, pilot_result, log_prefix="[predict]")
    if getattr(args, "dashboard", False):
        _launch_operator_dashboard(settings, runtime, args, "[predict]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Bootstrap, train when needed, predict, and show the dashboard (HTML + terminal)."""

    from kubera.bootstrap import bootstrap

    run_start = time.perf_counter()

    print("[run] bootstrap...")
    bootstrap()
    print("[run] bootstrap: ok")

    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)

    training_desc = ""
    if getattr(args, "no_train", False):
        print("[run] training: skipped (--no-train)")
        training_desc = "skipped (--no-train)"
    elif getattr(args, "retrain", False):
        print("[run] training: forced (--retrain)")
        training_desc = "ran (--retrain)"
        code = _execute_training_pipeline(args, settings, runtime)
        if code != 0:
            return code
    else:
        need_train, reason = should_run_training_for_current_features(
            settings,
            ticker=args.ticker,
            exchange=args.exchange,
        )
        if need_train:
            print(f"[run] training: required ({reason})")
            training_desc = f"ran (required: {reason})"
            code = _execute_training_pipeline(args, settings, runtime)
            if code != 0:
                return code
        else:
            print(f"[run] training: skipped ({reason})")
            training_desc = f"skipped ({reason})"

    code, pilot_result = _execute_live_predict(args, settings, runtime)
    if code != 0:
        return code

    data_refresh_desc = (
        "skipped (--no-refresh)" if getattr(args, "no_refresh", False) else "completed (market + news)"
    )

    backfill_desc = ""
    if getattr(args, "no_backfill", False):
        print("[run] backfill: skipped (--no-backfill)")
        backfill_desc = "skipped (--no-backfill)"
    elif pilot_result is not None:
        bf = _run_integrated_pilot_backfill(args, settings, pilot_result, log_prefix="[run]")
        if bf is None:
            backfill_desc = "warning (see stderr)"
        else:
            backfill_desc = (
                f"updated={bf.updated_row_count} | unresolved={bf.unresolved_row_count} | "
                f"errors={bf.error_count} | as_of={bf.effective_as_of.isoformat()}"
            )
    else:
        backfill_desc = "n/a"

    html_path, browser_opened = _launch_operator_dashboard(settings, runtime, args, "[run]")
    total_seconds = time.perf_counter() - run_start
    _print_kubera_run_complete_summary(
        training_desc=training_desc,
        data_refresh_desc=data_refresh_desc,
        pilot_result=pilot_result,
        backfill_desc=backfill_desc,
        html_path=html_path,
        browser_opened=browser_opened,
        no_html=bool(getattr(args, "no_html", False)),
        total_seconds=total_seconds,
    )
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    """Deprecated: Use 'kubera predict' instead."""

    print("Warning: 'kubera pilot' is deprecated. Use 'kubera predict' instead.")
    return cmd_predict(args)


def cmd_week(args: argparse.Namespace) -> int:
    """Unified weekly operations: plan, execute, and review."""

    settings = load_settings()
    now = _parse_optional_timestamp(getattr(args, "now", None))

    # Determine start/end dates (default: current or next trading week)
    if not args.start_date or not args.end_date:
        runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
        market_now = utc_to_market_time(datetime.now(timezone.utc), runtime.market)
        calendar = build_market_calendar(runtime.market)

        # Find next Monday
        current_date = market_now.date()
        days_until_monday = (7 - current_date.weekday()) % 7 or 7
        monday = current_date + pd.Timedelta(days=days_until_monday)
        while not calendar.is_trading_day(monday.date()):
            monday = monday + pd.Timedelta(days=1)

        # Find Friday of that week
        friday = monday + pd.Timedelta(days=4)
        while not calendar.is_trading_day(friday.date()):
            friday = friday - pd.Timedelta(days=1)

        start_date = args.start_date or monday.date()
        end_date = args.end_date or friday.date()
    else:
        start_date = args.start_date
        end_date = args.end_date

    # Step 1: Plan (unless --execute-only)
    if not getattr(args, "execute_only", False):
        print(f"\n=== Planning pilot week: {start_date} to {end_date} ===")
        plan_result = plan_pilot_week(
            settings,
            pilot_start_date=start_date,
            pilot_end_date=end_date,
            ticker=args.ticker,
            exchange=args.exchange,
        )
        print(f"Pilot week planned: {plan_result.slot_count} slots | {plan_result.manifest_path}")

    # Step 2: Execute due pilots (unless --plan-only)
    if not getattr(args, "plan_only", False):
        print(f"\n=== Executing pilot week: {start_date} to {end_date} ===")
        as_of = getattr(args, "as_of", None)
        operate_result = operate_pilot_week(
            settings,
            pilot_start_date=start_date,
            pilot_end_date=end_date,
            now=now,
            as_of=_parse_iso_date(as_of) if as_of else None,
            dry_run=getattr(args, "dry_run", False),
            ticker=args.ticker,
            exchange=args.exchange,
        )
        print(
            f"Executed: {operate_result.due_result.executed_slot_count}/{operate_result.due_result.due_slot_count} slots | "
            f"Backfilled: {operate_result.backfill_result.updated_row_count} rows"
        )

        # Step 3: Generate review (unless --plan-only or dry-run)
        if not getattr(args, "dry_run", False):
            print(f"\n=== Generating final review ===")
            review_result = generate_final_review(
                settings,
                pilot_start_date=start_date,
                pilot_end_date=end_date,
                refresh_offline_evaluation=False,
                ticker=args.ticker,
                exchange=args.exchange,
            )
            print(f"Review ready: {review_result.summary_markdown_path}")

    print(f"\n=== Week operations complete ===")
    return 0


def cmd_week_plan(args: argparse.Namespace) -> int:
    """[DEPRECATED] Use 'kubera week --plan-only' instead."""

    print("Warning: 'kubera week-plan' is deprecated. Use 'kubera week --plan-only' instead.")
    settings = load_settings()
    result = plan_pilot_week(
        settings,
        pilot_start_date=args.start_date,
        pilot_end_date=args.end_date,
        ticker=args.ticker,
        exchange=args.exchange,
    )
    print(f"Pilot week planned: slots={result.slot_count} manifest={result.manifest_path}")
    return 0


def cmd_due_run(args: argparse.Namespace) -> int:
    """[DEPRECATED] Use 'kubera week --execute-only' instead."""

    print("Warning: 'kubera due-run' is deprecated. Use 'kubera week --execute-only' instead.")
    settings = load_settings()
    now = _parse_optional_timestamp(args.now)
    result = run_due_pilot_week(
        settings,
        plan_path=args.plan_path,
        now=now,
        dry_run=bool(args.dry_run),
    )
    print(
        f"Due-run summary: due_slots={result.due_slot_count} executed={result.executed_slot_count} "
        f"dry_run={result.dry_run}"
    )
    return 0


def cmd_operate_week(args: argparse.Namespace) -> int:
    """[DEPRECATED] Use 'kubera week' instead."""

    print("Warning: 'kubera operate-week' is deprecated. Use 'kubera week' instead.")
    settings = load_settings()
    now = _parse_optional_timestamp(args.now)
    as_of = _parse_iso_date(args.as_of) if args.as_of else None
    result = operate_pilot_week(
        settings,
        pilot_start_date=args.start_date,
        pilot_end_date=args.end_date,
        now=now,
        as_of=as_of,
        dry_run=bool(args.dry_run),
        ticker=args.ticker,
        exchange=args.exchange,
    )
    print(
        f"Operate-week summary: due_slots={result.due_result.due_slot_count} "
        f"executed={result.due_result.executed_slot_count} "
        f"backfilled={result.backfill_result.updated_row_count}"
    )
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Backfill one prediction date or one full pilot week."""

    settings = load_settings()
    if args.start_date and args.end_date:
        result = backfill_due_pilot_week(
            settings,
            pilot_start_date=args.start_date,
            pilot_end_date=args.end_date,
            as_of=args.as_of,
            ticker=args.ticker,
            exchange=args.exchange,
        )
        print(
            f"Pilot-week backfill: updated={result.updated_row_count} "
            f"unresolved={result.unresolved_row_count}"
        )
        return 0

    if args.date is not None:
        prediction_date = args.date
    else:
        runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
        market_now = utc_to_market_time(datetime.now(timezone.utc), runtime.market)
        calendar = build_market_calendar(runtime.market)
        prediction_date = market_now.date()
        while not calendar.is_trading_day(prediction_date):
            prediction_date = pd.Timestamp(prediction_date) - pd.Timedelta(days=1)
            prediction_date = prediction_date.date()

    modes = [args.mode] if args.mode else ["pre_market", "after_close"]
    for prediction_mode in modes:
        try:
            result = backfill_pilot_actuals(
                settings,
                prediction_date=prediction_date,
                prediction_mode=prediction_mode,
                ticker=args.ticker,
                exchange=args.exchange,
            )
        except LivePilotError as exc:
            print(f"Backfill skipped for {prediction_mode}: {exc}")
            continue
        print(
            f"Backfill {prediction_mode}: updated={result.updated_row_count} "
            f"unresolved={result.unresolved_row_count}"
        )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the offline evaluation report."""

    from kubera.features.historical_features import HistoricalFeatureError
    from kubera.ingest.market_data import HistoricalMarketDataProviderError
    from kubera.models.train_baseline import BaselineModelError
    from kubera.models.train_enhanced import EnhancedModelError
    from kubera.utils.user_failure import describe_domain_error

    try:
        settings = load_settings()
        result = evaluate_offline(
            settings,
            ticker=args.ticker,
            exchange=args.exchange,
            force_stage8_refresh=bool(args.force_refresh),
        )
        print(f"Offline evaluation ready: metrics={result.metrics_path}")
        return 0
    except (
        BaselineModelError,
        EnhancedModelError,
        HistoricalFeatureError,
        HistoricalMarketDataProviderError,
    ) as exc:
        print(f"Offline evaluation stopped: {describe_domain_error(exc)}", file=sys.stderr)
        if os.environ.get("KUBERA_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"Offline evaluation failed: {exc}", file=sys.stderr)
        if os.environ.get("KUBERA_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            traceback.print_exc()
        return 1


def cmd_review(args: argparse.Namespace) -> int:
    """[DEPRECATED] Review is now auto-generated by 'kubera week' command."""

    print("Warning: 'kubera review' is deprecated.")
    print("Final review is now automatically generated at the end of 'kubera week'.")

    settings = load_settings()
    result = generate_final_review(
        settings,
        pilot_start_date=args.start_date,
        pilot_end_date=args.end_date,
        refresh_offline_evaluation=bool(args.refresh_offline),
        ticker=args.ticker,
        exchange=args.exchange,
    )
    print(
        f"Final review ready: json={result.summary_json_path} "
        f"markdown={result.summary_markdown_path}"
    )
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """Print recent stored pilot rows."""

    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
    path_manager = PathManager(runtime.paths)
    found_any = False

    for log_path in path_manager.list_existing_pilot_log_paths(
        runtime.ticker.symbol,
        runtime.ticker.exchange,
    ):
        frame = load_pilot_log_frame(log_path)
        if frame.empty:
            continue
        if args.mode:
            frame = frame.loc[frame["prediction_mode"].astype(str) == args.mode].copy()
        if args.date:
            frame = frame.loc[frame["prediction_date"].astype(str) == args.date.isoformat()].copy()
        frame = frame.head(args.limit) if args.date else frame.tail(args.limit)
        if frame.empty:
            continue
        found_any = True
        display_columns = [
            "prediction_date",
            "prediction_mode",
            "status",
            "selected_action",
            "data_quality_grade",
            "news_signal_state",
            "blended_predicted_probability_up",
            "actual_outcome_status",
        ]
        display_frame = frame.loc[:, [c for c in display_columns if c in frame.columns]].copy()
        print(f"\n{log_path.name}")
        print(display_frame.to_string(index=False))

    if not found_any:
        print("No matching pilot rows were found.")
    return 0


def cmd_dash(args: argparse.Namespace) -> int:
    """Launch the dashboard view."""

    from kubera.reporting.dashboard import launch_dashboard

    if args.view == "run" and args.prediction_key is None and args.date is None:
        print("Error: kubera dash --view run requires --prediction-key or --date.")
        return 1

    settings = load_settings()
    launch_dashboard(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        view=args.view,
        prediction_key=args.prediction_key,
        prediction_date=args.date.isoformat() if args.date else None,
        prediction_mode=args.mode,
        limit=args.limit,
    )
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """[DEPRECATED] Use 'kubera predict --explain' or 'kubera dash --view run' instead."""

    print("Warning: 'kubera explain' is deprecated.")
    print("Use 'kubera predict --explain' to see explanations during prediction,")
    print("or 'kubera dash --view run' to view past predictions with explanations.")

    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
    path_manager = PathManager(runtime.paths)
    target_row = None

    for log_path in path_manager.list_existing_pilot_log_paths(
        runtime.ticker.symbol,
        runtime.ticker.exchange,
    ):
        frame = load_pilot_log_frame(log_path)
        if args.prediction_key:
            matches = frame.loc[frame["prediction_key"].astype(str) == args.prediction_key].copy()
        elif args.date:
            matches = frame.loc[frame["prediction_date"].astype(str) == args.date.isoformat()].copy()
            if args.mode:
                matches = matches.loc[matches["prediction_mode"].astype(str) == args.mode].copy()
        else:
            matches = frame.head(1)
        if matches.empty:
            continue
        target_row = matches.iloc[0]
        break

    if target_row is None:
        print("Error: No matching pilot row found.")
        return 1

    snapshot_path = Path(str(target_row["pilot_snapshot_path"]))
    if not snapshot_path.exists():
        print(f"Error: Snapshot file not found at {snapshot_path}")
        return 1

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    print(format_pilot_summary(snapshot_payload))
    print()
    print(resolve_pilot_explanation_output(settings=settings, snapshot_payload=snapshot_payload))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check basic system health and environment setup."""

    del args
    import importlib.util
    import os
    import platform

    print("--- Kubera Health Check ---")
    py_ver = sys.version_info
    print(
        "Python Version: "
        f"{platform.python_version()} {'[OK]' if py_ver.major == 3 and py_ver.minor >= 12 else '[FAIL]'}"
    )

    llm_key = os.getenv("KUBERA_LLM_API_KEY")
    print(f"KUBERA_LLM_API_KEY: {'[SET]' if llm_key else '[MISSING]'}")

    for dependency in ("pandas", "sklearn", "shap", "numpy", "yfinance", "rich"):
        available = importlib.util.find_spec(dependency) is not None
        print(f"Dependency {dependency}: {'[OK]' if available else '[MISSING]'}")

    settings = load_settings()
    data_dir = PathManager(settings.paths).settings.data_dir
    print(f"Data Root: {data_dir} {'[EXISTS]' if data_dir.exists() else '[MISSING]'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree."""

    parser = argparse.ArgumentParser(
        prog="kubera",
        description="Kubera command-line interface. Use 'kubera run' for the default end-to-end flow.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    run_parser = subparsers.add_parser(
        "run",
        help="Default consumer flow: bootstrap, train if needed, predict, and show the dashboard.",
        description=(
            "Default consumer flow. Run 'kubera run' with no flags for the normal end-to-end "
            "experience. Use --mode or --timestamp only for advanced overrides."
        ),
    )
    run_parser.add_argument(
        "--mode",
        choices=["pre_market", "after_close"],
        help="Advanced override for auto-detected mode.",
    )
    run_parser.add_argument("--timestamp", help="Advanced ISO-8601 timestamp override.")
    run_parser.add_argument("--explain", action="store_true", help="Show SHAP feature importances.")
    run_parser.add_argument("--no-refresh", action="store_true", help="Skip automatic data refresh before predict.")
    run_parser.add_argument("--interactive", action="store_true", help="Prompt before refreshing data.")
    run_exclusive = run_parser.add_mutually_exclusive_group()
    run_exclusive.add_argument(
        "--no-train",
        action="store_true",
        help="Never run training; use existing models only.",
    )
    run_exclusive.add_argument(
        "--retrain",
        action="store_true",
        help="Always run the full training pipeline first.",
    )
    run_parser.add_argument("--no-html", action="store_true", help="Skip writing dashboard HTML file.")
    run_parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard HTML in a browser.")
    run_parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search hyperparameters when the training step runs.",
    )
    run_parser.add_argument(
        "--date",
        type=_parse_iso_date,
        help="Training cutoff date when training runs (default: yesterday).",
    )
    run_parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip training-phase data fetch when the training step runs.",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DASH_LIMIT,
        help="Dashboard limit for the stored-runs table.",
    )
    run_parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Skip post-predict pilot log actual-outcome backfill for eligible pending rows.",
    )
    run_parser.add_argument(
        "--backfill-as-of",
        type=_parse_iso_date,
        default=None,
        help="Override as-of date for integrated backfill (default: this run's historical cutoff).",
    )
    run_parser.add_argument(
        "--backfill-limit",
        type=int,
        default=None,
        help="Max distinct prediction dates to backfill per run (newest first; default: all eligible).",
    )
    _add_ticker_args(run_parser)

    subparsers.add_parser("setup", help="Initialize runtime directories.")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch market data and company news.")
    fetch_parser.add_argument("--lookback-days", type=int, default=None)
    _add_ticker_args(fetch_parser)

    train_parser = subparsers.add_parser("train", help="Run the end-to-end training pipeline with auto-fetch.")
    train_parser.add_argument("--skip-fetch", action="store_true", help="Skip automatic data refresh before training.")
    train_parser.add_argument("--date", type=_parse_iso_date, help="Training cutoff date (default: yesterday).")
    train_parser.add_argument("--tune", action="store_true", help="Grid-search hyperparameters using time-series CV (slow but improves accuracy).")
    _add_ticker_args(train_parser)

    pilot_parser = subparsers.add_parser("pilot", help="[DEPRECATED] Use 'predict' instead.")
    pilot_parser.add_argument("--mode", choices=["pre_market", "after_close"])
    pilot_parser.add_argument("--timestamp", help="Optional ISO-8601 timestamp override.")
    pilot_parser.add_argument("--explain", action="store_true")
    pilot_parser.add_argument("--no-refresh", action="store_true", help="Skip automatic data refresh.")
    pilot_parser.add_argument("--interactive", action="store_true", help="Prompt before refreshing data.")
    _add_ticker_args(pilot_parser)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Advanced: run only the live prediction step (dashboard/backfill optional).",
        description=(
            "Advanced single-step path. This skips bootstrap, train-if-needed, and dashboard/"
            "backfill unless you request them explicitly."
        ),
    )
    predict_parser.add_argument(
        "--mode",
        choices=["pre_market", "after_close"],
        help="Optional advanced override for auto-detected mode.",
    )
    predict_parser.add_argument("--timestamp", help="Optional advanced ISO-8601 timestamp override.")
    predict_parser.add_argument("--explain", action="store_true", help="Show SHAP feature importances.")
    predict_parser.add_argument("--no-refresh", action="store_true", help="Skip automatic data refresh.")
    predict_parser.add_argument("--interactive", action="store_true", help="Prompt before refreshing data.")
    predict_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="After predict, show the same terminal + HTML dashboard as 'kubera run' (off by default; use 'kubera run' for bootstrap + train-if-needed + dashboard).",
    )
    predict_parser.add_argument(
        "--no-html",
        action="store_true",
        help="With --dashboard: skip writing dashboard HTML file.",
    )
    predict_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="With --dashboard: do not open the dashboard HTML in a browser.",
    )
    predict_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DASH_LIMIT,
        help="With --dashboard: row limit for the stored-runs table.",
    )
    predict_parser.add_argument(
        "--backfill",
        action="store_true",
        help="After predict, run the same integrated pilot actual-outcome backfill as 'kubera run' (off by default).",
    )
    predict_parser.add_argument(
        "--backfill-as-of",
        type=_parse_iso_date,
        default=None,
        help="With --backfill: override as-of date for eligible pending rows.",
    )
    predict_parser.add_argument(
        "--backfill-limit",
        type=int,
        default=None,
        help="With --backfill: max distinct prediction dates to process per run (newest first).",
    )
    _add_ticker_args(predict_parser)

    week_parser = subparsers.add_parser("week", help="Unified weekly operations: plan, execute, and review.")
    week_parser.add_argument("--start-date", type=_parse_iso_date, help="Week start date (default: next Monday).")
    week_parser.add_argument("--end-date", type=_parse_iso_date, help="Week end date (default: Friday of start week).")
    week_parser.add_argument("--plan-only", action="store_true", help="Only generate the pilot plan, don't execute.")
    week_parser.add_argument("--execute-only", action="store_true", help="Only execute, skip planning phase.")
    week_parser.add_argument("--now", help="Optional ISO-8601 UTC timestamp override.")
    week_parser.add_argument("--as-of", help="Optional ISO date cutoff for backfill.")
    week_parser.add_argument("--dry-run", action="store_true", help="Simulate execution without running pilots.")
    _add_ticker_args(week_parser)

    week_plan_parser = subparsers.add_parser("week-plan", help="[DEPRECATED] Use 'kubera week --plan-only'.")
    week_plan_parser.add_argument("--start-date", type=_parse_iso_date, required=True)
    week_plan_parser.add_argument("--end-date", type=_parse_iso_date, required=True)
    _add_ticker_args(week_plan_parser)

    due_run_parser = subparsers.add_parser("due-run", help="Run due slots from a saved plan.")
    due_run_parser.add_argument("--plan-path", required=True)
    due_run_parser.add_argument("--now", help="Optional ISO-8601 UTC timestamp override.")
    due_run_parser.add_argument("--dry-run", action="store_true")

    operate_week_parser = subparsers.add_parser(
        "operate-week",
        help="Plan, execute due slots, and backfill a pilot week.",
    )
    operate_week_parser.add_argument("--start-date", type=_parse_iso_date, required=True)
    operate_week_parser.add_argument("--end-date", type=_parse_iso_date, required=True)
    operate_week_parser.add_argument("--now", help="Optional ISO-8601 UTC timestamp override.")
    operate_week_parser.add_argument("--as-of", help="Optional ISO date cutoff for backfill.")
    operate_week_parser.add_argument("--dry-run", action="store_true")
    _add_ticker_args(operate_week_parser)

    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Backfill one prediction date or a full pilot-week window.",
    )
    backfill_parser.add_argument("--date", type=_parse_iso_date)
    backfill_parser.add_argument("--mode", choices=["pre_market", "after_close"])
    backfill_parser.add_argument("--start-date", type=_parse_iso_date)
    backfill_parser.add_argument("--end-date", type=_parse_iso_date)
    backfill_parser.add_argument("--as-of", type=_parse_iso_date)
    _add_ticker_args(backfill_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the offline evaluation report.")
    evaluate_parser.add_argument("--force-refresh", action="store_true")
    _add_ticker_args(evaluate_parser)

    review_parser = subparsers.add_parser("review", help="Generate the final review package.")
    review_parser.add_argument("--start-date", type=_parse_iso_date, required=True)
    review_parser.add_argument("--end-date", type=_parse_iso_date, required=True)
    review_parser.add_argument("--refresh-offline", action="store_true")
    _add_ticker_args(review_parser)

    runs_parser = subparsers.add_parser("runs", help="Print recent stored pilot rows.")
    runs_parser.add_argument("--date", type=_parse_iso_date)
    runs_parser.add_argument("--mode", choices=["pre_market", "after_close"])
    runs_parser.add_argument("--limit", type=int, default=10)
    _add_ticker_args(runs_parser)

    dash_parser = subparsers.add_parser("dash", help="Open the terminal dashboard.")
    dash_parser.add_argument("--view", choices=["latest", "all", "run"], default="latest")
    dash_parser.add_argument("--prediction-key")
    dash_parser.add_argument("--date", type=_parse_iso_date)
    dash_parser.add_argument("--mode", choices=["pre_market", "after_close"])
    dash_parser.add_argument("--limit", type=int, default=DEFAULT_DASH_LIMIT)
    _add_ticker_args(dash_parser)

    explain_parser = subparsers.add_parser("explain", help="Print one run summary and explanation.")
    explain_parser.add_argument("--prediction-key")
    explain_parser.add_argument("--date", type=_parse_iso_date)
    explain_parser.add_argument("--mode", choices=["pre_market", "after_close"])
    _add_ticker_args(explain_parser)

    subparsers.add_parser("doctor", help="Check local environment health.")

    sync_holidays_parser = subparsers.add_parser(
        "sync-holidays",
        help="Validate or refresh NSE/BSE exchange closures JSON (see config/holiday_sync.json).",
    )
    sync_holidays_parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config/exchange_closures/india.json without downloading PDFs.",
    )
    sync_holidays_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse PDFs but do not write the output file.",
    )

    ingest_parser = subparsers.add_parser("ingest", help="Alias for `kubera fetch`.")
    ingest_parser.add_argument("--lookback-days", type=int, default=None)
    _add_ticker_args(ingest_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Kubera CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "setup": cmd_setup,
        "fetch": cmd_fetch,
        "train": cmd_train,
        "pilot": cmd_pilot,
        "predict": cmd_predict,
        "run": cmd_run,
        "week": cmd_week,
        "week-plan": cmd_week_plan,
        "due-run": cmd_due_run,
        "operate-week": cmd_operate_week,
        "backfill": cmd_backfill,
        "evaluate": cmd_evaluate,
        "review": cmd_review,
        "runs": cmd_runs,
        "dash": cmd_dash,
        "explain": cmd_explain,
        "doctor": cmd_doctor,
        "sync-holidays": cmd_sync_holidays,
        "ingest": cmd_fetch,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except LivePilotError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if os.environ.get("KUBERA_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
