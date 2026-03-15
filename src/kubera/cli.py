"""Unified Kubera command-line interface.

Usage:
    kubera predict  [--mode pre_market|after_close] [--ticker T] [--exchange E]
    kubera train    [--ticker T] [--exchange E]
    kubera ingest   [--ticker T] [--exchange E] [--lookback-days N]
    kubera backfill --date YYYY-MM-DD [--mode MODE] [--ticker T] [--exchange E]
    kubera review   [--date YYYY-MM-DD] [--ticker T] [--exchange E]
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import sys
from typing import Any

from kubera.config import load_settings, resolve_runtime_settings
from kubera.utils.paths import PathManager
from kubera.utils.calendar import build_market_calendar
from kubera.utils.time_utils import is_after_close, is_pre_market, utc_to_market_time

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_detect_prediction_mode(settings: Any) -> str:
    """Return 'pre_market' or 'after_close' based on the current market time."""
    now_utc = datetime.now(timezone.utc)
    market_now = utc_to_market_time(now_utc, settings.market)
    calendar = build_market_calendar(settings.market)
    
    if not calendar.is_trading_day(market_now.date()):
        return "pre_market"

    if is_pre_market(market_now, settings.market):
        return "pre_market"
    if is_after_close(market_now, settings.market):
        return "after_close"
    # During market hours or outside either window, default to after_close.
    return "after_close"


def _add_ticker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", help="NSE/BSE ticker symbol (default: INFY)")
    parser.add_argument("--exchange", help="Exchange code, e.g. NSE or BSE (default: NSE)")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    """Run a single live prediction."""
    from kubera.pilot.live_pilot import run_live_pilot

    settings = load_settings()
    if args.mode:
        mode = args.mode
    else:
        runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
        mode = _auto_detect_prediction_mode(runtime)

    run_live_pilot(
        settings,
        prediction_mode=mode,
        ticker=args.ticker,
        exchange=args.exchange,
        explain=bool(getattr(args, "explain", False)),
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run the full training pipeline: historical features -> baseline -> enhanced."""
    from kubera.features.historical_features import build_historical_features
    from kubera.models.train_baseline import train_baseline_model
    from kubera.models.train_enhanced import train_enhanced_models

    settings = load_settings()
    kwargs: dict[str, Any] = {}
    if args.ticker:
        kwargs["ticker"] = args.ticker
    if args.exchange:
        kwargs["exchange"] = args.exchange

    print(f"[kubera train] Building historical features...")
    build_historical_features(settings, **kwargs)
    print(f"[kubera train] Training baseline model...")
    train_baseline_model(settings, **kwargs)
    print(f"[kubera train] Training enhanced models...")
    train_enhanced_models(settings, **kwargs)
    print(f"[kubera train] Done.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Fetch and process company news."""
    from kubera.ingest.news_data import fetch_company_news

    settings = load_settings()
    result = fetch_company_news(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        lookback_days=args.lookback_days,
    )
    print(
        f"[kubera ingest] Done. rows={result.row_count} "
        f"dropped={result.dropped_row_count} "
        f"duplicates={result.duplicate_count}"
    )
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Backfill actual outcomes for a prediction date."""
    from kubera.pilot.live_pilot import backfill_pilot_actuals

    settings = load_settings()
    if not args.date:
        now_utc = datetime.now(timezone.utc)
        market_now = utc_to_market_time(now_utc, settings.market)
        prediction_date = market_now.date()
        calendar = build_market_calendar(settings.market)
        
        # If today is a weekend, backfill the last valid trading day
        while not calendar.is_trading_day(prediction_date):
            from datetime import timedelta
            prediction_date -= timedelta(days=1)
    else:
        prediction_date = date.fromisoformat(args.date)
    modes = [args.mode] if args.mode else ["pre_market", "after_close"]
    for mode in modes:
        backfill_pilot_actuals(
            settings,
            prediction_date=prediction_date,
            prediction_mode=mode,
            ticker=args.ticker,
            exchange=args.exchange,
        )
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Print recent pilot log entries, optionally filtered by date."""
    settings = load_settings()
    runtime = resolve_runtime_settings(settings, ticker=args.ticker, exchange=args.exchange)
    path_manager = PathManager(runtime.paths)

    target_date = args.date or None
    modes = ["pre_market", "after_close"]
    found_any = False
    for mode in modes:
        log_path = path_manager.build_pilot_log_path(runtime.ticker.symbol, runtime.ticker.exchange, mode)
        if not log_path.exists():
            continue
        try:
            frame = pd.read_csv(log_path)
        except Exception:
            continue
        if frame.empty:
            continue
        if target_date:
            frame = frame[frame["prediction_date"] == target_date]
        else:
            # Show the last 5 rows
            frame = frame.tail(5)
        if frame.empty:
            continue
        found_any = True
        print(f"\n--- {mode} ---")
        cols = [
            "prediction_date",
            "baseline_predicted_next_day_direction",
            "baseline_predicted_probability_up",
            "enhanced_predicted_next_day_direction",
            "enhanced_predicted_probability_up",
            "news_article_count",
            "status",
        ]
        display_cols = [c for c in cols if c in frame.columns]
        print(frame[display_cols].to_string(index=False))
    if not found_any:
        date_label = f" for {target_date}" if target_date else ""
        print(f"No pilot log entries found{date_label}.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kubera",
        description="Kubera: LLM-enhanced NSE stock movement prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # predict
    predict_parser = subparsers.add_parser("predict", help="Run a live prediction.")
    predict_parser.add_argument(
        "--mode",
        choices=["pre_market", "after_close"],
        help="Prediction mode. Auto-detected from current time if not given.",
    )
    predict_parser.add_argument(
        "--explain",
        action="store_true",
        help="Print feature contributions after prediction.",
    )
    _add_ticker_args(predict_parser)

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Build historical features, train baseline and enhanced models.",
    )
    _add_ticker_args(train_parser)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Fetch and process company news.")
    ingest_parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        dest="lookback_days",
        help="Number of days to look back for news (default: from settings).",
    )
    _add_ticker_args(ingest_parser)

    # backfill
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Backfill actual outcomes for a past prediction date.",
    )
    backfill_parser.add_argument(
        "--date",
        required=False,
        help="Prediction date to backfill in YYYY-MM-DD format (default: latest trading day).",
    )
    backfill_parser.add_argument(
        "--mode",
        choices=["pre_market", "after_close"],
        help="Backfill only this prediction mode (default: both).",
    )
    _add_ticker_args(backfill_parser)

    # review
    review_parser = subparsers.add_parser(
        "review",
        help="Print recent pilot log entries (last 5), or for a specific date.",
    )
    review_parser.add_argument(
        "--date",
        help="Filter to this YYYY-MM-DD prediction date.",
    )
    _add_ticker_args(review_parser)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the kubera CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "predict": cmd_predict,
        "train": cmd_train,
        "ingest": cmd_ingest,
        "backfill": cmd_backfill,
        "review": cmd_review,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
