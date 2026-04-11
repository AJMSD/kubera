"""Historical market-data ingestion for Kubera."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import logging
import time
from typing import Any

import pandas as pd

from kubera.config import (
    AppSettings,
    build_provider_symbol as build_config_provider_symbol,
    catalog_key_for_historical_provider,
    load_settings,
    resolve_exchange_calendar_name,
    resolve_runtime_settings,
)
from kubera.utils.calendar import build_market_calendar, load_exchange_closure_dates
from kubera.utils.logging import configure_logging
from kubera.utils.time_utils import is_after_close, utc_to_market_time
from kubera.utils.paths import PathManager
from kubera.utils.run_context import RunContext, create_run_context
from kubera.utils.serialization import write_json_file


CLEANED_COLUMNS = (
    "date",
    "ticker",
    "exchange",
    "provider",
    "provider_symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "fetched_at_utc",
    "raw_snapshot_path",
)
REQUIRED_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
PROVIDER_COLUMN_MAP = {
    "date": "date",
    "datetime": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}
class HistoricalMarketDataProviderError(RuntimeError):
    """Raised when historical market-data ingestion cannot continue."""


def cap_historical_end_date_before_session_close(
    settings: AppSettings,
    end_date: date,
    *,
    now: datetime | None = None,
) -> tuple[date, str | None]:
    """
    If the inclusive end targets today's session before the regular close, cap to
    the prior trading day so we do not persist or require an unfinalized daily bar.
    """

    clock = now or datetime.now(timezone.utc)
    market_now = utc_to_market_time(clock, settings.market)
    session_today = market_now.date()
    calendar = build_market_calendar(settings.market)
    if end_date != session_today:
        return (end_date, None)
    if not calendar.is_trading_day(end_date):
        return (end_date, None)
    if is_after_close(market_now, settings.market):
        return (end_date, None)
    capped = calendar.previous_trading_day(end_date)
    return (capped, "historical_end_date_capped_before_session_close")


def _cleaned_frame_coverage_end(cleaned_frame: pd.DataFrame) -> date:
    return date.fromisoformat(str(cleaned_frame.iloc[-1]["date"]))


def _after_close_missing_session_bar_retry_applies(
    settings: AppSettings,
    *,
    request_end: date,
    coverage_end: date,
    now: datetime,
) -> bool:
    if coverage_end >= request_end:
        return False
    market_now = utc_to_market_time(now, settings.market)
    if not is_after_close(market_now, settings.market):
        return False
    calendar = build_market_calendar(settings.market)
    if not calendar.is_trading_day(request_end):
        return False
    return True


@dataclass(frozen=True)
class HistoricalFetchRequest:
    ticker: str
    exchange: str
    provider: str
    provider_symbol: str
    start_date: date
    end_date: date
    lookback_months: int


@dataclass(frozen=True)
class HistoricalFetchResult:
    raw_snapshot_path: Path
    cleaned_table_path: Path
    metadata_path: Path
    row_count: int
    coverage_start: date
    coverage_end: date
    duplicate_count: int
    missing_trading_dates: tuple[str, ...]


def resolve_historical_provider_symbol(settings: AppSettings, *, provider_name: str) -> str:
    """Resolve provider_symbol for one historical provider from the ticker catalog."""

    key = catalog_key_for_historical_provider(provider_name)
    mapped = settings.ticker.provider_symbol_map.get(key)
    if mapped:
        return str(mapped).strip()
    if key == "yahoo_finance":
        return build_config_provider_symbol(
            settings.ticker.symbol,
            settings.ticker.exchange,
            provider_name="yahoo_finance",
        )
    raise HistoricalMarketDataProviderError(
        f"Missing ticker catalog provider_symbol_map['{key}'] "
        f"for historical provider '{provider_name}'."
    )


class HistoricalMarketDataProvider(ABC):
    """Boundary for daily OHLCV historical fetches."""

    provider_name: str

    @abstractmethod
    def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
        """Fetch daily OHLCV rows for the requested window."""


class YFinanceHistoricalDataProvider(HistoricalMarketDataProvider):
    """Daily OHLCV provider backed by yfinance."""

    provider_name = "yfinance"

    def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise HistoricalMarketDataProviderError(
                "yfinance is not installed. Install project dependencies before fetching market data."
            ) from exc

        frame = yf.download(
            tickers=request.provider_symbol,
            start=request.start_date.isoformat(),
            end=(request.end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        if not isinstance(frame, pd.DataFrame):
            raise HistoricalMarketDataProviderError(
                "Historical provider returned an unexpected payload type."
            )
        return frame


def fetch_historical_market_data(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    end_date: date | None = None,
    lookback_months: int | None = None,
    provider: HistoricalMarketDataProvider | None = None,
    full_refresh: bool = False,
    ensure_fresh_until: date | None = None,
) -> HistoricalFetchResult:
    """
    Fetch, validate, and persist historical OHLCV data.

    If ensure_fresh_until is set, checks existing data coverage.
    If existing data is fresh (covers up to ensure_fresh_until), returns existing artifacts.
    If stale, fetches missing gap only (incremental refresh).
    """

    settings = resolve_runtime_settings(settings, ticker=ticker, exchange=exchange)

    skip_session_cap = False
    # If ensure_fresh_until is set, check freshness and adjust end_date if needed
    if ensure_fresh_until is not None and end_date is None:
        is_fresh, actual_end_date, reason = check_market_data_freshness(
            settings,
            ticker=ticker,
            exchange=exchange,
            required_end_date=ensure_fresh_until,
        )
        # If fresh, use actual_end_date to enable reuse logic
        # If stale, fetch up to ensure_fresh_until
        if is_fresh and actual_end_date is not None:
            end_date = actual_end_date
            skip_session_cap = True
        elif not is_fresh:
            end_date = ensure_fresh_until

    clock_now = datetime.now(timezone.utc)
    if end_date is None:
        end_date = utc_to_market_time(clock_now, settings.market).date()
    end_date_cap_reason: str | None = None
    if not skip_session_cap:
        end_date, end_date_cap_reason = cap_historical_end_date_before_session_close(
            settings,
            end_date,
            now=clock_now,
        )

    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(settings, path_manager)
    logger = configure_logging(run_context, settings.run.log_level)
    stage_start = time.perf_counter()

    request = build_historical_fetch_request(
        settings,
        end_date=end_date,
        lookback_months=lookback_months,
        now=clock_now,
    )
    if end_date_cap_reason:
        logger.info(
            "Historical end date adjusted | reason=%s | effective_end_date=%s",
            end_date_cap_reason,
            request.end_date.isoformat(),
        )
    fetched_at_utc = run_context.started_at_utc
    raw_snapshot_path = path_manager.build_raw_market_data_path(
        request.ticker,
        run_context.run_id,
    )
    cleaned_table_path = path_manager.build_processed_market_data_path(
        request.ticker,
        request.exchange,
    )
    metadata_path = path_manager.build_processed_market_data_metadata_path(
        request.ticker,
        request.exchange,
    )

    refresh_strategy = "full_refresh"
    effective_fetch_request: HistoricalFetchRequest | None = request
    raw_frame: pd.DataFrame | None = None
    cleaned_frame: pd.DataFrame
    normalized_metadata: dict[str, Any]
    reused_existing_row_count = 0
    existing_artifacts = None
    reused_prefix: pd.DataFrame | None = None

    if not full_refresh:
        existing_artifacts = load_existing_market_artifacts(
            cleaned_table_path=cleaned_table_path,
            metadata_path=metadata_path,
        )
        if existing_artifacts is not None:
            existing_frame, existing_metadata = existing_artifacts
            existing_window = slice_market_window(
                existing_frame,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            if not existing_window.empty:
                existing_coverage_start = date.fromisoformat(str(existing_frame.iloc[0]["date"]))
                existing_coverage_end = date.fromisoformat(str(existing_frame.iloc[-1]["date"]))
                if (
                    existing_coverage_start <= request.start_date
                    and existing_coverage_end >= request.end_date
                ):
                    refresh_strategy = "reuse_existing"
                    effective_fetch_request = None
                    cleaned_frame = existing_frame.copy()
                    normalized_metadata = {
                        "duplicate_count": int(
                            cleaned_frame.duplicated(subset=["date"]).sum()
                        ),
                        "dropped_row_count": 0,
                        "dropped_rows": [],
                    }
                    reused_existing_row_count = int(len(existing_frame))
                elif existing_coverage_end >= request.end_date:
                    refresh_strategy = "full_refresh_missing_head_coverage"
                else:
                    refresh_strategy = "incremental_tail"
                    overlap_start = max(
                        request.start_date,
                        existing_coverage_start,
                        existing_coverage_end
                        - timedelta(days=settings.pilot.historical_incremental_overlap_days),
                    )
                    reused_prefix = existing_frame.loc[
                        pd.to_datetime(existing_frame["date"]).dt.date < overlap_start
                    ].copy()
                    reused_existing_row_count = int(len(reused_prefix))
                    effective_fetch_request = HistoricalFetchRequest(
                        ticker=request.ticker,
                        exchange=request.exchange,
                        provider=request.provider,
                        provider_symbol=resolve_historical_provider_symbol(
                            settings,
                            provider_name=request.provider,
                        ),
                        start_date=overlap_start,
                        end_date=request.end_date,
                        lookback_months=request.lookback_months,
                    )
                    data_provider = provider or resolve_historical_data_provider(settings)
                    raw_frame = data_provider.fetch_daily_ohlcv(effective_fetch_request)
                    write_json_file(
                        raw_snapshot_path,
                        build_raw_snapshot_payload(
                            raw_frame,
                            request=effective_fetch_request,
                            fetched_at_utc=fetched_at_utc,
                            run_context=run_context,
                            refresh_strategy=refresh_strategy,
                            reused_existing_row_count=reused_existing_row_count,
                            reused_metadata_path=metadata_path,
                        ),
                    )
                    incremental_frame, normalized_metadata = normalize_historical_market_data(
                        raw_frame,
                        request=effective_fetch_request,
                        fetched_at_utc=fetched_at_utc,
                        raw_snapshot_path=raw_snapshot_path,
                    )
                    cleaned_frame = pd.concat(
                        [reused_prefix, incremental_frame],
                        ignore_index=True,
                    )
                    cleaned_frame = (
                        cleaned_frame.sort_values("date")
                        .drop_duplicates(subset=["date"], keep="last")
                        .reset_index(drop=True)
                    )

    if effective_fetch_request is request:
        data_provider = provider or resolve_historical_data_provider(settings)
        raw_frame = data_provider.fetch_daily_ohlcv(request)
        write_json_file(
            raw_snapshot_path,
            build_raw_snapshot_payload(
                raw_frame,
                request=request,
                fetched_at_utc=fetched_at_utc,
                run_context=run_context,
                refresh_strategy=refresh_strategy,
                reused_existing_row_count=reused_existing_row_count,
                reused_metadata_path=metadata_path if existing_artifacts is not None else None,
            ),
        )
        cleaned_frame, normalized_metadata = normalize_historical_market_data(
            raw_frame,
            request=request,
            fetched_at_utc=fetched_at_utc,
            raw_snapshot_path=raw_snapshot_path,
        )
    elif effective_fetch_request is None:
        assert existing_artifacts is not None
        existing_metadata = existing_artifacts[1]
        write_json_file(
            raw_snapshot_path,
            build_reuse_snapshot_payload(
                request=request,
                run_context=run_context,
                fetched_at_utc=fetched_at_utc,
                existing_metadata=existing_metadata,
                reused_existing_row_count=reused_existing_row_count,
            ),
        )

    retry_clock = datetime.now(timezone.utc)
    coverage_end_pre_retry = _cleaned_frame_coverage_end(cleaned_frame)
    if _after_close_missing_session_bar_retry_applies(
        settings,
        request_end=request.end_date,
        coverage_end=coverage_end_pre_retry,
        now=retry_clock,
    ):
        logger.warning(
            "Stage 2: cleaned coverage_end %s before required session end %s after market close; "
            "retrying canonical provider once (transient incomplete daily bar).",
            coverage_end_pre_retry.isoformat(),
            request.end_date.isoformat(),
        )
        data_provider_retry = provider or resolve_historical_data_provider(settings)
        if refresh_strategy == "incremental_tail" and reused_prefix is not None:
            assert effective_fetch_request is not None
            raw_retry = data_provider_retry.fetch_daily_ohlcv(effective_fetch_request)
            incremental_retry, normalized_metadata = normalize_historical_market_data(
                raw_retry,
                request=effective_fetch_request,
                fetched_at_utc=fetched_at_utc,
                raw_snapshot_path=raw_snapshot_path,
            )
            cleaned_frame = pd.concat(
                [reused_prefix, incremental_retry],
                ignore_index=True,
            )
            cleaned_frame = (
                cleaned_frame.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
            raw_frame = raw_retry
        else:
            raw_retry = data_provider_retry.fetch_daily_ohlcv(request)
            cleaned_frame, normalized_metadata = normalize_historical_market_data(
                raw_retry,
                request=request,
                fetched_at_utc=fetched_at_utc,
                raw_snapshot_path=raw_snapshot_path,
            )
            raw_frame = raw_retry
        normalized_metadata = {
            **normalized_metadata,
            "stale_session_after_close_retry": True,
        }
        coverage_end_post = _cleaned_frame_coverage_end(cleaned_frame)
        if coverage_end_post < request.end_date:
            raise HistoricalMarketDataProviderError(
                "Historical data still missing a finalized daily bar for "
                f"{request.end_date.isoformat()} after one post-close retry. "
                "The feed may be delayed (e.g. Yahoo Finance); retry later or set "
                "KUBERA_HISTORICAL_CANONICAL_PROVIDER to an alternate source."
            )

    cleaned_table_path.parent.mkdir(parents=True, exist_ok=True)
    if cleaned_frame.empty:
        raise HistoricalMarketDataProviderError(
            "Historical ingestion produced no valid cleaned rows."
        )

    missing_trading_dates = find_missing_trading_dates(
        slice_market_window(
            cleaned_frame,
            start_date=request.start_date,
            end_date=request.end_date,
        )["date"].tolist(),
        exchange=request.exchange,
        start_date=request.start_date,
        end_date=request.end_date,
        closure_dates=load_exchange_closure_dates(settings.market),
    )
    parallel_raw_snapshots: list[dict[str, Any]] = []
    if settings.providers.historical_parallel_providers:
        parallel_raw_snapshots = append_parallel_historical_raw_snapshots(
            settings,
            base_request=request,
            run_context=run_context,
            fetched_at_utc=fetched_at_utc,
            refresh_strategy=refresh_strategy,
            reused_existing_row_count=reused_existing_row_count,
            reused_metadata_path=metadata_path if existing_artifacts is not None else None,
            path_manager=path_manager,
            logger=logger,
        )
    elapsed = round(time.perf_counter() - stage_start, 6)
    finished_at_utc = datetime.now(timezone.utc)
    metadata = build_market_metadata(
        request=request,
        cleaned_frame=cleaned_frame,
        raw_snapshot_path=raw_snapshot_path,
        missing_trading_dates=missing_trading_dates,
        refresh_strategy=refresh_strategy,
        reused_existing_row_count=reused_existing_row_count,
        fetched_row_count=(0 if raw_frame is None else int(len(raw_frame))),
        effective_fetch_request=effective_fetch_request,
        normalized_metadata=normalized_metadata,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
        started_at_utc=fetched_at_utc,
        finished_at_utc=finished_at_utc,
        elapsed_seconds=elapsed,
        parallel_raw_snapshots=parallel_raw_snapshots,
    )
    cleaned_frame.to_csv(cleaned_table_path, index=False)
    write_json_file(metadata_path, metadata)

    logger.info(
        "Historical market data ready | ticker=%s | exchange=%s | provider=%s | strategy=%s | rows=%s | coverage=%s..%s | missing_trading_dates=%s | elapsed=%.3fs | cleaned_csv=%s",
        request.ticker,
        request.exchange,
        request.provider,
        refresh_strategy,
        len(cleaned_frame),
        metadata["coverage_start"],
        metadata["coverage_end"],
        len(missing_trading_dates),
        elapsed,
        cleaned_table_path,
    )

    return HistoricalFetchResult(
        raw_snapshot_path=raw_snapshot_path,
        cleaned_table_path=cleaned_table_path,
        metadata_path=metadata_path,
        row_count=len(cleaned_frame),
        coverage_start=date.fromisoformat(metadata["coverage_start"]),
        coverage_end=date.fromisoformat(metadata["coverage_end"]),
        duplicate_count=metadata["duplicate_count"],
        missing_trading_dates=tuple(missing_trading_dates),
    )


def check_market_data_freshness(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    required_end_date: date | None = None,
) -> tuple[bool, date | None, str]:
    """
    Check if existing market data is fresh enough for the required date.

    Returns:
        (is_fresh, actual_end_date, reason) where:
        - is_fresh: True if data covers up to required_end_date
        - actual_end_date: The actual coverage end date from existing data (None if no data)
        - reason: Human-readable explanation of freshness status
    """
    settings = resolve_runtime_settings(settings, ticker=ticker, exchange=exchange)
    path_manager = PathManager(settings.paths)

    # Default required_end_date to T-1 (yesterday)
    if required_end_date is None:
        required_end_date = datetime.now(timezone.utc).date() - timedelta(days=1)

    cleaned_table_path = path_manager.build_processed_market_data_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )
    metadata_path = path_manager.build_processed_market_data_metadata_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )

    # Check if artifacts exist
    if not cleaned_table_path.exists() or not metadata_path.exists():
        return (False, None, "no existing market data found")

    # Load metadata to check coverage
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        coverage_end_str = metadata.get("coverage_end")
        if not coverage_end_str:
            return (False, None, "metadata missing coverage_end field")

        actual_end_date = date.fromisoformat(coverage_end_str)

        # Compare actual coverage to required
        if actual_end_date >= required_end_date:
            days_ahead = (actual_end_date - required_end_date).days
            if days_ahead > 0:
                return (
                    True,
                    actual_end_date,
                    f"data is fresh (covers {days_ahead} day(s) beyond required date)",
                )
            return (True, actual_end_date, "data is fresh (up to required date)")

        days_behind = (required_end_date - actual_end_date).days
        return (
            False,
            actual_end_date,
            f"data is stale (missing {days_behind} day(s), ends {actual_end_date})",
        )

    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return (False, None, f"failed to read metadata: {exc}")


def load_existing_market_artifacts(
    *,
    cleaned_table_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load saved Stage 2 artifacts when they can support reuse-aware refresh."""

    if not cleaned_table_path.exists() or not metadata_path.exists():
        return None
    try:
        existing_frame = pd.read_csv(cleaned_table_path)
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (pd.errors.EmptyDataError, json.JSONDecodeError, OSError):
        return None

    if existing_frame.empty or not isinstance(existing_metadata, dict):
        return None
    if not set(CLEANED_COLUMNS).issubset(existing_frame.columns):
        return None
    existing_frame = existing_frame.loc[:, CLEANED_COLUMNS].copy()
    parsed_dates = pd.to_datetime(existing_frame["date"], errors="coerce")
    if parsed_dates.isna().any():
        return None
    existing_frame["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    existing_frame = (
        existing_frame.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return existing_frame, existing_metadata


def slice_market_window(
    frame: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Filter one cleaned market-data frame to the requested date window."""

    if frame.empty:
        return frame.copy()
    working_dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    mask = (working_dates >= start_date) & (working_dates <= end_date)
    return frame.loc[mask].copy()


def build_historical_fetch_request(
    settings: AppSettings,
    *,
    end_date: date | None = None,
    lookback_months: int | None = None,
    now: datetime | None = None,
) -> HistoricalFetchRequest:
    """Build the normalized historical fetch request from settings."""

    clock = now or datetime.now(timezone.utc)
    resolved_end_date = (
        end_date
        if end_date is not None
        else utc_to_market_time(clock, settings.market).date()
    )
    resolved_lookback = lookback_months or settings.historical_data.default_lookback_months
    if resolved_lookback < settings.historical_data.minimum_lookback_months:
        raise HistoricalMarketDataProviderError(
            f"Historical lookback must be at least {settings.historical_data.minimum_lookback_months} months."
        )

    provider_symbol = resolve_historical_provider_symbol(
        settings,
        provider_name=settings.providers.historical_data_provider,
    )
    start_date = (
        pd.Timestamp(resolved_end_date) - pd.DateOffset(months=resolved_lookback)
    ).date()

    return HistoricalFetchRequest(
        ticker=settings.ticker.symbol,
        exchange=settings.ticker.exchange,
        provider=settings.providers.historical_data_provider,
        provider_symbol=provider_symbol,
        start_date=start_date,
        end_date=resolved_end_date,
        lookback_months=resolved_lookback,
    )


def resolve_historical_data_provider(settings: AppSettings) -> HistoricalMarketDataProvider:
    """Resolve the active historical provider from settings."""

    return resolve_historical_data_provider_by_name(
        settings,
        settings.providers.historical_data_provider,
    )


def resolve_historical_data_provider_by_name(
    settings: AppSettings,
    provider_name: str,
) -> HistoricalMarketDataProvider:
    """Instantiate one historical provider implementation."""

    name = provider_name.strip().lower()
    if name == "yfinance":
        return YFinanceHistoricalDataProvider()
    if name == "upstox":
        from kubera.ingest.providers.upstox_historical import UpstoxHistoricalDataProvider

        token = settings.providers.upstox_access_token
        if not token:
            raise HistoricalMarketDataProviderError(
                "Upstox requires KUBERA_UPSTOX_ACCESS_TOKEN when used as a historical provider."
            )
        return UpstoxHistoricalDataProvider(token)
    if name == "nsepython":
        from kubera.ingest.providers.nsepython_historical import NsePythonHistoricalDataProvider

        return NsePythonHistoricalDataProvider()
    raise HistoricalMarketDataProviderError(
        f"Unsupported historical data provider: {provider_name}"
    )


def append_parallel_historical_raw_snapshots(
    settings: AppSettings,
    *,
    base_request: HistoricalFetchRequest,
    run_context: RunContext,
    fetched_at_utc: datetime,
    refresh_strategy: str,
    reused_existing_row_count: int,
    reused_metadata_path: Path | None,
    path_manager: PathManager,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Best-effort extra raw JSON snapshots for configured parallel providers."""

    results: list[dict[str, Any]] = []
    for parallel_name in settings.providers.historical_parallel_providers:
        snapshot_path = path_manager.build_raw_market_data_path(
            base_request.ticker,
            run_context.run_id,
            source=parallel_name,
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if parallel_name == "nsepython" and base_request.exchange.strip().upper() == "BSE":
            payload: dict[str, Any] = {
                "provider": parallel_name,
                "provider_symbol": "",
                "ticker": base_request.ticker,
                "exchange": base_request.exchange,
                "requested_start_date": base_request.start_date.isoformat(),
                "requested_end_date": base_request.end_date.isoformat(),
                "lookback_months": base_request.lookback_months,
                "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
                "run_id": run_context.run_id,
                "refresh_strategy": refresh_strategy,
                "reused_existing_row_count": int(reused_existing_row_count),
                "reused_metadata_path": str(reused_metadata_path) if reused_metadata_path else None,
                "skipped": True,
                "skip_reason": "nsepython integration is NSE-only; BSE parallel fetch skipped.",
                "row_count": 0,
                "records": [],
            }
            write_json_file(snapshot_path, payload)
            results.append(
                {
                    "provider": parallel_name,
                    "raw_snapshot_path": str(snapshot_path),
                    "skipped": True,
                    "skip_reason": payload["skip_reason"],
                }
            )
            logger.warning(
                "Parallel historical fetch skipped | provider=%s | reason=%s",
                parallel_name,
                payload["skip_reason"],
            )
            continue

        try:
            parallel_symbol = resolve_historical_provider_symbol(
                settings,
                provider_name=parallel_name,
            )
        except HistoricalMarketDataProviderError as exc:
            err_payload = {
                "provider": parallel_name,
                "ticker": base_request.ticker,
                "exchange": base_request.exchange,
                "error": str(exc),
                "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
                "run_id": run_context.run_id,
            }
            write_json_file(snapshot_path, err_payload)
            results.append(
                {
                    "provider": parallel_name,
                    "raw_snapshot_path": str(snapshot_path),
                    "skipped": False,
                    "error": str(exc),
                }
            )
            logger.warning(
                "Parallel historical fetch failed before provider call | provider=%s | error=%s",
                parallel_name,
                exc,
            )
            continue

        parallel_request = HistoricalFetchRequest(
            ticker=base_request.ticker,
            exchange=base_request.exchange,
            provider=parallel_name,
            provider_symbol=parallel_symbol,
            start_date=base_request.start_date,
            end_date=base_request.end_date,
            lookback_months=base_request.lookback_months,
        )
        try:
            parallel_provider = resolve_historical_data_provider_by_name(settings, parallel_name)
            raw_parallel = parallel_provider.fetch_daily_ohlcv(parallel_request)
            write_json_file(
                snapshot_path,
                build_raw_snapshot_payload(
                    raw_parallel,
                    request=parallel_request,
                    fetched_at_utc=fetched_at_utc,
                    run_context=run_context,
                    refresh_strategy=refresh_strategy,
                    reused_existing_row_count=reused_existing_row_count,
                    reused_metadata_path=reused_metadata_path,
                ),
            )
            results.append(
                {
                    "provider": parallel_name,
                    "raw_snapshot_path": str(snapshot_path),
                    "skipped": False,
                }
            )
        except Exception as exc:  # noqa: BLE001; parallel providers must not fail the stage
            err_payload = {
                "provider": parallel_name,
                "provider_symbol": parallel_symbol,
                "ticker": base_request.ticker,
                "exchange": base_request.exchange,
                "requested_start_date": base_request.start_date.isoformat(),
                "requested_end_date": base_request.end_date.isoformat(),
                "error": str(exc),
                "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
                "run_id": run_context.run_id,
            }
            write_json_file(snapshot_path, err_payload)
            results.append(
                {
                    "provider": parallel_name,
                    "raw_snapshot_path": str(snapshot_path),
                    "skipped": False,
                    "error": str(exc),
                }
            )
            logger.warning(
                "Parallel historical fetch failed | provider=%s | error=%s",
                parallel_name,
                exc,
            )

    return results


def normalize_historical_market_data(
    raw_frame: pd.DataFrame,
    *,
    request: HistoricalFetchRequest,
    fetched_at_utc: datetime,
    raw_snapshot_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize, validate, and annotate raw provider rows.

    Rows missing any of open/high/low/close or valid volume are dropped (e.g. Yahoo
    may emit the current session date before Close is finalized), so persisted
    coverage_end can lag calendar 'today' versus the raw frame's last index date.
    """

    if raw_frame.empty:
        raise HistoricalMarketDataProviderError("Historical provider returned no rows.")

    normalized_columns = normalize_provider_columns(raw_frame)
    working_frame = normalized_columns.reset_index(drop=False)
    working_frame.columns = [
        PROVIDER_COLUMN_MAP.get(str(column).strip().lower(), str(column).strip().lower())
        for column in working_frame.columns
    ]
    if "date" not in working_frame.columns:
        raise HistoricalMarketDataProviderError(
            "Historical provider response does not include a date column."
        )

    for column_name in REQUIRED_NUMERIC_COLUMNS + ("adj_close",):
        if column_name in working_frame.columns:
            working_frame[column_name] = pd.to_numeric(
                working_frame[column_name],
                errors="coerce",
            )

    working_frame["date"] = pd.to_datetime(working_frame["date"], errors="coerce").dt.date
    working_frame["source_row_number"] = range(len(working_frame))

    dropped_rows: list[dict[str, Any]] = []
    valid_row_mask: list[bool] = []
    for row in working_frame.itertuples(index=False):
        reasons: list[str] = []
        row_date = getattr(row, "date")
        if pd.isna(row_date):
            reasons.append("invalid_date")

        for field_name in ("open", "high", "low", "close"):
            if field_name not in working_frame.columns or pd.isna(getattr(row, field_name)):
                reasons.append(f"invalid_{field_name}")

        volume_value = getattr(row, "volume", None)
        if volume_value is None or pd.isna(volume_value) or float(volume_value) < 0:
            reasons.append("invalid_volume")

        high_value = getattr(row, "high", None)
        low_value = getattr(row, "low", None)
        open_value = getattr(row, "open", None)
        close_value = getattr(row, "close", None)
        if not reasons and (
            float(high_value) < max(float(open_value), float(close_value), float(low_value))
            or float(low_value) > min(float(open_value), float(close_value), float(high_value))
        ):
            reasons.append("invalid_high_low_relationship")

        is_valid = not reasons
        valid_row_mask.append(is_valid)
        if not is_valid:
            dropped_rows.append(
                {
                    "source_row_number": getattr(row, "source_row_number"),
                    "date": row_date.isoformat() if isinstance(row_date, date) else None,
                    "reasons": reasons,
                }
            )

    valid_frame = working_frame.loc[valid_row_mask].copy()
    if valid_frame.empty:
        raise HistoricalMarketDataProviderError(
            "Historical provider response did not contain any valid OHLCV rows."
        )

    valid_frame = valid_frame.sort_values(["date", "source_row_number"]).reset_index(drop=True)
    duplicate_count = int(valid_frame.duplicated(subset=["date"]).sum())
    valid_frame = valid_frame.drop_duplicates(subset=["date"], keep="last")

    valid_frame["ticker"] = request.ticker
    valid_frame["exchange"] = request.exchange
    valid_frame["provider"] = request.provider
    valid_frame["provider_symbol"] = request.provider_symbol
    valid_frame["fetched_at_utc"] = fetched_at_utc.astimezone(timezone.utc).isoformat()
    valid_frame["raw_snapshot_path"] = str(raw_snapshot_path)
    if "adj_close" not in valid_frame.columns:
        valid_frame["adj_close"] = pd.NA

    cleaned_frame = valid_frame.loc[:, CLEANED_COLUMNS].copy()
    cleaned_frame["date"] = pd.to_datetime(cleaned_frame["date"]).dt.strftime("%Y-%m-%d")
    cleaned_frame = cleaned_frame.sort_values("date").reset_index(drop=True)

    metadata = {
        "ticker": request.ticker,
        "exchange": request.exchange,
        "provider": request.provider,
        "provider_symbol": request.provider_symbol,
        "requested_start_date": request.start_date.isoformat(),
        "requested_end_date": request.end_date.isoformat(),
        "lookback_months": request.lookback_months,
        "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "row_count": int(len(cleaned_frame)),
        "coverage_start": str(cleaned_frame.iloc[0]["date"]),
        "coverage_end": str(cleaned_frame.iloc[-1]["date"]),
        "duplicate_count": duplicate_count,
        "dropped_row_count": len(dropped_rows),
        "dropped_rows": dropped_rows,
    }
    return cleaned_frame, metadata


def normalize_provider_columns(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider columns into a flat daily OHLCV frame."""

    frame = raw_frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    normalized_columns: list[str] = []
    for column_name in frame.columns:
        normalized_name = PROVIDER_COLUMN_MAP.get(
            str(column_name).strip().lower(),
            str(column_name).strip().lower(),
        )
        normalized_columns.append(normalized_name)
    frame.columns = normalized_columns
    frame.index.name = "date"
    return frame


def build_raw_snapshot_payload(
    raw_frame: pd.DataFrame,
    *,
    request: HistoricalFetchRequest,
    fetched_at_utc: datetime,
    run_context: RunContext,
    refresh_strategy: str,
    reused_existing_row_count: int,
    reused_metadata_path: Path | None,
) -> dict[str, Any]:
    """Build a JSON-safe raw snapshot payload."""

    records = dataframe_to_records(raw_frame)
    return {
        "provider": request.provider,
        "provider_symbol": request.provider_symbol,
        "ticker": request.ticker,
        "exchange": request.exchange,
        "requested_start_date": request.start_date.isoformat(),
        "requested_end_date": request.end_date.isoformat(),
        "lookback_months": request.lookback_months,
        "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "run_id": run_context.run_id,
        "refresh_strategy": refresh_strategy,
        "reused_existing_row_count": int(reused_existing_row_count),
        "reused_metadata_path": str(reused_metadata_path) if reused_metadata_path else None,
        "row_count": len(records),
        "records": records,
    }


def build_reuse_snapshot_payload(
    *,
    request: HistoricalFetchRequest,
    run_context: RunContext,
    fetched_at_utc: datetime,
    existing_metadata: dict[str, Any],
    reused_existing_row_count: int,
) -> dict[str, Any]:
    """Build a raw Stage 2 snapshot payload for a reuse-only refresh."""

    return {
        "provider": request.provider,
        "provider_symbol": request.provider_symbol,
        "ticker": request.ticker,
        "exchange": request.exchange,
        "requested_start_date": request.start_date.isoformat(),
        "requested_end_date": request.end_date.isoformat(),
        "lookback_months": request.lookback_months,
        "fetched_at_utc": fetched_at_utc.astimezone(timezone.utc).isoformat(),
        "run_id": run_context.run_id,
        "refresh_strategy": "reuse_existing",
        "reused_existing_row_count": int(reused_existing_row_count),
        "reused_source_run_id": existing_metadata.get("run_id"),
        "reused_source_raw_snapshot_path": existing_metadata.get("raw_snapshot_path"),
        "row_count": 0,
        "records": [],
    }


def build_market_metadata(
    *,
    request: HistoricalFetchRequest,
    cleaned_frame: pd.DataFrame,
    raw_snapshot_path: Path,
    missing_trading_dates: list[str],
    refresh_strategy: str,
    reused_existing_row_count: int,
    fetched_row_count: int,
    effective_fetch_request: HistoricalFetchRequest | None,
    normalized_metadata: dict[str, Any],
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    elapsed_seconds: float,
    parallel_raw_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the persisted Stage 2 metadata payload."""

    duplicate_count = int(normalized_metadata.get("duplicate_count", 0) or 0)
    dropped_row_count = int(normalized_metadata.get("dropped_row_count", 0) or 0)
    return {
        "ticker": request.ticker,
        "exchange": request.exchange,
        "provider": request.provider,
        "provider_symbol": request.provider_symbol,
        "requested_start_date": request.start_date.isoformat(),
        "requested_end_date": request.end_date.isoformat(),
        "lookback_months": request.lookback_months,
        "fetched_at_utc": started_at_utc.astimezone(timezone.utc).isoformat(),
        "row_count": int(len(cleaned_frame)),
        "coverage_start": str(cleaned_frame.iloc[0]["date"]),
        "coverage_end": str(cleaned_frame.iloc[-1]["date"]),
        "duplicate_count": duplicate_count,
        "dropped_row_count": dropped_row_count,
        "dropped_rows": normalized_metadata.get("dropped_rows", []),
        "missing_trading_dates": missing_trading_dates,
        "raw_snapshot_path": str(raw_snapshot_path),
        "refresh_strategy": refresh_strategy,
        "reused_existing_row_count": int(reused_existing_row_count),
        "effective_fetch_start_date": (
            effective_fetch_request.start_date.isoformat()
            if effective_fetch_request is not None
            else None
        ),
        "effective_fetch_end_date": (
            effective_fetch_request.end_date.isoformat()
            if effective_fetch_request is not None
            else None
        ),
        "workload": {
            "fetched_provider_row_count": int(fetched_row_count),
            "output_row_count": int(len(cleaned_frame)),
            "missing_trading_date_count": int(len(missing_trading_dates)),
            "dropped_row_count": dropped_row_count,
            "stale_session_after_close_retry": bool(
                normalized_metadata.get("stale_session_after_close_retry", False)
            ),
        },
        "timing": {
            "started_at_utc": started_at_utc.astimezone(timezone.utc).isoformat(),
            "finished_at_utc": finished_at_utc.astimezone(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
        },
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
        "parallel_raw_snapshots": parallel_raw_snapshots or [],
    }


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe row records."""

    reset_frame = frame.reset_index(drop=False)
    records: list[dict[str, Any]] = []
    for row in reset_frame.to_dict(orient="records"):
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (datetime, pd.Timestamp)):
                safe_row[str(key)] = value.isoformat()
            elif pd.isna(value):
                safe_row[str(key)] = None
            elif isinstance(value, date):
                safe_row[str(key)] = value.isoformat()
            else:
                safe_row[str(key)] = value
        records.append(safe_row)
    return records


def find_missing_trading_dates(
    actual_dates: list[str],
    *,
    exchange: str,
    start_date: date,
    end_date: date,
    closure_dates: frozenset[date],
) -> list[str]:
    """Compare fetched dates against expected trading sessions."""

    if not actual_dates:
        return []

    expected_trading_days = build_expected_trading_days(
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        closure_dates=closure_dates,
    )
    actual_set = set(actual_dates)
    return [
        trading_day.isoformat()
        for trading_day in expected_trading_days
        if trading_day.isoformat() not in actual_set
    ]


def build_expected_trading_days(
    *,
    exchange: str,
    start_date: date,
    end_date: date,
    closure_dates: frozenset[date],
) -> list[date]:
    """Build the expected exchange trading sessions for the requested window."""

    try:
        import pandas_market_calendars as market_calendars
    except ImportError as exc:
        raise HistoricalMarketDataProviderError(
            "pandas_market_calendars is not installed. Install project dependencies before validating trading sessions."
        ) from exc

    calendar = market_calendars.get_calendar(resolve_exchange_calendar_name(exchange))
    schedule = calendar.schedule(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    return [
        session.date()
        for session in schedule.index.to_pydatetime()
        if session.date() not in closure_dates
    ]


def build_provider_symbol(ticker: str, exchange: str) -> str:
    """Build a provider symbol from the canonical ticker and exchange."""

    return build_config_provider_symbol(ticker, exchange)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse historical market-data ingestion arguments."""

    parser = argparse.ArgumentParser(description="Fetch Kubera historical market data.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--lookback-months",
        type=int,
        help="Override the configured historical lookback window in months.",
    )
    parser.add_argument(
        "--end-date",
        help="Use a specific inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore reusable Stage 2 outputs and refetch the full requested window.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the historical market-data ingestion command."""

    args = parse_args(argv)
    settings = load_settings()
    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
    )
    resolved_end_date = date.fromisoformat(args.end_date) if args.end_date else None
    fetch_historical_market_data(
        runtime_settings,
        end_date=resolved_end_date,
        lookback_months=args.lookback_months,
        full_refresh=args.full_refresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
