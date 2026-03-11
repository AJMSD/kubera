"""Historical market-data ingestion for Kubera."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from kubera.config import (
    AppSettings,
    build_provider_symbol as build_config_provider_symbol,
    load_settings,
    resolve_exchange_calendar_name,
    resolve_runtime_settings,
)
from kubera.utils.logging import configure_logging
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
    end_date: date | None = None,
    lookback_months: int | None = None,
    provider: HistoricalMarketDataProvider | None = None,
) -> HistoricalFetchResult:
    """Fetch, validate, and persist historical OHLCV data."""

    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(settings, path_manager)
    logger = configure_logging(run_context, settings.run.log_level)

    request = build_historical_fetch_request(
        settings,
        end_date=end_date,
        lookback_months=lookback_months,
    )
    data_provider = provider or resolve_historical_data_provider(settings)
    raw_frame = data_provider.fetch_daily_ohlcv(request)
    fetched_at_utc = run_context.started_at_utc

    raw_snapshot_path = path_manager.build_raw_market_data_path(
        request.ticker,
        run_context.run_id,
    )
    raw_snapshot_payload = build_raw_snapshot_payload(
        raw_frame,
        request=request,
        fetched_at_utc=fetched_at_utc,
        run_context=run_context,
    )
    write_json_file(raw_snapshot_path, raw_snapshot_payload)

    cleaned_frame, metadata = normalize_historical_market_data(
        raw_frame,
        request=request,
        fetched_at_utc=fetched_at_utc,
        raw_snapshot_path=raw_snapshot_path,
    )
    missing_trading_dates = find_missing_trading_dates(
        cleaned_frame["date"].tolist(),
        exchange=request.exchange,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    metadata["missing_trading_dates"] = missing_trading_dates
    metadata["raw_snapshot_path"] = str(raw_snapshot_path)
    metadata["run_id"] = run_context.run_id
    metadata["git_commit"] = run_context.git_commit
    metadata["git_is_dirty"] = run_context.git_is_dirty

    cleaned_table_path = path_manager.build_processed_market_data_path(
        request.ticker,
        request.exchange,
    )
    metadata_path = path_manager.build_processed_market_data_metadata_path(
        request.ticker,
        request.exchange,
    )
    cleaned_table_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_frame.to_csv(cleaned_table_path, index=False)
    write_json_file(metadata_path, metadata)

    if cleaned_frame.empty:
        raise HistoricalMarketDataProviderError(
            "Historical ingestion produced no valid cleaned rows."
        )

    logger.info(
        "Historical market data ready | ticker=%s | exchange=%s | provider=%s | rows=%s | coverage=%s..%s | duplicates=%s | dropped_rows=%s | missing_trading_dates=%s | cleaned_csv=%s",
        request.ticker,
        request.exchange,
        request.provider,
        len(cleaned_frame),
        metadata["coverage_start"],
        metadata["coverage_end"],
        metadata["duplicate_count"],
        metadata["dropped_row_count"],
        len(missing_trading_dates),
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


def build_historical_fetch_request(
    settings: AppSettings,
    *,
    end_date: date | None = None,
    lookback_months: int | None = None,
) -> HistoricalFetchRequest:
    """Build the normalized historical fetch request from settings."""

    resolved_end_date = end_date or datetime.now(timezone.utc).date()
    resolved_lookback = lookback_months or settings.historical_data.default_lookback_months
    if resolved_lookback < settings.historical_data.minimum_lookback_months:
        raise HistoricalMarketDataProviderError(
            f"Historical lookback must be at least {settings.historical_data.minimum_lookback_months} months."
        )

    provider_symbol = settings.ticker.provider_symbol_map.get(
        "yahoo_finance",
        build_provider_symbol(settings.ticker.symbol, settings.ticker.exchange),
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

    provider_name = settings.providers.historical_data_provider.strip().lower()
    if provider_name == "yfinance":
        return YFinanceHistoricalDataProvider()
    raise HistoricalMarketDataProviderError(
        f"Unsupported historical data provider: {settings.providers.historical_data_provider}"
    )


def normalize_historical_market_data(
    raw_frame: pd.DataFrame,
    *,
    request: HistoricalFetchRequest,
    fetched_at_utc: datetime,
    raw_snapshot_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize, validate, and annotate raw provider rows."""

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
        "row_count": len(records),
        "records": records,
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
) -> list[str]:
    """Compare fetched dates against expected trading sessions."""

    if not actual_dates:
        return []

    expected_trading_days = build_expected_trading_days(
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
