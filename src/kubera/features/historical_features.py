"""Historical feature engineering for Kubera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kubera.config import (
    AppSettings,
    HistoricalFeatureSettings,
    load_settings,
    resolve_runtime_settings,
)
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import create_run_context
from kubera.utils.calendar import build_market_calendar, MarketCalendar
from kubera.utils.hashing import compute_file_sha256 as _compute_file_sha256
from kubera.utils.serialization import write_json_file, write_settings_snapshot


FEATURE_FORMULA_VERSION = "4"
FEATURE_READY_COLUMNS = ("date", "ticker", "exchange", "close", "volume")
OUTPUT_IDENTITY_COLUMNS = ("date", "target_date", "ticker", "exchange", "close", "volume")
REQUIRED_SOURCE_COLUMNS = ("date", "ticker", "exchange", "close", "volume")
MARKET_DATA_GAP_FLAG_COLUMN = "market_data_gap_flag"
MARKET_DATA_GAP_COUNT_5D_COLUMN = "market_data_gap_count_5d"


class HistoricalFeatureError(RuntimeError):
    """Raised when historical feature engineering cannot continue."""


@dataclass(frozen=True)
class HistoricalFeatureComputation:
    """In-memory historical feature table plus drop counts."""

    feature_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    input_row_count: int
    warmup_rows_dropped: int
    label_rows_dropped: int


@dataclass(frozen=True)
class HistoricalFeaturePreparation:
    """Feature-ready rows before the final label filter is applied."""

    feature_ready_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    input_row_count: int
    warmup_rows_dropped: int


@dataclass(frozen=True)
class HistoricalFeatureBuildResult:
    """Persisted historical feature artifact summary."""

    feature_table_path: Path
    metadata_path: Path
    row_count: int
    coverage_start: date
    coverage_end: date
    warmup_rows_dropped: int
    label_rows_dropped: int


def build_historical_features(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    cleaned_table_path: str | Path | None = None,
    force: bool = False,
) -> HistoricalFeatureBuildResult:
    """Build, validate, and persist the historical feature table."""

    runtime_settings = resolve_runtime_settings(
        settings,
        ticker=ticker,
        exchange=exchange,
    )
    path_manager = PathManager(runtime_settings.paths)
    path_manager.ensure_managed_directories()
    run_context = create_run_context(runtime_settings, path_manager)
    write_settings_snapshot(runtime_settings, run_context.config_snapshot_path)
    logger = configure_logging(run_context, runtime_settings.run.log_level)

    source_path = resolve_cleaned_table_path(
        runtime_settings,
        path_manager=path_manager,
        cleaned_table_path=cleaned_table_path,
    )
    if not source_path.exists():
        raise HistoricalFeatureError(
            f"Cleaned historical market-data file does not exist: {source_path}"
        )

    source_metadata_path = infer_cleaned_metadata_path(source_path)
    source_hash = compute_file_sha256(source_path)
    feature_table_path = path_manager.build_historical_feature_table_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    metadata_path = path_manager.build_historical_feature_metadata_path(
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
    )
    feature_config = historical_feature_settings_to_dict(runtime_settings.historical_features)

    cached_result = load_cached_result(
        feature_table_path=feature_table_path,
        metadata_path=metadata_path,
        source_hash=source_hash,
        source_metadata_path=source_metadata_path,
        feature_config=feature_config,
        force=force,
    )
    if cached_result is not None:
        logger.info(
            "Historical features ready from cache | ticker=%s | exchange=%s | rows=%s | feature_csv=%s",
            runtime_settings.ticker.symbol,
            runtime_settings.ticker.exchange,
            cached_result.row_count,
            feature_table_path,
        )
        return cached_result

    cleaned_frame = read_cleaned_market_data(source_path)
    calendar = build_market_calendar(runtime_settings.market)
    validated_frame = validate_cleaned_market_data(
        cleaned_frame,
        ticker=runtime_settings.ticker.symbol,
        exchange=runtime_settings.ticker.exchange,
        feature_settings=runtime_settings.historical_features,
        calendar=calendar,
    )
    computation = compute_historical_feature_frame(
        validated_frame,
        runtime_settings.historical_features,
        calendar=calendar,
    )
    if computation.feature_frame.empty:
        raise HistoricalFeatureError(
            "Historical feature engineering produced no model-ready rows."
        )

    feature_table_path.parent.mkdir(parents=True, exist_ok=True)
    computation.feature_frame.to_csv(feature_table_path, index=False)

    metadata = build_feature_metadata(
        runtime_settings,
        source_path=source_path,
        source_metadata_path=source_metadata_path,
        source_hash=source_hash,
        feature_table_path=feature_table_path,
        run_id=run_context.run_id,
        git_commit=run_context.git_commit,
        git_is_dirty=run_context.git_is_dirty,
        computation=computation,
    )
    write_json_file(metadata_path, metadata)

    logger.info(
        "Historical features ready | ticker=%s | exchange=%s | rows=%s | coverage=%s..%s | warmup_rows=%s | label_rows=%s | feature_csv=%s",
        runtime_settings.ticker.symbol,
        runtime_settings.ticker.exchange,
        computation.feature_frame.shape[0],
        metadata["coverage_start"],
        metadata["coverage_end"],
        metadata["warmup_rows_dropped"],
        metadata["label_rows_dropped"],
        feature_table_path,
    )

    return HistoricalFeatureBuildResult(
        feature_table_path=feature_table_path,
        metadata_path=metadata_path,
        row_count=computation.feature_frame.shape[0],
        coverage_start=date.fromisoformat(metadata["coverage_start"]),
        coverage_end=date.fromisoformat(metadata["coverage_end"]),
        warmup_rows_dropped=computation.warmup_rows_dropped,
        label_rows_dropped=computation.label_rows_dropped,
    )


def resolve_cleaned_table_path(
    settings: AppSettings,
    *,
    path_manager: PathManager,
    cleaned_table_path: str | Path | None,
) -> Path:
    """Resolve the cleaned market-data table path for historical features."""

    if cleaned_table_path is not None:
        return Path(cleaned_table_path).expanduser().resolve()

    return path_manager.build_processed_market_data_path(
        settings.ticker.symbol,
        settings.ticker.exchange,
    )


def infer_cleaned_metadata_path(cleaned_table_path: Path) -> Path | None:
    """Infer the cleaned market-data metadata path when it follows the standard naming rule."""

    if cleaned_table_path.suffix.lower() != ".csv":
        return None

    candidate = cleaned_table_path.with_name(
        f"{cleaned_table_path.stem}.metadata.json"
    )
    if candidate.exists():
        return candidate
    return None


def read_cleaned_market_data(cleaned_table_path: Path) -> pd.DataFrame:
    """Read the cleaned Stage 2 market-data table."""

    try:
        return pd.read_csv(cleaned_table_path)
    except FileNotFoundError as exc:
        raise HistoricalFeatureError(
            f"Cleaned historical market-data file does not exist: {cleaned_table_path}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise HistoricalFeatureError(
            f"Cleaned historical market-data file is empty: {cleaned_table_path}"
        ) from exc


def validate_cleaned_market_data(
    cleaned_frame: pd.DataFrame,
    *,
    ticker: str,
    exchange: str,
    feature_settings: HistoricalFeatureSettings,
    calendar: MarketCalendar,
) -> pd.DataFrame:
    """Validate the cleaned OHLCV table before feature engineering."""

    missing_columns = [column for column in REQUIRED_SOURCE_COLUMNS if column not in cleaned_frame.columns]
    if missing_columns:
        raise HistoricalFeatureError(
            f"Cleaned historical market-data table is missing required columns: {missing_columns}"
        )

    working_frame = cleaned_frame.copy()
    working_frame["date"] = pd.to_datetime(working_frame["date"], errors="coerce")
    if working_frame["date"].isna().any():
        raise HistoricalFeatureError("Cleaned historical market-data table contains invalid dates.")

    if not working_frame["date"].is_monotonic_increasing:
        raise HistoricalFeatureError("Cleaned historical market-data dates must be strictly ascending.")

    if working_frame["date"].duplicated().any():
        raise HistoricalFeatureError("Cleaned historical market-data table contains duplicate dates.")

    if (pd.to_numeric(working_frame["close"], errors="coerce").isna()).any():
        raise HistoricalFeatureError("Cleaned historical market-data table contains invalid close values.")
    working_frame["close"] = pd.to_numeric(working_frame["close"], errors="raise")
    if (working_frame["close"] <= 0).any():
        raise HistoricalFeatureError("Cleaned historical market-data close values must be positive.")

    if (pd.to_numeric(working_frame["volume"], errors="coerce").isna()).any():
        raise HistoricalFeatureError("Cleaned historical market-data table contains invalid volume values.")
    working_frame["volume"] = pd.to_numeric(working_frame["volume"], errors="raise")
    if (working_frame["volume"] < 0).any():
        raise HistoricalFeatureError("Cleaned historical market-data volume values must be non-negative.")

    validate_source_identity(working_frame, ticker=ticker, exchange=exchange)

    minimum_row_count = minimum_required_row_count(feature_settings)
    if len(working_frame) < minimum_row_count:
        raise HistoricalFeatureError(
            "Cleaned historical market-data table does not contain enough rows for the configured feature windows."
        )

    # Reindex to strictly align with market sessions
    start_date = working_frame["date"].min().date()
    end_date = working_frame["date"].max().date()
    
    # We loop from start_date to end_date because valid_days expects strings or datetimes, but we can just use the calendar directly
    curr = start_date
    trading_days = []
    while curr <= end_date:
        if calendar.is_trading_day(curr):
            trading_days.append(pd.Timestamp(curr))
        curr += pd.Timedelta(days=1)
        
    working_frame = working_frame.set_index("date")
    # Reindex to the expected exchange trading sessions and keep a gap marker
    # before filling the synthetic rows for downstream quality scoring.
    reindexed = working_frame.reindex(trading_days)
    missing_session_mask = reindexed.loc[
        :,
        ["close", "volume", "ticker", "exchange"],
    ].isna().any(axis=1)
    reindexed[MARKET_DATA_GAP_FLAG_COLUMN] = missing_session_mask.astype(int)
    if reindexed["close"].isna().any():
        reindexed["close"] = reindexed["close"].ffill()
    if reindexed["volume"].isna().any():
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
    if reindexed["ticker"].isna().any():
        reindexed["ticker"] = reindexed["ticker"].ffill().bfill()
    if reindexed["exchange"].isna().any():
        reindexed["exchange"] = reindexed["exchange"].ffill().bfill()
    
    reindexed = reindexed.reset_index(names="date")

    return reindexed


def validate_source_identity(
    cleaned_frame: pd.DataFrame,
    *,
    ticker: str,
    exchange: str,
) -> None:
    """Ensure the cleaned table matches the requested ticker and exchange."""

    source_tickers = {
        str(value).strip().upper()
        for value in cleaned_frame["ticker"].dropna().unique().tolist()
    }
    source_exchanges = {
        str(value).strip().upper()
        for value in cleaned_frame["exchange"].dropna().unique().tolist()
    }

    if source_tickers != {ticker.upper()}:
        raise HistoricalFeatureError(
            f"Cleaned historical market-data ticker values do not match the requested ticker: {sorted(source_tickers)}"
        )
    if source_exchanges != {exchange.upper()}:
        raise HistoricalFeatureError(
            f"Cleaned historical market-data exchange values do not match the requested exchange: {sorted(source_exchanges)}"
        )


def minimum_required_row_count(feature_settings: HistoricalFeatureSettings) -> int:
    """Return the minimum input row count needed for at least one labeled feature row."""

    maximum_feature_window = max(
        max(feature_settings.return_windows),
        max(feature_settings.moving_average_windows),
        max(feature_settings.volatility_windows) + 1,
        feature_settings.rsi_window,
        feature_settings.volume_ratio_window,
        feature_settings.macd_slow_span + feature_settings.macd_signal_span - 1,
        feature_settings.rolling_year_window,
    )
    if feature_settings.lag_windows:
        maximum_feature_window += max(feature_settings.lag_windows)
    return maximum_feature_window + 1


def compute_historical_feature_frame(
    cleaned_frame: pd.DataFrame,
    feature_settings: HistoricalFeatureSettings,
    calendar: MarketCalendar,
) -> HistoricalFeatureComputation:
    """Compute the v1 historical features and next-day label."""

    preparation = prepare_historical_feature_rows(cleaned_frame, feature_settings, calendar)
    feature_ready_frame = preparation.feature_ready_frame.copy()
    feature_columns = preparation.feature_columns

    label_ready_mask = feature_ready_frame.loc[
        :,
        ("target_date", "target_next_day_direction"),
    ].notna().all(axis=1)
    invalid_label_positions = feature_ready_frame.index[~label_ready_mask].tolist()
    if invalid_label_positions and invalid_label_positions != [feature_ready_frame.index[-1]]:
        raise HistoricalFeatureError(
            "Historical labels are missing before the final source row."
        )

    label_rows_dropped = int((~label_ready_mask).sum())
    final_frame = feature_ready_frame.loc[label_ready_mask].copy()
    final_frame["date"] = pd.to_datetime(final_frame["date"]).dt.strftime("%Y-%m-%d")
    final_frame["target_date"] = pd.to_datetime(final_frame["target_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    final_frame["target_next_day_direction"] = (
        final_frame["target_next_day_direction"].astype(int)
    )
    final_frame = final_frame.reset_index(drop=True)

    return HistoricalFeatureComputation(
        feature_frame=final_frame,
        feature_columns=feature_columns,
        input_row_count=preparation.input_row_count,
        warmup_rows_dropped=preparation.warmup_rows_dropped,
        label_rows_dropped=label_rows_dropped,
    )


def prepare_historical_feature_rows(
    cleaned_frame: pd.DataFrame,
    feature_settings: HistoricalFeatureSettings,
    calendar: MarketCalendar,
) -> HistoricalFeaturePreparation:
    """Compute feature-ready historical rows before dropping the unlabeled tail row."""

    working_frame = cleaned_frame.copy()
    working_frame["close"] = working_frame["close"].astype(float)
    working_frame["volume"] = working_frame["volume"].astype(float)
    if MARKET_DATA_GAP_FLAG_COLUMN in working_frame.columns:
        working_frame[MARKET_DATA_GAP_FLAG_COLUMN] = (
            pd.to_numeric(working_frame[MARKET_DATA_GAP_FLAG_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        working_frame[MARKET_DATA_GAP_COUNT_5D_COLUMN] = (
            working_frame[MARKET_DATA_GAP_FLAG_COLUMN]
            .rolling(window=5, min_periods=1)
            .sum()
            .astype(int)
        )

    for window in feature_settings.return_windows:
        working_frame[f"ret_{window}d"] = (
            working_frame["close"] / working_frame["close"].shift(window) - 1.0
        )

    for window in feature_settings.moving_average_windows:
        working_frame[f"ma_{window}"] = working_frame["close"].rolling(window).mean()

    working_frame["ret_1d_base"] = working_frame["close"].pct_change()
    for window in feature_settings.volatility_windows:
        working_frame[f"volatility_{window}d"] = (
            working_frame["ret_1d_base"].rolling(window).std(ddof=0)
        )

    previous_volume = working_frame["volume"].shift(1)
    working_frame["volume_change_1d"] = calculate_safe_ratio(
        working_frame["volume"],
        previous_volume,
        neutral_value=1.0,
    ) - 1.0
    working_frame["volume_ma_ratio"] = calculate_safe_ratio(
        working_frame["volume"],
        working_frame["volume"].rolling(feature_settings.volume_ratio_window).mean(),
        neutral_value=1.0,
    )
    ema_fast = working_frame["close"].ewm(
        span=feature_settings.macd_fast_span,
        adjust=False,
        min_periods=feature_settings.macd_fast_span,
    ).mean()
    ema_slow = working_frame["close"].ewm(
        span=feature_settings.macd_slow_span,
        adjust=False,
        min_periods=feature_settings.macd_slow_span,
    ).mean()
    working_frame["macd"] = ema_fast - ema_slow
    working_frame["macd_signal"] = working_frame["macd"].ewm(
        span=feature_settings.macd_signal_span,
        adjust=False,
        min_periods=feature_settings.macd_signal_span,
    ).mean()
    rolling_high = working_frame["close"].rolling(feature_settings.rolling_year_window).max()
    rolling_low = working_frame["close"].rolling(feature_settings.rolling_year_window).min()
    working_frame["price_vs_52w_high"] = working_frame["close"] / rolling_high
    working_frame["price_vs_52w_low"] = working_frame["close"] / rolling_low
    if feature_settings.include_day_of_week:
        working_frame["day_of_week"] = working_frame["date"].dt.dayofweek.astype(float)
    working_frame[f"rsi_{feature_settings.rsi_window}"] = calculate_wilder_rsi(
        working_frame["close"],
        feature_settings.rsi_window,
    )

    base_feature_columns = _build_base_feature_columns(feature_settings)
    for window in feature_settings.lag_windows:
        for column in base_feature_columns:
            working_frame[f"{column}_lag{window}"] = working_frame[column].shift(window)

    next_close = working_frame["close"].shift(-1)
    working_frame["target_date"] = working_frame["date"].apply(lambda d: pd.Timestamp(calendar.next_trading_day(d.date())))
    working_frame["target_next_day_direction"] = pd.Series(pd.NA, index=working_frame.index, dtype="Int64")
    valid_target_mask = next_close.notna()
    working_frame.loc[valid_target_mask, "target_next_day_direction"] = (
        next_close.loc[valid_target_mask] > working_frame.loc[valid_target_mask, "close"]
    ).astype(int)

    feature_columns = build_feature_columns(feature_settings)
    auxiliary_output_columns = tuple(
        column
        for column in (
            MARKET_DATA_GAP_FLAG_COLUMN,
            MARKET_DATA_GAP_COUNT_5D_COLUMN,
        )
        if column in working_frame.columns
    )
    output_columns = (
        OUTPUT_IDENTITY_COLUMNS
        + auxiliary_output_columns
        + feature_columns
        + ("target_next_day_direction",)
    )
    output_frame = working_frame.loc[:, output_columns].copy()
    feature_ready_mask = output_frame.loc[
        :,
        FEATURE_READY_COLUMNS + feature_columns,
    ].notna().all(axis=1)
    if not feature_ready_mask.any():
        raise HistoricalFeatureError(
            "Historical feature engineering did not produce any fully populated feature rows."
        )

    first_feature_ready_index = int(feature_ready_mask.idxmax())
    warmup_rows_dropped = first_feature_ready_index
    if not feature_ready_mask.iloc[first_feature_ready_index:].all():
        raise HistoricalFeatureError(
            "Historical feature completeness became non-contiguous after warmup rows."
        )

    if feature_settings.drop_warmup_rows:
        feature_ready_frame = output_frame.iloc[first_feature_ready_index:].copy()
    else:
        feature_ready_frame = output_frame.copy()

    return HistoricalFeaturePreparation(
        feature_ready_frame=feature_ready_frame.reset_index(drop=True),
        feature_columns=feature_columns,
        input_row_count=len(cleaned_frame),
        warmup_rows_dropped=warmup_rows_dropped,
    )


def build_live_historical_feature_row(
    cleaned_frame: pd.DataFrame,
    feature_settings: HistoricalFeatureSettings,
    calendar: MarketCalendar,
    *,
    prediction_date: date,
) -> pd.DataFrame:
    """Build one unlabeled live snapshot row for the requested prediction date."""

    preparation = prepare_historical_feature_rows(cleaned_frame, feature_settings, calendar)
    if preparation.feature_ready_frame.empty:
        raise HistoricalFeatureError(
            "Historical feature engineering did not leave any feature-ready rows for live inference."
        )

    live_row = preparation.feature_ready_frame.iloc[[-1]].copy()
    historical_date = pd.Timestamp(live_row.iloc[0]["date"]).date()
    if prediction_date <= historical_date:
        raise HistoricalFeatureError(
            "Live prediction date must be later than the last available historical feature date."
        )

    live_row["date"] = pd.to_datetime(live_row["date"]).dt.strftime("%Y-%m-%d")
    live_row["target_date"] = prediction_date.isoformat()
    live_row["target_next_day_direction"] = pd.NA
    return live_row.reset_index(drop=True)


def calculate_wilder_rsi(close_series: pd.Series, window: int) -> pd.Series:
    """Compute RSI with Wilder smoothing."""

    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    both_zero_mask = (average_gain == 0) & (average_loss == 0)
    only_gain_mask = (average_loss == 0) & (average_gain > 0)
    only_loss_mask = (average_gain == 0) & (average_loss > 0)

    rsi = rsi.mask(both_zero_mask, 50.0)
    rsi = rsi.mask(only_gain_mask, 100.0)
    rsi = rsi.mask(only_loss_mask, 0.0)
    return rsi


def calculate_safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    neutral_value: float,
) -> pd.Series:
    """Divide one series by another and neutralize zero-denominator rows."""

    result = numerator / denominator.where(denominator != 0)
    return result.mask(denominator == 0, neutral_value)


def build_feature_metadata(
    settings: AppSettings,
    *,
    source_path: Path,
    source_metadata_path: Path | None,
    source_hash: str,
    feature_table_path: Path,
    run_id: str,
    git_commit: str | None,
    git_is_dirty: bool | None,
    computation: HistoricalFeatureComputation,
) -> dict[str, Any]:
    """Build the metadata payload for a persisted historical feature table."""

    feature_frame = computation.feature_frame
    return {
        "ticker": settings.ticker.symbol,
        "exchange": settings.ticker.exchange,
        "feature_table_path": str(feature_table_path),
        "source_cleaned_table_path": str(source_path),
        "source_cleaned_metadata_path": str(source_metadata_path) if source_metadata_path else None,
        "source_cleaned_table_hash": source_hash,
        "feature_config": historical_feature_settings_to_dict(settings.historical_features),
        "feature_columns": list(computation.feature_columns),
        "target_column": "target_next_day_direction",
        "formula_version": FEATURE_FORMULA_VERSION,
        "input_row_count": computation.input_row_count,
        "output_row_count": int(feature_frame.shape[0]),
        "warmup_rows_dropped": computation.warmup_rows_dropped,
        "label_rows_dropped": computation.label_rows_dropped,
        "gap_filled_row_count": int(
            pd.to_numeric(
                feature_frame.get(
                    MARKET_DATA_GAP_FLAG_COLUMN,
                    pd.Series([0] * len(feature_frame)),
                ),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "max_recent_gap_count_5d": int(
            pd.to_numeric(
                feature_frame.get(
                    MARKET_DATA_GAP_COUNT_5D_COLUMN,
                    pd.Series([0] * len(feature_frame)),
                ),
                errors="coerce",
            )
            .fillna(0)
            .max()
        ),
        "coverage_start": str(feature_frame.iloc[0]["date"]),
        "coverage_end": str(feature_frame.iloc[-1]["date"]),
        "run_id": run_id,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
    }


def historical_feature_settings_to_dict(
    feature_settings: HistoricalFeatureSettings,
) -> dict[str, Any]:
    """Serialize the historical feature settings into plain values."""

    return {
        "price_basis": feature_settings.price_basis,
        "return_windows": list(feature_settings.return_windows),
        "moving_average_windows": list(feature_settings.moving_average_windows),
        "volatility_windows": list(feature_settings.volatility_windows),
        "rsi_window": feature_settings.rsi_window,
        "volume_ratio_window": feature_settings.volume_ratio_window,
        "macd_fast_span": feature_settings.macd_fast_span,
        "macd_slow_span": feature_settings.macd_slow_span,
        "macd_signal_span": feature_settings.macd_signal_span,
        "rolling_year_window": feature_settings.rolling_year_window,
        "include_day_of_week": feature_settings.include_day_of_week,
        "drop_warmup_rows": feature_settings.drop_warmup_rows,
        "lag_windows": list(feature_settings.lag_windows),
    }


def _build_base_feature_columns(feature_settings: HistoricalFeatureSettings) -> tuple[str, ...]:
    """Build the base feature column names before lags."""

    columns = [
        *(f"ret_{window}d" for window in feature_settings.return_windows),
        *(f"ma_{window}" for window in feature_settings.moving_average_windows),
        *(f"volatility_{window}d" for window in feature_settings.volatility_windows),
        "volume_change_1d",
        "volume_ma_ratio",
        "macd",
        "macd_signal",
        "price_vs_52w_high",
        "price_vs_52w_low",
        f"rsi_{feature_settings.rsi_window}",
    ]
    if feature_settings.include_day_of_week:
        columns.append("day_of_week")
    return tuple(columns)


def build_feature_columns(feature_settings: HistoricalFeatureSettings) -> tuple[str, ...]:
    """Build all feature column names including configured lags."""

    base_columns = _build_base_feature_columns(feature_settings)
    columns = list(base_columns)
    for window in feature_settings.lag_windows:
        for column in base_columns:
            columns.append(f"{column}_lag{window}")
    return tuple(columns)


def compute_file_sha256(path: Path) -> str:
    """Hash a file so repeated feature builds can reuse cached artifacts."""

    return _compute_file_sha256(path)


def load_cached_result(
    *,
    feature_table_path: Path,
    metadata_path: Path,
    source_hash: str,
    source_metadata_path: Path | None,
    feature_config: dict[str, Any],
    force: bool,
) -> HistoricalFeatureBuildResult | None:
    """Return the cached feature result when the source and config still match."""

    if force or not feature_table_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    expected_metadata_path = str(source_metadata_path) if source_metadata_path else None
    if metadata.get("formula_version") != FEATURE_FORMULA_VERSION:
        return None
    if metadata.get("source_cleaned_table_hash") != source_hash:
        return None
    if metadata.get("feature_config") != feature_config:
        return None
    if metadata.get("source_cleaned_metadata_path") != expected_metadata_path:
        return None

    try:
        return HistoricalFeatureBuildResult(
            feature_table_path=feature_table_path,
            metadata_path=metadata_path,
            row_count=int(metadata["output_row_count"]),
            coverage_start=date.fromisoformat(metadata["coverage_start"]),
            coverage_end=date.fromisoformat(metadata["coverage_end"]),
            warmup_rows_dropped=int(metadata["warmup_rows_dropped"]),
            label_rows_dropped=int(metadata["label_rows_dropped"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse historical feature-engineering command arguments."""

    parser = argparse.ArgumentParser(description="Build Kubera historical features.")
    parser.add_argument("--ticker", help="Override the configured ticker symbol.")
    parser.add_argument("--exchange", help="Override the configured exchange code.")
    parser.add_argument(
        "--cleaned-path",
        help="Use a specific cleaned historical market-data CSV file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild features even if the cached artifact still matches the source table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the historical feature-engineering command."""

    args = parse_args(argv)
    settings = load_settings()
    build_historical_features(
        settings,
        ticker=args.ticker,
        exchange=args.exchange,
        cleaned_table_path=args.cleaned_path,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
