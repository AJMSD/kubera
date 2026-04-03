from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

from kubera.config import HistoricalFeatureSettings, load_settings
from kubera.features.historical_features import (
    HistoricalFeatureError,
    build_historical_features,
    calculate_wilder_rsi,
    compute_historical_feature_frame,
    compute_file_sha256,
    main,
    validate_cleaned_market_data,
)
from kubera.utils.calendar import build_market_calendar
from kubera.utils.paths import PathManager
from kubera.utils.serialization import write_json_file


def make_small_cleaned_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=12)
    close_values = [100.0, 102.0, 101.0, 104.0, 103.0, 103.0, 105.0, 104.0, 106.0, 105.0, 107.0, 108.0]
    volume_values = [1000, 1100, 1050, 1200, 1150, 1150, 1300, 1250, 1400, 1350, 1500, 1600]
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ticker": ["INFY"] * len(dates),
            "exchange": ["NSE"] * len(dates),
            "provider": ["yfinance"] * len(dates),
            "provider_symbol": ["INFY.NS"] * len(dates),
            "open": close_values,
            "high": [value + 1 for value in close_values],
            "low": [value - 1 for value in close_values],
            "close": close_values,
            "adj_close": close_values,
            "volume": volume_values,
            "fetched_at_utc": ["2026-03-10T00:00:00+00:00"] * len(dates),
            "raw_snapshot_path": ["data/raw/source.json"] * len(dates),
        }
    )


def make_default_cleaned_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=320)
    close_values = [
        100.0 + (index * 0.35) + ((index % 5) - 2) * 0.4
        for index in range(len(dates))
    ]
    volume_values = [1000 + (index * 25) + ((index % 3) * 10) for index in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ticker": ["INFY"] * len(dates),
            "exchange": ["NSE"] * len(dates),
            "provider": ["yfinance"] * len(dates),
            "provider_symbol": ["INFY.NS"] * len(dates),
            "open": close_values,
            "high": [value + 1 for value in close_values],
            "low": [value - 1 for value in close_values],
            "close": close_values,
            "adj_close": close_values,
            "volume": volume_values,
            "fetched_at_utc": ["2026-03-10T00:00:00+00:00"] * len(dates),
            "raw_snapshot_path": ["data/raw/source.json"] * len(dates),
        }
    )


def make_small_feature_settings() -> HistoricalFeatureSettings:
    return HistoricalFeatureSettings(
        price_basis="close",
        return_windows=(1, 3, 5),
        moving_average_windows=(5,),
        volatility_windows=(5,),
        rsi_window=5,
        volume_ratio_window=5,
        macd_fast_span=3,
        macd_slow_span=5,
        macd_signal_span=2,
        rolling_year_window=5,
        lag_windows=(1, 2),
        include_day_of_week=True,
        drop_warmup_rows=True,
    )


def write_default_cleaned_inputs(repo_root: Path) -> Path:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    cleaned_path = path_manager.build_processed_market_data_path("INFY", "NSE")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    default_frame = make_default_cleaned_market_data()
    default_frame.to_csv(cleaned_path, index=False)

    metadata_path = path_manager.build_processed_market_data_metadata_path("INFY", "NSE")
    write_json_file(
        metadata_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
            "provider": "yfinance",
            "coverage_start": str(default_frame["date"].min()),
            "coverage_end": str(default_frame["date"].max()),
        },
    )
    return cleaned_path


def reference_rsi(close_values: list[float], window: int) -> list[float | None]:
    deltas = [None]
    deltas.extend(current - previous for previous, current in zip(close_values, close_values[1:]))
    gains = [None if delta is None else max(delta, 0.0) for delta in deltas]
    losses = [None if delta is None else max(-delta, 0.0) for delta in deltas]

    alpha = 1 / window
    average_gain: float | None = None
    average_loss: float | None = None
    output: list[float | None] = []
    non_null_count = 0

    for gain, loss in zip(gains, losses):
        if gain is None or loss is None:
            output.append(None)
            continue

        non_null_count += 1
        if average_gain is None or average_loss is None:
            average_gain = gain
            average_loss = loss
        else:
            average_gain = average_gain + (alpha * (gain - average_gain))
            average_loss = average_loss + (alpha * (loss - average_loss))

        if non_null_count < window:
            output.append(None)
            continue

        if average_gain == 0 and average_loss == 0:
            output.append(50.0)
        elif average_loss == 0 and average_gain > 0:
            output.append(100.0)
        elif average_gain == 0 and average_loss > 0:
            output.append(0.0)
        else:
            relative_strength = average_gain / average_loss
            output.append(100.0 - (100.0 / (1.0 + relative_strength)))

    return output


def test_validate_cleaned_market_data_requires_expected_columns(isolated_repo) -> None:
    cleaned_frame = make_small_cleaned_market_data().drop(columns=["volume"])
    calendar = build_market_calendar(load_settings().market)

    with pytest.raises(HistoricalFeatureError, match="missing required columns"):
        validate_cleaned_market_data(
            cleaned_frame,
            ticker="INFY",
            exchange="NSE",
            feature_settings=make_small_feature_settings(),
            calendar=calendar,
        )


def test_validate_cleaned_market_data_marks_gap_filled_trading_days(isolated_repo) -> None:
    cleaned_frame = make_small_cleaned_market_data().drop(index=[3]).reset_index(drop=True)
    calendar = build_market_calendar(load_settings().market)

    validated_frame = validate_cleaned_market_data(
        cleaned_frame,
        ticker="INFY",
        exchange="NSE",
        feature_settings=make_small_feature_settings(),
        calendar=calendar,
    )

    assert "market_data_gap_flag" in validated_frame.columns
    assert int(validated_frame["market_data_gap_flag"].sum()) == 1
    inserted_row = validated_frame.loc[validated_frame["market_data_gap_flag"] == 1].iloc[0]
    assert inserted_row["volume"] == pytest.approx(0.0)


def test_compute_historical_feature_frame_matches_expected_calculations(
    isolated_repo,
) -> None:
    app_settings = load_settings()
    settings = app_settings.historical_features
    calendar = build_market_calendar(app_settings.market)
    cleaned_frame = validate_cleaned_market_data(
        make_default_cleaned_market_data(),
        ticker="INFY",
        exchange="NSE",
        feature_settings=settings,
        calendar=calendar,
    )

    result = compute_historical_feature_frame(
        cleaned_frame,
        settings,
        calendar=calendar,
    )

    feature_frame = result.feature_frame
    assert result.warmup_rows_dropped == 253
    assert result.label_rows_dropped == 1
    assert feature_frame.iloc[0]["date"] == cleaned_frame.iloc[253]["date"].strftime("%Y-%m-%d")

    working_frame = cleaned_frame.copy()
    working_frame["close"] = working_frame["close"].astype(float)
    working_frame["volume"] = working_frame["volume"].astype(float)
    expected_index = 253
    row = feature_frame.iloc[0]
    expected_date = working_frame.iloc[expected_index]["date"]
    expected_macd_fast = working_frame["close"].ewm(
        span=settings.macd_fast_span,
        adjust=False,
        min_periods=settings.macd_fast_span,
    ).mean()
    expected_macd_slow = working_frame["close"].ewm(
        span=settings.macd_slow_span,
        adjust=False,
        min_periods=settings.macd_slow_span,
    ).mean()
    expected_macd = expected_macd_fast - expected_macd_slow
    expected_macd_signal = expected_macd.ewm(
        span=settings.macd_signal_span,
        adjust=False,
        min_periods=settings.macd_signal_span,
    ).mean()
    expected_52w_high = working_frame["close"].rolling(settings.rolling_year_window).max()
    expected_52w_low = working_frame["close"].rolling(settings.rolling_year_window).min()

    assert row["ret_1d"] == pytest.approx(
        (working_frame.iloc[expected_index]["close"] / working_frame.iloc[expected_index - 1]["close"]) - 1.0
    )
    assert row["ma_20"] == pytest.approx(
        working_frame["close"].iloc[expected_index - 19 : expected_index + 1].mean()
    )
    assert row["macd"] == pytest.approx(float(expected_macd.iloc[expected_index]))
    assert row["macd_signal"] == pytest.approx(float(expected_macd_signal.iloc[expected_index]))
    assert row["price_vs_52w_high"] == pytest.approx(
        float(working_frame.iloc[expected_index]["close"] / expected_52w_high.iloc[expected_index])
    )
    assert row["price_vs_52w_low"] == pytest.approx(
        float(working_frame.iloc[expected_index]["close"] / expected_52w_low.iloc[expected_index])
    )
    assert row["day_of_week"] == expected_date.dayofweek
    assert "market_data_gap_count_5d" in feature_frame.columns
    assert row["target_next_day_direction"] in {0, 1}


def test_calculate_wilder_rsi_matches_reference_implementation() -> None:
    close_values = [100.0, 102.0, 101.0, 104.0, 103.0, 103.0, 105.0, 104.0]
    expected = reference_rsi(close_values, window=5)

    actual_series = calculate_wilder_rsi(pd.Series(close_values), window=5)
    actual = [None if pd.isna(value) else float(value) for value in actual_series.tolist()]

    assert actual[:4] == [None, None, None, None]
    assert actual[4] is None
    assert actual[5:] == pytest.approx(expected[5:])


def test_flat_next_day_close_maps_to_zero_in_final_feature_table(isolated_repo) -> None:
    cleaned_frame = make_default_cleaned_market_data()
    cleaned_frame.loc[281, "close"] = cleaned_frame.loc[280, "close"]
    app_settings = load_settings()
    calendar = build_market_calendar(app_settings.market)
    validated_frame = validate_cleaned_market_data(
        cleaned_frame,
        ticker="INFY",
        exchange="NSE",
        feature_settings=app_settings.historical_features,
        calendar=calendar,
    )

    result = compute_historical_feature_frame(validated_frame, app_settings.historical_features, calendar)
    flat_date = validated_frame.iloc[280]["date"].strftime("%Y-%m-%d")
    flat_target_date = validated_frame.iloc[281]["date"].strftime("%Y-%m-%d")
    flat_row = result.feature_frame.loc[result.feature_frame["date"] == flat_date].iloc[0]

    assert flat_row["target_date"] == flat_target_date
    assert flat_row["target_next_day_direction"] == 0


def test_zero_previous_volume_uses_neutral_volume_change(isolated_repo) -> None:
    cleaned_frame = make_small_cleaned_market_data()
    cleaned_frame.loc[6, "volume"] = 0
    app_settings = load_settings()
    calendar = build_market_calendar(app_settings.market)
    validated_frame = validate_cleaned_market_data(
        cleaned_frame,
        ticker="INFY",
        exchange="NSE",
        feature_settings=make_small_feature_settings(),
        calendar=calendar,
    )

    result = compute_historical_feature_frame(
        validated_frame,
        make_small_feature_settings(),
        calendar=calendar,
    )
    row = result.feature_frame.iloc[0]

    assert row["date"] == "2026-01-14"
    assert row["volume_change_1d"] == pytest.approx(0.0)


def test_historical_features_do_not_change_when_only_later_rows_change(
    isolated_repo,
) -> None:
    app_settings = load_settings()
    feature_settings = app_settings.historical_features
    calendar = build_market_calendar(app_settings.market)
    baseline_frame = validate_cleaned_market_data(
        make_default_cleaned_market_data(),
        ticker="INFY",
        exchange="NSE",
        feature_settings=feature_settings,
        calendar=calendar,
    )
    changed_frame = baseline_frame.copy()
    changed_frame.loc[280:, "close"] = changed_frame.loc[280:, "close"] + 25.0
    changed_frame.loc[280:, "volume"] = changed_frame.loc[280:, "volume"] * 2

    baseline_result = compute_historical_feature_frame(baseline_frame, feature_settings, calendar)
    changed_result = compute_historical_feature_frame(changed_frame, feature_settings, calendar)

    pd.testing.assert_frame_equal(
        baseline_result.feature_frame.iloc[:20].reset_index(drop=True),
        changed_result.feature_frame.iloc[:20].reset_index(drop=True),
    )


def test_build_historical_features_persists_outputs_and_metadata(isolated_repo) -> None:
    cleaned_path = write_default_cleaned_inputs(isolated_repo)
    settings = load_settings()
    source_frame = make_default_cleaned_market_data()
    expected_first_ready = pd.Timestamp(source_frame.iloc[253]["date"]).date()
    expected_last_ready = pd.Timestamp(source_frame.iloc[-2]["date"]).date()

    result = build_historical_features(settings)

    assert result.feature_table_path.exists()
    assert result.metadata_path.exists()
    assert result.warmup_rows_dropped == 253
    assert result.label_rows_dropped == 1

    feature_frame = pd.read_csv(result.feature_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert feature_frame.columns.tolist() == [
        "date",
        "target_date",
        "ticker",
        "exchange",
        "close",
        "volume",
        "market_data_gap_flag",
        "market_data_gap_count_5d",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5d",
        "volatility_10d",
        "volume_change_1d",
        "volume_ma_ratio",
        "macd",
        "macd_signal",
        "price_vs_52w_high",
        "price_vs_52w_low",
        "rsi_14",
        "day_of_week",
        "ret_1d_lag1",
        "ret_3d_lag1",
        "ret_5d_lag1",
        "ma_5_lag1",
        "ma_10_lag1",
        "ma_20_lag1",
        "volatility_5d_lag1",
        "volatility_10d_lag1",
        "volume_change_1d_lag1",
        "volume_ma_ratio_lag1",
        "macd_lag1",
        "macd_signal_lag1",
        "price_vs_52w_high_lag1",
        "price_vs_52w_low_lag1",
        "rsi_14_lag1",
        "day_of_week_lag1",
        "ret_1d_lag2",
        "ret_3d_lag2",
        "ret_5d_lag2",
        "ma_5_lag2",
        "ma_10_lag2",
        "ma_20_lag2",
        "volatility_5d_lag2",
        "volatility_10d_lag2",
        "volume_change_1d_lag2",
        "volume_ma_ratio_lag2",
        "macd_lag2",
        "macd_signal_lag2",
        "price_vs_52w_high_lag2",
        "price_vs_52w_low_lag2",
        "rsi_14_lag2",
        "day_of_week_lag2",
        "target_next_day_direction",
    ]
    assert metadata["source_cleaned_table_path"] == str(cleaned_path)
    assert metadata["gap_filled_row_count"] == 0
    assert metadata["max_recent_gap_count_5d"] == 0
    assert metadata["source_cleaned_metadata_path"].endswith("INFY_NSE_daily.metadata.json")
    assert metadata["source_cleaned_table_hash"] == compute_file_sha256(cleaned_path)
    assert metadata["feature_columns"] == [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5d",
        "volatility_10d",
        "volume_change_1d",
        "volume_ma_ratio",
        "macd",
        "macd_signal",
        "price_vs_52w_high",
        "price_vs_52w_low",
        "rsi_14",
        "day_of_week",
        "ret_1d_lag1",
        "ret_3d_lag1",
        "ret_5d_lag1",
        "ma_5_lag1",
        "ma_10_lag1",
        "ma_20_lag1",
        "volatility_5d_lag1",
        "volatility_10d_lag1",
        "volume_change_1d_lag1",
        "volume_ma_ratio_lag1",
        "macd_lag1",
        "macd_signal_lag1",
        "price_vs_52w_high_lag1",
        "price_vs_52w_low_lag1",
        "rsi_14_lag1",
        "day_of_week_lag1",
        "ret_1d_lag2",
        "ret_3d_lag2",
        "ret_5d_lag2",
        "ma_5_lag2",
        "ma_10_lag2",
        "ma_20_lag2",
        "volatility_5d_lag2",
        "volatility_10d_lag2",
        "volume_change_1d_lag2",
        "volume_ma_ratio_lag2",
        "macd_lag2",
        "macd_signal_lag2",
        "price_vs_52w_high_lag2",
        "price_vs_52w_low_lag2",
        "rsi_14_lag2",
        "day_of_week_lag2",
    ]


def test_build_historical_features_uses_cache_when_source_is_unchanged(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_default_cleaned_inputs(isolated_repo)
    settings = load_settings()
    first_result = build_historical_features(settings)

    def fail_if_recomputed(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Expected cached historical features to be reused.")

    monkeypatch.setattr(
        "kubera.features.historical_features.compute_historical_feature_frame",
        fail_if_recomputed,
    )

    second_result = build_historical_features(settings)

    assert second_result == first_result


def test_force_rebuild_bypasses_feature_cache(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_default_cleaned_inputs(isolated_repo)
    settings = load_settings()
    build_historical_features(settings)

    def fail_if_recomputed(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise HistoricalFeatureError("Recompute path hit")

    monkeypatch.setattr(
        "kubera.features.historical_features.compute_historical_feature_frame",
        fail_if_recomputed,
    )

    with pytest.raises(HistoricalFeatureError, match="Recompute path hit"):
        build_historical_features(settings, force=True)


def test_feature_command_smoke_builds_expected_artifacts(isolated_repo) -> None:
    write_default_cleaned_inputs(isolated_repo)

    exit_code = main([])

    assert exit_code == 0
    assert (
        isolated_repo
        / "data"
        / "features"
        / "historical"
        / "INFY_NSE_historical_features.csv"
    ).exists()
    assert (
        isolated_repo
        / "data"
        / "features"
        / "historical"
        / "INFY_NSE_historical_features.metadata.json"
    ).exists()
