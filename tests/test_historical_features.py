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
from kubera.utils.paths import PathManager
from kubera.utils.serialization import write_json_file


def make_small_cleaned_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=8)
    close_values = [100.0, 102.0, 101.0, 104.0, 103.0, 103.0, 105.0, 104.0]
    volume_values = [1000, 1100, 1050, 1200, 1150, 1150, 1300, 1250]
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

    with pytest.raises(HistoricalFeatureError, match="missing required columns"):
        validate_cleaned_market_data(
            cleaned_frame,
            ticker="INFY",
            exchange="NSE",
            feature_settings=make_small_feature_settings(),
        )


def test_compute_historical_feature_frame_matches_expected_calculations(
    isolated_repo,
) -> None:
    settings = load_settings().historical_features
    cleaned_frame = validate_cleaned_market_data(
        make_default_cleaned_market_data(),
        ticker="INFY",
        exchange="NSE",
        feature_settings=settings,
    )

    result = compute_historical_feature_frame(
        cleaned_frame,
        settings,
    )

    feature_frame = result.feature_frame
    assert result.warmup_rows_dropped == 251
    assert result.label_rows_dropped == 1
    assert feature_frame.iloc[0]["date"] == cleaned_frame.iloc[251]["date"].strftime("%Y-%m-%d")

    working_frame = cleaned_frame.copy()
    working_frame["close"] = working_frame["close"].astype(float)
    working_frame["volume"] = working_frame["volume"].astype(float)
    expected_index = 251
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
    validated_frame = validate_cleaned_market_data(
        cleaned_frame,
        ticker="INFY",
        exchange="NSE",
        feature_settings=load_settings().historical_features,
    )

    result = compute_historical_feature_frame(validated_frame, load_settings().historical_features)
    flat_date = validated_frame.iloc[280]["date"].strftime("%Y-%m-%d")
    flat_target_date = validated_frame.iloc[281]["date"].strftime("%Y-%m-%d")
    flat_row = result.feature_frame.loc[result.feature_frame["date"] == flat_date].iloc[0]

    assert flat_row["target_date"] == flat_target_date
    assert flat_row["target_next_day_direction"] == 0


def test_zero_previous_volume_uses_neutral_volume_change(isolated_repo) -> None:
    cleaned_frame = make_small_cleaned_market_data()
    cleaned_frame.loc[4, "volume"] = 0
    validated_frame = validate_cleaned_market_data(
        cleaned_frame,
        ticker="INFY",
        exchange="NSE",
        feature_settings=make_small_feature_settings(),
    )

    result = compute_historical_feature_frame(
        validated_frame,
        make_small_feature_settings(),
    )
    row = result.feature_frame.iloc[0]

    assert row["date"] == "2026-01-12"
    assert row["volume_change_1d"] == pytest.approx(0.0)


def test_historical_features_do_not_change_when_only_later_rows_change(
    isolated_repo,
) -> None:
    feature_settings = load_settings().historical_features
    baseline_frame = validate_cleaned_market_data(
        make_default_cleaned_market_data(),
        ticker="INFY",
        exchange="NSE",
        feature_settings=feature_settings,
    )
    changed_frame = baseline_frame.copy()
    changed_frame.loc[280:, "close"] = changed_frame.loc[280:, "close"] + 25.0
    changed_frame.loc[280:, "volume"] = changed_frame.loc[280:, "volume"] * 2

    baseline_result = compute_historical_feature_frame(baseline_frame, feature_settings)
    changed_result = compute_historical_feature_frame(changed_frame, feature_settings)

    pd.testing.assert_frame_equal(
        baseline_result.feature_frame.iloc[:20].reset_index(drop=True),
        changed_result.feature_frame.iloc[:20].reset_index(drop=True),
    )


def test_build_historical_features_persists_outputs_and_metadata(isolated_repo) -> None:
    cleaned_path = write_default_cleaned_inputs(isolated_repo)
    settings = load_settings()
    source_frame = make_default_cleaned_market_data()
    expected_first_ready = pd.Timestamp(source_frame.iloc[251]["date"]).date()
    expected_last_ready = pd.Timestamp(source_frame.iloc[-2]["date"]).date()

    result = build_historical_features(settings)

    assert result.feature_table_path.exists()
    assert result.metadata_path.exists()
    assert result.row_count == len(source_frame) - 251 - 1
    assert result.warmup_rows_dropped == 251
    assert result.label_rows_dropped == 1
    assert result.coverage_start == expected_first_ready
    assert result.coverage_end == expected_last_ready

    feature_frame = pd.read_csv(result.feature_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert feature_frame.columns.tolist() == [
        "date",
        "target_date",
        "ticker",
        "exchange",
        "close",
        "volume",
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
        "target_next_day_direction",
    ]
    assert metadata["source_cleaned_table_path"] == str(cleaned_path)
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
