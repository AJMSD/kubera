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
    dates = pd.bdate_range("2026-01-05", periods=30)
    close_values = [
        100.0,
        101.0,
        102.0,
        101.0,
        103.0,
        104.0,
        105.0,
        104.0,
        106.0,
        107.0,
        108.0,
        110.0,
        109.0,
        111.0,
        112.0,
        113.0,
        115.0,
        114.0,
        116.0,
        117.0,
        118.0,
        119.0,
        121.0,
        121.0,
        122.0,
        123.0,
        124.0,
        125.0,
        126.0,
        127.0,
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
        drop_warmup_rows=True,
    )


def write_default_cleaned_inputs(repo_root: Path) -> Path:
    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    cleaned_path = path_manager.build_processed_market_data_path("INFY", "NSE")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    make_default_cleaned_market_data().to_csv(cleaned_path, index=False)

    metadata_path = path_manager.build_processed_market_data_metadata_path("INFY", "NSE")
    write_json_file(
        metadata_path,
        {
            "ticker": "INFY",
            "exchange": "NSE",
            "provider": "yfinance",
            "coverage_start": "2026-01-05",
            "coverage_end": "2026-02-13",
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
    cleaned_frame = validate_cleaned_market_data(
        make_small_cleaned_market_data(),
        ticker="INFY",
        exchange="NSE",
        feature_settings=make_small_feature_settings(),
    )

    result = compute_historical_feature_frame(
        cleaned_frame,
        make_small_feature_settings(),
    )

    feature_frame = result.feature_frame
    assert result.warmup_rows_dropped == 5
    assert result.label_rows_dropped == 1
    assert feature_frame["date"].tolist() == ["2026-01-12", "2026-01-13"]

    row = feature_frame.iloc[0]
    assert row["ret_1d"] == pytest.approx(0.0)
    assert row["ret_3d"] == pytest.approx((103.0 / 101.0) - 1.0)
    assert row["ret_5d"] == pytest.approx((103.0 / 100.0) - 1.0)
    assert row["ma_5"] == pytest.approx((102.0 + 101.0 + 104.0 + 103.0 + 103.0) / 5)

    expected_returns = pd.Series(
        [0.02, (101.0 / 102.0) - 1.0, (104.0 / 101.0) - 1.0, (103.0 / 104.0) - 1.0, 0.0]
    )
    assert row["volatility_5d"] == pytest.approx(expected_returns.std(ddof=0))
    assert row["volume_change_1d"] == pytest.approx(0.0)
    assert row["volume_ma_ratio"] == pytest.approx(1150.0 / ((1100 + 1050 + 1200 + 1150 + 1150) / 5))
    assert row["target_next_day_direction"] == 1


def test_calculate_wilder_rsi_matches_reference_implementation() -> None:
    close_values = [100.0, 102.0, 101.0, 104.0, 103.0, 103.0, 105.0, 104.0]
    expected = reference_rsi(close_values, window=5)

    actual_series = calculate_wilder_rsi(pd.Series(close_values), window=5)
    actual = [None if pd.isna(value) else float(value) for value in actual_series.tolist()]

    assert actual[:4] == [None, None, None, None]
    assert actual[4] is None
    assert actual[5:] == pytest.approx(expected[5:])


def test_flat_next_day_close_maps_to_zero_in_final_feature_table(isolated_repo) -> None:
    cleaned_frame = validate_cleaned_market_data(
        make_default_cleaned_market_data(),
        ticker="INFY",
        exchange="NSE",
        feature_settings=load_settings().historical_features,
    )

    result = compute_historical_feature_frame(cleaned_frame, load_settings().historical_features)
    flat_row = result.feature_frame.loc[result.feature_frame["date"] == "2026-02-04"].iloc[0]

    assert flat_row["target_date"] == "2026-02-05"
    assert flat_row["target_next_day_direction"] == 0


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
    changed_frame.loc[24:, "close"] = changed_frame.loc[24:, "close"] + 25.0
    changed_frame.loc[24:, "volume"] = changed_frame.loc[24:, "volume"] * 2

    baseline_result = compute_historical_feature_frame(baseline_frame, feature_settings)
    changed_result = compute_historical_feature_frame(changed_frame, feature_settings)

    unchanged_rows = baseline_result.feature_frame["date"] <= "2026-02-04"
    pd.testing.assert_frame_equal(
        baseline_result.feature_frame.loc[unchanged_rows].reset_index(drop=True),
        changed_result.feature_frame.loc[unchanged_rows].reset_index(drop=True),
    )


def test_build_historical_features_persists_outputs_and_metadata(isolated_repo) -> None:
    cleaned_path = write_default_cleaned_inputs(isolated_repo)
    settings = load_settings()

    result = build_historical_features(settings)

    assert result.feature_table_path.exists()
    assert result.metadata_path.exists()
    assert result.row_count == 10
    assert result.warmup_rows_dropped == 19
    assert result.label_rows_dropped == 1
    assert result.coverage_start == date(2026, 1, 30)
    assert result.coverage_end == date(2026, 2, 12)

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
        "rsi_14",
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
        "rsi_14",
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
