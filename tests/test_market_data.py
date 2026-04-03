from __future__ import annotations

from datetime import date
import json

import pandas as pd
import pytest

from kubera.config import load_settings
from kubera.ingest.market_data import (
    HistoricalFetchRequest,
    HistoricalMarketDataProvider,
    HistoricalMarketDataProviderError,
    build_expected_trading_days,
    build_historical_fetch_request,
    build_provider_symbol,
    check_market_data_freshness,
    fetch_historical_market_data,
    main,
    normalize_historical_market_data,
)


class FakeHistoricalProvider(HistoricalMarketDataProvider):
    provider_name = "fake_provider"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.call_count = 0

    def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
        self.call_count += 1
        return self._frame.copy()


def make_ohlcv_frame(date_values: list[str]) -> pd.DataFrame:
    base_values = list(range(len(date_values)))
    return pd.DataFrame(
        {
            "Open": [100.0 + value for value in base_values],
            "High": [101.0 + value for value in base_values],
            "Low": [99.0 + value for value in base_values],
            "Close": [100.5 + value for value in base_values],
            "Adj Close": [100.5 + value for value in base_values],
            "Volume": [1000 + (value * 10) for value in base_values],
        },
        index=pd.to_datetime(date_values),
    )


def make_provider_frame() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2026-03-09",
            "2026-03-10",
            "2026-03-10",
            "2026-03-13",
        ]
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 110.0],
            "High": [102.0, 103.0, 105.0, 108.0],
            "Low": [99.0, 100.0, 101.0, 111.0],
            "Close": [101.0, 102.0, 104.0, 107.0],
            "Adj Close": [101.0, 102.0, 104.0, 107.0],
            "Volume": [1000, 1500, 1600, 1700],
        },
        index=index,
    )


def test_build_historical_request_uses_yahoo_symbol_mapping(isolated_repo) -> None:
    settings = load_settings()

    request = build_historical_fetch_request(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
    )

    assert request.ticker == "INFY"
    assert request.exchange == "NSE"
    assert request.provider == "yfinance"
    assert request.provider_symbol == "INFY.NS"


def test_normalize_historical_market_data_dedupes_and_drops_invalid_rows(
    isolated_repo,
) -> None:
    settings = load_settings()
    request = build_historical_fetch_request(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
    )

    cleaned_frame, metadata = normalize_historical_market_data(
        make_provider_frame(),
        request=request,
        fetched_at_utc=pd.Timestamp("2026-03-13T12:00:00Z").to_pydatetime(),
        raw_snapshot_path=isolated_repo / "data" / "raw" / "market_data" / "INFY" / "run.json",
    )

    assert list(cleaned_frame.columns) == [
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
    ]
    assert cleaned_frame["date"].tolist() == ["2026-03-09", "2026-03-10"]
    assert cleaned_frame.iloc[1]["close"] == 104.0
    assert metadata["duplicate_count"] == 1
    assert metadata["dropped_row_count"] == 1
    assert metadata["dropped_rows"][0]["reasons"] == ["invalid_high_low_relationship"]


def test_fetch_historical_market_data_persists_outputs_and_missing_dates(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(make_provider_frame())
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            date(2026, 3, 9),
            date(2026, 3, 10),
            date(2026, 3, 11),
            date(2026, 3, 12),
            date(2026, 3, 13),
        ],
    )

    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
        provider=provider,
    )

    assert result.row_count == 2
    assert result.duplicate_count == 1
    assert set(result.missing_trading_dates) == {"2026-03-11", "2026-03-12", "2026-03-13"}
    assert result.raw_snapshot_path.exists()
    assert result.cleaned_table_path.exists()
    assert result.metadata_path.exists()

    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert cleaned_frame["ticker"].tolist() == ["INFY", "INFY"]
    assert metadata["provider_symbol"] == "INFY.NS"
    assert metadata["duplicate_count"] == 1
    assert metadata["missing_trading_dates"] == ["2026-03-11", "2026-03-12", "2026-03-13"]
    assert metadata["refresh_strategy"] == "full_refresh"
    assert metadata["timing"]["elapsed_seconds"] >= 0.0
    assert metadata["workload"]["fetched_provider_row_count"] == 4


def test_weekend_gaps_are_not_treated_as_missing_for_nse_calendar() -> None:
    trading_days = build_expected_trading_days(
        exchange="NSE",
        start_date=date(2026, 3, 6),
        end_date=date(2026, 3, 9),
    )

    assert trading_days == [date(2026, 3, 6), date(2026, 3, 9)]


def test_known_nse_holiday_is_not_returned_as_trading_day() -> None:
    trading_days = build_expected_trading_days(
        exchange="NSE",
        start_date=date(2026, 1, 23),
        end_date=date(2026, 1, 27),
    )

    assert date(2026, 1, 26) not in trading_days
    assert date(2026, 1, 27) in trading_days


def test_command_smoke_writes_market_data_outputs(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_provider = FakeHistoricalProvider(
        pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Adj Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=pd.to_datetime(["2026-03-09", "2026-03-10"]),
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.resolve_historical_data_provider",
        lambda settings: fake_provider,
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 9), date(2026, 3, 10)],
    )

    exit_code = main(["--end-date", "2026-03-10"])

    assert exit_code == 0
    assert (isolated_repo / "data" / "processed" / "market_data" / "INFY_NSE_daily.csv").exists()
    assert (
        isolated_repo
        / "data"
        / "processed"
        / "market_data"
        / "INFY_NSE_daily.metadata.json"
    ).exists()


def test_build_provider_symbol_maps_supported_exchanges() -> None:
    assert build_provider_symbol("INFY", "NSE") == "INFY.NS"
    assert build_provider_symbol("INFY", "BSE") == "INFY.BO"


def test_fetch_historical_market_data_reuses_existing_coverage(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2024-03-11", "2026-03-10")
        ],
    )

    first_result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
    )
    second_result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
    )

    metadata = json.loads(second_result.metadata_path.read_text(encoding="utf-8"))

    assert first_result.cleaned_table_path == second_result.cleaned_table_path
    assert provider.call_count == 1
    assert metadata["refresh_strategy"] == "reuse_existing"
    assert metadata["reused_existing_row_count"] == 523
    assert metadata["workload"]["fetched_provider_row_count"] == 0


def test_fetch_historical_market_data_full_refresh_bypasses_reuse(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2024-03-11", "2026-03-10")
        ],
    )

    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
    )
    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
        full_refresh=True,
    )

    assert provider.call_count == 2


def test_fetch_historical_market_data_full_refreshes_when_cached_head_is_too_short(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(make_provider_frame())
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 9), date(2026, 3, 10)],
    )

    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
    )
    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=36,
        provider=provider,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert provider.call_count == 2
    assert metadata["refresh_strategy"] == "full_refresh_missing_head_coverage"
    assert metadata["reused_existing_row_count"] == 0
    assert metadata["effective_fetch_start_date"] == "2023-03-10"
    assert metadata["effective_fetch_end_date"] == "2026-03-10"
    assert metadata["workload"]["fetched_provider_row_count"] == 4


def test_fetch_historical_market_data_refreshes_missing_tail_with_overlap_merge(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    initial_provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            [
                "2026-03-02",
                "2026-03-03",
                "2026-03-04",
                "2026-03-05",
                "2026-03-06",
                "2026-03-09",
                "2026-03-10",
            ]
        )
    )
    incremental_provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            [
                "2026-03-05",
                "2026-03-06",
                "2026-03-09",
                "2026-03-10",
                "2026-03-11",
                "2026-03-12",
                "2026-03-13",
            ]
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            date(2026, 3, 2),
            date(2026, 3, 3),
            date(2026, 3, 4),
            date(2026, 3, 5),
            date(2026, 3, 6),
            date(2026, 3, 9),
            date(2026, 3, 10),
            date(2026, 3, 11),
            date(2026, 3, 12),
            date(2026, 3, 13),
        ],
    )

    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=initial_provider,
    )
    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
        provider=incremental_provider,
    )

    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert initial_provider.call_count == 1
    assert incremental_provider.call_count == 1
    assert cleaned_frame["date"].tolist() == [
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
        "2026-03-05",
        "2026-03-06",
        "2026-03-09",
        "2026-03-10",
        "2026-03-11",
        "2026-03-12",
        "2026-03-13",
    ]
    assert metadata["refresh_strategy"] == "incremental_tail"
    assert metadata["reused_existing_row_count"] == 3
    assert metadata["effective_fetch_start_date"] == "2026-03-05"
    assert metadata["effective_fetch_end_date"] == "2026-03-13"
    assert metadata["workload"]["fetched_provider_row_count"] == 7


def test_invalid_short_lookback_is_rejected(isolated_repo) -> None:
    settings = load_settings()

    with pytest.raises(HistoricalMarketDataProviderError, match="at least"):
        build_historical_fetch_request(
            settings,
            end_date=date(2026, 3, 10),
            lookback_months=6,
        )


def test_check_market_data_freshness_returns_false_when_no_data_exists(
    isolated_repo,
) -> None:
    settings = load_settings()

    is_fresh, actual_end_date, reason = check_market_data_freshness(
        settings,
        required_end_date=date(2026, 3, 10),
    )

    assert is_fresh is False
    assert actual_end_date is None
    assert "no existing market data" in reason


def test_check_market_data_freshness_returns_true_when_data_is_current(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2024-03-11", "2026-03-10")
        ],
    )

    # First, fetch some data
    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
    )

    # Check freshness for same date
    is_fresh, actual_end_date, reason = check_market_data_freshness(
        settings,
        required_end_date=date(2026, 3, 10),
    )

    assert is_fresh is True
    assert actual_end_date == date(2026, 3, 10)
    assert "fresh" in reason


def test_check_market_data_freshness_returns_false_when_data_is_stale(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2024-03-11", "2026-03-10")
        ],
    )

    # First, fetch some data up to March 10
    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=provider,
    )

    # Check freshness for later date (March 13)
    is_fresh, actual_end_date, reason = check_market_data_freshness(
        settings,
        required_end_date=date(2026, 3, 13),
    )

    assert is_fresh is False
    assert actual_end_date == date(2026, 3, 10)
    assert "stale" in reason
    assert "missing 3 day(s)" in reason


def test_ensure_fresh_until_triggers_refetch_when_stale(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    initial_provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    extended_provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-13").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2024-03-11", "2026-03-13")
        ],
    )

    # First fetch up to March 10
    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=initial_provider,
    )

    # Use ensure_fresh_until to request data up to March 13
    result = fetch_historical_market_data(
        settings,
        lookback_months=24,
        provider=extended_provider,
        ensure_fresh_until=date(2026, 3, 13),
    )

    # Should have triggered incremental fetch
    cleaned_frame = pd.read_csv(result.cleaned_table_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert extended_provider.call_count == 1
    assert cleaned_frame["date"].max() == "2026-03-13"
    assert metadata["refresh_strategy"] == "incremental_tail"


def test_ensure_fresh_until_reuses_when_already_fresh(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2024-03-08", "2026-03-13").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2024-03-11", "2026-03-13")
        ],
    )

    # First fetch up to March 13
    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
        provider=provider,
    )

    # Use ensure_fresh_until for March 10 (already covered)
    result = fetch_historical_market_data(
        settings,
        lookback_months=24,
        provider=provider,
        ensure_fresh_until=date(2026, 3, 10),
    )

    # Should have reused existing data
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert provider.call_count == 1  # Only the first fetch
    assert metadata["refresh_strategy"] == "reuse_existing"
