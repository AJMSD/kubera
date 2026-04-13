from __future__ import annotations

from datetime import date, datetime, timezone
import json
import sys
import types

import pandas as pd
import pytest
import requests

from kubera.config import load_settings
from kubera.utils.calendar import load_exchange_closure_dates

from kubera.ingest.market_data import (
    HistoricalFetchRequest,
    HistoricalMarketDataProvider,
    HistoricalMarketDataProviderError,
    build_historical_provider_precedence,
    build_expected_trading_days,
    build_historical_fetch_request,
    build_provider_symbol,
    cap_historical_end_date_before_session_close,
    check_market_data_freshness,
    fetch_historical_market_data,
    main,
    normalize_historical_market_data,
    resolve_historical_provider_symbol,
)
from kubera.ingest.providers.bhavcopy_historical import (
    BseBhavcopyHistoricalDataProvider,
    NseBhavcopyHistoricalDataProvider,
)


class FakeHistoricalProvider(HistoricalMarketDataProvider):
    provider_name = "fake_provider"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.call_count = 0

    def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
        self.call_count += 1
        return self._frame.copy()


class FakeBhavcopyResponse:
    def __init__(self, *, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")


class FakeBhavcopySession:
    def __init__(self, responses: list[FakeBhavcopyResponse]) -> None:
        self._responses = list(responses)

    def get(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        if not self._responses:
            return FakeBhavcopyResponse(status_code=404)
        return self._responses.pop(0)


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


def _zip_csv_payload(text: str) -> bytes:
    import io
    import zipfile

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bhavcopy.csv", text)
    return output.getvalue()


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


def test_normalize_historical_market_data_drops_yahoo_style_incomplete_session_row(
    isolated_repo,
) -> None:
    """Yahoo daily data can include the current session date with NaN Close/Adj Close."""
    settings = load_settings()
    request = build_historical_fetch_request(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
    )
    index = pd.to_datetime(["2026-03-12", "2026-03-13"])
    raw = pd.DataFrame(
        {
            "Open": [100.0, 100.5],
            "High": [102.0, 101.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, float("nan")],
            "Adj Close": [101.0, float("nan")],
            "Volume": [1000, 500],
        },
        index=index,
    )

    cleaned_frame, metadata = normalize_historical_market_data(
        raw,
        request=request,
        fetched_at_utc=pd.Timestamp("2026-03-13T10:00:00Z").to_pydatetime(),
        raw_snapshot_path=isolated_repo / "data" / "raw" / "market_data" / "INFY" / "run.json",
    )

    assert cleaned_frame["date"].tolist() == ["2026-03-12"]
    assert metadata["coverage_end"] == "2026-03-12"
    assert metadata["dropped_row_count"] == 1
    assert metadata["dropped_rows"][0]["date"] == "2026-03-13"
    assert "invalid_close" in metadata["dropped_rows"][0]["reasons"]


def test_check_market_data_freshness_stale_when_last_cleaned_day_lags_required(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the incomplete 'today' row is dropped, coverage_end is T-1; freshness may fail."""
    settings = load_settings()
    index = pd.to_datetime(["2026-03-12", "2026-03-13"])
    partial_last = pd.DataFrame(
        {
            "Open": [100.0, 100.5],
            "High": [102.0, 101.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, float("nan")],
            "Adj Close": [101.0, float("nan")],
            "Volume": [1000, 500],
        },
        index=index,
    )
    provider = FakeHistoricalProvider(partial_last)
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 12), date(2026, 3, 13)],
    )

    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
        provider=provider,
    )

    is_fresh, actual_end, reason = check_market_data_freshness(
        settings,
        required_end_date=date(2026, 3, 13),
    )

    assert is_fresh is False
    assert actual_end == date(2026, 3, 12)
    assert "stale" in reason


def test_cap_historical_end_date_before_session_close_friday_morning_ist(
    isolated_repo,
) -> None:
    settings = load_settings()
    # 2026-03-13 Friday 10:00 IST = 04:30 UTC, before NSE close.
    now = datetime(2026, 3, 13, 4, 30, tzinfo=timezone.utc)
    capped, reason = cap_historical_end_date_before_session_close(
        settings,
        date(2026, 3, 13),
        now=now,
    )
    assert reason == "historical_end_date_capped_before_session_close"
    assert capped == date(2026, 3, 12)


def test_cap_historical_end_date_unchanged_after_session_close(isolated_repo) -> None:
    settings = load_settings()
    # 2026-03-13 Friday 17:31 IST = 12:01 UTC, after NSE close.
    now = datetime(2026, 3, 13, 12, 1, tzinfo=timezone.utc)
    capped, reason = cap_historical_end_date_before_session_close(
        settings,
        date(2026, 3, 13),
        now=now,
    )
    assert reason is None
    assert capped == date(2026, 3, 13)


def test_after_close_retry_refetches_when_coverage_short(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    monkeypatch.setattr(
        "kubera.ingest.market_data.cap_historical_end_date_before_session_close",
        lambda s, d, now=None: (d, None),
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.is_after_close",
        lambda dt, m: True,
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 12), date(2026, 3, 13)],
    )

    index = pd.to_datetime(["2026-03-12", "2026-03-13"])
    partial_last = pd.DataFrame(
        {
            "Open": [100.0, 100.5],
            "High": [102.0, 101.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, float("nan")],
            "Adj Close": [101.0, float("nan")],
            "Volume": [1000, 500],
        },
        index=index,
    )

    class FlakyProvider(HistoricalMarketDataProvider):
        provider_name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
            self.calls += 1
            if self.calls == 1:
                return partial_last.copy()
            return make_ohlcv_frame(["2026-03-12", "2026-03-13"])

    provider = FlakyProvider()
    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
        provider=provider,
    )

    assert provider.calls == 2
    assert result.coverage_end == date(2026, 3, 13)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["workload"]["stale_session_after_close_retry"] is True


def test_after_close_retry_raises_when_still_missing_bar(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    monkeypatch.setattr(
        "kubera.ingest.market_data.cap_historical_end_date_before_session_close",
        lambda s, d, now=None: (d, None),
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.is_after_close",
        lambda dt, m: True,
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 12), date(2026, 3, 13)],
    )

    index = pd.to_datetime(["2026-03-12", "2026-03-13"])
    partial_last = pd.DataFrame(
        {
            "Open": [100.0, 100.5],
            "High": [102.0, 101.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, float("nan")],
            "Adj Close": [101.0, float("nan")],
            "Volume": [1000, 500],
        },
        index=index,
    )

    provider = FakeHistoricalProvider(partial_last)
    with pytest.raises(HistoricalMarketDataProviderError, match="after one post-close retry"):
        fetch_historical_market_data(
            settings,
            end_date=date(2026, 3, 13),
            lookback_months=24,
            provider=provider,
        )

    assert provider.call_count == 2


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
    settings = load_settings()
    closures = load_exchange_closure_dates(settings.market)
    trading_days = build_expected_trading_days(
        exchange="NSE",
        start_date=date(2026, 3, 6),
        end_date=date(2026, 3, 9),
        closure_dates=closures,
    )

    assert trading_days == [date(2026, 3, 6), date(2026, 3, 9)]


def test_known_nse_holiday_is_not_returned_as_trading_day() -> None:
    settings = load_settings()
    closures = load_exchange_closure_dates(settings.market)
    trading_days = build_expected_trading_days(
        exchange="NSE",
        start_date=date(2026, 1, 23),
        end_date=date(2026, 1, 27),
        closure_dates=closures,
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


def test_resolve_historical_provider_symbol_maps_yfinance_to_yahoo_catalog_key(
    isolated_repo,
) -> None:
    settings = load_settings()
    assert (
        resolve_historical_provider_symbol(settings, provider_name="yfinance") == "INFY.NS"
    )


def test_parallel_upstox_without_catalog_records_error_in_metadata(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_PARALLEL_PROVIDERS", "upstox")
    monkeypatch.setenv("KUBERA_UPSTOX_ACCESS_TOKEN", "dummy-token")
    settings = load_settings()
    fake = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2026-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2026-03-09", "2026-03-10")
        ],
    )

    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=fake,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    parallel = metadata.get("parallel_raw_snapshots") or []
    assert len(parallel) == 1
    assert parallel[0]["provider"] == "upstox"
    assert "error" in parallel[0]
    assert "upstox" in parallel[0]["raw_snapshot_path"]


def test_parallel_nsepython_skipped_for_bse(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_EXCHANGE", "BSE")
    monkeypatch.setenv("KUBERA_HISTORICAL_PARALLEL_PROVIDERS", "nsepython")
    settings = load_settings()
    fake = FakeHistoricalProvider(
        make_ohlcv_frame(
            pd.bdate_range("2026-03-08", "2026-03-10").strftime("%Y-%m-%d").tolist()
        )
    )
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [
            value.date() for value in pd.bdate_range("2026-03-09", "2026-03-10")
        ],
    )

    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 10),
        lookback_months=24,
        provider=fake,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    parallel = metadata.get("parallel_raw_snapshots") or []
    assert len(parallel) == 1
    assert parallel[0]["skipped"] is True
    assert parallel[0]["provider"] == "nsepython"


def test_nsepython_equity_history_keyerror_maps_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mod = types.ModuleType("nsepython")

    def _bad_equity_history(*_args: object, **_kwargs: object) -> None:
        raise KeyError("data")

    fake_mod.equity_history = _bad_equity_history
    monkeypatch.setitem(sys.modules, "nsepython", fake_mod)

    from kubera.ingest.providers.nsepython_historical import (
        NsePythonHistoricalDataProvider,
    )

    provider = NsePythonHistoricalDataProvider()
    request = HistoricalFetchRequest(
        ticker="INFY",
        exchange="NSE",
        provider="nsepython",
        provider_symbol="INFY",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        lookback_months=24,
    )
    with pytest.raises(HistoricalMarketDataProviderError, match="equity_history"):
        provider.fetch_daily_ohlcv(request)


def test_nse_bhavcopy_provider_parses_zipped_csv_row() -> None:
    csv_text = "\n".join(
        [
            "SYMBOL,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TOTTRDQTY",
            "INFY,100,105,99,104,123456",
        ]
    )
    provider = NseBhavcopyHistoricalDataProvider(
        session=FakeBhavcopySession(
            [FakeBhavcopyResponse(status_code=200, content=_zip_csv_payload(csv_text))]
        )
    )
    frame = provider.fetch_daily_ohlcv(
        HistoricalFetchRequest(
            ticker="INFY",
            exchange="NSE",
            provider="nse_bhavcopy",
            provider_symbol="INFY",
            start_date=date(2026, 3, 10),
            end_date=date(2026, 3, 10),
            lookback_months=24,
        )
    )

    assert not frame.empty
    assert float(frame.iloc[0]["Close"]) == pytest.approx(104.0)
    assert float(frame.iloc[0]["Volume"]) == pytest.approx(123456.0)


def test_bse_bhavcopy_provider_parses_zipped_csv_row() -> None:
    csv_text = "\n".join(
        [
            "SC_CODE,OPEN,HIGH,LOW,CLOSE,NO_OF_SHRS",
            "500209,1400,1420,1390,1415,999",
        ]
    )
    provider = BseBhavcopyHistoricalDataProvider(
        session=FakeBhavcopySession(
            [FakeBhavcopyResponse(status_code=200, content=_zip_csv_payload(csv_text))]
        )
    )
    frame = provider.fetch_daily_ohlcv(
        HistoricalFetchRequest(
            ticker="INFY",
            exchange="BSE",
            provider="bse_bhavcopy",
            provider_symbol="500209",
            start_date=date(2026, 3, 10),
            end_date=date(2026, 3, 10),
            lookback_months=24,
        )
    )

    assert not frame.empty
    assert float(frame.iloc[0]["Close"]) == pytest.approx(1415.0)


def test_historical_provider_precedence_defaults_to_official_then_yfinance(
    isolated_repo,
) -> None:
    settings = load_settings()
    assert build_historical_provider_precedence(settings) == (
        "nse_bhavcopy",
        "bse_bhavcopy",
        "yfinance",
    )


def test_historical_provider_precedence_honors_official_only(
    monkeypatch: pytest.MonkeyPatch,
    isolated_repo,
) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_OFFICIAL_ONLY", "true")
    monkeypatch.setenv("KUBERA_HISTORICAL_PROVIDER_PRIORITY", "nse_bhavcopy,yfinance")
    settings = load_settings()
    assert build_historical_provider_precedence(settings) == ("nse_bhavcopy",)


def test_check_market_data_freshness_treats_non_trading_required_day_as_previous_session(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings()
    provider = FakeHistoricalProvider(make_ohlcv_frame(["2026-03-13"]))
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 13)],
    )
    fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
        provider=provider,
    )

    is_fresh, actual_end_date, reason = check_market_data_freshness(
        settings,
        required_end_date=date(2026, 3, 14),  # Saturday
    )
    assert is_fresh is True
    assert actual_end_date == date(2026, 3, 13)
    assert "non-trading" in reason


def test_fetch_historical_market_data_falls_back_to_next_priority_provider(
    isolated_repo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBERA_HISTORICAL_PROVIDER_PRIORITY", "nse_bhavcopy,yfinance")
    settings = load_settings()
    calls: list[str] = []

    class _FailingProvider(HistoricalMarketDataProvider):
        provider_name = "nse_bhavcopy"

        def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
            del request
            calls.append("nse_bhavcopy")
            raise HistoricalMarketDataProviderError("missing official file")

    class _SuccessProvider(HistoricalMarketDataProvider):
        provider_name = "yfinance"

        def fetch_daily_ohlcv(self, request: HistoricalFetchRequest) -> pd.DataFrame:
            del request
            calls.append("yfinance")
            return make_ohlcv_frame(["2026-03-12", "2026-03-13"])

    def _resolve_provider(_settings, provider_name: str):
        if provider_name == "nse_bhavcopy":
            return _FailingProvider()
        if provider_name == "yfinance":
            return _SuccessProvider()
        raise AssertionError(provider_name)

    monkeypatch.setattr("kubera.ingest.market_data.resolve_historical_data_provider_by_name", _resolve_provider)
    monkeypatch.setattr(
        "kubera.ingest.market_data.build_expected_trading_days",
        lambda **_: [date(2026, 3, 12), date(2026, 3, 13)],
    )

    result = fetch_historical_market_data(
        settings,
        end_date=date(2026, 3, 13),
        lookback_months=24,
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert calls == ["nse_bhavcopy", "yfinance"]
    assert metadata["provider"] == "yfinance"
