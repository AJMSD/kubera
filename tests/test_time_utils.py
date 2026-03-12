from __future__ import annotations

from datetime import date, datetime, timezone
import json

from kubera.config import load_settings
from kubera.utils.calendar import build_market_calendar
from kubera.utils.time_utils import (
    is_after_close,
    is_intraday,
    is_pre_market,
    market_time_to_utc,
    utc_to_market_time,
)


def test_market_time_helpers_respect_nse_session_windows(isolated_repo) -> None:
    settings = load_settings()

    pre_market_utc = datetime(2026, 3, 10, 3, 20, tzinfo=timezone.utc)
    at_open_utc = datetime(2026, 3, 10, 3, 45, tzinfo=timezone.utc)
    at_close_utc = datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc)

    assert is_pre_market(pre_market_utc, settings.market)
    assert is_intraday(at_open_utc, settings.market)
    assert is_after_close(at_close_utc, settings.market)


def test_market_timezone_round_trip_is_stable(isolated_repo) -> None:
    settings = load_settings()
    at_open_utc = datetime(2026, 3, 10, 3, 45, tzinfo=timezone.utc)

    market_dt = utc_to_market_time(at_open_utc, settings.market)
    round_trip_utc = market_time_to_utc(market_dt, settings.market)

    assert market_dt.hour == 9
    assert market_dt.minute == 15
    assert round_trip_utc == at_open_utc


def test_local_holiday_override_marks_non_trading_days(isolated_repo) -> None:
    config_dir = isolated_repo / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    override_path = config_dir / "market_holidays.local.json"
    override_path.write_text(
        json.dumps({"holidays": ["2026-03-11"]}),
        encoding="utf-8",
    )

    settings = load_settings()
    calendar = build_market_calendar(settings.market)

    assert not calendar.is_trading_day(date(2026, 3, 11))
    assert calendar.next_trading_day(date(2026, 3, 10)) == date(2026, 3, 12)


def test_builtin_exchange_holiday_is_non_trading_day(isolated_repo) -> None:
    settings = load_settings()
    calendar = build_market_calendar(settings.market)

    assert not calendar.is_trading_day(date(2026, 1, 26))


def test_builtin_and_local_holiday_overrides_are_merged(isolated_repo) -> None:
    config_dir = isolated_repo / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    override_path = config_dir / "market_holidays.local.json"
    override_path.write_text(
        json.dumps({"holidays": ["2026-03-11"]}),
        encoding="utf-8",
    )

    settings = load_settings()
    calendar = build_market_calendar(settings.market)

    assert not calendar.is_trading_day(date(2026, 1, 26))
    assert not calendar.is_trading_day(date(2026, 3, 11))
