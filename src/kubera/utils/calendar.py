"""Market calendar helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from kubera.config import MarketSettings, SettingsError


INDIA_EXCHANGE_FIXED_HOLIDAYS = (
    (1, 26),   # Republic Day
    (5, 1),    # Maharashtra Day / Labour Day
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
)


class MarketCalendar(ABC):
    """Trading-day interface for market-aware utilities."""

    timezone: ZoneInfo

    @abstractmethod
    def is_trading_day(self, value: date) -> bool:
        """Return True when the given date is a trading day."""

    @abstractmethod
    def next_trading_day(self, value: date) -> date:
        """Return the next trading day after the given date."""

    @abstractmethod
    def previous_trading_day(self, value: date) -> date:
        """Return the previous trading day before the given date."""


def first_trading_day_on_or_after(value: date, calendar: MarketCalendar) -> date:
    """Return the first trading day that lands on or after the given date."""

    current = value
    while not calendar.is_trading_day(current):
        current += timedelta(days=1)
    return current


def first_trading_day_after(value: date, calendar: MarketCalendar) -> date:
    """Return the first trading day that lands strictly after the given date."""

    if calendar.is_trading_day(value):
        return calendar.next_trading_day(value)
    return first_trading_day_on_or_after(value, calendar)


@dataclass(frozen=True)
class PandasMarketCalendar(MarketCalendar):
    """Market calendar backed by pandas_market_calendars."""

    timezone: ZoneInfo
    calendar_name: str
    holiday_overrides: frozenset[date] = field(default_factory=frozenset)
    _valid_days: set[date] = field(default_factory=set, repr=False, hash=False, init=False)

    def __post_init__(self) -> None:
        try:
            cal = mcal.get_calendar(self.calendar_name)
        except Exception as exc:
            raise SettingsError(f"Unsupported market calendar: {self.calendar_name}") from exc
        
        valid_dts = cal.valid_days(start_date="2000-01-01", end_date="2050-12-31")
        valid_dates = {dt.date() for dt in valid_dts}
        
        valid_dates -= self.holiday_overrides
        object.__setattr__(self, "_valid_days", valid_dates)

    def is_trading_day(self, value: date) -> bool:
        return value in self._valid_days

    def next_trading_day(self, value: date) -> date:
        next_value = value + timedelta(days=1)
        while next_value not in self._valid_days:
            next_value += timedelta(days=1)
        return next_value

    def previous_trading_day(self, value: date) -> date:
        prev_value = value - timedelta(days=1)
        while prev_value not in self._valid_days:
            prev_value -= timedelta(days=1)
        return prev_value


def build_market_calendar(settings: MarketSettings) -> MarketCalendar:
    """Create the default calendar for the active market settings."""

    holiday_overrides = (
        load_builtin_exchange_holidays(settings.exchange_code)
        | load_local_holiday_overrides(settings.local_holiday_override_path)
    )
    return PandasMarketCalendar(
        timezone=ZoneInfo(settings.timezone_name),
        calendar_name=settings.calendar_name,
        holiday_overrides=holiday_overrides,
    )


def load_builtin_exchange_holidays(exchange_code: str) -> frozenset[date]:
    """Return conservative built-in closures that supplement calendar packages."""

    normalized_exchange = exchange_code.strip().upper()
    if normalized_exchange not in {"NSE", "BSE"}:
        return frozenset()

    built_in_dates = {
        date(year, month, day)
        for year in range(2000, 2101)
        for month, day in INDIA_EXCHANGE_FIXED_HOLIDAYS
    }
    return frozenset(built_in_dates)


def load_local_holiday_overrides(path: Path) -> frozenset[date]:
    """Load optional local trading-day overrides from JSON."""

    if not path.exists():
        return frozenset()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Invalid local holiday override file: {path}") from exc

    raw_dates: list[str]
    if isinstance(payload, list):
        raw_dates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("holidays"), list):
        raw_dates = payload["holidays"]
    else:
        raise SettingsError(
            "Local holiday override file must be a list of ISO dates or an object with a 'holidays' list."
        )

    parsed_dates: set[date] = set()
    for raw_date in raw_dates:
        try:
            parsed_dates.add(date.fromisoformat(raw_date))
        except ValueError as exc:
            raise SettingsError(
                f"Invalid holiday date in local override file: {raw_date}"
            ) from exc

    return frozenset(parsed_dates)
