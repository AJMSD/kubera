"""Market calendar helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from kubera.config import MarketSettings, SettingsError


class MarketCalendar(ABC):
    """Trading-day interface for market-aware utilities."""

    timezone: ZoneInfo

    @abstractmethod
    def is_trading_day(self, value: date) -> bool:
        """Return True when the given date is a trading day."""

    @abstractmethod
    def next_trading_day(self, value: date) -> date:
        """Return the next trading day after the given date."""


@dataclass(frozen=True)
class WeekendHolidayMarketCalendar(MarketCalendar):
    """Market calendar with weekend logic and optional local holiday overrides."""

    timezone: ZoneInfo
    holiday_overrides: frozenset[date] = field(default_factory=frozenset)
    weekend_days: frozenset[int] = field(
        default_factory=lambda: frozenset({5, 6})
    )

    def is_trading_day(self, value: date) -> bool:
        return value.weekday() not in self.weekend_days and value not in self.holiday_overrides

    def next_trading_day(self, value: date) -> date:
        next_value = value + timedelta(days=1)
        while not self.is_trading_day(next_value):
            next_value += timedelta(days=1)
        return next_value


def build_market_calendar(settings: MarketSettings) -> MarketCalendar:
    """Create the default calendar for the active market settings."""

    holiday_overrides = load_local_holiday_overrides(settings.local_holiday_override_path)
    return WeekendHolidayMarketCalendar(
        timezone=ZoneInfo(settings.timezone_name),
        holiday_overrides=holiday_overrides,
    )


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
