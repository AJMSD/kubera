"""Market calendar helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
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

    return PandasMarketCalendar(
        timezone=ZoneInfo(settings.timezone_name),
        calendar_name=settings.calendar_name,
        holiday_overrides=load_exchange_closure_dates(settings),
    )


def load_exchange_closure_dates(market: MarketSettings) -> frozenset[date]:
    """Union of built-in, synced, and local closure dates (exchange non-trading days)."""

    return (
        load_builtin_exchange_holidays(market.exchange_code)
        | load_synced_exchange_closures(market.exchange_closures_path)
        | load_local_holiday_overrides(market.local_holiday_override_path)
    )


def load_synced_exchange_closures(path: Path) -> frozenset[date]:
    """Load repository-shipped exchange closures from JSON (optional file)."""

    if not path.exists():
        return frozenset()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Invalid exchange closures file: {path}") from exc

    return _closure_dates_from_json_payload(payload, label=f"exchange closures file {path}")


def format_live_pilot_cutoff_error(
    *,
    calendar: MarketCalendar,
    latest: date,
    cutoff: date,
    synced_as_of: date | None,
) -> str:
    """Build a categorized LivePilotError message for Stage 2 cutoff shortfalls."""

    if latest >= cutoff:
        return (
            "cutoff_calendar_mismatch: Latest historical bar meets or exceeds the cutoff "
            f"({latest.isoformat()} vs {cutoff.isoformat()}); this should not happen."
        )

    parts: list[str] = []
    if synced_as_of is not None and cutoff.year > synced_as_of.year:
        parts.append("cutoff_stale_holiday_cache:")
    next_after_latest = calendar.next_trading_day(latest)
    if next_after_latest == cutoff:
        parts.append(
            "cutoff_provider_lag: "
            f"Latest historical bar is {latest.isoformat()}; the next trading session on this calendar "
            f"is {cutoff.isoformat()} (the data provider may not have published that session yet)."
        )
    else:
        parts.append(
            "cutoff_calendar_mismatch: "
            f"Latest historical bar is {latest.isoformat()} but the configured calendar "
            f"requires coverage through {cutoff.isoformat()} (next session after latest bar is "
            f"{next_after_latest.isoformat()}). "
            "Run `kubera sync-holidays` or update `config/exchange_closures/india.json` "
            "or `config/market_holidays.local.json`."
        )
    return " ".join(parts)


def load_exchange_closures_as_of(path: Path) -> date | None:
    """Return optional publication / coverage date from synced closures metadata."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("as_of")
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


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


def _closure_dates_from_json_payload(payload: Any, *, label: str) -> frozenset[date]:
    raw_dates: list[str]
    if isinstance(payload, list):
        raw_dates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("holidays"), list):
        raw_dates = payload["holidays"]
    else:
        raise SettingsError(
            f"{label} must be a list of ISO dates or an object with a 'holidays' list."
        )

    parsed_dates: set[date] = set()
    for raw_date in raw_dates:
        try:
            parsed_dates.add(date.fromisoformat(str(raw_date)))
        except ValueError as exc:
            raise SettingsError(f"Invalid holiday date in {label}: {raw_date}") from exc

    return frozenset(parsed_dates)


def load_local_holiday_overrides(path: Path) -> frozenset[date]:
    """Load optional local trading-day overrides from JSON."""

    if not path.exists():
        return frozenset()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Invalid local holiday override file: {path}") from exc

    return _closure_dates_from_json_payload(payload, label=f"local holiday override file {path}")
