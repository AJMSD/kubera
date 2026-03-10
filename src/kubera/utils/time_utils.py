"""Timezone and market window helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kubera.config import MarketSettings, SettingsError


def get_timezone(timezone_name: str) -> ZoneInfo:
    """Resolve a timezone name into a ZoneInfo object."""

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise SettingsError(f"Unknown timezone: {timezone_name}") from exc


def normalize_datetime(
    value: datetime,
    *,
    target_timezone_name: str,
    assume_timezone_name: str | None = None,
) -> datetime:
    """Normalize any datetime into the target timezone."""

    target_timezone = get_timezone(target_timezone_name)
    if value.tzinfo is None:
        assumed_timezone = get_timezone(assume_timezone_name or target_timezone_name)
        value = value.replace(tzinfo=assumed_timezone)

    return value.astimezone(target_timezone)


def utc_to_market_time(value: datetime, market: MarketSettings) -> datetime:
    """Convert a UTC datetime into market local time."""

    return normalize_datetime(
        value,
        target_timezone_name=market.timezone_name,
        assume_timezone_name="UTC",
    )


def market_time_to_utc(value: datetime, market: MarketSettings) -> datetime:
    """Convert a market local datetime into UTC."""

    return normalize_datetime(
        value,
        target_timezone_name="UTC",
        assume_timezone_name=market.timezone_name,
    )


def is_pre_market(value: datetime, market: MarketSettings) -> bool:
    """Return True when the datetime lands before the regular session open."""

    market_dt = normalize_datetime(
        value,
        target_timezone_name=market.timezone_name,
        assume_timezone_name=market.timezone_name,
    )
    return market_dt.timetz().replace(tzinfo=None) < market.market_open


def is_intraday(value: datetime, market: MarketSettings) -> bool:
    """Return True when the datetime lands during the regular session."""

    market_dt = normalize_datetime(
        value,
        target_timezone_name=market.timezone_name,
        assume_timezone_name=market.timezone_name,
    )
    local_time = market_dt.timetz().replace(tzinfo=None)
    return market.market_open <= local_time < market.market_close


def is_after_close(value: datetime, market: MarketSettings) -> bool:
    """Return True when the datetime lands at or after the regular close."""

    market_dt = normalize_datetime(
        value,
        target_timezone_name=market.timezone_name,
        assume_timezone_name=market.timezone_name,
    )
    return market_dt.timetz().replace(tzinfo=None) >= market.market_close
