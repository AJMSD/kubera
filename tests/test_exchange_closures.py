"""Tests for India exchange closure loading and calendar alignment."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from kubera.config import load_settings
from kubera.ingest.market_data import build_expected_trading_days
from kubera.utils.calendar import (
    build_market_calendar,
    load_exchange_closure_dates,
    load_exchange_closures_as_of,
)


def test_load_exchange_closures_as_of_reads_metadata(isolated_repo) -> None:
    path = Path(isolated_repo) / "config" / "exchange_closures" / "india.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"holidays": [], "as_of": "2026-01-15"}),
        encoding="utf-8",
    )
    settings = load_settings(isolated_repo)
    assert load_exchange_closures_as_of(settings.market.exchange_closures_path) == date(2026, 1, 15)


def test_good_friday_2026_not_a_trading_day_when_listed_in_synced_closures(
    isolated_repo,
) -> None:
    path = Path(isolated_repo) / "config" / "exchange_closures" / "india.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"holidays": ["2026-04-03"]}),
        encoding="utf-8",
    )
    settings = load_settings(isolated_repo)
    calendar = build_market_calendar(settings.market)
    assert not calendar.is_trading_day(date(2026, 4, 3))


def test_build_expected_trading_days_matches_closure_union(isolated_repo) -> None:
    path = Path(isolated_repo) / "config" / "exchange_closures" / "india.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"holidays": ["2026-04-03"]}),
        encoding="utf-8",
    )
    settings = load_settings(isolated_repo)
    closures = load_exchange_closure_dates(settings.market)
    days = build_expected_trading_days(
        exchange="NSE",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 7),
        closure_dates=closures,
    )
    assert date(2026, 4, 3) not in days
    assert date(2026, 4, 6) in days
    assert date(2026, 4, 7) in days
