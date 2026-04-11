"""Tests for holiday PDF text extraction helpers."""

from __future__ import annotations

from datetime import date

from kubera.sync_holidays import extract_iso_dates_from_text


def test_extract_iso_dates_from_text_finds_common_formats() -> None:
    text = """
    NSE Circular
    03-04-2026 Good Friday
    2026-01-26 Republic Day
    15/08/2026 Independence Day
    """
    dates = extract_iso_dates_from_text(text)
    assert date(2026, 4, 3) in dates
    assert date(2026, 1, 26) in dates
    assert date(2026, 8, 15) in dates
