"""Tests for DST-aware timezone utilities."""

from datetime import datetime, timezone

import pytest

from core.timezone import et_day_bounds_utc, utc_to_et_hour


def test_est_bounds():
    """January date: EST offset = UTC-5, so midnight ET = 05:00 UTC."""
    start, end = et_day_bounds_utc("2026-01-15")
    assert start.hour == 5
    assert start.day == 15
    assert end.hour == 5
    assert end.day == 16


def test_edt_bounds():
    """July date: EDT offset = UTC-4, so midnight ET = 04:00 UTC."""
    start, end = et_day_bounds_utc("2026-07-15")
    assert start.hour == 4
    assert start.day == 15
    assert end.hour == 4
    assert end.day == 16


def test_utc_to_et_hour_est():
    """13 UTC in January (EST) = 8am ET."""
    dt = datetime(2026, 1, 15, 13, 0, tzinfo=timezone.utc)
    assert utc_to_et_hour(dt) == 8


def test_utc_to_et_hour_edt():
    """13 UTC in July (EDT) = 9am ET."""
    dt = datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc)
    assert utc_to_et_hour(dt) == 9


def test_default_today():
    """Calling with no argument should not raise."""
    start, end = et_day_bounds_utc()
    assert start < end
    assert (end - start).total_seconds() == 86400
