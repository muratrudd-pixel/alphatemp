"""Tests for the NWS ACIS daily fetcher."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from core.db import init_db, get_connection

TEST_DB = "data/test_nws.duckdb"


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create and tear down a test database."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _make_acis_response(data_rows):
    """Build a mock ACIS StnData response."""
    return {
        "meta": {"name": "TEST STATION"},
        "data": data_rows,
    }


@pytest.mark.asyncio
async def test_fetcher_stores_daily_data():
    """Mock ACIS response should produce rows in nws_daily."""
    from services.nws_fetcher import NWSFetcher

    fetcher = NWSFetcher(db_path=TEST_DB)

    mock_response = _make_acis_response([
        ["2026-02-21", "42", "28"],
        ["2026-02-22", "38", "25"],
    ])

    async def fake_fetch(client, payload):
        return mock_response

    with patch.object(fetcher, "_fetch_acis", side_effect=fake_fetch):
        inserted = await fetcher.poll_once()

    # 5 stations * 2 days = 10 rows
    assert inserted == 10

    con = get_connection(TEST_DB)
    rows = con.execute("SELECT COUNT(*) FROM nws_daily").fetchone()
    assert rows[0] == 10
    con.close()


@pytest.mark.asyncio
async def test_fetcher_skips_missing_data():
    """ACIS 'M' values should not produce rows."""
    from services.nws_fetcher import NWSFetcher

    fetcher = NWSFetcher(db_path=TEST_DB)

    mock_response = _make_acis_response([
        ["2026-02-21", "M", "28"],
        ["2026-02-22", "38", "M"],
    ])

    async def fake_fetch(client, payload):
        return mock_response

    with patch.object(fetcher, "_fetch_acis", side_effect=fake_fetch):
        inserted = await fetcher.poll_once()

    assert inserted == 0

    con = get_connection(TEST_DB)
    rows = con.execute("SELECT COUNT(*) FROM nws_daily").fetchone()
    assert rows[0] == 0
    con.close()


@pytest.mark.asyncio
async def test_fetcher_deduplicates():
    """Polling twice with same data should not create duplicates."""
    from services.nws_fetcher import NWSFetcher

    fetcher = NWSFetcher(db_path=TEST_DB)

    mock_response = _make_acis_response([
        ["2026-02-22", "38", "25"],
    ])

    async def fake_fetch(client, payload):
        return mock_response

    with patch.object(fetcher, "_fetch_acis", side_effect=fake_fetch):
        first = await fetcher.poll_once()
        second = await fetcher.poll_once()

    assert first == 5   # 5 stations * 1 day
    assert second == 0  # all duplicates

    con = get_connection(TEST_DB)
    rows = con.execute("SELECT COUNT(*) FROM nws_daily").fetchone()
    assert rows[0] == 5
    con.close()
