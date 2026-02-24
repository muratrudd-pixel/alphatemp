"""Tests for the NWS daily fetcher — ACIS + CLI channels."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from core.db import init_db, get_connection
from services.nws_fetcher import NWSFetcher, CLI_STATIONS

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

    # 1 station * 2 days = 2 rows
    assert inserted == 2

    con = get_connection(TEST_DB)
    rows = con.execute("SELECT COUNT(*) FROM nws_daily").fetchone()
    assert rows[0] == 2
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

    assert first == 1   # 1 station * 1 day
    assert second == 0  # all duplicates

    con = get_connection(TEST_DB)
    rows = con.execute("SELECT COUNT(*) FROM nws_daily").fetchone()
    assert rows[0] == 1
    con.close()


# ------------------------------------------------------------------
# NWS CLI tests
# ------------------------------------------------------------------

def _make_cli_product(cli_code: str, office: str, issuance: str, body: str) -> dict:
    """Build a mock NWS CLI product dict."""
    return {
        "productText": f"CLI{cli_code}\n\n{body}",
        "issuanceTime": issuance,
        "issuingOffice": office,
    }


CLI_TODAY_TEXT = """\
WEATHER ITEM   OBSERVED   RECORD

TODAY
  MAXIMUM         39   1206 AM
  MINIMUM         28   1159 PM
"""

CLI_YESTERDAY_TEXT = """\
WEATHER ITEM   OBSERVED   RECORD

YESTERDAY
  MAXIMUM         42   215 PM
  MINIMUM         30   605 AM
"""

CLI_MISSING_TEXT = """\
WEATHER ITEM   OBSERVED   RECORD

TODAY
  MAXIMUM         MM
  MINIMUM         28   1159 PM
"""


@pytest.mark.asyncio
async def test_cli_parses_today_maximum():
    """CLI product with TODAY section should yield correct max_temp_f and obs_date."""
    fetcher = NWSFetcher(db_path=TEST_DB)
    product = _make_cli_product("NYC", "KOKX", "2026-02-23T22:00:00+00:00", CLI_TODAY_TEXT)

    rows = fetcher._parse_cli_product(product)

    assert len(rows) == 1
    assert rows[0]["station_id"] == "KNYC"
    assert rows[0]["max_temp_f"] == 39.0
    assert rows[0]["min_temp_f"] == 28.0
    # 22:00 UTC = 5 PM ET → obs_date should be 2026-02-23
    assert rows[0]["obs_date"] == "2026-02-23"
    assert rows[0]["source"] == "NWS_CLI"


@pytest.mark.asyncio
async def test_cli_parses_yesterday_maximum():
    """CLI YESTERDAY section should set obs_date to day before issuance (in ET)."""
    fetcher = NWSFetcher(db_path=TEST_DB)
    product = _make_cli_product("NYC", "KOKX", "2026-02-23T12:00:00+00:00", CLI_YESTERDAY_TEXT)

    rows = fetcher._parse_cli_product(product)

    assert len(rows) == 1
    assert rows[0]["station_id"] == "KNYC"
    assert rows[0]["max_temp_f"] == 42.0
    # 12:00 UTC = 7 AM ET on 2/23 → YESTERDAY = 2/22
    assert rows[0]["obs_date"] == "2026-02-22"


@pytest.mark.asyncio
async def test_cli_skips_missing_data():
    """CLI with MM for max temp should produce no rows."""
    fetcher = NWSFetcher(db_path=TEST_DB)
    product = _make_cli_product("NYC", "KOKX", "2026-02-23T22:00:00+00:00", CLI_MISSING_TEXT)

    rows = fetcher._parse_cli_product(product)

    assert len(rows) == 0


@pytest.mark.asyncio
async def test_cli_overwrites_acis():
    """CLI upsert should overwrite a prior ACIS row for same station+date."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert an ACIS row first
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-23", 37.0, 25.0, "ACIS", datetime(2026, 2, 23, 12, 0)],
    )
    con.close()

    # Now upsert a CLI row for the same station+date
    cli_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-23",
        "max_temp_f": 39.0,
        "min_temp_f": 28.0,
        "source": "NWS_CLI",
        "ingested_at": datetime(2026, 2, 23, 22, 0),
    }
    fetcher._upsert_row(cli_row)

    con = get_connection(TEST_DB)
    row = con.execute(
        "SELECT max_temp_f, source FROM nws_daily WHERE station_id = 'KNYC' AND obs_date = '2026-02-23'"
    ).fetchone()
    con.close()

    assert row[0] == 39.0      # CLI value wins
    assert row[1] == "NWS_CLI"  # source updated
