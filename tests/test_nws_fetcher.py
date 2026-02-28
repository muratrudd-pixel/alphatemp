"""Tests for the NWS daily fetcher — ACIS + CLI + DSM channels."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from core.db import init_db, get_connection
from services.nws_fetcher import (
    NWSFetcher,
    CLI_STATIONS,
    DSM_STATIONS,
    SOURCE_PRIORITY,
    parse_dsm_product,
)

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


# ------------------------------------------------------------------
# DSM tests
# ------------------------------------------------------------------

def _make_dsm_product(text: str, office: str, issuance: str) -> dict:
    """Build a mock NWS DSM product dict."""
    return {
        "productText": text,
        "issuanceTime": issuance,
        "issuingOffice": office,
    }


# Real KOKX DSM product text — afternoon partial (1600 observation)
DSM_PARTIAL_TEXT = """
000
CXUS41 KOKX 272115
DSMNYC
KNYC DS 1600 27/02 441449/ 290527// 44/ 29//9950019/00/00/00/00/00/
00/00/00/00/00/00/00/00/00/00/00/00/-/-/-/-/-/-/-/-/-/02100002/
13151555
"""

# Real KOKX DSM product text — midnight final (no time prefix)
DSM_FINAL_TEXT = """
000
CXUS41 KOKX 270515
DSMNYC
KNYC DS 26/02 491421/ 352359// 49/ 38//9710020/00/00/00/00/00/00/00/
00/00/00/00/00/00/00/00/00/00/00/00/00/00/00/00/00/00/39/01122230/
31190910/N/NN/N/N/NN/EW
"""

# Real KOKX DSM correction text
DSM_COR_TEXT = """
000
CXUS41 KOKX 270530
DSMNYC
KNYC DS COR 22/02 390006/ 302359// 35/ 34//9642359/83/00/00/00/00/00/
00/T/02/01/02/01/02/02/T/03/01/03/07/08/11/13/10/10/07/115/05241600/
06391517/1/NN/88/0/NN/EW
"""


def test_parse_dsm_product_partial():
    """DSM partial (afternoon) product should extract calendar-day max/min."""
    product = _make_dsm_product(DSM_PARTIAL_TEXT, "KOKX", "2026-02-27T21:15:00+00:00")
    rows = parse_dsm_product(product)

    assert len(rows) == 1
    assert rows[0]["station_id"] == "KNYC"
    assert rows[0]["obs_date"] == "2026-02-27"
    assert rows[0]["max_temp_f"] == 44.0
    assert rows[0]["min_temp_f"] == 29.0
    assert rows[0]["source"] == "DSM"


def test_parse_dsm_product_final():
    """DSM final (midnight) product should extract correct date and temps."""
    product = _make_dsm_product(DSM_FINAL_TEXT, "KOKX", "2026-02-27T05:15:00+00:00")
    rows = parse_dsm_product(product)

    assert len(rows) == 1
    assert rows[0]["station_id"] == "KNYC"
    assert rows[0]["obs_date"] == "2026-02-26"
    assert rows[0]["max_temp_f"] == 49.0
    assert rows[0]["min_temp_f"] == 38.0
    assert rows[0]["source"] == "DSM"


def test_parse_dsm_product_correction():
    """DSM COR (correction) product should parse correctly."""
    product = _make_dsm_product(DSM_COR_TEXT, "KOKX", "2026-02-27T05:30:00+00:00")
    rows = parse_dsm_product(product)

    assert len(rows) == 1
    assert rows[0]["station_id"] == "KNYC"
    assert rows[0]["obs_date"] == "2026-02-22"
    assert rows[0]["max_temp_f"] == 35.0
    assert rows[0]["min_temp_f"] == 34.0


def test_parse_dsm_product_wrong_office():
    """DSM from an untracked office should return empty list."""
    product = _make_dsm_product(DSM_PARTIAL_TEXT, "KPHI", "2026-02-27T21:15:00+00:00")
    rows = parse_dsm_product(product)
    assert len(rows) == 0


def test_parse_dsm_product_bad_issuance():
    """DSM with unparseable issuanceTime should return empty list."""
    product = _make_dsm_product(DSM_PARTIAL_TEXT, "KOKX", "not-a-date")
    rows = parse_dsm_product(product)
    assert len(rows) == 0


# ------------------------------------------------------------------
# Source hierarchy tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_source_hierarchy_dsm_over_acis():
    """DSM should overwrite a prior ACIS row."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert ACIS row
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-27", 43.0, 28.0, "ACIS", datetime(2026, 2, 27, 12, 0)],
    )
    con.close()

    # Upsert DSM row — should overwrite
    dsm_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-27",
        "max_temp_f": 44.0,
        "min_temp_f": 29.0,
        "source": "DSM",
        "ingested_at": datetime(2026, 2, 27, 21, 15),
    }
    result = fetcher._upsert_row(dsm_row)
    assert result is True

    con = get_connection(TEST_DB)
    try:
        row = con.execute(
            "SELECT max_temp_f, min_temp_f, source FROM nws_daily "
            "WHERE station_id = 'KNYC' AND obs_date = '2026-02-27'"
        ).fetchone()
    finally:
        con.close()

    assert row[0] == 44.0     # DSM max
    assert row[1] == 29.0     # DSM min
    assert row[2] == "DSM"    # source updated


@pytest.mark.asyncio
async def test_upsert_source_hierarchy_cli_over_dsm():
    """CLI should overwrite a prior DSM row."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert DSM row
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-27", 44.0, 29.0, "DSM", datetime(2026, 2, 27, 21, 15)],
    )
    con.close()

    # Upsert CLI row — should overwrite
    cli_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-27",
        "max_temp_f": 45.0,
        "min_temp_f": 30.0,
        "source": "NWS_CLI",
        "ingested_at": datetime(2026, 2, 28, 6, 30),
    }
    result = fetcher._upsert_row(cli_row)
    assert result is True

    con = get_connection(TEST_DB)
    try:
        row = con.execute(
            "SELECT max_temp_f, min_temp_f, source FROM nws_daily "
            "WHERE station_id = 'KNYC' AND obs_date = '2026-02-27'"
        ).fetchone()
    finally:
        con.close()

    assert row[0] == 45.0      # CLI max
    assert row[1] == 30.0      # CLI min
    assert row[2] == "NWS_CLI"  # source updated


@pytest.mark.asyncio
async def test_dsm_does_not_overwrite_cli():
    """DSM must NOT overwrite an existing CLI row — lower priority."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert CLI row first (the authoritative source)
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-27", 45.0, 30.0, "NWS_CLI", datetime(2026, 2, 28, 6, 30)],
    )
    con.close()

    # Attempt to upsert DSM row — should be rejected
    dsm_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-27",
        "max_temp_f": 44.0,
        "min_temp_f": 29.0,
        "source": "DSM",
        "ingested_at": datetime(2026, 2, 27, 21, 15),
    }
    result = fetcher._upsert_row(dsm_row)
    assert result is False  # Rejected — lower priority

    # Verify CLI data is untouched
    con = get_connection(TEST_DB)
    try:
        row = con.execute(
            "SELECT max_temp_f, min_temp_f, source FROM nws_daily "
            "WHERE station_id = 'KNYC' AND obs_date = '2026-02-27'"
        ).fetchone()
    finally:
        con.close()

    assert row[0] == 45.0      # CLI max preserved
    assert row[1] == 30.0      # CLI min preserved
    assert row[2] == "NWS_CLI"  # source unchanged


@pytest.mark.asyncio
async def test_acis_does_not_overwrite_dsm():
    """ACIS must NOT overwrite an existing DSM row — lower priority."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert DSM row first
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-27", 44.0, 29.0, "DSM", datetime(2026, 2, 27, 21, 15)],
    )
    con.close()

    # Attempt ACIS upsert — should be rejected
    acis_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-27",
        "max_temp_f": 43.0,
        "min_temp_f": 28.0,
        "source": "ACIS",
        "ingested_at": datetime(2026, 2, 28, 4, 0),
    }
    result = fetcher._upsert_row(acis_row)
    assert result is False

    con = get_connection(TEST_DB)
    try:
        row = con.execute(
            "SELECT max_temp_f, source FROM nws_daily "
            "WHERE station_id = 'KNYC' AND obs_date = '2026-02-27'"
        ).fetchone()
    finally:
        con.close()

    assert row[0] == 44.0   # DSM value preserved
    assert row[1] == "DSM"  # source unchanged


@pytest.mark.asyncio
async def test_acis_does_not_overwrite_cli():
    """ACIS must NOT overwrite an existing CLI row — lowest priority."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert CLI row first
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-27", 45.0, 30.0, "NWS_CLI", datetime(2026, 2, 28, 6, 30)],
    )
    con.close()

    # Attempt ACIS upsert — should be rejected
    acis_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-27",
        "max_temp_f": 43.0,
        "min_temp_f": 28.0,
        "source": "ACIS",
        "ingested_at": datetime(2026, 2, 28, 4, 0),
    }
    result = fetcher._upsert_row(acis_row)
    assert result is False

    con = get_connection(TEST_DB)
    try:
        row = con.execute(
            "SELECT max_temp_f, source FROM nws_daily "
            "WHERE station_id = 'KNYC' AND obs_date = '2026-02-27'"
        ).fetchone()
    finally:
        con.close()

    assert row[0] == 45.0      # CLI value preserved
    assert row[1] == "NWS_CLI"  # source unchanged


@pytest.mark.asyncio
async def test_same_source_updates():
    """Same-priority source should update (refresh) the row."""
    fetcher = NWSFetcher(db_path=TEST_DB)

    # Insert initial DSM row
    con = get_connection(TEST_DB)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", "2026-02-27", 43.0, 28.0, "DSM", datetime(2026, 2, 27, 20, 15)],
    )
    con.close()

    # Upsert a newer DSM with updated temps — should overwrite
    dsm_row = {
        "station_id": "KNYC",
        "obs_date": "2026-02-27",
        "max_temp_f": 44.0,
        "min_temp_f": 29.0,
        "source": "DSM",
        "ingested_at": datetime(2026, 2, 27, 21, 15),
    }
    result = fetcher._upsert_row(dsm_row)
    assert result is True

    con = get_connection(TEST_DB)
    try:
        row = con.execute(
            "SELECT max_temp_f, min_temp_f, source FROM nws_daily "
            "WHERE station_id = 'KNYC' AND obs_date = '2026-02-27'"
        ).fetchone()
        count = con.execute("SELECT COUNT(*) FROM nws_daily").fetchone()[0]
    finally:
        con.close()

    assert row[0] == 44.0   # Updated max
    assert row[1] == 29.0   # Updated min
    assert row[2] == "DSM"  # Same source
    assert count == 1        # Still one row, not duplicated
