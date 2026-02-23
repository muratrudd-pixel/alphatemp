"""Tests for AWC METAR+SPECI ingestor (services/iem_ingestor.py)."""

import os
from datetime import datetime, timezone

import duckdb
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from core.db import init_db
from services.iem_ingestor import IEMIngestor

TEST_DB = "data/test_iem.duckdb"

# Standard METAR JSON from AWC
AWC_JSON_METAR = [
    {
        "icaoId": "KNYC",
        "obsTime": 1771876260,  # 2026-02-23T19:51:00Z
        "reportTime": "2026-02-23T20:00:00.000Z",  # AWC rounds to :00
        "temp": 0.0,
        "dewp": -1.1,
        "wdir": "VRB",
        "wspd": 4,
        "visib": 1.25,
        "altim": 1003.1,
        "metarType": "METAR",
        "rawOb": "METAR KNYC 231951Z AUTO VRB04KT 1 1/4SM BR 00/M01 A2961 RMK AO2 SLP018 T00001011",
    }
]

# SPECI at non-standard time (the whole point of this ingestor)
AWC_JSON_SPECI = [
    {
        "icaoId": "KNYC",
        "obsTime": 1771880880,  # 2026-02-23T21:08:00Z
        "reportTime": "2026-02-23T21:08:00.000Z",
        "temp": 0.6,
        "dewp": -1.1,
        "wdir": "VRB",
        "wspd": 6,
        "wgst": 17,
        "visib": 3,
        "altim": 1003.5,
        "metarType": "SPECI",
        "rawOb": "SPECI KNYC 232108Z AUTO VRB06G17KT 3SM BR 01/M01 A2963 RMK AO2 T00061011 $",
    }
]

# Stub METAR — no Zulu timestamp, garbage temp
AWC_JSON_STUB = [
    {
        "icaoId": "KMIA",
        "obsTime": 1771848000,  # 2026-02-23T12:00:00Z
        "reportTime": "2026-02-23T12:00:00.000Z",
        "temp": 22.5,
        "dewp": 18.0,
        "metarType": "METAR",
        "rawOb": "METAR KMIA AUTO",
    }
]


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _mock_awc_response(json_data):
    """Build a mock httpx response returning the given JSON."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@pytest.mark.asyncio
async def test_iem_parses_metar_row(test_db):
    """Standard METAR JSON → correct station_id, observed_at, temp_f, temp_c_tenth."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_awc_response(AWC_JSON_METAR)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        ingestor = IEMIngestor(db_path=test_db)
        inserted = await ingestor.poll_once()

    assert inserted == 1

    con = duckdb.connect(test_db)
    row = con.execute(
        "SELECT station_id, observed_at, temp_f, temp_c_tenth FROM observations"
    ).fetchone()
    con.close()

    assert row[0] == "KNYC"
    assert row[1].hour == 19 and row[1].minute == 51
    assert row[2] == 32.0  # 0.0°C → 32.0°F
    # T00001011 → 0.0°C
    assert row[3] == 0.0


@pytest.mark.asyncio
async def test_iem_parses_speci_row(test_db):
    """SPECI at :08 (non-standard time) → row captured with correct T-group."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_awc_response(AWC_JSON_SPECI)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        ingestor = IEMIngestor(db_path=test_db)
        inserted = await ingestor.poll_once()

    assert inserted == 1

    con = duckdb.connect(test_db)
    row = con.execute(
        "SELECT station_id, observed_at, temp_f, temp_c_tenth FROM observations"
    ).fetchone()
    con.close()

    assert row[0] == "KNYC"
    assert row[1].hour == 21 and row[1].minute == 8  # Non-standard SPECI time
    assert row[2] == 33.1  # 0.6°C → 33.08 → 33.1°F
    # T00061011 → 0.6°C
    assert row[3] == 0.6


@pytest.mark.asyncio
async def test_iem_skips_stub_metar(test_db):
    """Stub METAR → temp_f and temp_c_tenth are None."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_awc_response(AWC_JSON_STUB)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        ingestor = IEMIngestor(db_path=test_db)
        inserted = await ingestor.poll_once()

    assert inserted == 1

    con = duckdb.connect(test_db)
    row = con.execute(
        "SELECT station_id, temp_f, temp_c_tenth FROM observations"
    ).fetchone()
    con.close()

    assert row[0] == "KMIA"
    assert row[1] is None, f"Stub METAR should have temp_f=None, got {row[1]}"
    assert row[2] is None, f"Stub METAR should have temp_c_tenth=None, got {row[2]}"


@pytest.mark.asyncio
async def test_iem_skips_empty_json(test_db):
    """Empty JSON array → 0 rows inserted."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_awc_response([])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        ingestor = IEMIngestor(db_path=test_db)
        inserted = await ingestor.poll_once()

    assert inserted == 0

    con = duckdb.connect(test_db)
    count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    con.close()
    assert count == 0


@pytest.mark.asyncio
async def test_iem_dedup_with_synoptic(test_db):
    """Synoptic row first, then AWC for same (station, time) → only 1 row in DB."""
    # Pre-insert a Synoptic observation (naive UTC, matching DB convention)
    observed = datetime(2026, 2, 23, 19, 51)
    con = duckdb.connect(test_db)
    con.execute(
        """INSERT INTO observations
           (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", observed, 32.0, 0.0,
         "METAR KNYC 231951Z AUTO VRB04KT 1 1/4SM BR 00/M01 A2961 RMK AO2 SLP018 T00001011",
         datetime.now()],
    )
    con.close()

    # Now poll AWC with the same observation
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_awc_response(AWC_JSON_METAR)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        ingestor = IEMIngestor(db_path=test_db)
        inserted = await ingestor.poll_once()

    assert inserted == 0  # Duplicate skipped

    con = duckdb.connect(test_db)
    count = con.execute(
        "SELECT COUNT(*) FROM observations WHERE station_id = 'KNYC'"
    ).fetchone()[0]
    con.close()
    assert count == 1  # Only the original Synoptic row
