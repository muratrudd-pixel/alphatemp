"""Tests for IEM ASOS ingestor."""

import os
from datetime import datetime, timezone

import duckdb
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from core.db import init_db
from services.iem_ingestor import IEMIngestor

TEST_DB = "data/test_iem.duckdb"

# Standard METAR CSV row from IEM
IEM_CSV_METAR = (
    "station,valid,tmpf,metar\n"
    'NYC,2026-02-23 19:51,33.0,'
    '"METAR KNYC 231951Z 31008KT 10SM FEW250 01/M06 A3032 RMK AO2 SLP283 T00060061"\n'
)

# SPECI at non-standard time (the whole point of this ingestor)
IEM_CSV_SPECI = (
    "station,valid,tmpf,metar\n"
    'NYC,2026-02-23 20:34,33.1,'
    '"SPECI KNYC 232034Z 28006KT 10SM FEW250 01/M03 A3030 RMK AO2 T00501028"\n'
)

# Stub METAR — no Zulu timestamp, garbage temp
IEM_CSV_STUB = (
    "station,valid,tmpf,metar\n"
    'MIA,2026-02-23 12:00,72.5,"METAR KMIA AUTO"\n'
)

# Empty response — header only
IEM_CSV_EMPTY = "station,valid,tmpf,metar\n"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _mock_iem_response(csv_text: str):
    """Build a mock httpx response returning the given CSV text."""
    mock_resp = MagicMock()
    mock_resp.text = csv_text
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@pytest.mark.asyncio
async def test_iem_parses_metar_row(test_db):
    """Standard METAR CSV row → correct station_id, observed_at, temp_f, temp_c_tenth."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_iem_response(IEM_CSV_METAR)
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
    assert row[2] == 33.0
    # T00060061 → +0.6°C
    assert row[3] == 0.6


@pytest.mark.asyncio
async def test_iem_parses_speci_row(test_db):
    """SPECI at :34 (non-standard time) → row captured with correct T-group."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_iem_response(IEM_CSV_SPECI)
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
    assert row[1].minute == 34  # Non-standard SPECI time
    assert row[2] == 33.1
    # T00501028 → +5.0°C
    assert row[3] == 5.0


@pytest.mark.asyncio
async def test_iem_skips_stub_metar(test_db):
    """Stub METAR → temp_f and temp_c_tenth are None."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_iem_response(IEM_CSV_STUB)
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
async def test_iem_skips_empty_csv(test_db):
    """Empty/header-only CSV → 0 rows inserted."""
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_iem_response(IEM_CSV_EMPTY)
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
    """Synoptic row first, then IEM for same (station, time) → only 1 row in DB."""
    # Pre-insert a Synoptic observation (naive UTC, matching DB convention)
    observed = datetime(2026, 2, 23, 19, 51)
    con = duckdb.connect(test_db)
    con.execute(
        """INSERT INTO observations
           (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["KNYC", observed, 33.0, 0.6,
         "METAR KNYC 231951Z 31008KT 10SM FEW250 01/M06 A3032 RMK AO2 SLP283 T00060061",
         datetime.now()],
    )
    con.close()

    # Now poll IEM with the same observation
    with patch("services.iem_ingestor.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_iem_response(IEM_CSV_METAR)
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
