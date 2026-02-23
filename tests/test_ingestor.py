import os

import duckdb
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.ingestor import parse_t_group, _is_metar_stub, SynopticIngestor
from core.db import init_db


def test_parse_positive_temp():
    assert parse_t_group("T02280167") == 22.8


def test_parse_negative_temp():
    assert parse_t_group("T10051012") == -0.5


def test_parse_zero():
    assert parse_t_group("T00000000") == 0.0


def test_parse_no_t_group():
    assert parse_t_group("RMK AO2 SLP135") is None


def test_parse_embedded_in_metar():
    metar = "RMK AO2 SLP135 T02280167 10272 20228 53012"
    assert parse_t_group(metar) == 22.8


def test_parse_negative_freezing():
    assert parse_t_group("T10171028") == -1.7


def test_parse_hot_day():
    assert parse_t_group("T03890350") == 38.9


TEST_DB = "data/test_alphatemp.duckdb"

MOCK_SYNOPTIC_RESPONSE = {
    "SUMMARY": {"RESPONSE_CODE": 1, "RESPONSE_MESSAGE": "OK", "NUMBER_OF_OBJECTS": 2},
    "STATION": [
        {
            "STID": "KNYC",
            "OBSERVATIONS": {
                "date_time": ["2026-02-22T12:00:00Z", "2026-02-22T12:01:00Z"],
                "air_temp_set_1": [7.2, 7.3],
                "metar_set_1": [
                    "METAR KNYC 221200Z RMK AO2 T00720056",
                    "METAR KNYC 221201Z RMK AO2 T00730058",
                ],
            },
        },
        {
            "STID": "KMDW",
            "OBSERVATIONS": {
                "date_time": ["2026-02-22T12:00:00Z"],
                "air_temp_set_1": [-2.5],
                "metar_set_1": [
                    "METAR KMDW 221200Z RMK AO2 T10251033",
                ],
            },
        },
    ]
}


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.mark.asyncio
async def test_ingestor_stores_observations(test_db):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SYNOPTIC_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("services.ingestor.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        ingestor = SynopticIngestor(token="test_token", db_path=test_db)
        await ingestor.poll_once()

    con = duckdb.connect(test_db)
    rows = con.execute("SELECT station_id, temp_c_tenth FROM observations ORDER BY station_id, observed_at").fetchall()
    con.close()

    # KMDW: -2.5, KNYC: 7.2, 7.3
    assert len(rows) == 3
    station_ids = [r[0] for r in rows]
    assert "KNYC" in station_ids
    assert "KMDW" in station_ids

    # Check T-group parsing worked
    kmdw_row = [r for r in rows if r[0] == "KMDW"][0]
    assert kmdw_row[1] == -2.5


@pytest.mark.asyncio
async def test_ingestor_deduplicates(test_db):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SYNOPTIC_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("services.ingestor.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        ingestor = SynopticIngestor(token="test_token", db_path=test_db)
        await ingestor.poll_once()
        await ingestor.poll_once()  # Second poll — same data

    con = duckdb.connect(test_db)
    count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    con.close()
    assert count == 3  # No duplicates


# --- METAR stub detection ---

def test_stub_detection_empty_string():
    assert _is_metar_stub("") is True


def test_stub_detection_none_like():
    assert _is_metar_stub("") is True


def test_stub_detection_bare_stub():
    assert _is_metar_stub("METAR KMIA AUTO") is True


def test_stub_detection_stub_with_station_only():
    assert _is_metar_stub("METAR KNYC") is True


def test_stub_detection_real_metar():
    assert _is_metar_stub("METAR KNYC 221200Z 31008KT 10SM FEW250 07/M06 A3032 RMK AO2 T00720056") is False


def test_stub_detection_real_metar_minimal():
    assert _is_metar_stub("METAR KMIA 230505Z AUTO") is False


# --- Stub rejection in poll_once ---

MOCK_STUB_RESPONSE = {
    "SUMMARY": {"RESPONSE_CODE": 1, "RESPONSE_MESSAGE": "OK", "NUMBER_OF_OBJECTS": 1},
    "STATION": [
        {
            "STID": "KMIA",
            "OBSERVATIONS": {
                "date_time": ["2026-02-22T12:00:00Z", "2026-02-22T12:01:00Z"],
                "air_temp_set_1": [19.8, 7.5],
                "metar_set_1": [
                    "METAR KMIA AUTO",                          # stub — garbage temp
                    "METAR KMIA 221201Z RMK AO2 T00750060",    # real
                ],
            },
        },
    ],
}


@pytest.mark.asyncio
async def test_stub_metar_rejects_api_temp(test_db):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_STUB_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("services.ingestor.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        ingestor = SynopticIngestor(token="test_token", db_path=test_db)
        await ingestor.poll_once()

    con = duckdb.connect(test_db)
    rows = con.execute(
        "SELECT observed_at, temp_f, temp_c_tenth, raw_metar FROM observations WHERE station_id='KMIA' ORDER BY observed_at"
    ).fetchall()
    con.close()

    assert len(rows) == 2

    # First row: stub METAR — temp_f and temp_c_tenth should be NULL
    stub_row = rows[0]
    assert stub_row[1] is None, f"Stub METAR should have temp_f=None, got {stub_row[1]}"
    assert stub_row[2] is None, f"Stub METAR should have temp_c_tenth=None, got {stub_row[2]}"

    # Second row: real METAR — temps should be preserved
    real_row = rows[1]
    assert real_row[1] is not None, "Real METAR should have temp_f"
    assert real_row[2] is not None, "Real METAR should have temp_c_tenth"


@pytest.mark.asyncio
async def test_real_metar_keeps_api_temp(test_db):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SYNOPTIC_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("services.ingestor.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        ingestor = SynopticIngestor(token="test_token", db_path=test_db)
        await ingestor.poll_once()

    con = duckdb.connect(test_db)
    rows = con.execute(
        "SELECT temp_f, temp_c_tenth FROM observations ORDER BY station_id, observed_at"
    ).fetchall()
    con.close()

    # All 3 rows from MOCK_SYNOPTIC_RESPONSE have real METARs — all should have temps
    for row in rows:
        assert row[0] is not None, "Real METAR should preserve temp_f"
        assert row[1] is not None, "Real METAR should preserve temp_c_tenth"


# --- Gap recovery logging ---

@pytest.mark.asyncio
async def test_recover_gap_empty_db(test_db):
    """Fresh DB with no observations — should log 'starting fresh'."""
    ingestor = SynopticIngestor(token="test_token", db_path=test_db)
    with patch("services.ingestor.logger") as mock_logger:
        await ingestor._recover_gap()
        mock_logger.info.assert_called_once()
        assert "starting fresh" in mock_logger.info.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_recover_gap_large_gap(test_db):
    """Observation 3 hours ago — should warn about unrecoverable gap."""
    from datetime import datetime, timezone, timedelta

    con = duckdb.connect(test_db)
    old_time = datetime.now(timezone.utc) - timedelta(hours=3)
    con.execute(
        "INSERT INTO observations (station_id, observed_at, temp_f, raw_metar, ingested_at) VALUES (?, ?, ?, ?, ?)",
        ["KNYC", old_time, 45.0, "METAR KNYC 221200Z RMK AO2", old_time],
    )
    con.close()

    ingestor = SynopticIngestor(token="test_token", db_path=test_db)
    with patch("services.ingestor.logger") as mock_logger:
        await ingestor._recover_gap()
        mock_logger.warning.assert_called_once()
        assert "unrecoverable" in mock_logger.warning.call_args[0][0].lower()
