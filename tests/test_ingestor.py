import os

import duckdb
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.ingestor import parse_t_group, SynopticIngestor
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
