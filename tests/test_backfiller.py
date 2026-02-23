"""Tests for historical ASOS observation backfiller."""

import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from core.db import init_db, get_connection
from services.backfiller import backfill

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _synoptic_response(stations_data):
    """Build a mock Synoptic API JSON response."""
    return {
        "SUMMARY": {"RESPONSE_CODE": 1, "RESPONSE_MESSAGE": "OK"},
        "STATION": stations_data,
    }


def _mock_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


SINGLE_STATION_OBS = [
    {
        "STID": "KNYC",
        "OBSERVATIONS": {
            "date_time": [
                "2026-01-15T12:00:00Z",
                "2026-01-15T12:01:00Z",
            ],
            "air_temp_set_1": [5.0, 5.5],
            "metar_set_1": [
                "METAR KNYC RMK T0050",
                "METAR KNYC RMK T0055",
            ],
        },
    }
]


@patch("services.backfiller.httpx.get")
def test_correct_api_params(mock_get, test_db):
    """Should pass correct stid, date range, and vars to Synoptic."""
    mock_get.return_value = _mock_response(
        _synoptic_response(SINGLE_STATION_OBS)
    )

    backfill(token="test-token", days_back=3, db_path=test_db)

    # Verify API was called with expected params
    call_args = mock_get.call_args_list[0]
    params = call_args.kwargs.get("params") or call_args[1].get("params")
    assert "KNYC" in params["stid"]
    assert params["vars"] == "air_temp,metar"
    assert params["token"] == "test-token"
    assert params["obtimezone"] == "UTC"


@patch("services.backfiller.httpx.get")
def test_rows_inserted_for_valid_response(mock_get, test_db):
    """Should insert rows when API returns valid observation data."""
    mock_get.return_value = _mock_response(
        _synoptic_response(SINGLE_STATION_OBS)
    )

    total = backfill(token="test-token", days_back=1, db_path=test_db)

    assert total == 2
    con = get_connection(test_db)
    rows = con.execute("SELECT COUNT(*) FROM observations WHERE station_id = 'KNYC'").fetchone()[0]
    con.close()
    assert rows == 2


@patch("services.backfiller.httpx.get")
def test_deduplication(mock_get, test_db):
    """Re-running should not double-insert existing observations."""
    mock_get.return_value = _mock_response(
        _synoptic_response(SINGLE_STATION_OBS)
    )

    first_run = backfill(token="test-token", days_back=1, db_path=test_db)
    second_run = backfill(token="test-token", days_back=1, db_path=test_db)

    assert first_run == 2
    assert second_run == 0  # All dupes

    con = get_connection(test_db)
    rows = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    con.close()
    assert rows == 2


@patch("services.backfiller.httpx.get")
def test_handles_api_errors_gracefully(mock_get, test_db):
    """Should continue to next chunk when API returns an error."""
    import httpx as real_httpx

    error_resp = MagicMock()
    error_resp.raise_for_status.side_effect = real_httpx.HTTPStatusError(
        "500 Server Error", request=MagicMock(), response=error_resp
    )
    ok_resp = _mock_response(_synoptic_response(SINGLE_STATION_OBS))

    mock_get.side_effect = [error_resp, ok_resp]

    # 10-day range = 2 chunks (7 + 3), first fails, second succeeds
    total = backfill(token="test-token", days_back=10, db_path=test_db)

    assert total == 2  # Only second chunk's data
    assert mock_get.call_count == 2


@patch("services.backfiller.httpx.get")
def test_handles_empty_response(mock_get, test_db):
    """Should handle responses with no station data."""
    empty_response = {
        "SUMMARY": {"RESPONSE_CODE": 2, "RESPONSE_MESSAGE": "No stations found"},
    }
    mock_get.return_value = _mock_response(empty_response)

    total = backfill(token="test-token", days_back=3, db_path=test_db)

    assert total == 0


@patch("services.backfiller.httpx.get")
def test_t_group_parsing(mock_get, test_db):
    """Should parse T-group from METAR and store temp_c_tenth."""
    station_with_tgroup = [
        {
            "STID": "KNYC",
            "OBSERVATIONS": {
                "date_time": ["2026-01-15T12:00:00Z"],
                "air_temp_set_1": [22.8],
                "metar_set_1": ["METAR KNYC 151200Z RMK T0228"],
            },
        }
    ]
    mock_get.return_value = _mock_response(
        _synoptic_response(station_with_tgroup)
    )

    backfill(token="test-token", days_back=1, db_path=test_db)

    con = get_connection(test_db)
    row = con.execute(
        "SELECT temp_c_tenth FROM observations WHERE station_id = 'KNYC'"
    ).fetchone()
    con.close()
    assert row[0] == 22.8  # T0228 → +22.8°C
