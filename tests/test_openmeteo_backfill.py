"""Tests for Open-Meteo historical forecast backfill script."""

import os
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock

import duckdb
import pytest

from core.db import init_db

TEST_DB = "data/test_openmeteo_backfill.duckdb"


def _cleanup_db():
    """Remove test DB and WAL file."""
    for path in [TEST_DB, TEST_DB + ".wal"]:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def test_db():
    """Create a fresh test DB, clean up after."""
    _cleanup_db()
    init_db(TEST_DB)
    yield TEST_DB
    _cleanup_db()


def _mock_one_day_response():
    """Build a mock Open-Meteo response for 2024-01-01 (24 hourly values)."""
    times = ["2024-01-01T{:02d}:00".format(h) for h in range(24)]
    # Temperatures in Fahrenheit: 30.0, 31.0, ..., 53.0
    temps = [30.0 + float(h) for h in range(24)]
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
        }
    }


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_backfill_inserts_24_rows_for_one_day(mock_sleep, mock_get, test_db):
    """A 1-day mock response should produce exactly 24 forecast rows."""
    from scripts.backfill_openmeteo import backfill_model

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_one_day_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        assert count == 24, "Expected 24 rows for 1 day, got {}".format(count)
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_all_rows_have_correct_model_name(mock_sleep, mock_get, test_db):
    """Every row should have model_name='gfs'."""
    from scripts.backfill_openmeteo import backfill_model

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_one_day_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        distinct = con.execute(
            "SELECT DISTINCT model_name FROM forecasts"
        ).fetchall()
        assert len(distinct) == 1
        assert distinct[0][0] == "gfs"
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_model_run_is_midnight_utc(mock_sleep, mock_get, test_db):
    """model_run should be midnight UTC for every row."""
    from scripts.backfill_openmeteo import backfill_model

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_one_day_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        runs = con.execute("SELECT DISTINCT model_run FROM forecasts").fetchall()
        for (run_ts,) in runs:
            # DuckDB returns datetime objects; hour/minute/second should be 0
            assert run_ts.hour == 0
            assert run_ts.minute == 0
            assert run_ts.second == 0
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_temperature_conversion_correct(mock_sleep, mock_get, test_db):
    """F-to-C conversion should be correct for known values."""
    from scripts.backfill_openmeteo import backfill_model

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_one_day_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        # First row: 30.0 F -> (30-32)*5/9 = -1.11 C
        row = con.execute(
            "SELECT temp_f, temp_c FROM forecasts ORDER BY valid_at LIMIT 1"
        ).fetchone()
        assert row[0] == 30.0
        expected_c = round((30.0 - 32.0) * 5.0 / 9.0, 2)
        assert row[1] == expected_c, "Expected {}, got {}".format(expected_c, row[1])
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_idempotent_no_duplicates(mock_sleep, mock_get, test_db):
    """Running backfill twice with the same data should not create duplicates."""
    from scripts.backfill_openmeteo import backfill_model

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_one_day_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    # Run twice
    for _ in range(2):
        backfill_model(
            model="gfs",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            db_path=test_db,
        )

    con = duckdb.connect(test_db, read_only=True)
    try:
        count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        assert count == 24, "Expected 24 rows (no duplicates), got {}".format(count)
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_station_id_is_knyc(mock_sleep, mock_get, test_db):
    """All rows should have station_id='KNYC'."""
    from scripts.backfill_openmeteo import backfill_model

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_one_day_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        stations = con.execute(
            "SELECT DISTINCT station_id FROM forecasts"
        ).fetchall()
        assert len(stations) == 1
        assert stations[0][0] == "KNYC"
    finally:
        con.close()


# --- Extended variable backfill tests ---

def _mock_extended_response():
    """Build a mock Open-Meteo response with all extended variables for 1 day."""
    times = ["2024-01-01T{:02d}:00".format(h) for h in range(24)]
    return {
        "hourly": {
            "time": times,
            "dewpoint_2m": [25.0 + float(h) for h in range(24)],
            "relative_humidity_2m": [70.0 + float(h) * 0.5 for h in range(24)],
            "wind_speed_10m": [5.0 + float(h) * 0.2 for h in range(24)],
            "wind_direction_10m": [180.0 + float(h) for h in range(24)],
            "wind_gusts_10m": [10.0 + float(h) * 0.3 for h in range(24)],
            "pressure_msl": [1013.0 + float(h) * 0.1 for h in range(24)],
            "cloud_cover": [50.0 + float(h) for h in range(24)],
            "precipitation": [0.0] * 20 + [0.1, 0.2, 0.3, 0.0],
            "shortwave_radiation": [0.0] * 6 + [100.0 + float(h) * 10 for h in range(12)] + [0.0] * 6,
            "cape": [0.0] * 12 + [50.0 + float(h) for h in range(12)],
        }
    }


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_extended_backfill_inserts_rows(mock_sleep, mock_get, test_db):
    """Extended backfill should insert 24 rows into forecast_extended."""
    from scripts.backfill_openmeteo import backfill_model_extended

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_extended_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    count = backfill_model_extended(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )
    assert count == 24

    con = duckdb.connect(test_db, read_only=True)
    try:
        rows = con.execute("SELECT COUNT(*) FROM forecast_extended").fetchone()[0]
        assert rows == 24
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_extended_has_correct_model_name(mock_sleep, mock_get, test_db):
    """Extended rows should have correct model_name."""
    from scripts.backfill_openmeteo import backfill_model_extended

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_extended_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model_extended(
        model="ecmwf",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        names = con.execute(
            "SELECT DISTINCT model_name FROM forecast_extended"
        ).fetchall()
        assert len(names) == 1
        assert names[0][0] == "ecmwf"
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_extended_values_stored_correctly(mock_sleep, mock_get, test_db):
    """Verify specific variable values are stored correctly."""
    from scripts.backfill_openmeteo import backfill_model_extended

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_extended_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model_extended(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        # First row (hour 0): dewpoint=25.0, humidity=70.0, wind=5.0
        row = con.execute(
            "SELECT dewpoint_2m_f, humidity_2m, wind_speed_10m, pressure_msl "
            "FROM forecast_extended ORDER BY valid_at LIMIT 1"
        ).fetchone()
        assert row[0] == 25.0   # dewpoint
        assert row[1] == 70.0   # humidity
        assert row[2] == 5.0    # wind speed
        assert row[3] == 1013.0 # pressure
    finally:
        con.close()


@patch("scripts.backfill_openmeteo.requests.get")
@patch("scripts.backfill_openmeteo.time.sleep")
def test_extended_model_run_is_midnight(mock_sleep, mock_get, test_db):
    """Extended rows should have model_run at midnight UTC."""
    from scripts.backfill_openmeteo import backfill_model_extended

    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_extended_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    backfill_model_extended(
        model="gfs",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        db_path=test_db,
    )

    con = duckdb.connect(test_db, read_only=True)
    try:
        runs = con.execute("SELECT DISTINCT model_run FROM forecast_extended").fetchall()
        for (run_ts,) in runs:
            assert run_ts.hour == 0
            assert run_ts.minute == 0
    finally:
        con.close()
