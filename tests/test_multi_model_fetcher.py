# tests/test_multi_model_fetcher.py
"""Tests for MultiModelFetcher — GFS/ECMWF live ingestion via Open-Meteo."""
from datetime import date, datetime

import duckdb
import pytest
from unittest.mock import patch, MagicMock

from services.multi_model_fetcher import MultiModelFetcher


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE forecasts (
            station_id VARCHAR NOT NULL,
            model_run TIMESTAMP NOT NULL,
            valid_at TIMESTAMP NOT NULL,
            temp_f DOUBLE,
            temp_c DOUBLE,
            ingested_at TIMESTAMP,
            model_name VARCHAR DEFAULT 'hrrr',
            fxx INTEGER,
            is_spinup BOOLEAN DEFAULT FALSE,
            UNIQUE (station_id, model_run, valid_at, model_name)
        )
    """)
    con.execute("""
        CREATE TABLE forecast_extended (
            station_id VARCHAR NOT NULL,
            model_run TIMESTAMP NOT NULL,
            valid_at TIMESTAMP NOT NULL,
            model_name VARCHAR NOT NULL,
            dewpoint_2m_f DOUBLE,
            humidity_2m DOUBLE,
            wind_speed_10m DOUBLE,
            wind_dir_10m DOUBLE,
            wind_gusts_10m DOUBLE,
            pressure_msl DOUBLE,
            cloud_cover DOUBLE,
            precipitation DOUBLE,
            shortwave_rad DOUBLE,
            cape DOUBLE,
            ingested_at TIMESTAMP NOT NULL,
            UNIQUE (station_id, model_run, valid_at, model_name)
        )
    """)
    con.close()
    return db_path


@pytest.fixture
def fetcher(test_db):
    return MultiModelFetcher(db_path=test_db)


def _mock_temp_response():
    """Minimal Open-Meteo temp response."""
    return {
        "hourly": {
            "time": [
                "2026-03-12T00:00", "2026-03-12T01:00", "2026-03-12T02:00",
            ],
            "temperature_2m": [35.0, 34.5, 33.8],
        }
    }


def _mock_extended_response():
    """Minimal Open-Meteo extended response."""
    return {
        "hourly": {
            "time": ["2026-03-12T12:00", "2026-03-12T13:00"],
            "dewpoint_2m": [28.0, 29.0],
            "relative_humidity_2m": [55.0, 60.0],
            "wind_speed_10m": [8.0, 10.0],
            "wind_direction_10m": [180.0, 200.0],
            "wind_gusts_10m": [15.0, 18.0],
            "pressure_msl": [1013.0, 1012.5],
            "cloud_cover": [40.0, 50.0],
            "precipitation": [0.0, 0.1],
            "shortwave_radiation": [300.0, 350.0],
            "cape": [100.0, 150.0],
        }
    }


class TestHasData:
    def test_no_data_returns_false(self, fetcher):
        assert fetcher._has_data_for_date("gfs", date(2026, 3, 12)) is False

    def test_with_data_returns_true(self, test_db, fetcher):
        con = duckdb.connect(test_db)
        con.execute(
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, model_name, fxx) "
            "VALUES ('KNYC', '2026-03-12 00:00', '2026-03-12 06:00', 40.0, 'gfs', 6)"
        )
        con.close()
        assert fetcher._has_data_for_date("gfs", date(2026, 3, 12)) is True

    def test_no_extended_returns_false(self, fetcher):
        assert fetcher._has_extended_for_date("ecmwf", date(2026, 3, 12)) is False


class TestInsertTemp:
    def test_insert_temp_rows(self, test_db, fetcher):
        data = _mock_temp_response()
        count = fetcher._insert_temp("gfs", data)
        assert count == 3

        con = duckdb.connect(test_db)
        rows = con.execute("SELECT COUNT(*) FROM forecasts WHERE model_name = 'gfs'").fetchone()
        con.close()
        assert rows[0] == 3

    def test_insert_temp_idempotent(self, fetcher):
        data = _mock_temp_response()
        fetcher._insert_temp("gfs", data)
        count2 = fetcher._insert_temp("gfs", data)
        assert count2 == 0  # All duplicates

    def test_model_run_is_midnight(self, test_db, fetcher):
        data = _mock_temp_response()
        fetcher._insert_temp("ecmwf", data)

        con = duckdb.connect(test_db)
        row = con.execute(
            "SELECT DISTINCT EXTRACT(HOUR FROM model_run) FROM forecasts WHERE model_name = 'ecmwf'"
        ).fetchone()
        con.close()
        assert row[0] == 0  # All model_run hours are 0 (midnight)


class TestInsertExtended:
    def test_insert_extended_rows(self, test_db, fetcher):
        data = _mock_extended_response()
        count = fetcher._insert_extended("ecmwf", data)
        assert count == 2

        con = duckdb.connect(test_db)
        rows = con.execute(
            "SELECT COUNT(*) FROM forecast_extended WHERE model_name = 'ecmwf'"
        ).fetchone()
        con.close()
        assert rows[0] == 2

    def test_extended_columns_populated(self, test_db, fetcher):
        data = _mock_extended_response()
        fetcher._insert_extended("gfs", data)

        con = duckdb.connect(test_db)
        row = con.execute(
            "SELECT dewpoint_2m_f, humidity_2m, shortwave_rad, cape, precipitation "
            "FROM forecast_extended WHERE model_name = 'gfs' LIMIT 1"
        ).fetchone()
        con.close()
        assert row[0] == 28.0  # dewpoint
        assert row[1] == 55.0  # humidity
        assert row[2] == 300.0  # shortwave_rad
        assert row[3] == 100.0  # cape
        assert row[4] == 0.0   # precipitation


class TestFetchModelForDate:
    @patch("services.multi_model_fetcher.requests.get")
    def test_fetches_when_no_data(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = [_mock_temp_response(), _mock_extended_response()]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        total = fetcher.fetch_model_for_date("gfs", "gfs_seamless", date(2026, 3, 12))
        assert total == 5  # 3 temp + 2 extended
        assert mock_get.call_count == 2

    @patch("services.multi_model_fetcher.requests.get")
    def test_skips_when_data_exists(self, mock_get, test_db, fetcher):
        # Pre-populate both tables
        con = duckdb.connect(test_db)
        con.execute(
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, model_name, fxx) "
            "VALUES ('KNYC', '2026-03-12 00:00', '2026-03-12 06:00', 40.0, 'gfs', 6)"
        )
        con.execute(
            "INSERT INTO forecast_extended "
            "(station_id, model_run, valid_at, model_name, ingested_at) "
            "VALUES ('KNYC', '2026-03-12 00:00', '2026-03-12 12:00', 'gfs', '2026-03-12 08:00')"
        )
        con.close()

        total = fetcher.fetch_model_for_date("gfs", "gfs_seamless", date(2026, 3, 12))
        assert total == 0
        assert mock_get.call_count == 0  # No API calls — data already exists
