# tests/test_forecast.py
import os
import pytest
import duckdb
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from core.db import init_db
from services.forecast import HRRRFetcher

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_missing_runs_cold_start(test_db):
    """On empty DB, should look back COLD_START_LOOKBACK_HOURS."""
    fetcher = HRRRFetcher(db_path=test_db)
    now = datetime(2026, 2, 22, 15, 30, tzinfo=timezone.utc)
    with patch("services.forecast.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        runs = fetcher._get_missing_runs()
    # Should include current hour back to 6 hours ago = 7 runs
    assert len(runs) == 7
    assert runs[0] == datetime(2026, 2, 22, 15, 0, tzinfo=timezone.utc)  # newest
    assert runs[-1] == datetime(2026, 2, 22, 9, 0, tzinfo=timezone.utc)  # oldest


def test_missing_runs_with_stored_data(test_db):
    """Should only return runs newer than what's already stored."""
    # Insert a fake forecast for 12:00z run
    con = duckdb.connect(test_db)
    con.execute(
        """INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
           VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 35.0, 1.7, '2026-02-22 14:00:00')"""
    )
    con.close()

    fetcher = HRRRFetcher(db_path=test_db)
    now = datetime(2026, 2, 22, 15, 30, tzinfo=timezone.utc)
    with patch("services.forecast.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        runs = fetcher._get_missing_runs()
    # 13z, 14z, 15z — should NOT include 12z (already stored)
    assert len(runs) == 3
    assert runs[0] == datetime(2026, 2, 22, 15, 0, tzinfo=timezone.utc)
    assert runs[-1] == datetime(2026, 2, 22, 13, 0, tzinfo=timezone.utc)


def test_missing_runs_all_caught_up(test_db):
    """If latest stored run is the current hour, nothing to fetch."""
    con = duckdb.connect(test_db)
    con.execute(
        """INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
           VALUES ('KNYC', '2026-02-22 15:00:00', '2026-02-22 16:00:00', 35.0, 1.7, '2026-02-22 16:30:00')"""
    )
    con.close()

    fetcher = HRRRFetcher(db_path=test_db)
    now = datetime(2026, 2, 22, 15, 30, tzinfo=timezone.utc)
    with patch("services.forecast.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        runs = fetcher._get_missing_runs()
    assert len(runs) == 0


def _make_mock_grib_msg():
    """Create a mock pygrib message with 2m temp data at a single grid point."""
    import numpy as np
    mock_msg = MagicMock()
    mock_msg.values = np.array([[280.0]])
    mock_msg.latlons.return_value = (np.array([[40.78]]), np.array([[-73.97]]))
    return mock_msg


def test_fetcher_stores_forecasts(test_db):
    """Mock Herbie to verify forecasts land in DuckDB."""
    mock_msg = _make_mock_grib_msg()

    mock_herbie = MagicMock()
    mock_herbie.download.return_value = "/tmp/fake.grib2"

    mock_grbs = MagicMock()
    mock_grbs.select.return_value = [mock_msg]

    with patch("services.forecast.Herbie", return_value=mock_herbie), \
         patch("services.forecast.pygrib.open", return_value=mock_grbs):
        fetcher = HRRRFetcher(db_path=test_db)
        model_run = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
        inserted = fetcher.fetch_run(model_run, fxx_range=range(1, 3))

    assert inserted > 0

    con = duckdb.connect(test_db)
    rows = con.execute("SELECT station_id, temp_f, temp_c FROM forecasts").fetchall()
    con.close()

    assert len(rows) > 0
    for row in rows:
        assert row[2] is not None  # temp_c exists
        assert row[1] is not None  # temp_f exists


def test_fetcher_deduplicates(test_db):
    """Running fetch_run twice with same data should not create duplicates."""
    mock_msg = _make_mock_grib_msg()

    mock_herbie = MagicMock()
    mock_herbie.download.return_value = "/tmp/fake.grib2"

    mock_grbs = MagicMock()
    mock_grbs.select.return_value = [mock_msg]

    with patch("services.forecast.Herbie", return_value=mock_herbie), \
         patch("services.forecast.pygrib.open", return_value=mock_grbs):
        fetcher = HRRRFetcher(db_path=test_db)
        model_run = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
        fetcher.fetch_run(model_run, fxx_range=range(1, 3))
        fetcher.fetch_run(model_run, fxx_range=range(1, 3))

    con = duckdb.connect(test_db)
    count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    con.close()

    # 1 station x 2 forecast hours = 2 rows, no duplicates
    assert count == 2
