# tests/test_forecast.py
import os
import pytest
import duckdb
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from core.db import init_db
from services.forecast import HRRRFetcher, get_recent_model_runs

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_get_recent_model_runs():
    """Should return 3 model run datetimes, each 1 hour apart, offset by publication delay."""
    ref_time = datetime(2026, 2, 22, 15, 30, tzinfo=timezone.utc)
    runs = get_recent_model_runs(ref_time, count=3, delay_hours=2)
    assert len(runs) == 3
    assert runs[0] == datetime(2026, 2, 22, 13, 0, tzinfo=timezone.utc)
    assert runs[1] == datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    assert runs[2] == datetime(2026, 2, 22, 11, 0, tzinfo=timezone.utc)


def test_get_recent_model_runs_crosses_midnight():
    """Model runs should cross midnight correctly."""
    ref_time = datetime(2026, 2, 22, 2, 30, tzinfo=timezone.utc)
    runs = get_recent_model_runs(ref_time, count=3, delay_hours=2)
    assert runs[0] == datetime(2026, 2, 22, 0, 0, tzinfo=timezone.utc)
    assert runs[1] == datetime(2026, 2, 21, 23, 0, tzinfo=timezone.utc)
    assert runs[2] == datetime(2026, 2, 21, 22, 0, tzinfo=timezone.utc)


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

    # 5 stations x 2 forecast hours = 10 rows, no duplicates
    assert count == 10
