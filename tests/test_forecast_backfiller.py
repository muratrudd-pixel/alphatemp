"""Tests for HRRR forecast backfiller."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from core.db import init_db, get_connection
from services.forecast_backfiller import backfill_forecasts

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@patch("services.forecast_backfiller.HRRRFetcher")
def test_backfiller_calls_fetch_run_for_each_day(mock_fetcher_cls, test_db):
    """Should call fetch_run once per day for 5-day backfill."""
    mock_instance = MagicMock()
    mock_instance.fetch_run.return_value = 90  # 18 hours * 5 stations
    mock_fetcher_cls.return_value = mock_instance

    total = backfill_forecasts(days_back=5, db_path=test_db, delay_seconds=0)

    assert mock_instance.fetch_run.call_count == 5
    assert total == 450  # 5 days * 90 rows


@patch("services.forecast_backfiller.HRRRFetcher")
def test_backfiller_uses_12z_runs(mock_fetcher_cls, test_db):
    """All model runs should be 12z (hour=12)."""
    mock_instance = MagicMock()
    mock_instance.fetch_run.return_value = 90
    mock_fetcher_cls.return_value = mock_instance

    backfill_forecasts(days_back=3, db_path=test_db, delay_seconds=0)

    for call in mock_instance.fetch_run.call_args_list:
        model_run = call[0][0]
        assert model_run.hour == 12
        assert model_run.tzinfo == timezone.utc


@patch("services.forecast_backfiller.HRRRFetcher")
def test_backfiller_skips_existing_runs(mock_fetcher_cls, test_db):
    """Should skip days that already have forecast data."""
    mock_instance = MagicMock()
    mock_instance.fetch_run.return_value = 90
    mock_fetcher_cls.return_value = mock_instance

    # Pre-seed one day's worth of data
    con = get_connection(test_db)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    model_run = datetime(yesterday.year, yesterday.month, yesterday.day, 12)
    con.execute(
        "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at) "
        "VALUES ('KNYC', ?, ?, 45.0, 7.2, CURRENT_TIMESTAMP)",
        [model_run, model_run],
    )
    con.close()

    backfill_forecasts(days_back=3, db_path=test_db, delay_seconds=0)

    # Should have called fetch_run for 2 days, not 3
    assert mock_instance.fetch_run.call_count == 2


@patch("services.forecast_backfiller.HRRRFetcher")
def test_backfiller_handles_archive_gaps(mock_fetcher_cls, test_db):
    """Should continue past failed runs without crashing."""
    mock_instance = MagicMock()
    mock_instance.fetch_run.side_effect = [Exception("archive gap"), 90, 90]
    mock_fetcher_cls.return_value = mock_instance

    total = backfill_forecasts(days_back=3, db_path=test_db, delay_seconds=0)

    assert mock_instance.fetch_run.call_count == 3
    assert total == 180  # Only 2 successful days
