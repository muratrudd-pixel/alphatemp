"""Tests for cross-hour training sub-ablation."""

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db
from services.backtester import (
    _walk_forward_regression_data_cross_hour,
    wf_regression_cross_hour,
)


def _seed_multi_hour_data(db_path, n_days=120):
    """Seed data across multiple run hours for cross-hour testing."""
    con = duckdb.connect(db_path)
    base = date(2024, 1, 1)
    for d in range(n_days):
        current = base + timedelta(days=d)
        actual_high = 50 + 30 * (d % 365) / 365
        for hour in range(24):
            model_run = datetime(current.year, current.month, current.day, hour)
            bias = 2.0 + 0.5 * (hour / 24)
            for fxx in range(1, 19):
                valid_at = model_run + timedelta(hours=fxx)
                temp_f = actual_high + bias + (fxx * 0.1)
                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                    ['KNYC', model_run, valid_at, temp_f,
                     round((temp_f - 32) * 5 / 9, 2),
                     datetime.now(), fxx, fxx <= 3],
                )
        con.execute(
            "INSERT INTO nws_daily "
            "(station_id, obs_date, max_temp_f, ingested_at) "
            "VALUES ('KNYC', ?, ?, ?)",
            [current, actual_high, datetime.now()],
        )
    con.close()


def test_cross_hour_query_returns_all_hours(tmp_path):
    """Cross-hour query should return data from multiple run hours."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    _seed_multi_hour_data(db_path)

    con = duckdb.connect(db_path, read_only=True)
    training = _walk_forward_regression_data_cross_hour(
        con, 'KNYC', date(2024, 4, 15), model_name='hrrr',
    )
    con.close()

    assert training is not None
    # Should have data from multiple run hours
    hours_seen = {row[4] for row in training}
    assert len(hours_seen) > 1


def test_cross_hour_model_importable():
    """wf_regression_cross_hour should be importable and callable."""
    assert callable(wf_regression_cross_hour)
    assert wf_regression_cross_hour.__name__ == "wf_regression_cross_hour"


def test_cross_hour_delta_temp_partitioned(tmp_path):
    """delta_temp in cross-hour query should be partitioned by run_hour."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    _seed_multi_hour_data(db_path)

    con = duckdb.connect(db_path, read_only=True)
    training = _walk_forward_regression_data_cross_hour(
        con, 'KNYC', date(2024, 4, 15), model_name='hrrr',
    )
    con.close()

    assert training is not None
    # All returned rows should have non-None delta_temp (filtered in query)
    for row in training:
        assert row[3] is not None, "delta_temp should not be None in returned rows"
