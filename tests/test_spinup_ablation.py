"""Tests for spin-up exclusion ablation."""

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db


def test_spinup_exclusion_changes_fcst_high(tmp_path):
    """Excluding spin-up fxx should change the forecast high."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)

    model_run = datetime(2024, 6, 15, 12)
    # Insert spin-up hours with higher temps (simulating artifacts)
    for fxx in range(1, 19):
        temp = 80.0 if fxx <= 3 else 75.0  # Spin-up artificially warm
        con.execute(
            "INSERT INTO forecasts "
            "(station_id, model_run, valid_at, temp_f, temp_c, "
            "ingested_at, model_name, fxx, is_spinup) "
            "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
            ['KNYC', model_run, model_run + timedelta(hours=fxx),
             temp, round((temp - 32) * 5 / 9, 2),
             datetime.now(), fxx, fxx <= 3],
        )
    con.close()

    con = duckdb.connect(db_path, read_only=True)
    # With spin-up
    high_all = con.execute(
        "SELECT MAX(temp_f) FROM forecasts "
        "WHERE station_id = 'KNYC' AND model_run = ?",
        [model_run],
    ).fetchone()[0]

    # Without spin-up
    high_no_spinup = con.execute(
        "SELECT MAX(temp_f) FROM forecasts "
        "WHERE station_id = 'KNYC' AND model_run = ? AND is_spinup = FALSE",
        [model_run],
    ).fetchone()[0]

    con.close()

    assert high_all == 80.0  # Includes spin-up
    assert high_no_spinup == 75.0  # Excludes spin-up
    assert high_all != high_no_spinup


def test_no_spinup_gold_view_exists(tmp_path):
    """gold_hrrr_bias_features_no_spinup view should exist and filter spinup."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)

    model_run = datetime(2024, 6, 15, 12)
    for fxx in range(1, 19):
        temp = 85.0 if fxx <= 3 else 72.0
        con.execute(
            "INSERT INTO forecasts "
            "(station_id, model_run, valid_at, temp_f, temp_c, "
            "ingested_at, model_name, fxx, is_spinup) "
            "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
            ['KNYC', model_run, model_run + timedelta(hours=fxx),
             temp, round((temp - 32) * 5 / 9, 2),
             datetime.now(), fxx, fxx <= 3],
        )

    # Gold view with spinup should have fcst_high = 85
    row = con.execute(
        "SELECT fcst_high FROM gold_hrrr_bias_features "
        "WHERE forecast_date = '2024-06-15' AND run_hour = 12"
    ).fetchone()
    assert row[0] == 85.0

    # Gold view without spinup should have fcst_high = 72
    row = con.execute(
        "SELECT fcst_high FROM gold_hrrr_bias_features_no_spinup "
        "WHERE forecast_date = '2024-06-15' AND run_hour = 12"
    ).fetchone()
    assert row[0] == 72.0

    con.close()


def test_no_spinup_model_exists():
    """wf_regression_full_no_spinup should be importable."""
    from services.backtester import wf_regression_full_no_spinup
    assert callable(wf_regression_full_no_spinup)
    assert wf_regression_full_no_spinup.__name__ == "wf_regression_full_no_spinup"
