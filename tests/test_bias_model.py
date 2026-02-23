"""Tests for historical bias model."""

import os
from datetime import datetime, timedelta

import pytest

from core.db import init_db, get_connection
from services.bias_model import BiasModel

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _seed_day(con, station_id: str, date_str: str, fcst_temps: list, obs_high: float):
    """Seed one day of forecast + observation data.

    fcst_temps: list of hourly forecast temps (fxx 1-18).
    obs_high: the max observed temperature for that window.
    """
    model_run = datetime.fromisoformat(f"{date_str} 12:00:00")

    for i, temp_f in enumerate(fcst_temps):
        fxx = i + 1
        valid_hour = 12 + fxx
        valid_at = datetime.fromisoformat(f"{date_str} {valid_hour:02d}:00:00")
        temp_c = round((temp_f - 32) * 5 / 9, 2)
        con.execute(
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [station_id, model_run, valid_at, temp_f, temp_c],
        )

    # Seed enough observations to pass MIN_OBS_COUNT completeness check.
    # 120 obs at 1-min intervals starting at 13:00, all within the forecast window.
    window_start = datetime.fromisoformat(f"{date_str} 13:00:00")
    for i in range(120):
        obs_time = window_start + timedelta(minutes=i)
        # Vary temps so the max is obs_high (peak around obs #60)
        temp = obs_high - abs(i - 60) * 0.1 if obs_high is not None else None
        con.execute(
            "INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at) "
            "VALUES (?, ?, ?, NULL, '', CURRENT_TIMESTAMP)",
            [station_id, obs_time, temp],
        )


@pytest.fixture
def seeded_db(test_db):
    """Seed 3 days of paired data for KNYC with known bias."""
    con = get_connection(test_db)

    # Day 1: forecast high 52, observed high 51 -> error = +1.0
    _seed_day(con, "KNYC", "2026-01-01", [45, 47, 49, 51, 52, 51, 49, 47], 51.0)
    # Day 2: forecast high 55, observed high 53 -> error = +2.0
    _seed_day(con, "KNYC", "2026-01-02", [48, 50, 52, 54, 55, 54, 52, 50], 53.0)
    # Day 3: forecast high 50, observed high 50 -> error = 0.0
    _seed_day(con, "KNYC", "2026-01-03", [43, 45, 47, 49, 50, 49, 47, 45], 50.0)

    con.close()
    return test_db


def test_compute_station_bias_mean(seeded_db):
    """Mean bias should be average of peak errors: (1 + 2 + 0) / 3 = 1.0."""
    model = BiasModel(db_path=seeded_db)
    bias = model.compute_station_bias("KNYC")

    assert bias is not None
    assert bias.station_id == "KNYC"
    assert bias.mean_bias == pytest.approx(1.0, abs=0.01)
    assert bias.sample_days == 3


def test_compute_station_bias_std(seeded_db):
    """Std error for [1, 2, 0] with mean=1.0 using Bessel's correction: sqrt(((0+1+1)/2)) = 1.0."""
    model = BiasModel(db_path=seeded_db)
    bias = model.compute_station_bias("KNYC")

    assert bias is not None
    assert bias.std_error == pytest.approx(1.0, abs=0.01)


def test_compute_station_bias_no_data(test_db):
    """Returns None for station with no forecast data."""
    model = BiasModel(db_path=test_db)
    bias = model.compute_station_bias("KNYC")
    assert bias is None


def test_compute_station_bias_no_observations(test_db):
    """Returns None when forecasts exist but no matching observations."""
    con = get_connection(test_db)
    # Seed forecast only, no observations
    _seed_day(con, "KNYC", "2026-01-01", [45, 47, 49, 51, 52, 51, 49, 47], obs_high=None)
    # Undo the observation that _seed_day inserted (it was None, so remove it)
    con.execute("DELETE FROM observations")
    con.close()

    model = BiasModel(db_path=test_db)
    bias = model.compute_station_bias("KNYC")
    assert bias is None


def test_compute_and_store(seeded_db):
    """Bias stats should be stored in station_bias table."""
    model = BiasModel(db_path=seeded_db)
    stored = model.compute_and_store()

    assert stored >= 1  # At least KNYC

    con = get_connection(seeded_db)
    rows = con.execute(
        "SELECT station_id, mean_bias, std_error, sample_days FROM station_bias WHERE station_id = 'KNYC'"
    ).fetchall()
    con.close()

    assert len(rows) == 1
    assert rows[0][1] == pytest.approx(1.0, abs=0.01)  # mean_bias
    assert rows[0][3] == 3  # sample_days


def test_positive_bias_means_forecast_runs_hot(seeded_db):
    """Positive mean_bias means forecast consistently overshoots observed high."""
    model = BiasModel(db_path=seeded_db)
    bias = model.compute_station_bias("KNYC")

    assert bias is not None
    assert bias.mean_bias > 0, "Forecast ran hotter than observations — bias should be positive"
