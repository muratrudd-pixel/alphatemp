"""Tests for services/feature_builder.py — FeatureBuilder class.

Validates that feature construction matches autoresearch/experiment.py:get_training_data()
exactly: same SQL logic, same feature ordering, same defaults for missing data.

Uses an in-memory DuckDB with synthetic data inserted directly.
"""

import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict

import duckdb
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_tables(con):
    # type: (duckdb.DuckDBPyConnection) -> None
    """Create the minimal schema needed by FeatureBuilder."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            station_id  VARCHAR NOT NULL,
            model_run   TIMESTAMP NOT NULL,
            valid_at    TIMESTAMP NOT NULL,
            temp_f      DOUBLE,
            temp_c      DOUBLE,
            ingested_at TIMESTAMP NOT NULL,
            model_name  VARCHAR NOT NULL DEFAULT 'hrrr',
            fxx         INTEGER,
            is_spinup   BOOLEAN DEFAULT FALSE,
            UNIQUE (station_id, model_run, valid_at, model_name)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_extended (
            station_id      VARCHAR NOT NULL,
            model_run       TIMESTAMP NOT NULL,
            valid_at        TIMESTAMP NOT NULL,
            model_name      VARCHAR NOT NULL,
            dewpoint_2m_f   DOUBLE,
            humidity_2m     DOUBLE,
            wind_speed_10m  DOUBLE,
            wind_dir_10m    DOUBLE,
            wind_gusts_10m  DOUBLE,
            pressure_msl    DOUBLE,
            cloud_cover     DOUBLE,
            precipitation   DOUBLE,
            shortwave_rad   DOUBLE,
            cape            DOUBLE,
            ingested_at     TIMESTAMP NOT NULL,
            UNIQUE (station_id, model_run, valid_at, model_name)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS nws_daily (
            station_id   VARCHAR NOT NULL,
            obs_date     DATE NOT NULL,
            max_temp_f   DOUBLE,
            min_temp_f   DOUBLE,
            source       VARCHAR DEFAULT 'ACIS',
            ingested_at  TIMESTAMP NOT NULL,
            UNIQUE (station_id, obs_date)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            station_id   VARCHAR NOT NULL,
            observed_at  TIMESTAMP NOT NULL,
            temp_f       DOUBLE,
            UNIQUE (station_id, observed_at)
        )
    """)


def _insert_hrrr_forecasts(con, station_id, target_date, run_hour, temps_by_fxx):
    # type: (duckdb.DuckDBPyConnection, str, date, int, Dict[int, float]) -> None
    """Insert HRRR forecasts for a given date and run hour.

    temps_by_fxx: {fxx: temp_f} — each fxx is hours after model_run.
    """
    model_run = datetime(target_date.year, target_date.month, target_date.day,
                         run_hour, 0, 0)
    now = datetime.utcnow()
    for fxx, temp in temps_by_fxx.items():
        valid_at = model_run + timedelta(hours=fxx)
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c,
                                   ingested_at, model_name, fxx, is_spinup)
            VALUES (?, ?, ?, ?, NULL, ?, 'hrrr', ?, FALSE)
        """, [station_id, model_run, valid_at, temp, now, fxx])


def _insert_model_forecasts(con, station_id, target_date, run_hour,
                             model_name, temps_by_fxx):
    # type: (duckdb.DuckDBPyConnection, str, date, int, str, Dict[int, float]) -> None
    """Insert GFS or ECMWF forecasts for a given date and run hour."""
    model_run = datetime(target_date.year, target_date.month, target_date.day,
                         run_hour, 0, 0)
    now = datetime.utcnow()
    for fxx, temp in temps_by_fxx.items():
        valid_at = model_run + timedelta(hours=fxx)
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c,
                                   ingested_at, model_name, fxx, is_spinup)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?, FALSE)
        """, [station_id, model_run, valid_at, temp, now, model_name, fxx])


def _insert_extended(con, station_id, target_date, run_hour, model_name,
                     valid_hours, dewpoint=None, humidity=None, gusts=None,
                     precip=None, solar=None, cape=None):
    # type: (...) -> None
    """Insert forecast_extended rows for valid_hours (UTC hours of valid_at)."""
    model_run = datetime(target_date.year, target_date.month, target_date.day,
                         run_hour, 0, 0)
    now = datetime.utcnow()
    for vh in valid_hours:
        valid_at = datetime(target_date.year, target_date.month, target_date.day,
                            vh, 0, 0)
        # Handle next-day valid hours
        if vh < run_hour:
            valid_at += timedelta(days=1)
        con.execute("""
            INSERT INTO forecast_extended
                (station_id, model_run, valid_at, model_name,
                 dewpoint_2m_f, humidity_2m, wind_gusts_10m, precipitation,
                 shortwave_rad, cape, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [station_id, model_run, valid_at, model_name,
              dewpoint, humidity, gusts, precip, solar, cape, now])


def _insert_nws_daily(con, station_id, obs_date, max_temp):
    # type: (duckdb.DuckDBPyConnection, str, date, float) -> None
    """Insert a single nws_daily row."""
    now = datetime.utcnow()
    con.execute("""
        INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f,
                               source, ingested_at)
        VALUES (?, ?, ?, NULL, 'test', ?)
    """, [station_id, obs_date, max_temp, now])


def _insert_observation(con, station_id, observed_at, temp_f):
    # type: (duckdb.DuckDBPyConnection, str, datetime, float) -> None
    """Insert a single observation row."""
    con.execute("""
        INSERT INTO observations (station_id, observed_at, temp_f)
        VALUES (?, ?, ?)
    """, [station_id, observed_at, temp_f])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db(tmp_path):
    """Create a test DuckDB with all required tables, populated with 90 days
    of synthetic data — enough to pass MIN_SAMPLES=60.

    Data characteristics:
    - 90 days of HRRR forecasts (run_hour=0, fxx 5-28)
    - 90 days of NWS daily settlements
    - 90 days of ECMWF 00z forecasts
    - 90 days of GFS 00z forecasts
    - 90 days of observations (hourly 0-23 UTC)
    - 90 days of forecast_extended for ECMWF and GFS
    """
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    _create_tables(con)

    station = "KNYC"
    base_date = date(2026, 1, 15)

    for day_offset in range(90):
        d = base_date + timedelta(days=day_offset)
        # Realistic seasonal temp: ~40F in Jan, ~55F in Mar
        base_temp = 40.0 + (day_offset / 90.0) * 15.0
        actual_high = base_temp + 2.0  # actual always slightly higher

        # HRRR 00z run: fxx 5-28 cover settlement window
        hrrr_temps = {}
        for fxx in range(5, 29):
            # Parabolic temp curve peaking around fxx=17 (noon-ish ET)
            t = base_temp + 5.0 * math.sin(math.pi * (fxx - 5) / 23.0)
            hrrr_temps[fxx] = t
        _insert_hrrr_forecasts(con, station, d, 0, hrrr_temps)

        # ECMWF 00z: similar curve, slightly offset
        ecmwf_temps = {}
        for fxx in range(5, 29):
            t = base_temp + 5.0 * math.sin(math.pi * (fxx - 5) / 23.0) + 1.0
            ecmwf_temps[fxx] = t
        _insert_model_forecasts(con, station, d, 0, 'ecmwf', ecmwf_temps)

        # GFS 00z: similar curve, slightly different offset
        gfs_temps = {}
        for fxx in range(5, 29):
            t = base_temp + 5.0 * math.sin(math.pi * (fxx - 5) / 23.0) - 0.5
            gfs_temps[fxx] = t
        _insert_model_forecasts(con, station, d, 0, 'gfs', gfs_temps)

        # NWS daily settlement
        _insert_nws_daily(con, station, d, actual_high)

        # Observations: hourly 0-23 UTC, temp ramps up then down
        for h in range(24):
            obs_time = datetime(d.year, d.month, d.day, h, 51, 0)
            obs_temp = base_temp + 4.0 * math.sin(math.pi * (h - 5) / 18.0)
            _insert_observation(con, station, obs_time, obs_temp)

        # forecast_extended: ECMWF 00z, valid hours 10-22 UTC
        _insert_extended(con, station, d, 0, 'ecmwf',
                         list(range(10, 23)),
                         dewpoint=base_temp - 10.0,
                         humidity=55.0,
                         gusts=15.0,
                         precip=0.02,
                         solar=250.0,
                         cape=100.0)

        # forecast_extended: GFS 00z, valid hours 10-22 UTC
        _insert_extended(con, station, d, 0, 'gfs',
                         list(range(10, 23)),
                         dewpoint=base_temp - 12.0,
                         humidity=50.0,
                         gusts=12.0,
                         precip=0.01,
                         solar=230.0,
                         cape=80.0)

    con.close()
    return db_path


@pytest.fixture
def builder(test_db):
    """FeatureBuilder pointing at the test database."""
    from services.feature_builder import FeatureBuilder
    return FeatureBuilder(test_db)


# ---------------------------------------------------------------------------
# Test: FEATURE_NAMES has exactly 23 entries in correct order
# ---------------------------------------------------------------------------

def test_feature_names_count():
    """FEATURE_NAMES must have exactly 23 entries."""
    from services.feature_builder import FeatureBuilder
    assert len(FeatureBuilder.FEATURE_NAMES) == 23


def test_feature_names_order():
    """Feature names must be in the exact order matching experiment.py."""
    from services.feature_builder import FeatureBuilder
    expected = [
        'update_hour', 'fcst_high', 'sin_month', 'cos_month',
        'running_max_div', 'slope_div', 'cum_div',
        'ecmwf_spread', 'diurnal_range', 'gfs_spread',
        'solar_rad', 'dp_depression', 'total_precip',
        'humidity', 'max_gusts', 'lag_error', 'cape',
        'dp_spread', 'gfs_precip', 'rain_day',
        'precip_agree', 'solar_spread', 'abs(lag_error)',
    ]
    assert FeatureBuilder.FEATURE_NAMES == expected


# ---------------------------------------------------------------------------
# Test: get_training_data returns correct shapes
# ---------------------------------------------------------------------------

def test_get_training_data_shapes(builder):
    """get_training_data returns (X, y, dates) with correct shapes.

    With 90 days of data, 4 update hours each, we expect up to 360 rows.
    X should have 23 columns.
    """
    target_date = date(2026, 4, 15)
    result = builder.get_training_data(target_date, update_hour=0, window_days=180)

    assert result is not None, "Should return data with 90 days in window"
    X, y, dates = result
    assert X.ndim == 2, "X should be 2D"
    assert X.shape[1] == 23, f"X should have 23 features, got {X.shape[1]}"
    assert len(y) == X.shape[0], "y length should match X rows"
    assert len(dates) == X.shape[0], "dates length should match X rows"
    assert X.shape[0] >= 60, f"Should have >= 60 samples, got {X.shape[0]}"


def test_get_training_data_returns_none_insufficient(builder):
    """get_training_data returns None when window has < MIN_SAMPLES days."""
    # Use a date before most data exists
    target_date = date(2026, 1, 20)
    result = builder.get_training_data(target_date, update_hour=0, window_days=2)

    assert result is None, "Should return None with < 60 samples in tiny window"


# ---------------------------------------------------------------------------
# Test: build_features returns correct shape
# ---------------------------------------------------------------------------

def test_build_features_shape(builder, test_db):
    """build_features returns (features_array, fcst_high) with 23 features."""
    # Use a date we have data for
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=12)

    assert result is not None, "Should return features for a date with data"
    features, fcst_high = result
    assert features.shape == (23,), f"features should be shape (23,), got {features.shape}"
    assert isinstance(fcst_high, float), "fcst_high should be a float"
    assert fcst_high > 0, "fcst_high should be positive (temperature)"


def test_build_features_returns_none_no_data(builder):
    """build_features returns None when there's no HRRR data for the date."""
    target_date = date(2030, 1, 1)
    result = builder.build_features(target_date, update_hour=12)

    assert result is None, "Should return None for a date with no data"


# ---------------------------------------------------------------------------
# Test: feature values match expected calculations
# ---------------------------------------------------------------------------

def test_update_hour_is_first_feature(builder):
    """Feature index 0 should be the update_hour passed in."""
    target_date = date(2026, 2, 15)

    result_0 = builder.build_features(target_date, update_hour=0)
    result_12 = builder.build_features(target_date, update_hour=12)

    assert result_0 is not None and result_12 is not None
    assert result_0[0][0] == 0.0, "update_hour=0 should be feature[0]=0.0"
    assert result_12[0][0] == 12.0, "update_hour=12 should be feature[0]=12.0"


def test_fcst_high_is_second_feature(builder):
    """Feature index 1 should be the HRRR forecast high."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    features, fcst_high = result
    assert features[1] == fcst_high, "Feature[1] should equal fcst_high"


def test_seasonality_features(builder):
    """Features 2 and 3 should be sin(2pi*month/12) and cos(2pi*month/12)."""
    target_date = date(2026, 2, 15)  # month=2
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    features, _ = result
    expected_sin = math.sin(2.0 * math.pi * 2.0 / 12.0)
    expected_cos = math.cos(2.0 * math.pi * 2.0 / 12.0)
    assert abs(features[2] - expected_sin) < 1e-10, "sin_month mismatch"
    assert abs(features[3] - expected_cos) < 1e-10, "cos_month mismatch"


def test_ecmwf_spread_is_difference(builder):
    """Feature 7 (ecmwf_spread) = ECMWF high - HRRR high."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    features, fcst_high = result
    # Our test data has ECMWF = HRRR + 1.0, so spread should be ~1.0
    ecmwf_spread = features[7]
    assert abs(ecmwf_spread - 1.0) < 0.5, (
        f"ecmwf_spread should be ~1.0 (ECMWF offset), got {ecmwf_spread}"
    )


def test_gfs_spread_is_difference(builder):
    """Feature 9 (gfs_spread) = GFS high - HRRR high."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    features, fcst_high = result
    # Our test data has GFS = HRRR - 0.5, so spread should be ~-0.5
    gfs_spread = features[9]
    assert abs(gfs_spread - (-0.5)) < 0.5, (
        f"gfs_spread should be ~-0.5 (GFS offset), got {gfs_spread}"
    )


def test_rain_day_binary(builder, test_db):
    """Feature 19 (rain_day) should be binary: 0.0 or 1.0."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    rain_day = result[0][19]
    assert rain_day in (0.0, 1.0), f"rain_day should be 0 or 1, got {rain_day}"


def test_precip_agree_binary(builder, test_db):
    """Feature 20 (precip_agree) should be binary: 0.0 or 1.0."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    precip_agree = result[0][20]
    assert precip_agree in (0.0, 1.0), f"precip_agree should be 0 or 1, got {precip_agree}"


def test_abs_lag_error_is_last_feature(builder):
    """Feature 22 (abs(lag_error)) should be abs of feature 15 (lag_error)."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    features, _ = result
    lag_error = features[15]
    abs_lag = features[22]
    assert abs(abs_lag - abs(lag_error)) < 1e-10, (
        f"abs(lag_error)={abs_lag} should equal abs(lag_error={lag_error})"
    )


# ---------------------------------------------------------------------------
# Test: training data matches build_features for same date
# ---------------------------------------------------------------------------

def test_training_and_live_features_consistent(builder):
    """For a historical date, training data features at a specific update_hour
    should match build_features for the same date and update_hour.

    This is the CRITICAL invariant — if these diverge, the model trained on
    backtest data won't work correctly on live data.
    """
    # target_date must be AFTER enough data to have >= 60 daily rows.
    # Test data runs 90 days from 2026-01-15 to 2026-04-14.
    # Use target_date=2026-04-15 so the entire 90-day window is included.
    target_date = date(2026, 4, 15)
    check_date = date(2026, 3, 1)
    update_hour = 12

    # Get training data that includes check_date
    train_result = builder.get_training_data(target_date, update_hour,
                                             window_days=180)
    assert train_result is not None

    X, y, dates = train_result

    # Find the row for check_date at update_hour=12
    matching_indices = [i for i, d in enumerate(dates) if d == check_date]
    assert len(matching_indices) > 0, f"check_date {check_date} not found in training dates"

    # The training data includes all 4 update hours per date
    # Find the one with update_hour=12
    train_features = None
    for idx in matching_indices:
        if abs(X[idx, 0] - float(update_hour)) < 0.01:
            train_features = X[idx]
            break

    assert train_features is not None, (
        f"No training row for {check_date} at update_hour={update_hour}"
    )

    # Get live features for the same date
    live_result = builder.build_features(check_date, update_hour)
    assert live_result is not None
    live_features, _ = live_result

    # Compare all 23 features
    for i in range(23):
        assert abs(train_features[i] - live_features[i]) < 1e-6, (
            f"Feature {i} ({builder.FEATURE_NAMES[i]}) mismatch: "
            f"training={train_features[i]:.6f} vs live={live_features[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# Test: defaults for missing multi-model data
# ---------------------------------------------------------------------------

def test_missing_ecmwf_defaults(tmp_path):
    """When ECMWF data is missing, spreads default to 0.0, humidity to 50.0."""
    db_path = str(tmp_path / "sparse.duckdb")
    con = duckdb.connect(db_path)
    _create_tables(con)

    station = "KNYC"
    d = date(2026, 2, 15)

    # Only HRRR data — no ECMWF, no GFS
    hrrr_temps = {fxx: 50.0 + fxx * 0.1 for fxx in range(5, 29)}
    _insert_hrrr_forecasts(con, station, d, 0, hrrr_temps)
    _insert_nws_daily(con, station, d, 52.0)

    # Observations for divergence
    for h in range(24):
        _insert_observation(con, station,
                            datetime(d.year, d.month, d.day, h, 51, 0), 50.0)

    con.close()

    from services.feature_builder import FeatureBuilder
    fb = FeatureBuilder(db_path)
    result = fb.build_features(d, update_hour=12)

    assert result is not None
    features, _ = result

    # ecmwf_spread should be 0.0 (no ECMWF data)
    assert features[7] == 0.0, f"ecmwf_spread should be 0.0, got {features[7]}"
    # gfs_spread should be 0.0 (no GFS data)
    assert features[9] == 0.0, f"gfs_spread should be 0.0, got {features[9]}"
    # humidity should be 50.0 (default)
    assert features[13] == 50.0, f"humidity should be 50.0, got {features[13]}"
    # solar_rad should be 0.0 (default)
    assert features[10] == 0.0, f"solar_rad should be 0.0, got {features[10]}"


# ---------------------------------------------------------------------------
# Test: no NaN in output
# ---------------------------------------------------------------------------

def test_no_nans_in_features(builder):
    """build_features should never return NaN values."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=12)

    assert result is not None
    features, fcst_high = result
    assert not np.any(np.isnan(features)), "Features contain NaN values"
    assert not math.isnan(fcst_high), "fcst_high is NaN"


def test_no_nans_in_training_data(builder):
    """get_training_data should never return NaN values."""
    target_date = date(2026, 4, 15)
    result = builder.get_training_data(target_date, update_hour=0, window_days=180)

    assert result is not None
    X, y, dates = result
    assert not np.any(np.isnan(X)), "Training X contains NaN values"
    assert not np.any(np.isnan(y)), "Training y contains NaN values"


# ---------------------------------------------------------------------------
# Test: y values are error = fcst_high - actual_high
# ---------------------------------------------------------------------------

def test_y_is_forecast_error(builder):
    """y should be (fcst_high - actual_high), matching experiment.py."""
    target_date = date(2026, 4, 15)
    result = builder.get_training_data(target_date, update_hour=0, window_days=180)

    assert result is not None
    X, y, dates = result

    # All y values should be reasonable forecast errors (not huge)
    assert np.all(np.abs(y) < 30), f"Some y values are unreasonably large: max={np.max(np.abs(y))}"

    # y should not be all zeros (our test data has actual_high != fcst_high)
    assert np.any(y != 0.0), "All y values are zero — error computation is wrong"


# ---------------------------------------------------------------------------
# Test: diurnal_range is non-negative
# ---------------------------------------------------------------------------

def test_diurnal_range_nonnegative(builder):
    """Feature 8 (diurnal_range) should be >= 0 (max - min of HRRR forecast)."""
    target_date = date(2026, 2, 15)
    result = builder.build_features(target_date, update_hour=0)

    assert result is not None
    diurnal_range = result[0][8]
    assert diurnal_range >= 0.0, f"diurnal_range should be >= 0, got {diurnal_range}"
