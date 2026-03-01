"""Tests for Phase 2 walk-forward regression models.

Uses synthetic data with known temperature-dependent bias:
  - Hot days (actual > 55F): forecast = actual + 3.0
  - Cold days (actual <= 55F): forecast = actual + 1.0

This gives a clear signal for fcst_high vs error that the regression should detect.
"""

import os
import math
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from core.db import init_db, get_connection
from services.backtester import (
    _walk_forward_regression_data,
    _fit_and_predict,
    _get_delta_temp,
    _encode_month,
    _make_regression_model,
    wf_regression_full,
    wf_regression_fcst,
    WALK_FORWARD_MIN_DAYS,
)
from services.data_provider import BacktestDataProvider, HRRR_AVAILABILITY_LAG_HOURS

TEST_DB = "data/test_phase2.duckdb"


def _cleanup_db():
    """Remove test DB and its WAL file."""
    for path in [TEST_DB, TEST_DB + ".wal"]:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def test_db():
    _cleanup_db()
    init_db(TEST_DB)
    yield TEST_DB
    _cleanup_db()


def _seed_synthetic_data(db_path, n_days=200, run_hour=12):
    """Seed synthetic data with known temperature-dependent bias.

    Hot days (actual > 55): forecast = actual + 3.0  (bias = +3.0)
    Cold days (actual <= 55): forecast = actual + 1.0  (bias = +1.0)

    Creates n_days of consecutive data starting from 2023-01-01.
    Temperatures oscillate seasonally (30-80F range).
    """
    con = get_connection(db_path)

    start = date(2023, 1, 1)
    for i in range(n_days):
        obs_date = start + timedelta(days=i)
        # Sinusoidal actual temp: cold in winter, hot in summer
        day_of_year = obs_date.timetuple().tm_yday
        actual = 55.0 + 25.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
        actual = round(actual, 1)

        # Apply known bias
        if actual > 55.0:
            fcst_high = actual + 3.0
        else:
            fcst_high = actual + 1.0

        # Insert NWS daily
        con.execute(
            "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
            "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
            [obs_date, actual],
        )

        # Insert forecast (single row with temp_f = fcst_high for that model_run)
        model_run = datetime(obs_date.year, obs_date.month, obs_date.day, run_hour, 0)
        fxx = 6  # hour offset from model_run to valid_at
        valid_at = model_run + timedelta(hours=fxx)
        con.execute(
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, fxx, ingested_at) "
            "VALUES ('KNYC', ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [model_run, valid_at, fcst_high, round((fcst_high - 32) * 5 / 9, 2), fxx],
        )

    con.close()


@pytest.fixture
def seeded_db(test_db):
    _seed_synthetic_data(test_db, n_days=200, run_hour=12)
    return test_db


# ---------------------------------------------------------------------------
# Test: regression query is walk-forward safe
# ---------------------------------------------------------------------------

class TestRegressionQueryWalkForward:
    """Verify _walk_forward_regression_data only uses dates < current_date."""

    def test_excludes_current_date(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)
        check_date = date(2023, 7, 1)  # ~180 days in

        rows = _walk_forward_regression_data(con, 12, "KNYC", check_date)
        assert rows is not None

        # All training data should be from BEFORE check_date
        # We can verify by count: days from Jan 1 to Jun 30 = 181 days
        # minus 2 for delta_temp nulls = 179 max
        assert len(rows) <= 181
        assert len(rows) >= WALK_FORWARD_MIN_DAYS
        con.close()

    def test_later_date_has_more_data(self, seeded_db):
        """Expanding window: later dates see more training data."""
        con = duckdb.connect(seeded_db, read_only=True)
        early = _walk_forward_regression_data(con, 12, "KNYC", date(2023, 5, 1))
        late = _walk_forward_regression_data(con, 12, "KNYC", date(2023, 7, 1))
        assert early is not None
        assert late is not None
        assert len(late) > len(early)
        con.close()


# ---------------------------------------------------------------------------
# Test: regression detects temperature-dependent bias
# ---------------------------------------------------------------------------

class TestRegressionDetectsBias:
    """Regression should predict higher bias for hot days than cold days."""

    def test_fcst_model_predicts_higher_bias_for_hot(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)

        # Train on all data before July
        training = _walk_forward_regression_data(con, 12, "KNYC", date(2023, 7, 1))
        assert training is not None

        # Predict for a hot day (fcst=80, July, delta=+2)
        hot_features = (80.0, 7.0, 2.0)
        hot_result = _fit_and_predict(training, hot_features, [0])  # fcst_high only
        assert hot_result is not None

        # Predict for a cold day (fcst=35, January, delta=-2)
        cold_features = (35.0, 1.0, -2.0)
        cold_result = _fit_and_predict(training, cold_features, [0])
        assert cold_result is not None

        hot_bias, _ = hot_result
        cold_bias, _ = cold_result

        # Hot day predicted bias should be higher than cold day
        assert hot_bias > cold_bias, (
            f"Expected hot bias ({hot_bias:.3f}) > cold bias ({cold_bias:.3f})"
        )
        con.close()

    def test_full_model_also_detects_bias(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)
        training = _walk_forward_regression_data(con, 12, "KNYC", date(2023, 7, 1))
        assert training is not None

        hot_result = _fit_and_predict(training, (80.0, 7.0, 2.0), [0, 1, 2, 3])
        cold_result = _fit_and_predict(training, (35.0, 1.0, -2.0), [0, 1, 2, 3])
        assert hot_result is not None
        assert cold_result is not None
        assert hot_result[0] > cold_result[0]
        con.close()


# ---------------------------------------------------------------------------
# Test: regression model returns valid probabilities
# ---------------------------------------------------------------------------

class TestRegressionValidProbs:
    """Model output should sum to ~1.0 and contain no negatives."""

    def test_probs_sum_to_one(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)
        model_run = datetime(2023, 7, 1, 12, 0, tzinfo=timezone.utc)
        ref_time = model_run + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

        provider = BacktestDataProvider(
            db_path=seeded_db,
            station_id="KNYC",
            model_run=model_run,
            ref_time=ref_time,
            connection=con,
        )

        probs = wf_regression_fcst(provider, ref_time)
        assert probs is not None
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1.0"
        con.close()

    def test_no_negative_probs(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)
        model_run = datetime(2023, 7, 1, 12, 0, tzinfo=timezone.utc)
        ref_time = model_run + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

        provider = BacktestDataProvider(
            db_path=seeded_db,
            station_id="KNYC",
            model_run=model_run,
            ref_time=ref_time,
            connection=con,
        )

        probs = wf_regression_fcst(provider, ref_time)
        assert probs is not None
        for k, p in probs.items():
            assert p >= 0, f"Negative prob {p} at bracket {k}"
        con.close()


# ---------------------------------------------------------------------------
# Test: minimum training window enforced
# ---------------------------------------------------------------------------

class TestRegressionMinWindow:
    """Regression should return None with < 90 days of training data."""

    def test_returns_none_with_few_days(self, test_db):
        """Seed only 30 days — regression should refuse."""
        _seed_synthetic_data(test_db, n_days=30, run_hour=12)

        con = duckdb.connect(test_db, read_only=True)
        # Try to get regression data for the last day
        check_date = date(2023, 1, 31)
        rows = _walk_forward_regression_data(con, 12, "KNYC", check_date)
        assert rows is None, f"Expected None, got {len(rows)} rows"
        con.close()

    def test_returns_none_from_model_fn(self, test_db):
        """Full model function returns None when not enough data."""
        _seed_synthetic_data(test_db, n_days=30, run_hour=12)

        con = duckdb.connect(test_db, read_only=True)
        model_run = datetime(2023, 1, 25, 12, 0, tzinfo=timezone.utc)
        ref_time = model_run + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

        provider = BacktestDataProvider(
            db_path=test_db,
            station_id="KNYC",
            model_run=model_run,
            ref_time=ref_time,
            connection=con,
        )

        result = wf_regression_full(provider, ref_time)
        assert result is None
        con.close()


# ---------------------------------------------------------------------------
# Test: delta_temp is walk-forward safe
# ---------------------------------------------------------------------------

class TestDeltaTempWalkForward:
    """_get_delta_temp should only use data before current_date."""

    def test_uses_only_prior_dates(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)
        # For date 2023-03-01, delta should be actual(Feb 28) - actual(Feb 27)
        check_date = date(2023, 3, 1)
        delta = _get_delta_temp(con, "KNYC", check_date)
        assert delta is not None

        # Verify against known values
        rows = con.execute("""
            SELECT obs_date, max_temp_f
            FROM nws_daily
            WHERE station_id = 'KNYC'
                AND obs_date IN (?, ?)
            ORDER BY obs_date DESC
        """, [date(2023, 2, 28), date(2023, 2, 27)]).fetchall()

        expected = rows[0][1] - rows[1][1]  # Feb 28 - Feb 27
        assert abs(delta - expected) < 0.01
        con.close()

    def test_returns_none_at_start(self, test_db):
        """With only 1 day of data, delta_temp should be None."""
        _seed_synthetic_data(test_db, n_days=1, run_hour=12)
        con = duckdb.connect(test_db, read_only=True)
        delta = _get_delta_temp(con, "KNYC", date(2023, 1, 2))
        # Only 1 day seeded, need 2 prior days
        assert delta is None
        con.close()

    def test_does_not_use_current_date(self, seeded_db):
        """Delta for date D should not include D's actual."""
        con = duckdb.connect(seeded_db, read_only=True)
        check_date = date(2023, 6, 15)

        delta = _get_delta_temp(con, "KNYC", check_date)
        assert delta is not None

        # Should be actual(Jun 14) - actual(Jun 13), NOT involving Jun 15
        d_minus_1 = con.execute(
            "SELECT max_temp_f FROM nws_daily WHERE station_id = 'KNYC' AND obs_date = ?",
            [date(2023, 6, 14)],
        ).fetchone()[0]
        d_minus_2 = con.execute(
            "SELECT max_temp_f FROM nws_daily WHERE station_id = 'KNYC' AND obs_date = ?",
            [date(2023, 6, 13)],
        ).fetchone()[0]

        expected = d_minus_1 - d_minus_2
        assert abs(delta - expected) < 0.01
        con.close()


# ---------------------------------------------------------------------------
# Test: month encoding
# ---------------------------------------------------------------------------

class TestMonthEncoding:
    """sin/cos encoding should be periodic and distinct."""

    def test_january_and_july_opposite(self):
        sin_jan, cos_jan = _encode_month(1.0)
        sin_jul, cos_jul = _encode_month(7.0)
        # January and July are 6 months apart — sin values should have opposite sign
        assert sin_jan * sin_jul < 0 or abs(sin_jan) < 0.01 or abs(sin_jul) < 0.01

    def test_december_near_january(self):
        sin_dec, cos_dec = _encode_month(12.0)
        sin_jan, cos_jan = _encode_month(1.0)
        # Dec and Jan are adjacent — their encodings should be close
        dist = math.sqrt((sin_dec - sin_jan) ** 2 + (cos_dec - cos_jan) ** 2)
        # Distance between adjacent months on the unit circle
        expected = 2 * math.sin(math.pi / 12)
        assert abs(dist - expected) < 0.01
