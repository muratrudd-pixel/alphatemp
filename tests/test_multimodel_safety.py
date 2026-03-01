"""Multi-model safety tests for backtester queries.

Seeds BOTH HRRR and GFS forecast data into the same DB with DIFFERENT biases,
then verifies that backtester queries filter by model_name and don't mix models.

HRRR bias = +2.0 F (forecast = actual + 2.0)
GFS bias  = -4.0 F (forecast = actual - 4.0)

If queries lack model_name filters, the mean bias will be contaminated.
"""

import math
import os
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from core.db import init_db
from services.backtester import (
    _walk_forward_bias_query,
    _walk_forward_regression_data,
    _ensure_level1,
    _p2b_level1,
    _p2b_level2,
    WALK_FORWARD_MIN_DAYS,
)

TEST_DB = "data/test_multimodel.duckdb"


def _cleanup_db():
    """Remove test DB and its WAL file."""
    for path in [TEST_DB, TEST_DB + ".wal"]:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(autouse=True)
def test_db():
    """Create a fresh test DB for every test, clean up after."""
    _cleanup_db()
    _p2b_level1.clear()
    _p2b_level2.clear()
    init_db(TEST_DB)
    yield TEST_DB
    _cleanup_db()


def _seed_dual_model_data(db_path, n_days=120):
    """Seed HRRR (run_hour=12, bias=+2.0) and GFS (run_hour=0, bias=-4.0).

    Both models forecast for the same station and date range.
    NWS daily actuals are shared between them.
    """
    con = duckdb.connect(db_path)
    try:
        start = date(2023, 1, 1)
        for i in range(n_days):
            obs_date = start + timedelta(days=i)
            day_of_year = obs_date.timetuple().tm_yday
            actual = 55.0 + 20.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
            actual = round(actual, 1)

            # NWS daily (shared truth)
            con.execute(
                "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
                "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
                [obs_date, actual],
            )

            # HRRR forecast: run_hour=12, bias = +2.0, fxx=6 (12+6=18, within 5..28)
            hrrr_fcst = actual + 2.0
            hrrr_run = datetime(obs_date.year, obs_date.month, obs_date.day, 12, 0)
            hrrr_fxx = 6
            hrrr_valid = hrrr_run + timedelta(hours=hrrr_fxx)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name, fxx) "
                "VALUES ('KNYC', ?, ?, ?, ?, CURRENT_TIMESTAMP, 'hrrr', ?)",
                [hrrr_run, hrrr_valid, hrrr_fcst, round((hrrr_fcst - 32) * 5 / 9, 2), hrrr_fxx],
            )

            # GFS forecast: run_hour=0, bias = -4.0, fxx=6 (0+6=6, within 5..28)
            gfs_fcst = actual - 4.0
            gfs_run = datetime(obs_date.year, obs_date.month, obs_date.day, 0, 0)
            gfs_fxx = 6
            gfs_valid = gfs_run + timedelta(hours=gfs_fxx)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name, fxx) "
                "VALUES ('KNYC', ?, ?, ?, ?, CURRENT_TIMESTAMP, 'gfs', ?)",
                [gfs_run, gfs_valid, gfs_fcst, round((gfs_fcst - 32) * 5 / 9, 2), gfs_fxx],
            )
    finally:
        con.close()


@pytest.fixture
def seeded_db(test_db):
    _seed_dual_model_data(test_db, n_days=120)
    return test_db


# ---------------------------------------------------------------------------
# Test: _walk_forward_bias_query filters by model_name
# ---------------------------------------------------------------------------

class TestBiasQueryModelSafety:
    """Verify _walk_forward_bias_query returns model-specific bias, not a mix."""

    def test_hrrr_bias_is_positive_two(self, seeded_db):
        """HRRR bias should be ~+2.0, not contaminated by GFS (-4.0)."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            # Query for HRRR (run_hour=12)
            stats = _walk_forward_bias_query(con, 12, "KNYC", date(2023, 5, 1))
            assert stats is not None, "Expected bias stats, got None"
            mean_bias = stats[0]
            # HRRR bias should be ~+2.0
            assert abs(mean_bias - 2.0) < 0.5, (
                "HRRR mean bias = {:.3f}, expected ~2.0 "
                "(contaminated by GFS if far off)".format(mean_bias)
            )
        finally:
            con.close()

    def test_gfs_bias_is_negative_four(self, seeded_db):
        """GFS bias should be ~-4.0 when queried with model_name='gfs'."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            # Query for GFS (run_hour=0, model_name='gfs')
            stats = _walk_forward_bias_query(con, 0, "KNYC", date(2023, 5, 1),
                                             model_name='gfs')
            assert stats is not None, "Expected GFS bias stats, got None"
            mean_bias = stats[0]
            assert abs(mean_bias - (-4.0)) < 0.5, (
                "GFS mean bias = {:.3f}, expected ~-4.0".format(mean_bias)
            )
        finally:
            con.close()

    def test_hrrr_not_contaminated_by_gfs(self, seeded_db):
        """If model_name filter is missing, bias would be somewhere between +2 and -4.
        This test passes ONLY if the filter is working."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            stats = _walk_forward_bias_query(con, 12, "KNYC", date(2023, 5, 1))
            assert stats is not None
            mean_bias = stats[0]
            # If contaminated, bias would drift toward -1.0 (average of +2 and -4)
            # We assert it stays close to +2.0
            assert mean_bias > 1.0, (
                "HRRR bias = {:.3f}, suspiciously low — GFS contamination?".format(mean_bias)
            )
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Test: _walk_forward_regression_data filters by model_name
# ---------------------------------------------------------------------------

class TestRegressionDataModelSafety:
    """Verify regression training data is model-specific."""

    def test_hrrr_regression_data_has_positive_errors(self, seeded_db):
        """HRRR errors should all be ~+2.0 (forecast - actual)."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            rows = _walk_forward_regression_data(con, 12, "KNYC", date(2023, 5, 1))
            assert rows is not None, "Expected regression data, got None"
            errors = [r[0] for r in rows]
            mean_error = sum(errors) / len(errors)
            assert abs(mean_error - 2.0) < 0.5, (
                "HRRR mean error = {:.3f}, expected ~2.0".format(mean_error)
            )
        finally:
            con.close()

    def test_gfs_regression_data_has_negative_errors(self, seeded_db):
        """GFS errors should all be ~-4.0."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            rows = _walk_forward_regression_data(con, 0, "KNYC", date(2023, 5, 1),
                                                 model_name='gfs')
            assert rows is not None, "Expected GFS regression data, got None"
            errors = [r[0] for r in rows]
            mean_error = sum(errors) / len(errors)
            assert abs(mean_error - (-4.0)) < 0.5, (
                "GFS mean error = {:.3f}, expected ~-4.0".format(mean_error)
            )
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Test: _ensure_level1 filters by model_name
# ---------------------------------------------------------------------------

class TestEnsureLevel1ModelSafety:
    """Verify level-1 cache is model-specific."""

    def test_hrrr_curves_not_mixed_with_gfs(self, seeded_db):
        """HRRR level-1 should only contain 12z curves, not 00z GFS."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            l1 = _ensure_level1(con, 12, "KNYC")
            errors = l1["errors_list"]
            assert len(errors) > 0, "Expected errors_list data"

            # Check that all errors are consistent with HRRR bias (+2.0)
            actual_errors = [e for _, e, _, _, _ in errors if e is not None]
            mean_error = sum(actual_errors) / len(actual_errors)
            assert abs(mean_error - 2.0) < 0.5, (
                "Level-1 mean error = {:.3f}, expected ~2.0 for HRRR".format(mean_error)
            )
        finally:
            con.close()

    def test_cache_key_includes_model_name(self, seeded_db):
        """Different model_name values should produce separate cache entries."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            l1_hrrr = _ensure_level1(con, 12, "KNYC", model_name='hrrr')
            l1_gfs = _ensure_level1(con, 0, "KNYC", model_name='gfs')

            # They should be separate cache entries
            hrrr_errors = [e for _, e, _, _, _ in l1_hrrr["errors_list"]]
            gfs_errors = [e for _, e, _, _, _ in l1_gfs["errors_list"]]

            hrrr_mean = sum(hrrr_errors) / len(hrrr_errors)
            gfs_mean = sum(gfs_errors) / len(gfs_errors)

            # HRRR ~+2, GFS ~-4 — they must be clearly different
            assert abs(hrrr_mean - gfs_mean) > 3.0, (
                "HRRR mean={:.2f}, GFS mean={:.2f} — not distinct enough".format(
                    hrrr_mean, gfs_mean)
            )
        finally:
            con.close()
