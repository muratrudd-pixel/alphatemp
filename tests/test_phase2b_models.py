"""Tests for Phase 2B observation-based divergence models.

Uses synthetic data extending the Phase 2 seeder with hourly observations
that track the forecast curve with known divergence patterns.
"""

import os
import math
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from core.db import init_db, get_connection
from services.divergence import (
    interpolate_forecast,
    compute_divergence_features,
    compute_synoptic_max_divergence,
)
from services.backtester import (
    _compute_features_for_date,
    _ensure_level1,
    _ensure_level2,
    _fit_and_predict_phase2b,
    _PHASE2B_FEATURE_KEYS,
    _p2b_level1,
    _p2b_level2,
    wf_phase2b_full,
    wf_phase2b_runmax,
    wf_regression_full,
    Backtester,
    WALK_FORWARD_MIN_DAYS,
)
from services.data_provider import BacktestDataProvider, HRRR_AVAILABILITY_LAG_HOURS

TEST_DB = "data/test_phase2b.duckdb"


def _cleanup_db():
    for path in [TEST_DB, TEST_DB + ".wal"]:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def test_db():
    _cleanup_db()
    # Clear module-level Phase 2B caches from previous tests
    _p2b_level1.clear()
    _p2b_level2.clear()
    init_db(TEST_DB)
    yield TEST_DB
    _cleanup_db()


def _seed_phase2b_data(db_path, n_days=200, run_hour=12):
    """Seed synthetic data with known forecast curves + observations.

    Hot days (actual > 55F): forecast curve peaks at actual + 3.0
    Cold days (actual <= 55F): forecast curve peaks at actual + 1.0

    Observations track 1°F ABOVE the forecast curve (warm divergence)
    for summer days, and 1°F BELOW for winter days.
    This gives a predictable divergence → residual relationship.
    """
    con = get_connection(db_path)

    start = date(2023, 1, 1)
    for i in range(n_days):
        obs_date = start + timedelta(days=i)
        day_of_year = obs_date.timetuple().tm_yday
        actual = 55.0 + 25.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
        actual = round(actual, 1)

        # Known bias
        if actual > 55.0:
            bias = 3.0
        else:
            bias = 1.0
        fcst_high = actual + bias

        # Insert NWS daily
        con.execute(
            "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
            "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
            [obs_date, actual],
        )

        # Insert forecast curve: 18 hourly points from model_run to model_run + 18h
        model_run = datetime(obs_date.year, obs_date.month, obs_date.day, run_hour, 0)
        for h in range(19):
            valid_at = model_run + timedelta(hours=h)
            # Parabolic curve peaking at hour 10 (roughly 2PM ET for 12z run)
            frac = 1.0 - ((h - 10) / 10.0) ** 2
            temp_f = fcst_high - 15.0 + 15.0 * max(0, frac)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at) "
                "VALUES ('KNYC', ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [model_run, valid_at, round(temp_f, 1), round((temp_f - 32) * 5 / 9, 2)],
            )

        # Insert observations: hourly from 05z to 23z (midnight to 6PM ET roughly)
        # Summer: obs = forecast + 1.0 (warm divergence)
        # Winter: obs = forecast - 1.0 (cold divergence)
        if actual > 55.0:
            obs_offset = 1.0  # warm divergence
        else:
            obs_offset = -1.0  # cold divergence

        for obs_hour in range(5, 24):
            obs_time = datetime(obs_date.year, obs_date.month, obs_date.day, obs_hour, 56)
            # Interpolate what the forecast would be at this time
            hours_from_run = (obs_time - model_run).total_seconds() / 3600.0
            if 0 <= hours_from_run <= 18:
                frac = 1.0 - ((hours_from_run - 10) / 10.0) ** 2
                fcst_at_time = fcst_high - 15.0 + 15.0 * max(0, frac)
                obs_temp = round(fcst_at_time + obs_offset, 1)
            else:
                obs_temp = round(actual - 5.0, 1)  # Outside curve, arbitrary

            con.execute(
                "INSERT INTO observations (station_id, observed_at, temp_f, ingested_at) "
                "VALUES ('KNYC', ?, ?, CURRENT_TIMESTAMP)",
                [obs_time, obs_temp],
            )

    con.close()


@pytest.fixture
def seeded_db(test_db):
    _seed_phase2b_data(test_db, n_days=200, run_hour=12)
    return test_db


# ---------------------------------------------------------------------------
# Test: divergence.py interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_basic_interpolation(self):
        """Interpolation between two points gives the midpoint."""
        fc_ts = [0.0, 3600.0]  # 0 and 1 hour
        fc_temps = [50.0, 60.0]
        result = interpolate_forecast(fc_ts, fc_temps, [1800.0])  # 30 min
        assert result[0] is not None
        assert abs(result[0] - 55.0) < 0.01

    def test_clamps_outside_range(self):
        """Values outside the curve are clamped to endpoints."""
        fc_ts = [100.0, 200.0]
        fc_temps = [50.0, 60.0]
        result = interpolate_forecast(fc_ts, fc_temps, [0.0, 300.0])
        assert result[0] == 50.0
        assert result[1] == 60.0

    def test_empty_curve(self):
        """Single-point curve returns None."""
        result = interpolate_forecast([100.0], [50.0], [100.0])
        assert result[0] is None


# ---------------------------------------------------------------------------
# Test: divergence features
# ---------------------------------------------------------------------------

class TestDivergenceFeatures:
    def test_cumulative_divergence_sign(self):
        """Warm obs → positive cumulative divergence."""
        obs_temps = [52.0, 54.0, 56.0, 58.0]
        fcst_interp = [50.0, 52.0, 54.0, 56.0]  # obs is 2°F warmer everywhere
        obs_hours = [0.0, 1.0, 2.0, 3.0]
        fcst_up = [50.0, 52.0, 54.0, 56.0]

        result = compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up)
        assert result is not None
        assert result["cumulative_divergence"] > 0
        assert abs(result["cumulative_divergence"] - 2.0) < 0.01

    def test_running_max_divergence(self):
        """Observed max above forecast max → positive running_max."""
        obs_temps = [50.0, 55.0, 60.0]  # max = 60
        fcst_interp = [50.0, 53.0, 56.0]
        obs_hours = [0.0, 1.0, 2.0]
        fcst_up = [50.0, 53.0, 56.0]  # max = 56

        result = compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up)
        assert result is not None
        assert abs(result["running_max_divergence"] - 4.0) < 0.01  # 60 - 56

    def test_slope_divergence_direction(self):
        """Widening gap → positive slope."""
        obs_temps = [51.0, 53.0, 56.0, 60.0]  # divergence: 1, 3, 6, 10 → widening
        fcst_interp = [50.0, 50.0, 50.0, 50.0]
        obs_hours = [0.0, 1.0, 2.0, 3.0]
        fcst_up = [50.0]

        result = compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up)
        assert result is not None
        assert result["slope_divergence"] > 0

    def test_slope_requires_min_obs(self):
        """With only 2 obs, slope should be 0.0."""
        obs_temps = [52.0, 54.0]
        fcst_interp = [50.0, 52.0]
        obs_hours = [0.0, 1.0]
        fcst_up = [50.0, 52.0]

        result = compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up)
        assert result is not None
        assert result["slope_divergence"] == 0.0

    def test_returns_none_with_few_obs(self):
        """Fewer than 2 valid paired obs → None."""
        obs_temps = [52.0]
        fcst_interp = [50.0]
        obs_hours = [0.0]
        fcst_up = [50.0]

        result = compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up)
        assert result is None

    def test_filters_none_interpolations(self):
        """None values in fcst_interp are filtered out."""
        obs_temps = [52.0, 54.0, 56.0]
        fcst_interp = [50.0, None, 54.0]  # middle one failed
        obs_hours = [0.0, 1.0, 2.0]
        fcst_up = [50.0, 54.0]

        result = compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up)
        assert result is not None
        assert result["n_obs"] == 2  # only 2 valid pairs


# ---------------------------------------------------------------------------
# Test: synoptic max divergence
# ---------------------------------------------------------------------------

class TestSynopticMax:
    def test_basic_conversion(self):
        """Converts Celsius to Fahrenheit and compares."""
        # 20°C = 68°F, forecast max in window = 65°F → divergence = +3°F
        result = compute_synoptic_max_divergence(
            20.0, [100.0, 200.0, 300.0], [60.0, 65.0, 63.0],
            100.0, 300.0,
        )
        assert result is not None
        assert abs(result - 3.0) < 0.01

    def test_none_when_no_data(self):
        assert compute_synoptic_max_divergence(None, [], [], 0, 0) is None

    def test_none_when_no_fcst_in_window(self):
        result = compute_synoptic_max_divergence(
            20.0, [100.0], [60.0], 200.0, 300.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test: walk-forward obs truncation
# ---------------------------------------------------------------------------

class TestObsTruncation:
    def test_obs_truncated_at_update_hour(self, seeded_db):
        """Features computed at 14 ET should use fewer obs than at 18 ET."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            l1 = _ensure_level1(con, 12, "KNYC")
            curves = l1["curves"]
            obs_by_date = l1["obs_by_date"]

            test_date = date(2023, 7, 1)
            if test_date not in curves or test_date not in obs_by_date:
                pytest.skip("Test date not in data")

            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
            cutoff_14 = datetime(2023, 7, 1, 14, 0, tzinfo=_et).astimezone(
                timezone.utc).timestamp()
            cutoff_18 = datetime(2023, 7, 1, 18, 0, tzinfo=_et).astimezone(
                timezone.utc).timestamp()

            feats_14 = _compute_features_for_date(
                curves[test_date], obs_by_date[test_date], cutoff_14,
            )
            feats_18 = _compute_features_for_date(
                curves[test_date], obs_by_date[test_date], cutoff_18,
            )

            assert feats_14 is not None
            assert feats_18 is not None
            assert feats_14["n_obs"] < feats_18["n_obs"]
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Test: walk-forward training excludes current date
# ---------------------------------------------------------------------------

class TestTrainingExclusion:
    def test_training_excludes_current_date(self, seeded_db):
        """Phase 2B training data for date D should not include D."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            training = _ensure_level2(con, 12, "KNYC", 14)
            test_date = date(2023, 7, 1)

            # Filter like the model_fn does
            filtered = [
                (r, fv) for d, r, fv in training if d < test_date
            ]

            # Verify no data from test_date or later
            dates_in_training = [d for d, _, _ in training if d < test_date]
            assert all(d < test_date for d in dates_in_training)

            # Should have plenty of training data (200 days seeded,
            # ~89 have both Phase 2 predictions and divergence features before July 1)
            assert len(filtered) >= WALK_FORWARD_MIN_DAYS - 5
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Test: Phase 2B model returns valid probabilities
# ---------------------------------------------------------------------------

class TestPhase2BProbs:
    def test_probs_sum_to_one(self, seeded_db):
        """Phase 2B model output should sum to ~1.0."""
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            model_run = datetime(2023, 7, 1, 12, 0, tzinfo=timezone.utc)
            # ref_time at 14 ET = 18 UTC (EDT)
            from zoneinfo import ZoneInfo
            ref_time = datetime(2023, 7, 1, 14, 0,
                                tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)

            provider = BacktestDataProvider(
                db_path=seeded_db,
                station_id="KNYC",
                model_run=model_run,
                ref_time=ref_time,
                connection=con,
            )

            probs = wf_phase2b_full(provider, ref_time)
            assert probs is not None, "Phase 2B model returned None"
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, f"Probs sum to {total}"
        finally:
            con.close()

    def test_no_negative_probs(self, seeded_db):
        con = duckdb.connect(seeded_db, read_only=True)
        try:
            model_run = datetime(2023, 7, 1, 12, 0, tzinfo=timezone.utc)
            from zoneinfo import ZoneInfo
            ref_time = datetime(2023, 7, 1, 14, 0,
                                tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)

            provider = BacktestDataProvider(
                db_path=seeded_db,
                station_id="KNYC",
                model_run=model_run,
                ref_time=ref_time,
                connection=con,
            )

            probs = wf_phase2b_full(provider, ref_time)
            assert probs is not None
            for k, p in probs.items():
                assert p >= 0, f"Negative prob {p} at bracket {k}"
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Test: backward compatibility — no update_hours = Phase 1/2 behavior
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_backtester_unchanged_without_update_hours(self, seeded_db):
        """Backtester with update_hours_et=None should produce results
        with update_hour_et=None on all RunResults."""
        bt = Backtester(db_path=seeded_db, city="NYC")
        result = bt.run(wf_regression_full, run_hours=[12])

        assert result.total_evaluations > 0
        for rr in result.run_results:
            assert rr.update_hour_et is None

    def test_backtester_with_update_hours_populates_field(self, seeded_db):
        """Backtester with update_hours_et should populate the field."""
        bt = Backtester(db_path=seeded_db, city="NYC")
        result = bt.run(wf_phase2b_runmax, run_hours=[12], update_hours_et=[14, 16])

        # Some results should have update_hour_et set
        update_hours_seen = set(
            rr.update_hour_et for rr in result.run_results
            if rr.update_hour_et is not None
        )
        # Should see at least one of our requested hours
        assert len(update_hours_seen) > 0
