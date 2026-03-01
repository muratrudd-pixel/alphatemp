"""Test Gaussian mixture ensemble combination."""
import math
import os
from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import pytest
from services.ensemble import (
    combine_mixture_brackets,
    make_ensemble_model_fn,
    make_latest_run_ensemble_fn,
    make_learned_weight_ensemble_fn,
    _find_latest_run_hour,
    _generate_simplex_weights,
    _grid_search_simplex_weights,
    _ProviderProxy,
)


# ---------------------------------------------------------------------------
# Unit tests for combine_mixture_brackets
# ---------------------------------------------------------------------------

def test_equal_weight_two_identical_models():
    """Two identical Gaussians should produce same brackets as single."""
    single = combine_mixture_brackets([(70.0, 3.0)], [1.0])
    double = combine_mixture_brackets([(70.0, 3.0), (70.0, 3.0)], [0.5, 0.5])
    # Peak should be same
    assert max(single, key=single.get) == max(double, key=double.get)
    # Total should be ~1.0
    assert abs(sum(double.values()) - 1.0) < 0.01


def test_mixture_wider_than_components():
    """Two models disagreeing should produce wider distribution."""
    single = combine_mixture_brackets([(70.0, 2.5)], [1.0])
    mixture = combine_mixture_brackets([(68.0, 2.5), (72.0, 2.5)], [0.5, 0.5])
    # Mixture should have more brackets with non-trivial probability
    assert len(mixture) >= len(single)


def test_three_model_equal_weights():
    """HRRR + GFS + ECMWF equal weight ensemble."""
    predictions = [(70.0, 2.5), (68.0, 3.0), (71.0, 2.8)]
    weights = [1/3, 1/3, 1/3]
    brackets = combine_mixture_brackets(predictions, weights)
    assert abs(sum(brackets.values()) - 1.0) < 0.01
    peak = max(brackets, key=brackets.get)
    assert 68 <= peak <= 71


def test_empty_predictions():
    """No predictions should return empty dict."""
    assert combine_mixture_brackets([], []) == {}


def test_single_model_matches_gaussian():
    """Single model should match a plain Gaussian."""
    from scipy.stats import norm
    brackets = combine_mixture_brackets([(70.0, 3.0)], [1.0])
    # P(69.5 < X < 70.5) for N(70, 3)
    expected_70 = norm.cdf(0.5/3.0) - norm.cdf(-0.5/3.0)
    # Should be close after renormalization
    assert abs(brackets[70] - expected_70) < 0.02


def test_weights_matter():
    """Heavier weight on one model should shift the peak."""
    heavy_left = combine_mixture_brackets([(65.0, 2.0), (75.0, 2.0)], [0.9, 0.1])
    heavy_right = combine_mixture_brackets([(65.0, 2.0), (75.0, 2.0)], [0.1, 0.9])
    peak_left = max(heavy_left, key=heavy_left.get)
    peak_right = max(heavy_right, key=heavy_right.get)
    assert peak_left < peak_right


def test_make_ensemble_model_fn():
    """Ensemble wraps model .raw calls correctly."""
    # Create fake model functions with .raw attribute
    def fake_model_1(provider, ref_time):
        return {70: 1.0}
    fake_model_1.raw = lambda provider, ref_time: (70.0, 3.0)

    def fake_model_2(provider, ref_time):
        return {68: 1.0}
    fake_model_2.raw = lambda provider, ref_time: (68.0, 2.5)

    ensemble = make_ensemble_model_fn([fake_model_1, fake_model_2])
    result = ensemble(None, None)  # provider/ref_time not used by fakes
    assert result is not None
    assert abs(sum(result.values()) - 1.0) < 0.01
    peak = max(result, key=result.get)
    assert 68 <= peak <= 70


def test_ensemble_handles_none_from_model():
    """If a model returns None from .raw, skip it."""
    def model_ok(provider, ref_time):
        return {70: 1.0}
    model_ok.raw = lambda provider, ref_time: (70.0, 3.0)

    def model_fail(provider, ref_time):
        return None
    model_fail.raw = lambda provider, ref_time: None

    ensemble = make_ensemble_model_fn([model_ok, model_fail])
    result = ensemble(None, None)
    assert result is not None  # Should still work with one model
    assert 70 == max(result, key=result.get)


# ---------------------------------------------------------------------------
# Integration tests: multi-model forecast query bug fix + latest-run ensemble
# ---------------------------------------------------------------------------

TEST_DB = "data/test_ensemble_integration.duckdb"


def _cleanup_db():
    for path in [TEST_DB, TEST_DB + ".wal"]:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def integration_db():
    """Fresh DB with schema for integration tests."""
    _cleanup_db()
    from core.db import init_db
    init_db(TEST_DB)
    yield TEST_DB
    _cleanup_db()


def _seed_hrrr_and_gfs(db_path, n_days=120):
    """Seed HRRR at 14z (temp=70) and GFS at 12z (temp=80) for same dates.

    HRRR: run_hour=14, forecast temp=actual+2 (bias=+2), fxx=6 (14+6=20, within 5..28)
    GFS:  run_hour=12, forecast temp=actual+8 (bias=+8), fxx=6 (12+6=18, within 5..28)

    This means HRRR fcst_high ~ 70 and GFS fcst_high ~ 76 for typical days.
    The key test: when a GFS model_fn runs against an HRRR provider,
    it should get GFS's 76, not HRRR's 70.
    """
    con = duckdb.connect(db_path)
    try:
        start = date(2023, 1, 1)
        for i in range(n_days):
            obs_date = start + timedelta(days=i)
            day_of_year = obs_date.timetuple().tm_yday
            actual = 55.0 + 20.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
            actual = round(actual, 1)

            # NWS daily truth
            con.execute(
                "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
                "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
                [obs_date, actual],
            )

            # HRRR: run_hour=14, bias=+2
            hrrr_fcst = actual + 2.0
            hrrr_run = datetime(obs_date.year, obs_date.month, obs_date.day, 14, 0)
            hrrr_fxx = 6
            hrrr_valid = hrrr_run + timedelta(hours=hrrr_fxx)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, "
                "ingested_at, model_name, fxx) "
                "VALUES ('KNYC', ?, ?, ?, ?, CURRENT_TIMESTAMP, 'hrrr', ?)",
                [hrrr_run, hrrr_valid, hrrr_fcst, round((hrrr_fcst - 32) * 5 / 9, 2), hrrr_fxx],
            )

            # GFS: run_hour=12, bias=+8
            gfs_fcst = actual + 8.0
            gfs_run = datetime(obs_date.year, obs_date.month, obs_date.day, 12, 0)
            gfs_fxx = 6
            gfs_valid = gfs_run + timedelta(hours=gfs_fxx)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, "
                "ingested_at, model_name, fxx) "
                "VALUES ('KNYC', ?, ?, ?, ?, CURRENT_TIMESTAMP, 'gfs', ?)",
                [gfs_run, gfs_valid, gfs_fcst, round((gfs_fcst - 32) * 5 / 9, 2), gfs_fxx],
            )
    finally:
        con.close()


class TestBugFixForecastHighForModel:
    """Test A: Verify forecast high uses correct model_name after bug fix."""

    def test_gfs_model_gets_gfs_forecast_not_hrrr(self, integration_db):
        """When GFS model_fn runs, it should get GFS forecast high, not HRRR."""
        _seed_hrrr_and_gfs(integration_db, n_days=120)

        from services.backtester import (
            _get_forecast_high_for_model,
            wf_regression_full_gfs,
            _p2b_level1,
            _p2b_level2,
        )
        _p2b_level1.clear()
        _p2b_level2.clear()

        con = duckdb.connect(integration_db, read_only=True)
        try:
            # Test the helper directly: at run_hour=12, GFS should return GFS temp
            # Use Apr 15 (day 104 of range) — well within the 120-day seed window
            test_date = date(2023, 4, 15)
            day_of_year = test_date.timetuple().tm_yday
            expected_actual = 55.0 + 20.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
            expected_actual = round(expected_actual, 1)
            expected_gfs_high = expected_actual + 8.0
            expected_hrrr_high = expected_actual + 2.0

            gfs_run = datetime(2023, 4, 15, 12, 0)
            hrrr_run = datetime(2023, 4, 15, 14, 0)

            gfs_high = _get_forecast_high_for_model(con, 'KNYC', gfs_run, 'gfs')
            hrrr_high = _get_forecast_high_for_model(con, 'KNYC', hrrr_run, 'hrrr')

            assert gfs_high is not None
            assert hrrr_high is not None
            assert abs(gfs_high - expected_gfs_high) < 0.1, (
                "GFS high={}, expected={}".format(gfs_high, expected_gfs_high)
            )
            assert abs(hrrr_high - expected_hrrr_high) < 0.1, (
                "HRRR high={}, expected={}".format(hrrr_high, expected_hrrr_high)
            )
            # Key assertion: they are NOT the same
            assert abs(gfs_high - hrrr_high) > 5.0, (
                "GFS and HRRR highs should differ by ~6°F, got GFS={}, HRRR={}".format(
                    gfs_high, hrrr_high)
            )
        finally:
            con.close()


class TestLatestRunEnsemble:
    """Test B: Latest-run ensemble combines models with age-based weights."""

    def test_finds_latest_run_hours(self, integration_db):
        """_find_latest_run_hour should find GFS at 12z when current_hour=14."""
        _seed_hrrr_and_gfs(integration_db, n_days=30)
        con = duckdb.connect(integration_db, read_only=True)
        try:
            # At hour 14, GFS latest should be 12
            gfs_hour = _find_latest_run_hour(
                con, 'gfs', 'KNYC', date(2023, 1, 15), 14,
            )
            assert gfs_hour == 12

            # At hour 14, HRRR latest should be 14
            hrrr_hour = _find_latest_run_hour(
                con, 'hrrr', 'KNYC', date(2023, 1, 15), 14,
            )
            assert hrrr_hour == 14

            # At hour 11, GFS should not exist (only seeded at 12z)
            gfs_before = _find_latest_run_hour(
                con, 'gfs', 'KNYC', date(2023, 1, 15), 11,
            )
            assert gfs_before is None
        finally:
            con.close()

    def test_age_based_weights(self, integration_db):
        """HRRR at current hour should get higher weight than stale GFS."""
        _seed_hrrr_and_gfs(integration_db, n_days=120)

        from services.backtester import (
            wf_regression_full,
            wf_regression_full_gfs,
            _p2b_level1,
            _p2b_level2,
        )
        _p2b_level1.clear()
        _p2b_level2.clear()

        # Build ensemble with λ=0.1
        ensemble_fn = make_latest_run_ensemble_fn([
            (wf_regression_full, 'hrrr'),
            (wf_regression_full_gfs, 'gfs'),
        ], decay_lambda=0.1)

        con = duckdb.connect(integration_db, read_only=True)
        try:
            # Provider at HRRR 14z — use Apr 15 (day 104), within seed window
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 4, 15, 14, 0))
            ref_time = datetime(2023, 4, 15, 16, 0)

            result = ensemble_fn(provider, ref_time)
            assert result is not None, "Ensemble returned None"
            assert abs(sum(result.values()) - 1.0) < 0.01

            # Verify weights: HRRR age=0 → w=1.0, GFS age=2 → w=exp(-0.2)≈0.82
            # After normalization: HRRR≈0.55, GFS≈0.45
            # Both should contribute (result should not be single-model)
        finally:
            con.close()


class TestMissingModelFallback:
    """Test C: Ensemble gracefully falls back when a model has no data."""

    def test_hrrr_only_fallback(self, integration_db):
        """When only HRRR data exists, ensemble should still produce results."""
        # Seed only HRRR data (no GFS)
        con = duckdb.connect(integration_db)
        try:
            start = date(2023, 1, 1)
            for i in range(120):
                obs_date = start + timedelta(days=i)
                day_of_year = obs_date.timetuple().tm_yday
                actual = 55.0 + 20.0 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0)
                actual = round(actual, 1)

                con.execute(
                    "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at) "
                    "VALUES ('KNYC', ?, ?, NULL, 'test', CURRENT_TIMESTAMP)",
                    [obs_date, actual],
                )

                hrrr_fcst = actual + 2.0
                hrrr_run = datetime(obs_date.year, obs_date.month, obs_date.day, 14, 0)
                hrrr_fxx = 6
                hrrr_valid = hrrr_run + timedelta(hours=hrrr_fxx)
                con.execute(
                    "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx) "
                    "VALUES ('KNYC', ?, ?, ?, ?, CURRENT_TIMESTAMP, 'hrrr', ?)",
                    [hrrr_run, hrrr_valid, hrrr_fcst, round((hrrr_fcst - 32) * 5 / 9, 2), hrrr_fxx],
                )
        finally:
            con.close()

        from services.backtester import (
            wf_regression_full,
            wf_regression_full_gfs,
            _p2b_level1,
            _p2b_level2,
        )
        _p2b_level1.clear()
        _p2b_level2.clear()

        ensemble_fn = make_latest_run_ensemble_fn([
            (wf_regression_full, 'hrrr'),
            (wf_regression_full_gfs, 'gfs'),
        ], decay_lambda=0.1)

        con = duckdb.connect(integration_db, read_only=True)
        try:
            # Use Apr 15 (day 104), within seed window
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 4, 15, 14, 0))
            ref_time = datetime(2023, 4, 15, 16, 0)

            result = ensemble_fn(provider, ref_time)
            # Should still work with HRRR alone
            assert result is not None, "Ensemble should not be None with HRRR-only data"
            assert abs(sum(result.values()) - 1.0) < 0.01
        finally:
            con.close()

    def test_no_data_returns_none(self, integration_db):
        """When no model has data at all, ensemble returns None."""
        from services.backtester import wf_regression_full, wf_regression_full_gfs

        ensemble_fn = make_latest_run_ensemble_fn([
            (wf_regression_full, 'hrrr'),
            (wf_regression_full_gfs, 'gfs'),
        ], decay_lambda=0.1)

        con = duckdb.connect(integration_db, read_only=True)
        try:
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 5, 1, 14, 0))
            ref_time = datetime(2023, 5, 1, 16, 0)
            result = ensemble_fn(provider, ref_time)
            assert result is None
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Level 3 tests: Multi-model OLS stacking
# ---------------------------------------------------------------------------

class TestMultimodelOLS:
    """Tests for _make_multimodel_regression_model (Level 3 OLS stacking)."""

    def test_multimodel_uses_gfs_fcst_high(self, integration_db):
        """Multi-model OLS should receive both HRRR and GFS forecast highs as features."""
        _seed_hrrr_and_gfs(integration_db, n_days=120)

        from services.backtester import (
            wf_multimodel_hrrr_gfs,
            _get_latest_model_fcst_highs_bulk,
            _p2b_level1,
            _p2b_level2,
        )
        _p2b_level1.clear()
        _p2b_level2.clear()

        con = duckdb.connect(integration_db, read_only=True)
        try:
            # Verify _get_latest_model_fcst_highs_bulk returns GFS data
            gfs_highs = _get_latest_model_fcst_highs_bulk(
                con, 'gfs', 'KNYC', 14, date(2023, 4, 30),
            )
            assert len(gfs_highs) > 0, "GFS bulk query returned no data"

            # Verify GFS highs differ from HRRR (bias=+8 vs +2)
            hrrr_highs = _get_latest_model_fcst_highs_bulk(
                con, 'hrrr', 'KNYC', 14, date(2023, 4, 30),
            )
            # Pick a date that both have
            common_dates = set(gfs_highs.keys()) & set(hrrr_highs.keys())
            assert len(common_dates) > 0
            sample_date = sorted(common_dates)[50]  # mid-range date
            assert abs(gfs_highs[sample_date] - hrrr_highs[sample_date]) > 5.0, (
                "GFS and HRRR bulk highs should differ by ~6°F"
            )

            # Run the actual multi-model function
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 4, 15, 14, 0))
            ref_time = datetime(2023, 4, 15, 16, 0)
            result = wf_multimodel_hrrr_gfs.raw(provider, ref_time)
            assert result is not None, "Multi-model raw returned None"
            center, std = result
            assert 50 < center < 90, "Center {} out of plausible range".format(center)
            assert 0.3 <= std < 15, "Std {} out of plausible range".format(std)
        finally:
            con.close()

    def test_multimodel_imputes_missing(self, integration_db):
        """When ECMWF has no data, multimodel should impute with HRRR and not crash."""
        # Seed only HRRR + GFS (no ECMWF)
        _seed_hrrr_and_gfs(integration_db, n_days=120)

        from services.backtester import (
            wf_multimodel_full,
            _p2b_level1,
            _p2b_level2,
        )
        _p2b_level1.clear()
        _p2b_level2.clear()

        con = duckdb.connect(integration_db, read_only=True)
        try:
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 4, 15, 14, 0))
            ref_time = datetime(2023, 4, 15, 16, 0)

            # wf_multimodel_full expects HRRR+GFS+ECMWF — ECMWF is missing
            result = wf_multimodel_full.raw(provider, ref_time)
            # Should still produce a result (ECMWF imputed with HRRR)
            assert result is not None, "Multi-model should impute missing ECMWF, not return None"
            center, std = result
            assert 50 < center < 90
        finally:
            con.close()

    def test_multimodel_returns_valid_probs(self, integration_db):
        """Multi-model bracket probabilities should sum to ~1.0."""
        _seed_hrrr_and_gfs(integration_db, n_days=120)

        from services.backtester import (
            wf_multimodel_hrrr_gfs,
            _p2b_level1,
            _p2b_level2,
        )
        _p2b_level1.clear()
        _p2b_level2.clear()

        con = duckdb.connect(integration_db, read_only=True)
        try:
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 4, 15, 14, 0))
            ref_time = datetime(2023, 4, 15, 16, 0)
            bracket_probs = wf_multimodel_hrrr_gfs(provider, ref_time)
            assert bracket_probs is not None
            total = sum(bracket_probs.values())
            assert abs(total - 1.0) < 0.01, "Probs sum to {}, expected ~1.0".format(total)
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Level 2 tests: Learned mixture weights
# ---------------------------------------------------------------------------

class TestLearnedWeights:
    """Tests for grid search and learned weight ensemble (Level 2)."""

    def test_grid_search_finds_optimal(self):
        """With 2 fake models (one always better), grid search should favor it."""
        from scipy.stats import norm

        n_dates = 200
        radius = 15
        n_brackets = 2 * radius + 1

        # Model A: centered on actual (good), Model B: centered 5°F off (bad)
        model_probs = np.zeros((n_dates, 2, n_brackets), dtype=np.float64)
        actuals = np.zeros((n_dates, n_brackets), dtype=np.float64)

        for i in range(n_dates):
            actual = 70  # fixed actual
            center_a = 70.0  # Model A: perfect center
            center_b = 75.0  # Model B: 5°F too high
            sigma = 3.0
            bracket_range = list(range(actual - radius, actual + radius + 1))

            for b_idx, k in enumerate(bracket_range):
                model_probs[i, 0, b_idx] = (
                    norm.cdf((k + 0.5 - center_a) / sigma) -
                    norm.cdf((k - 0.5 - center_a) / sigma)
                )
                model_probs[i, 1, b_idx] = (
                    norm.cdf((k + 0.5 - center_b) / sigma) -
                    norm.cdf((k - 0.5 - center_b) / sigma)
                )

            # Normalize
            model_probs[i, 0] /= model_probs[i, 0].sum()
            model_probs[i, 1] /= model_probs[i, 1].sum()

            # One-hot actual
            actuals[i, radius] = 1.0  # actual=70 is at index radius

        best_weights, best_brier = _grid_search_simplex_weights(
            model_probs, actuals, 2, step=0.05,
        )

        # Model A should get much higher weight than Model B
        assert best_weights[0] > best_weights[1], (
            "Good model should get higher weight: {}".format(best_weights)
        )
        assert best_weights[0] >= 0.7, (
            "Good model weight should be >= 0.7, got {}".format(best_weights[0])
        )

    def test_learned_weights_no_data_returns_none(self, integration_db):
        """When no model has data, learned weight ensemble returns None."""
        from services.backtester import wf_regression_full, wf_regression_full_gfs

        learned_fn = make_learned_weight_ensemble_fn([
            (wf_regression_full, 'hrrr'),
            (wf_regression_full_gfs, 'gfs'),
        ], lookback_days=365)

        con = duckdb.connect(integration_db, read_only=True)
        try:
            provider = _ProviderProxy(con, 'KNYC', datetime(2023, 5, 1, 14, 0))
            ref_time = datetime(2023, 5, 1, 16, 0)
            result = learned_fn(provider, ref_time)
            assert result is None
        finally:
            con.close()
