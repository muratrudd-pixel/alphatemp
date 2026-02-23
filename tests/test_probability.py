"""Tests for the probability engine."""

import os
from datetime import datetime, timedelta, timezone

import pytest

from core.db import init_db, get_connection
from services.bias_model import StationBias
from services.probability import ProbabilityEngine, CityForecast

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _seed_forecast(con, station_id, model_run_str, temps):
    """Seed forecast temps for a model run (fxx 1..len(temps))."""
    model_run = datetime.fromisoformat(model_run_str)
    for i, temp_f in enumerate(temps):
        fxx = i + 1
        valid_hour = model_run.hour + fxx
        valid_at = model_run.replace(hour=valid_hour % 24)
        temp_c = round((temp_f - 32) * 5 / 9, 2)
        con.execute(
            "INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [station_id, model_run, valid_at, temp_f, temp_c],
        )


@pytest.fixture
def seeded_db(test_db):
    """Seed forecast data, drift signal, and bias stats for NYC."""
    con = get_connection(test_db)

    # Forecast: 12z run, temps climbing to 52°F high
    _seed_forecast(con, "KNYC", "2026-02-22 12:00:00",
                   [44, 46, 48, 50, 52, 51, 49, 47])

    # Drift signal: +0.5°F warm drift
    con.execute(
        """INSERT INTO drift_signals
           (city, calculated_at, model_run, drift_score, slope_divergence,
            forecast_trend, confidence, projected_high)
           VALUES ('NYC', '2026-02-22 14:00:00', '2026-02-22 12:00:00',
                   0.5, 0.1, 0.2, 0.85, 52.5)"""
    )

    # Historical bias: forecast runs +0.8°F hot, std 1.2°F
    con.execute(
        """INSERT INTO station_bias (station_id, calculated_at, mean_bias, std_error, sample_days)
           VALUES ('KNYC', '2026-02-22 00:00:00', 0.8, 1.2, 85)"""
    )

    con.close()
    return test_db


def test_bracket_probs_sum_to_one(seeded_db):
    """Bracket probabilities must sum to ~1.0."""
    engine = ProbabilityEngine(db_path=seeded_db)
    engine.load_bias_cache()
    forecast = engine.calculate_city("NYC", ref_time=datetime(2026, 2, 22, 14, 0, tzinfo=timezone.utc))

    assert forecast is not None
    total = sum(forecast.bracket_probs.values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_center_adjusts_for_bias_and_drift(seeded_db):
    """Center = fcst_high(52) - bias(0.8) + drift(0.5) = 51.7."""
    engine = ProbabilityEngine(db_path=seeded_db)
    engine.load_bias_cache()
    forecast = engine.calculate_city("NYC", ref_time=datetime(2026, 2, 22, 14, 0, tzinfo=timezone.utc))

    assert forecast is not None
    assert forecast.center == pytest.approx(51.7, abs=0.1)


def test_interval_narrows_with_lower_std():
    """Smaller std -> narrower 90% confidence interval."""
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    # Compute brackets at two different std values
    probs_wide = engine._compute_bracket_probs(center=50.0, std=2.0)
    probs_narrow = engine._compute_bracket_probs(center=50.0, std=0.5)

    # Wide should have more brackets with meaningful probability
    wide_brackets = [k for k, v in probs_wide.items() if v > 0.05]
    narrow_brackets = [k for k, v in probs_narrow.items() if v > 0.05]

    assert len(wide_brackets) > len(narrow_brackets)


def test_bracket_probs_degenerate_zero_std():
    """Zero std should put all probability on nearest integer."""
    engine = ProbabilityEngine.__new__(ProbabilityEngine)
    probs = engine._compute_bracket_probs(center=50.3, std=0.0)

    assert probs == {50: 1.0}


def test_time_factor_morning_is_high():
    """8am ET (13 UTC) should have time_factor near 1.0."""
    engine = ProbabilityEngine.__new__(ProbabilityEngine)
    # 13 UTC = 8am ET
    factor = engine._compute_time_factor(datetime(2026, 2, 22, 13, 0, tzinfo=timezone.utc))
    assert factor == pytest.approx(1.0, abs=0.05)


def test_time_factor_afternoon_is_low():
    """3pm ET (20 UTC) should have time_factor near 0.3."""
    engine = ProbabilityEngine.__new__(ProbabilityEngine)
    # 20 UTC = 3pm ET
    factor = engine._compute_time_factor(datetime(2026, 2, 22, 20, 0, tzinfo=timezone.utc))
    assert factor == pytest.approx(0.3, abs=0.05)


def test_city_forecast_has_all_fields(seeded_db):
    """CityForecast should include city, center, std, interval_90, bracket_probs."""
    engine = ProbabilityEngine(db_path=seeded_db)
    engine.load_bias_cache()
    forecast = engine.calculate_city("NYC", ref_time=datetime(2026, 2, 22, 14, 0, tzinfo=timezone.utc))

    assert forecast is not None
    assert isinstance(forecast, CityForecast)
    assert forecast.city == "NYC"
    assert isinstance(forecast.center, float)
    assert isinstance(forecast.std, float)
    assert isinstance(forecast.interval_90, tuple)
    assert len(forecast.interval_90) == 2
    assert forecast.interval_90[0] < forecast.interval_90[1]
    assert isinstance(forecast.bracket_probs, dict)
    assert len(forecast.bracket_probs) > 0


def test_no_forecast_data_returns_none(test_db):
    """Returns None when no forecast data exists for a city."""
    engine = ProbabilityEngine(db_path=test_db)
    engine.load_bias_cache()
    forecast = engine.calculate_city("NYC")
    assert forecast is None


def test_calculate_all_returns_dict(seeded_db):
    """calculate_all should return a dict of CityForecasts."""
    engine = ProbabilityEngine(db_path=seeded_db)
    engine.load_bias_cache()
    results = engine.calculate_all(ref_time=datetime(2026, 2, 22, 14, 0, tzinfo=timezone.utc))

    assert isinstance(results, dict)
    assert "NYC" in results
    assert isinstance(results["NYC"], CityForecast)


def test_bias_cache_auto_refreshes(seeded_db):
    """Cache should auto-refresh when TTL expires."""
    engine = ProbabilityEngine(db_path=seeded_db)
    engine.load_bias_cache()

    original_bias = engine._bias_cache["KNYC"].mean_bias
    assert original_bias == pytest.approx(0.8, abs=0.01)

    # Insert newer bias data
    con = get_connection(seeded_db)
    con.execute(
        """INSERT INTO station_bias (station_id, calculated_at, mean_bias, std_error, sample_days)
           VALUES ('KNYC', '2026-02-22 12:00:00', 1.5, 1.0, 90)"""
    )
    con.close()

    # Cache not stale yet — should still return old value
    engine.calculate_city("NYC", ref_time=datetime(2026, 2, 22, 14, 0, tzinfo=timezone.utc))
    assert engine._bias_cache["KNYC"].mean_bias == pytest.approx(0.8, abs=0.01)

    # Force cache to appear stale
    engine._cache_loaded_at = datetime.now(timezone.utc) - timedelta(hours=2)

    # Next call should trigger refresh
    engine.calculate_city("NYC", ref_time=datetime(2026, 2, 22, 14, 0, tzinfo=timezone.utc))
    assert engine._bias_cache["KNYC"].mean_bias == pytest.approx(1.5, abs=0.01)
