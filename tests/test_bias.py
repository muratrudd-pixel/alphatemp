# tests/test_bias.py
import os
import pytest
import duckdb
from datetime import datetime, timezone

from core.db import init_db, get_connection
from services.bias import BiasEngine, compute_drift, compute_slope_divergence

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def seeded_db(test_db):
    """Seed observations and forecasts for testing."""
    con = get_connection(test_db)

    # Seed 5 observations for KNYC over 5 hours
    # Observations show temps slightly above forecast (+0.8F)
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", f"2026-02-22 {hour}:00:00", 44.0 + i * 1.5 + 0.8,
             None, "", "2026-02-22 15:00:00"],
        )

    # Seed HRRR forecast curve (model run 06z)
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO forecasts
               (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", "2026-02-22 06:00:00", f"2026-02-22 {hour}:00:00",
             44.0 + i * 1.5, (44.0 + i * 1.5 - 32) * 5 / 9, "2026-02-22 07:00:00"],
        )

    # Seed second HRRR run (09z) with slightly higher temps
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO forecasts
               (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", "2026-02-22 09:00:00", f"2026-02-22 {hour}:00:00",
             44.0 + i * 1.5 + 0.3, (44.3 + i * 1.5 - 32) * 5 / 9, "2026-02-22 10:00:00"],
        )

    # Seed third HRRR run (12z) with even higher temps
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO forecasts
               (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", "2026-02-22 12:00:00", f"2026-02-22 {hour}:00:00",
             44.0 + i * 1.5 + 0.5, (44.5 + i * 1.5 - 32) * 5 / 9, "2026-02-22 13:00:00"],
        )

    con.close()
    return test_db


def test_compute_drift_positive_when_warmer():
    """Positive drift = observations warmer than forecast."""
    obs_temps = [45.0, 46.5, 48.0]
    fcst_temps = [44.0, 45.5, 47.0]
    drift = compute_drift(obs_temps, fcst_temps)
    assert drift == pytest.approx(1.0, abs=0.01)


def test_compute_drift_negative_when_cooler():
    obs_temps = [43.0, 44.5, 46.0]
    fcst_temps = [44.0, 45.5, 47.0]
    drift = compute_drift(obs_temps, fcst_temps)
    assert drift == pytest.approx(-1.0, abs=0.01)


def test_compute_drift_zero_when_matching():
    obs_temps = [44.0, 45.5, 47.0]
    fcst_temps = [44.0, 45.5, 47.0]
    drift = compute_drift(obs_temps, fcst_temps)
    assert drift == pytest.approx(0.0, abs=0.01)


def test_compute_slope_divergence_widening():
    """Positive slope = gap is widening (obs pulling further above forecast)."""
    obs_temps = [44.5, 46.5, 48.5]
    fcst_temps = [44.0, 45.5, 47.0]
    hours = [0.0, 1.0, 2.0]
    slope = compute_slope_divergence(obs_temps, fcst_temps, hours)
    assert slope > 0


def test_compute_slope_divergence_closing():
    """Negative slope = gap is closing."""
    obs_temps = [45.5, 46.5, 47.5]
    fcst_temps = [44.0, 45.5, 47.0]
    hours = [0.0, 1.0, 2.0]
    slope = compute_slope_divergence(obs_temps, fcst_temps, hours)
    assert slope < 0


def test_bias_engine_produces_drift_report(seeded_db):
    """Full integration: seeded data should produce a drift signal for NYC."""
    engine = BiasEngine(db_path=seeded_db)
    reports = engine.calculate_all(
        ref_time=datetime(2026, 2, 22, 15, 0, tzinfo=timezone.utc)
    )

    assert "NYC" in reports
    report = reports["NYC"]
    assert report["drift_score"] > 0  # Obs are warmer
    assert report["projected_high"] is not None
    assert report["magnet_proximity"] is not None
    assert 0.0 <= report["confidence"] <= 1.0


def test_bias_engine_stores_signals(seeded_db):
    """Drift signals should be written to the drift_signals table."""
    engine = BiasEngine(db_path=seeded_db)
    engine.calculate_and_store(
        ref_time=datetime(2026, 2, 22, 15, 0, tzinfo=timezone.utc)
    )

    con = get_connection(seeded_db)
    count = con.execute("SELECT COUNT(*) FROM drift_signals").fetchone()[0]
    con.close()

    assert count > 0
