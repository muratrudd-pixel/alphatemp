"""Tests for the web dashboard endpoints."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.db import init_db, get_connection

TEST_DB = "data/test_dashboard.duckdb"


@pytest.fixture(autouse=True)
def setup_test_db():
    """Redirect dashboard to test DB, init tables, clean up after."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)

    # Patch get_connection and engine before importing app
    with patch("ui.web_dashboard.get_connection", lambda: get_connection(TEST_DB)), \
         patch("ui.web_dashboard.init_db", lambda: None):
        from ui.web_dashboard import app, engine
        engine.db_path = TEST_DB
        engine._bias_cache = {}
        engine._cache_loaded_at = datetime.now(timezone.utc)
        yield app
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def client(setup_test_db):
    return TestClient(setup_test_db)


# --- Health endpoint ---


def test_health_returns_freshness_fields(client):
    """Health endpoint should include obs/fcst age and stale flags."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "obs_age_minutes" in data
    assert "fcst_age_minutes" in data
    assert "obs_stale" in data
    assert "fcst_stale" in data
    assert data["db_connected"] is True


def test_health_stale_when_no_data(client):
    """With empty tables, data should be marked stale."""
    resp = client.get("/api/health")
    data = resp.json()
    assert data["obs_stale"] is True
    assert data["fcst_stale"] is True
    assert data["status"] == "degraded"


# --- Error handling ---


def test_unknown_city_observations(client):
    """Unknown city should return error, not crash."""
    resp = client.get("/api/observations/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_unknown_city_forecast_points(client):
    """Unknown city should return error for forecast-points."""
    resp = client.get("/api/forecast-points/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_unknown_city_forecast_curve(client):
    """Unknown city should return error for forecast-curve."""
    resp = client.get("/api/forecast-curve/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


# --- Empty responses ---


def test_empty_observations(client):
    """Empty observations should return empty list, not crash."""
    resp = client.get("/api/observations/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["observations"] == []
    assert data["running_high"] is None


def test_empty_forecast_curve(client):
    """Empty forecast curve should return empty lists."""
    resp = client.get("/api/forecast-curve/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["forecasts"] == []


def test_empty_forecast_points(client):
    """Empty forecast points should return empty list."""
    resp = client.get("/api/forecast-points/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["points"] == []
    assert data["forecast_high"] is None


def test_empty_market(client):
    """Empty market should return available=False."""
    resp = client.get("/api/market/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


# --- Index page ---


def test_index_returns_html(client):
    """Index should return HTML."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# --- Observations with date param ---


def test_observations_with_date_param(client):
    """Observations endpoint should accept a date param without crashing."""
    resp = client.get("/api/observations/NYC?date=2026-02-22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"
    assert isinstance(data["observations"], list)


# --- Forecast curve with date param ---


def test_forecast_curve_with_date_param(client):
    """Forecast curve should accept a date param without crashing."""
    resp = client.get("/api/forecast-curve/NYC?date=2026-02-22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"
    assert isinstance(data["forecasts"], list)


# --- Forecast points with date param ---


def test_forecast_points_with_date_param(client):
    """Forecast points should accept a date param."""
    resp = client.get("/api/forecast-points/NYC?date=2026-02-22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"


# --- Market with date param ---


def test_market_with_date_param(client):
    """Market endpoint should accept a date param."""
    resp = client.get("/api/market/NYC?date=2026-02-22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"


# --- Multi-station observations ---


def _strip_tz(dt):
    """Strip timezone for DuckDB TIMESTAMP columns (stores naive values)."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _insert_obs(db_path, station_id, observed_at, temp_f):
    """Helper to insert a single observation row."""
    observed_at = _strip_tz(observed_at)
    con = get_connection(db_path)
    con.execute(
        """INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
           VALUES (?, ?, ?, NULL, NULL, ?)""",
        [station_id, observed_at, temp_f, observed_at],
    )
    con.close()


def _insert_forecast(db_path, station_id, model_run, valid_at, temp_f):
    """Helper to insert a single forecast row."""
    model_run = _strip_tz(model_run)
    valid_at = _strip_tz(valid_at)
    con = get_connection(db_path)
    con.execute(
        """INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
           VALUES (?, ?, ?, ?, NULL, ?)""",
        [station_id, model_run, valid_at, temp_f, valid_at],
    )
    con.close()


def test_observations_include_neighbor_stations(client):
    """Observation feed should return obs from settlement + neighbor stations."""
    # NYC: settlement=KNYC, neighbors=[KLGA, KEWR]
    now = datetime.now(timezone.utc)
    base = now.replace(hour=12, minute=0, second=0, microsecond=0)

    _insert_obs(TEST_DB, "KNYC", base, 30.0)
    _insert_obs(TEST_DB, "KLGA", base + timedelta(minutes=1), 31.5)
    _insert_obs(TEST_DB, "KEWR", base + timedelta(minutes=2), 29.0)

    today = base.strftime("%Y-%m-%d")
    resp = client.get(f"/api/observations/NYC?date={today}")
    assert resp.status_code == 200
    data = resp.json()

    stations_returned = {o["station_id"] for o in data["observations"]}
    assert "KNYC" in stations_returned
    assert "KLGA" in stations_returned
    assert "KEWR" in stations_returned
    assert len(data["observations"]) == 3


def test_running_high_settlement_only(client):
    """Running high should only consider the settlement station, not neighbors."""
    now = datetime.now(timezone.utc)
    base = now.replace(hour=12, minute=0, second=0, microsecond=0)

    # Neighbor has higher temp — should NOT be running high
    _insert_obs(TEST_DB, "KNYC", base, 30.0)
    _insert_obs(TEST_DB, "KLGA", base + timedelta(minutes=1), 50.0)

    today = base.strftime("%Y-%m-%d")
    resp = client.get(f"/api/observations/NYC?date={today}")
    data = resp.json()

    assert data["running_high"] == 30.0
    assert data["running_high_station"] == "KNYC"


# --- Ribbon collapse ---


def test_ribbon_collapses_for_past_timestamps(client):
    """Confidence ribbon should have near-zero std for past forecast timestamps."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=6)

    # Past timestamp (3 hours ago)
    past_valid = now - timedelta(hours=3)
    # Future timestamp (3 hours ahead)
    future_valid = now + timedelta(hours=3)

    _insert_forecast(TEST_DB, "KNYC", model_run, past_valid, 32.0)
    _insert_forecast(TEST_DB, "KNYC", model_run, future_valid, 35.0)

    resp = client.get("/api/forecast-curve/NYC")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["ribbon"]) == 2

    past_ribbon = data["ribbon"][0]
    future_ribbon = data["ribbon"][1]

    # Past should have near-zero std (0.01)
    assert past_ribbon["std"] == 0.01
    # Future should have std >= 0.3 (the floor)
    assert future_ribbon["std"] >= 0.3


# --- now_utc in forecast_curve ---


def test_forecast_curve_includes_now_utc(client):
    """forecast_curve response should include now_utc field."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=1)
    valid = now + timedelta(hours=1)

    _insert_forecast(TEST_DB, "KNYC", model_run, valid, 33.0)

    resp = client.get("/api/forecast-curve/NYC")
    data = resp.json()

    assert "now_utc" in data
    # Should be parseable as ISO datetime
    parsed = datetime.fromisoformat(data["now_utc"])
    assert parsed.tzinfo is not None


# --- Multi-station obs in forecast_curve ---


def test_forecast_curve_obs_include_neighbors(client):
    """forecast_curve observations should include data from neighbor stations."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=2)
    valid = now + timedelta(hours=1)

    _insert_forecast(TEST_DB, "KNYC", model_run, valid, 33.0)

    # Insert obs from multiple stations within the same day
    base = now - timedelta(hours=1)
    _insert_obs(TEST_DB, "KNYC", base, 30.0)
    _insert_obs(TEST_DB, "KLGA", base + timedelta(minutes=1), 31.0)
    _insert_obs(TEST_DB, "KEWR", base + timedelta(minutes=2), 29.5)

    # Use date param so the day-bounds query includes past observations
    today = now.strftime("%Y-%m-%d")
    resp = client.get(f"/api/forecast-curve/NYC?date={today}")
    data = resp.json()

    stations_in_obs = {o["station_id"] for o in data["observations"]}
    assert "KNYC" in stations_in_obs
    assert "KLGA" in stations_in_obs
    assert "KEWR" in stations_in_obs
