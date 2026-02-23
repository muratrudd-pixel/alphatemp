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
