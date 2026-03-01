"""Tests for the /api/health/detailed endpoint."""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.db import init_db, get_connection

TEST_DB = "data/test_health_detailed.duckdb"


@pytest.fixture(autouse=True)
def setup_test_db():
    """Redirect dashboard to test DB, init tables, clean up after."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)

    with patch("ui.web_dashboard.get_connection", lambda: get_connection(TEST_DB)), \
         patch("ui.web_dashboard.init_db", lambda: None):
        from ui.web_dashboard import app, engine
        engine.provider.db_path = TEST_DB
        engine.provider._bias_cache = {}
        engine.provider._cache_loaded_at = datetime.now(timezone.utc)
        yield app
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def client(setup_test_db):
    return TestClient(setup_test_db)


# --- /api/health/detailed ---


def test_health_detailed_returns_sections(client):
    """Response must contain all four top-level sections."""
    resp = client.get("/api/health/detailed")
    assert resp.status_code == 200
    data = resp.json()
    assert "freshness" in data
    assert "pipelines" in data
    assert "database" in data
    assert "config" in data


def test_health_detailed_freshness_keys(client):
    """Freshness section must include all expected data source keys."""
    resp = client.get("/api/health/detailed")
    data = resp.json()
    expected_keys = [
        "observations_synoptic", "observations_awc", "forecasts_hrrr",
        "market_ticks", "drift_signals", "nws_cli", "nws_dsm",
    ]
    for key in expected_keys:
        assert key in data["freshness"], "Missing freshness key: {}".format(key)


def test_health_detailed_freshness_structure(client):
    """Each freshness entry must have last_seen and age_minutes fields."""
    resp = client.get("/api/health/detailed")
    data = resp.json()
    for key, entry in data["freshness"].items():
        assert "last_seen" in entry, "Missing last_seen in {}".format(key)
        assert "age_minutes" in entry, "Missing age_minutes in {}".format(key)


def test_health_detailed_config_values(client):
    """Config section must include known static values."""
    resp = client.get("/api/health/detailed")
    data = resp.json()
    assert data["config"]["settlement_station"] == "KNYC"
    assert data["config"]["stale_threshold_min"] == 30
    assert data["config"]["model"] == "HRRR"
    assert "polling_intervals" in data["config"]


def test_health_detailed_database_tables(client):
    """Database section must include row counts for core tables."""
    resp = client.get("/api/health/detailed")
    data = resp.json()
    tables = data["database"]["tables"]
    for table in ["observations", "forecasts", "market_ticks", "drift_signals", "nws_daily"]:
        assert table in tables, "Missing table: {}".format(table)
        assert "rows" in tables[table]


def test_health_detailed_empty_db_no_crash(client):
    """Endpoint must not crash on an empty database."""
    resp = client.get("/api/health/detailed")
    assert resp.status_code == 200
    data = resp.json()
    # All freshness entries should be null/None for empty DB
    for key, entry in data["freshness"].items():
        assert entry["last_seen"] is None
        assert entry["age_minutes"] is None
