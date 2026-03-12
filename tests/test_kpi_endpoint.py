"""Tests for the KPI summary endpoint and new page routes."""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.db import init_db, get_connection

TEST_DB = "data/test_kpi.duckdb"


@pytest.fixture(autouse=True)
def setup_test_db():
    """Redirect dashboard to test DB, init tables, clean up after."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)

    with patch("ui.web_dashboard.get_connection", lambda: get_connection(TEST_DB)), \
         patch("ui.web_dashboard.init_db", lambda: None):
        from ui.web_dashboard import app
        yield app
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def client(setup_test_db):
    return TestClient(setup_test_db)


# --- KPI Summary endpoint ---


def test_kpi_summary_returns_expected_fields(client):
    """KPI summary should return all required fields."""
    resp = client.get("/api/kpi-summary?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert "system_status" in data
    assert "model_high" in data
    assert "settlement" in data
    assert "open_positions" in data
    assert "day_pnl" in data
    assert "total_pnl" in data
    assert "drift" in data
    assert "market_consensus" in data


def test_kpi_summary_system_status_values(client):
    """System status should be one of green, amber, red."""
    resp = client.get("/api/kpi-summary?city=nyc")
    data = resp.json()
    assert data["system_status"] in ("green", "amber", "red")


def test_kpi_summary_settlement_structure(client):
    """Settlement should have temp and source fields."""
    resp = client.get("/api/kpi-summary?city=nyc")
    data = resp.json()
    assert "temp" in data["settlement"]
    assert "source" in data["settlement"]


def test_kpi_summary_empty_db_defaults(client):
    """With empty DB, should return safe defaults (no crashes)."""
    resp = client.get("/api/kpi-summary?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["open_positions"] == 0
    assert data["day_pnl"] == 0
    assert data["total_pnl"] == 0
    assert data["drift"] == 0.0


# --- New page routes ---


def test_health_page(client):
    """Health page should return HTML."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert b"Health" in resp.content


def test_blotter_page(client):
    """Blotter page should return HTML."""
    resp = client.get("/blotter")
    assert resp.status_code == 200
    assert b"Blotter" in resp.content
