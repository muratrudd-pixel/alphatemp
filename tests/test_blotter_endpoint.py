"""Tests for the /api/blotter/{city} endpoint."""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.db import init_db, get_connection

TEST_DB = "data/test_blotter.duckdb"


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


# --- Blotter endpoint ---


def test_blotter_returns_expected_sections(client):
    """Blotter should return countdown, brackets, positions, and summary."""
    resp = client.get("/api/blotter/nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert "countdown" in data
    assert "brackets" in data
    assert "positions" in data
    assert "positions_summary" in data


def test_blotter_countdown_has_fields(client):
    """Countdown section should contain all expected fields."""
    resp = client.get("/api/blotter/nyc")
    data = resp.json()
    cd = data["countdown"]
    assert "event_date" in cd
    assert "model_high" in cd
    assert "model_bracket" in cd
    assert "running_obs_max" in cd
    assert "settlement" in cd
    assert "close_time" in cd


def test_blotter_settlement_structure(client):
    """Settlement within countdown should have temp and source."""
    resp = client.get("/api/blotter/nyc")
    data = resp.json()
    settlement = data["countdown"]["settlement"]
    assert "temp" in settlement
    assert "source" in settlement


def test_blotter_positions_summary_defaults(client):
    """With empty DB, positions summary should show zero defaults."""
    resp = client.get("/api/blotter/nyc")
    data = resp.json()
    assert data["positions_summary"]["open_count"] == 0
    assert data["positions_summary"]["day_pnl"] == 0
    assert data["positions_summary"]["day_exposure"] == 0


def test_blotter_brackets_has_expected_structure(client):
    """Brackets section should mirror the brackets endpoint format."""
    resp = client.get("/api/blotter/nyc")
    data = resp.json()
    brackets = data["brackets"]
    # Should have the standard bracket_spread keys
    assert "city" in brackets
    assert "brackets" in brackets
    assert "liquidity" in brackets


def test_blotter_with_date_param(client):
    """Blotter should accept an explicit date parameter."""
    resp = client.get("/api/blotter/nyc?date=2026-01-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["countdown"]["event_date"] == "2026-01-15"
