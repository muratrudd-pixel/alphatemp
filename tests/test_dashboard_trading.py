"""Tests for the Trading tab API endpoints."""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.db import get_connection

TEST_DB = "data/test_trading.duckdb"


def _init_trading_tables(db_path):
    """Create only the tables needed for trading endpoints."""
    import duckdb

    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            id INTEGER,
            city VARCHAR,
            event_date DATE,
            bracket_floor INTEGER,
            bracket_cap INTEGER,
            direction VARCHAR,
            model_prob DOUBLE,
            market_price DOUBLE,
            edge DOUBLE,
            entry_price DOUBLE,
            entry_time TIMESTAMP,
            exit_price DOUBLE,
            exit_time TIMESTAMP,
            settled_yes BOOLEAN,
            gross_pnl DOUBLE,
            fees DOUBLE,
            net_pnl DOUBLE,
            status VARCHAR DEFAULT 'open',
            exit_reason VARCHAR,
            contracts INTEGER DEFAULT 1,
            unrealized_pnl DOUBLE DEFAULT 0.0
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_config (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        INSERT INTO paper_config (key, value) VALUES
        ('max_daily_loss_cents', '-1000'),
        ('max_open_positions', '5'),
        ('max_per_bracket', '2'),
        ('min_edge_pct', '5.0'),
        ('cooldown_minutes', '30'),
        ('kill_switch', 'False')
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS market_ticks (
            market_id    VARCHAR NOT NULL,
            city         VARCHAR NOT NULL,
            captured_at  TIMESTAMP NOT NULL,
            yes_bid      DOUBLE,
            yes_ask      DOUBLE,
            no_bid       DOUBLE,
            no_ask       DOUBLE,
            last_trade   DOUBLE,
            volume       INTEGER,
            floor_strike DOUBLE,
            cap_strike   DOUBLE,
            open_interest BIGINT,
            liquidity    BIGINT,
            volume_24h   BIGINT
        )
    """)

    # Minimal tables so imports don't crash
    con.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            station_id VARCHAR, observed_at TIMESTAMP, temp_f DOUBLE,
            temp_c_tenth DOUBLE, raw_metar VARCHAR, ingested_at TIMESTAMP,
            six_hr_max_c DOUBLE, six_hr_min_c DOUBLE, ingest_source VARCHAR,
            obs_type VARCHAR DEFAULT 'metar'
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            station_id VARCHAR, model_run TIMESTAMP, valid_at TIMESTAMP,
            temp_f DOUBLE, temp_c DOUBLE, ingested_at TIMESTAMP,
            model_name VARCHAR DEFAULT 'hrrr', fxx INTEGER,
            is_spinup BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS drift_signals (
            city VARCHAR, calculated_at TIMESTAMP, model_run TIMESTAMP,
            drift_score DOUBLE, slope_divergence DOUBLE, forecast_trend DOUBLE,
            confidence DOUBLE, projected_high DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS station_bias (
            station_id VARCHAR, calculated_at TIMESTAMP, mean_bias DOUBLE,
            std_error DOUBLE, sample_days INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS nws_daily (
            station_id VARCHAR, obs_date DATE, max_temp_f DOUBLE,
            min_temp_f DOUBLE, source VARCHAR DEFAULT 'ACIS',
            ingested_at TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS model_state (
            city VARCHAR NOT NULL,
            target_date DATE NOT NULL,
            update_hour INTEGER,
            bracket_probs VARCHAR,
            fcst_high DOUBLE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (city, target_date)
        )
    """)

    con.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create test DB with trading tables, patch dashboard, clean up."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    _init_trading_tables(TEST_DB)

    with patch("ui.web_dashboard.get_connection", lambda: get_connection(TEST_DB)), \
         patch("ui.web_dashboard.init_db", lambda: None):
        from ui.web_dashboard import app
        yield app
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def client(setup_test_db):
    return TestClient(setup_test_db)


# ---------------------------------------------------------------------------
# GET /api/trading/positions
# ---------------------------------------------------------------------------


def test_trading_positions_empty(client):
    """GET /api/trading/positions returns empty list with no positions."""
    resp = client.get("/api/trading/positions?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert "positions" in data
    assert data["positions"] == []


def test_trading_positions_returns_open(client):
    """GET /api/trading/positions returns open positions with expected fields."""
    con = get_connection(TEST_DB)
    con.execute("""
        INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             model_prob, market_price, edge, entry_price, entry_time,
             status, contracts, unrealized_pnl)
        VALUES
            (1, 'NYC', '2026-03-11', 72, 74, 'YES',
             0.38, 30.0, 0.08, 30.0, '2026-03-11 14:00:00',
             'open', 1, 5.0)
    """)
    con.execute("""
        INSERT INTO market_ticks
            (market_id, city, captured_at, yes_bid, yes_ask,
             floor_strike, cap_strike, volume)
        VALUES
            ('MKT-72-74', 'NYC', '2026-03-11 16:00:00', 35.0, 37.0,
             72.0, 74.0, 100)
    """)
    con.close()

    resp = client.get("/api/trading/positions?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["positions"]) == 1
    pos = data["positions"][0]
    assert pos["id"] == 1
    assert pos["bracket"] == "72-74"
    assert pos["direction"] == "YES"
    assert pos["entry"] == 30.0
    assert pos["current_bid"] == 35.0
    assert pos["current_ask"] == 37.0
    assert pos["unrealized"] == 5.0


def test_trading_positions_excludes_closed(client):
    """GET /api/trading/positions does not return closed positions."""
    con = get_connection(TEST_DB)
    con.execute("""
        INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, status, contracts, net_pnl)
        VALUES
            (1, 'NYC', '2026-03-11', 72, 74, 'YES',
             30.0, '2026-03-11 14:00:00', 'closed', 1, 10.0)
    """)
    con.close()

    resp = client.get("/api/trading/positions?city=nyc")
    assert resp.status_code == 200
    assert resp.json()["positions"] == []


# ---------------------------------------------------------------------------
# GET /api/trading/pnl
# ---------------------------------------------------------------------------


def test_trading_pnl_empty(client):
    """GET /api/trading/pnl returns zeros with no positions."""
    resp = client.get("/api/trading/pnl?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["realized"] == 0
    assert data["unrealized"] == 0
    assert data["total"] == 0
    assert "daily" in data


def test_trading_pnl_with_positions(client):
    """GET /api/trading/pnl returns correct realized/unrealized totals."""
    con = get_connection(TEST_DB)
    # Closed position with realized P&L
    con.execute("""
        INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             status, contracts, net_pnl, unrealized_pnl)
        VALUES
            (1, 'NYC', '2026-03-10', 70, 72, 'YES',
             25.0, '2026-03-10 14:00:00', 50.0, '2026-03-10 20:00:00',
             'closed', 1, 200.0, 0.0)
    """)
    # Open position with unrealized P&L
    con.execute("""
        INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time,
             status, contracts, net_pnl, unrealized_pnl)
        VALUES
            (2, 'NYC', '2026-03-11', 72, 74, 'YES',
             30.0, '2026-03-11 14:00:00',
             'open', 1, 0.0, 50.0)
    """)
    con.close()

    resp = client.get("/api/trading/pnl?city=nyc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["realized"] == 200.0
    assert data["unrealized"] == 50.0
    assert data["total"] == 250.0
    assert len(data["daily"]) >= 1


def test_trading_pnl_daily_breakdown(client):
    """GET /api/trading/pnl daily array has date, realized, unrealized, trades, wins."""
    con = get_connection(TEST_DB)
    con.execute("""
        INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             status, contracts, net_pnl, unrealized_pnl, settled_yes)
        VALUES
            (1, 'NYC', '2026-03-10', 70, 72, 'YES',
             25.0, '2026-03-10 14:00:00', 100.0, '2026-03-10 20:00:00',
             'closed', 1, 200.0, 0.0, TRUE)
    """)
    con.close()

    resp = client.get("/api/trading/pnl?city=nyc")
    data = resp.json()
    assert len(data["daily"]) == 1
    day = data["daily"][0]
    assert "date" in day
    assert "realized" in day
    assert "unrealized" in day
    assert "trades" in day
    assert "wins" in day
    assert day["wins"] == 1


# ---------------------------------------------------------------------------
# GET /api/trading/breakers
# ---------------------------------------------------------------------------


def test_trading_breakers_returns_status(client):
    """GET /api/trading/breakers returns circuit breaker status."""
    resp = client.get("/api/trading/breakers")
    assert resp.status_code == 200
    data = resp.json()
    breakers = data["breakers"]
    assert "kill_switch" in breakers
    assert breakers["kill_switch"] is False
    assert "max_daily_loss" in breakers
    assert breakers["max_daily_loss"]["threshold"] == -1000
    assert "max_open" in breakers
    assert breakers["max_open"]["threshold"] == 5
    assert "min_edge_pct" in breakers
    assert "cooldown_minutes" in breakers


def test_trading_breakers_reflects_positions(client):
    """GET /api/trading/breakers current counts reflect DB state."""
    con = get_connection(TEST_DB)
    con.execute("""
        INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, status, contracts, net_pnl)
        VALUES
            (1, 'NYC', '2026-03-11', 72, 74, 'YES',
             30.0, '2026-03-11 14:00:00', 'open', 1, 0.0),
            (2, 'NYC', '2026-03-11', 74, 76, 'NO',
             40.0, '2026-03-11 15:00:00', 'open', 1, 0.0)
    """)
    con.close()

    resp = client.get("/api/trading/breakers")
    data = resp.json()
    assert data["breakers"]["max_open"]["current"] == 2


# ---------------------------------------------------------------------------
# POST /api/trading/kill-switch
# ---------------------------------------------------------------------------


def test_kill_switch_toggle_on(client):
    """POST /api/trading/kill-switch enables the kill switch."""
    resp = client.post("/api/trading/kill-switch", json={"enabled": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kill_switch"] is True

    # Verify it persisted
    resp2 = client.get("/api/trading/breakers")
    assert resp2.json()["breakers"]["kill_switch"] is True


def test_kill_switch_toggle_off(client):
    """POST /api/trading/kill-switch disables the kill switch."""
    # Enable first
    client.post("/api/trading/kill-switch", json={"enabled": True})
    # Then disable
    resp = client.post("/api/trading/kill-switch", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["kill_switch"] is False


def test_kill_switch_missing_field(client):
    """POST /api/trading/kill-switch with missing field returns 422."""
    resp = client.post("/api/trading/kill-switch", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /trading (page route)
# ---------------------------------------------------------------------------


def test_trading_page_returns_html(client):
    """Trading page should return HTML."""
    resp = client.get("/trading")
    assert resp.status_code == 200
    assert b"Trading" in resp.content
