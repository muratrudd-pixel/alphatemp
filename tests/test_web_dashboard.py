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
        engine.provider.db_path = TEST_DB
        engine.provider._bias_cache = {}
        engine.provider._cache_loaded_at = datetime.now(timezone.utc)
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
    assert data["runs"] == []
    assert data["forecast_high"] is None


def test_empty_market(client):
    """Empty market should return available=False."""
    resp = client.get("/api/market/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


# --- Root redirect and tab pages ---


def test_root_redirects_to_operations(client):
    """Root should redirect to /operations."""
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert "/operations" in resp.headers["location"]


def test_operations_page(client):
    """Operations tab should return HTML with 'Operations' in content."""
    resp = client.get("/operations")
    assert resp.status_code == 200
    assert b"Operations" in resp.content


def test_performance_page(client):
    """Performance tab should return HTML with 'Performance' in content."""
    resp = client.get("/performance")
    assert resp.status_code == 200
    assert b"Performance" in resp.content


def test_review_page(client):
    """Review tab should return HTML with 'Review' in content."""
    resp = client.get("/review")
    assert resp.status_code == 200
    assert b"Review" in resp.content


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
    # Compute fxx from hour difference for settlement-day filtering
    fxx = int((valid_at - model_run).total_seconds() // 3600)
    con = get_connection(db_path)
    con.execute(
        """INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, fxx, ingested_at)
           VALUES (?, ?, ?, ?, NULL, ?, ?)""",
        [station_id, model_run, valid_at, temp_f, fxx, valid_at],
    )
    con.close()


def test_observations_settlement_only(client):
    """Observation feed should only return the settlement station, not neighbors."""
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
    assert stations_returned == {"KNYC"}
    assert len(data["observations"]) == 1


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


# --- Settlement-only obs in forecast_curve ---


def test_forecast_curve_obs_settlement_only(client):
    """forecast_curve observations should only include the settlement station."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=2)
    valid = now + timedelta(hours=1)

    _insert_forecast(TEST_DB, "KNYC", model_run, valid, 33.0)

    base = now - timedelta(hours=1)
    _insert_obs(TEST_DB, "KNYC", base, 30.0)
    _insert_obs(TEST_DB, "KLGA", base + timedelta(minutes=1), 31.0)
    _insert_obs(TEST_DB, "KEWR", base + timedelta(minutes=2), 29.5)

    today = now.strftime("%Y-%m-%d")
    resp = client.get(f"/api/forecast-curve/NYC?date={today}")
    data = resp.json()

    # Only KNYC obs should appear — neighbors excluded
    assert len(data["observations"]) == 1
    assert data["observations"][0]["temp_f"] == 30.0


# --- NWS Settlement source ---


def _insert_nws_daily(db_path, station_id, obs_date, max_temp_f, min_temp_f):
    """Helper to insert a single NWS daily row."""
    con = get_connection(db_path)
    con.execute(
        """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
           VALUES (?, ?, ?, ?, 'ACIS', CURRENT_TIMESTAMP)""",
        [station_id, obs_date, max_temp_f, min_temp_f],
    )
    con.close()


def test_forecast_curve_prefers_nws_over_obs(client):
    """When NWS daily data exists, it should be used over running obs max."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=2)
    valid = now + timedelta(hours=1)
    today = now.strftime("%Y-%m-%d")

    _insert_forecast(TEST_DB, "KNYC", model_run, valid, 40.0)

    # Obs running high = 38°F
    base = now - timedelta(hours=1)
    _insert_obs(TEST_DB, "KNYC", base, 38.0)

    # NWS daily high = 42°F (should win)
    _insert_nws_daily(TEST_DB, "KNYC", today, 42.0, 28.0)

    resp = client.get(f"/api/forecast-curve/NYC?date={today}")
    data = resp.json()

    assert data["observed_high"] == 42.0
    assert data["settlement_source"] == "nws_cli"


def test_forecast_curve_falls_back_to_obs(client):
    """When no NWS data exists, fall back to running obs max."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=2)
    valid = now + timedelta(hours=1)
    today = now.strftime("%Y-%m-%d")

    _insert_forecast(TEST_DB, "KNYC", model_run, valid, 40.0)

    # Only obs data, no NWS
    base = now - timedelta(hours=1)
    _insert_obs(TEST_DB, "KNYC", base, 38.0)

    resp = client.get(f"/api/forecast-curve/NYC?date={today}")
    data = resp.json()

    assert data["observed_high"] == 38.0
    assert data["settlement_source"] == "obs_running"


def test_forecast_curve_includes_settlement_source(client):
    """forecast_curve response should include settlement_source field."""
    now = datetime.now(timezone.utc)
    model_run = now - timedelta(hours=1)
    valid = now + timedelta(hours=1)

    _insert_forecast(TEST_DB, "KNYC", model_run, valid, 33.0)

    resp = client.get("/api/forecast-curve/NYC")
    data = resp.json()

    assert "settlement_source" in data


# --- Brackets endpoint ---


def test_brackets_endpoint(client):
    """GET /api/brackets/NYC should return bracket comparison data."""
    resp = client.get("/api/brackets/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert "brackets" in data
    assert "city" in data
    assert "liquidity" in data


def test_brackets_unknown_city(client):
    """Unknown city should return error for brackets endpoint."""
    resp = client.get("/api/brackets/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


# --- Positions endpoint ---


def test_positions_endpoint(client):
    """GET /api/positions/NYC should return positions data with expected fields."""
    resp = client.get("/api/positions/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert "active" in data
    assert "near_misses" in data
    assert "daily_pnl" in data
    assert "total_pnl" in data
    assert data["city"] == "NYC"


def test_positions_empty_table(client):
    """With no paper_positions rows, should return empty lists and zero P&L."""
    resp = client.get("/api/positions/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] == []
    assert isinstance(data["near_misses"], list)
    assert data["daily_pnl"] == 0.0
    assert data["total_pnl"] == 0.0


def test_positions_unknown_city(client):
    """Unknown city should return error for positions endpoint."""
    resp = client.get("/api/positions/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_positions_with_date_param(client):
    """Positions endpoint should accept a date param without crashing."""
    resp = client.get("/api/positions/NYC?date=2026-02-22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"
    assert data["date"] == "2026-02-22"


def _insert_paper_position(db_path, city, event_date, floor, cap, direction,
                           status="open", net_pnl=None, entry_price=0.30,
                           model_prob=0.42, market_price=0.29, edge=0.13):
    """Helper to insert a paper position row."""
    con = get_connection(db_path)
    # Get next ID
    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM paper_positions").fetchone()[0]
    con.execute(
        """INSERT INTO paper_positions
           (id, city, event_date, bracket_floor, bracket_cap, direction,
            model_prob, market_price, edge, entry_price, entry_time, status, net_pnl)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)""",
        [max_id + 1, city, event_date, floor, cap, direction,
         model_prob, market_price, edge, entry_price, status, net_pnl],
    )
    con.close()


def test_positions_active_positions(client):
    """Active positions should be returned for the correct date."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES")
    _insert_paper_position(TEST_DB, "NYC", today, 44, 46, "NO")

    resp = client.get(f"/api/positions/NYC?date={today}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["active"]) == 2
    assert data["active"][0]["direction"] == "YES"
    assert data["active"][0]["bracket"] == "42-44°F"


def test_positions_pnl(client):
    """P&L should sum settled positions correctly."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=5.50)
    _insert_paper_position(TEST_DB, "NYC", today, 44, 46, "NO",
                           status="settled", net_pnl=-1.30)
    # Different date — should only count in total, not daily
    _insert_paper_position(TEST_DB, "NYC", "2026-01-01", 40, 42, "YES",
                           status="settled", net_pnl=10.00)

    resp = client.get(f"/api/positions/NYC?date={today}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["daily_pnl"] == 4.20
    assert data["total_pnl"] == 14.20


# --- Market swings endpoint ---


def test_market_swings_endpoint(client):
    """GET /api/market-swings/NYC should return swings data."""
    resp = client.get("/api/market-swings/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert "swings" in data
    assert data["city"] == "NYC"


def test_market_swings_empty(client):
    """With no market_ticks, should return empty swings list."""
    resp = client.get("/api/market-swings/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["swings"] == []


def test_market_swings_unknown_city(client):
    """Unknown city should return error for market-swings endpoint."""
    resp = client.get("/api/market-swings/ZZZZ")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_market_swings_with_date_param(client):
    """Market swings endpoint should accept a date param without crashing."""
    resp = client.get("/api/market-swings/NYC?date=2026-02-22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"
    assert data["date"] == "2026-02-22"


def _insert_market_tick(db_path, city, market_id, floor_strike, cap_strike,
                        yes_bid, yes_ask, captured_at):
    """Helper to insert a market tick row."""
    con = get_connection(db_path)
    con.execute(
        """INSERT INTO market_ticks
           (market_id, city, captured_at, yes_bid, yes_ask, floor_strike, cap_strike)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [market_id, city, _strip_tz(captured_at), yes_bid, yes_ask,
         floor_strike, cap_strike],
    )
    con.close()


def test_market_swings_detects_swing(client):
    """Should detect a >10c swing within 2 hours."""
    now = datetime.now(timezone.utc)
    base = now.replace(hour=14, minute=0, second=0, microsecond=0)
    today = base.strftime("%Y-%m-%d")

    # Two ticks 30 min apart, mid changes from 0.20 to 0.35 (15c swing)
    _insert_market_tick(TEST_DB, "NYC", "KXHIGHNY-26FEB27-T42-T44",
                        42, 44, 0.18, 0.22, base)
    _insert_market_tick(TEST_DB, "NYC", "KXHIGHNY-26FEB27-T42-T44",
                        42, 44, 0.33, 0.37, base + timedelta(minutes=30))

    resp = client.get(f"/api/market-swings/NYC?date={today}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["swings"]) == 1
    assert data["swings"][0]["bracket"] == "42-44°F"
    assert abs(data["swings"][0]["change"] - 0.15) < 0.01


# --- Performance endpoint ---


def test_performance_endpoint(client):
    """GET /api/performance should return P&L data with expected fields."""
    resp = client.get("/api/performance?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "cumulative_pnl" in data
    assert "daily_bars" in data
    assert "win_rate" in data
    assert "total_trades" in data
    assert "by_bracket" in data
    assert "by_edge" in data
    assert data["range"] == "30d"


def test_performance_empty_table(client):
    """With no settled paper_positions, should return zeros and empty arrays."""
    resp = client.get("/api/performance?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cumulative_pnl"] == []
    assert data["daily_bars"] == []
    assert data["win_rate"] == 0.0
    assert data["total_trades"] == 0
    assert data["total_gross"] == 0.0
    assert data["total_fees"] == 0.0
    assert data["total_net"] == 0.0
    assert data["by_bracket"] == []
    assert data["by_edge"] == []


def test_performance_with_data(client):
    """Performance endpoint should compute correct P&L from settled positions."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=5.50,
                           model_prob=0.42, market_price=0.29, edge=0.13)
    _insert_paper_position(TEST_DB, "NYC", today, 44, 46, "NO",
                           status="settled", net_pnl=-1.30,
                           model_prob=0.35, market_price=0.29, edge=0.06)

    resp = client.get("/api/performance?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_trades"] == 2
    assert data["win_rate"] == 0.5
    assert len(data["daily_bars"]) == 1
    assert len(data["cumulative_pnl"]) == 1
    assert len(data["by_bracket"]) == 2


def test_performance_range_all(client):
    """Performance endpoint should accept range=all."""
    resp = client.get("/api/performance?range=all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["range"] == "all"


# --- Brier comparison endpoint ---


def test_brier_comparison_endpoint(client):
    """GET /api/brier-comparison should return stub with expected fields."""
    resp = client.get("/api/brier-comparison?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "by_hour" in data
    assert "note" in data
    assert data["range"] == "30d"
    assert isinstance(data["by_hour"], list)


# --- Edge heatmap endpoint ---


def test_edge_heatmap_endpoint(client):
    """GET /api/edge-heatmap should return cells data."""
    resp = client.get("/api/edge-heatmap?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "cells" in data
    assert data["range"] == "30d"


def test_edge_heatmap_empty_table(client):
    """With no settled positions, should return empty cells list."""
    resp = client.get("/api/edge-heatmap?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cells"] == []


def test_edge_heatmap_with_data(client):
    """Edge heatmap should return cell data for settled positions."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=5.50,
                           model_prob=0.42, market_price=0.29, edge=0.13)

    resp = client.get("/api/edge-heatmap?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["cells"]) >= 1
    cell = data["cells"][0]
    assert "bracket" in cell
    assert "hour_et" in cell
    assert "avg_edge" in cell
    assert "trades" in cell
    assert "win_rate" in cell


# --- Review incidents endpoint ---


def test_review_incidents_endpoint(client):
    """GET /api/review/incidents should return incidents data with expected fields."""
    resp = client.get("/api/review/incidents?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "incidents" in data
    assert data["range"] == "30d"
    assert data["filter"] == "all"


def test_review_incidents_empty_table(client):
    """With no settled paper_positions, should return empty incidents list."""
    resp = client.get("/api/review/incidents?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["incidents"] == []


def test_review_incidents_with_lost_bet(client):
    """Should generate an incident card for a lost bet (net_pnl < 0)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=-3.20,
                           model_prob=0.42, market_price=0.29, edge=0.13)
    # Insert NWS settlement temp so categorization works
    _insert_nws_daily(TEST_DB, "KNYC", today, 47.0, 28.0)

    resp = client.get("/api/review/incidents?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["incidents"]) == 1
    inc = data["incidents"][0]
    assert inc["type"] == "lost_bet"
    assert inc["net_pnl"] == -3.20
    assert inc["settlement_temp"] == 47.0
    assert inc["category"] == "model_miss"
    assert "narrative" in inc
    assert inc["bracket"] == "42-44°F"


def test_review_incidents_filter_worst(client):
    """Filter 'worst' should only return incidents with severity >= 0.5."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # High severity (large loss)
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=-8.00,
                           model_prob=0.42, market_price=0.29, edge=0.13)
    # Low severity (small loss)
    _insert_paper_position(TEST_DB, "NYC", today, 44, 46, "NO",
                           status="settled", net_pnl=-0.10,
                           model_prob=0.35, market_price=0.29, edge=0.06)

    resp = client.get("/api/review/incidents?range=30d&filter=worst")
    assert resp.status_code == 200
    data = resp.json()
    # Only the high-severity incident should pass
    for inc in data["incidents"]:
        assert inc["severity"] >= 0.5


def test_review_incidents_winning_bets_excluded(client):
    """Winning bets (net_pnl >= 0) should not appear as incidents."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=5.50,
                           model_prob=0.42, market_price=0.29, edge=0.13)

    resp = client.get("/api/review/incidents?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["incidents"] == []


def test_review_incidents_range_all(client):
    """Range 'all' should accept and return correctly."""
    resp = client.get("/api/review/incidents?range=all")
    assert resp.status_code == 200
    data = resp.json()
    assert data["range"] == "all"


# --- Review patterns endpoint ---


def test_review_patterns_endpoint(client):
    """GET /api/review/patterns should return patterns data with expected fields."""
    resp = client.get("/api/review/patterns?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert "patterns" in data
    assert data["range"] == "30d"


def test_review_patterns_empty_table(client):
    """With no incidents, should return empty patterns list."""
    resp = client.get("/api/review/patterns?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["patterns"] == []


def test_review_patterns_groups_by_category(client):
    """Should aggregate incidents by category with correct counts and P&L."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    # Two model_miss incidents
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=-3.20,
                           model_prob=0.42, market_price=0.29, edge=0.13)
    _insert_paper_position(TEST_DB, "NYC", yesterday, 44, 46, "NO",
                           status="settled", net_pnl=-2.10,
                           model_prob=0.38, market_price=0.29, edge=0.09)
    # NWS data for categorization
    _insert_nws_daily(TEST_DB, "KNYC", today, 47.0, 28.0)
    _insert_nws_daily(TEST_DB, "KNYC", yesterday, 45.0, 27.0)

    resp = client.get("/api/review/patterns?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["patterns"]) >= 1
    # Each pattern should have required fields
    for p in data["patterns"]:
        assert "category" in p
        assert "count" in p
        assert "total_pnl" in p
        assert "suggested_action" in p


def test_review_patterns_includes_suggested_action(client):
    """Each pattern should map to a known suggested action."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _insert_paper_position(TEST_DB, "NYC", today, 42, 44, "YES",
                           status="settled", net_pnl=-3.20,
                           model_prob=0.42, market_price=0.29, edge=0.13)
    _insert_nws_daily(TEST_DB, "KNYC", today, 47.0, 28.0)

    resp = client.get("/api/review/patterns?range=30d")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["patterns"]) >= 1
    # model_miss should map to its known action
    model_miss_patterns = [p for p in data["patterns"] if p["category"] == "model_miss"]
    if model_miss_patterns:
        assert "backtester" in model_miss_patterns[0]["suggested_action"].lower()
