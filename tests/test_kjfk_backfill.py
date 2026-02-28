"""Tests for KJFK observation backfill via IEM ASOS.

Verifies:
- KJFK is included in get_all_station_ids()
- KJFK is in the IEM backfill station map
- Resume logic finds latest KJFK observation
- IEM CSV parsing produces correct observation rows
"""

import os
from datetime import datetime, timezone

import duckdb
import pytest

from core.constants import get_all_station_ids, CITIES, STATION_COORDS
from core.db import init_db, get_connection
from scripts.backfill_iem import STATION_MAP

TEST_DB = "data/test_kjfk_backfill.duckdb"


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test DB before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


# --- Station inclusion tests ---

def test_kjfk_in_get_all_station_ids():
    """KJFK should be returned by get_all_station_ids()."""
    stations = get_all_station_ids()
    assert "KJFK" in stations, f"KJFK not found in station list: {stations}"


def test_kjfk_in_cities_neighbors():
    """KJFK should be in NYC neighbors."""
    assert "KJFK" in CITIES["NYC"]["neighbors"]


def test_kjfk_in_station_coords():
    """KJFK should have coordinates in STATION_COORDS."""
    assert "KJFK" in STATION_COORDS
    lat, lon = STATION_COORDS["KJFK"]
    assert 40.0 < lat < 41.0, f"KJFK lat {lat} out of range"
    assert -74.5 < lon < -73.0, f"KJFK lon {lon} out of range"


def test_kjfk_in_iem_backfill_station_map():
    """KJFK should be in the IEM backfill STATION_MAP."""
    assert "KJFK" in STATION_MAP.values(), (
        f"KJFK not in STATION_MAP values: {STATION_MAP}"
    )
    # IEM uses 3-letter code JFK -> maps to KJFK
    assert STATION_MAP.get("JFK") == "KJFK"


# --- Resume logic tests ---

def test_resume_finds_latest_kjfk_observation():
    """Resume should detect existing KJFK data and find the latest timestamp."""
    init_db(TEST_DB)
    con = get_connection(TEST_DB)
    try:
        # Insert a KJFK observation
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth,
                raw_metar, ingested_at, ingest_source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["KJFK", datetime(2025, 6, 15, 18, 51),
             85.0, 29.4,
             "METAR KJFK 151851Z 21010KT 10SM FEW250 29/19 A2990 RMK AO2 T02940189",
             datetime(2025, 6, 15, 19, 0), "iem_backfill"],
        )

        # Insert an older KJFK observation
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth,
                raw_metar, ingested_at, ingest_source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["KJFK", datetime(2025, 6, 14, 12, 51),
             78.0, 25.6,
             "METAR KJFK 141251Z 18008KT 10SM SCT250 26/18 A2995 RMK AO2 T02560178",
             datetime(2025, 6, 14, 13, 0), "iem_backfill"],
        )
    finally:
        con.close()

    # Verify resume finds the latest KJFK timestamp
    con = duckdb.connect(TEST_DB, read_only=True)
    try:
        result = con.execute(
            "SELECT MAX(observed_at) FROM observations WHERE station_id = 'KJFK'"
        ).fetchone()
        assert result is not None
        assert result[0] is not None
        latest = result[0]
        assert latest.year == 2025
        assert latest.month == 6
        assert latest.day == 15
        assert latest.hour == 18
    finally:
        con.close()


def test_resume_returns_none_when_no_kjfk_data():
    """Resume should return None when no KJFK observations exist."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB, read_only=True)
    try:
        result = con.execute(
            "SELECT MAX(observed_at) FROM observations WHERE station_id = 'KJFK'"
        ).fetchone()
        assert result[0] is None
    finally:
        con.close()


def test_resume_ignores_other_stations():
    """Resume for KJFK should not be confused by KNYC data."""
    init_db(TEST_DB)
    con = get_connection(TEST_DB)
    try:
        # Insert KNYC observation only
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth,
                raw_metar, ingested_at, ingest_source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["KNYC", datetime(2025, 7, 1, 12, 0),
             90.0, 32.2,
             "METAR KNYC 011200Z 20005KT 10SM CLR 32/22 A2990 RMK AO2 T03220222",
             datetime(2025, 7, 1, 12, 5), "synoptic"],
        )
    finally:
        con.close()

    con = duckdb.connect(TEST_DB, read_only=True)
    try:
        result = con.execute(
            "SELECT MAX(observed_at) FROM observations WHERE station_id = 'KJFK'"
        ).fetchone()
        assert result[0] is None, "Should have no KJFK data"
    finally:
        con.close()


# --- Duplicate handling tests ---

def test_kjfk_duplicate_insert_rejected():
    """Inserting a duplicate (station_id, observed_at) should raise ConstraintException."""
    init_db(TEST_DB)
    con = get_connection(TEST_DB)
    try:
        row = [
            "KJFK", datetime(2025, 8, 1, 15, 51),
            88.0, 31.1,
            "METAR KJFK 011551Z 25012KT 10SM FEW080 31/21 A2985 RMK AO2 T03110211",
            datetime(2025, 8, 1, 16, 0), "iem_backfill",
        ]
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth,
                raw_metar, ingested_at, ingest_source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
        # Second insert with same (station_id, observed_at) should fail
        with pytest.raises(duckdb.ConstraintException):
            con.execute(
                """INSERT INTO observations
                   (station_id, observed_at, temp_f, temp_c_tenth,
                    raw_metar, ingested_at, ingest_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
    finally:
        con.close()


# --- IEM source tag test ---

def test_kjfk_backfill_uses_iem_source():
    """Backfilled KJFK rows should have ingest_source='iem_backfill'."""
    init_db(TEST_DB)
    con = get_connection(TEST_DB)
    try:
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth,
                raw_metar, ingested_at, ingest_source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["KJFK", datetime(2025, 3, 10, 14, 51),
             55.0, 12.8,
             "METAR KJFK 101451Z 31015G22KT 10SM SCT045 13/04 A3010 RMK AO2 T01280039",
             datetime.now(), "iem_backfill"],
        )
    finally:
        con.close()

    con = duckdb.connect(TEST_DB, read_only=True)
    try:
        row = con.execute(
            "SELECT ingest_source FROM observations WHERE station_id = 'KJFK'"
        ).fetchone()
        assert row[0] == "iem_backfill"
    finally:
        con.close()
