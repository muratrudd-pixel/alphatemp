import duckdb
import pytest
from datetime import datetime

from core.db import init_db
from scripts.phase1_gate_check import check_gate

TEST_DB = "data/test_gate.duckdb"


@pytest.fixture(autouse=True)
def clean_test_db():
    import os
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _seed_full_data(db_path):
    """Insert minimal data that satisfies all gate checks."""
    init_db(db_path)
    con = duckdb.connect(db_path)
    now = datetime(2026, 1, 1)

    # HRRR: all 24 run hours
    for hour in range(24):
        mr = datetime(2025, 6, 15, hour)
        va = datetime(2025, 6, 15, hour + 4 if hour < 20 else hour - 20 + 4)
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", mr, va, 75.0, 23.89, now, "hrrr", 4, False],
        )

    # GFS: 4 run hours
    for hour in [0, 6, 12, 18]:
        mr = datetime(2025, 6, 15, hour)
        va = datetime(2025, 6, 15, hour + 6 if hour < 18 else hour - 18 + 6)
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", mr, va, 72.0, 22.22, now, "gfs", 6, False],
        )

    # ECMWF: 00z and 12z
    for hour in [0, 12]:
        mr = datetime(2025, 6, 15, hour)
        va = datetime(2025, 6, 15, hour + 6)
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", mr, va, 71.0, 21.67, now, "ecmwf", 6, False],
        )

    # KJFK observation
    con.execute(
        "INSERT INTO observations (station_id, observed_at, temp_f, ingested_at, obs_type) "
        "VALUES (?, ?, ?, ?, ?)",
        ["KJFK", datetime(2025, 6, 15, 12), 78.0, now, "metar"],
    )

    # Market tick (for dedup check)
    con.execute(
        "INSERT INTO market_ticks (market_id, city, captured_at, yes_bid, yes_ask, no_bid, no_ask, last_trade, volume, floor_strike, cap_strike) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["MKT1", "NYC", now, 0.5, 0.6, 0.4, 0.5, 0.55, 100, 70.0, 72.0],
    )

    con.close()


def test_gate_passes_with_full_data():
    """Gate should pass when all data is present."""
    _seed_full_data(TEST_DB)
    assert check_gate(TEST_DB) is True


def test_gate_fails_missing_hrrr_hours():
    """Gate should fail if fewer than 24 HRRR run hours."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    now = datetime(2026, 1, 1)
    # Only 4 HRRR hours (synoptic only)
    for hour in [0, 6, 12, 18]:
        mr = datetime(2025, 6, 15, hour)
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", mr, datetime(2025, 6, 15, hour + 4), 75.0, 23.89,
             now, "hrrr", 4, False],
        )
    # Still need GFS + ECMWF + KJFK to isolate the HRRR failure
    for hour in [0, 6, 12, 18]:
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", datetime(2025, 6, 15, hour),
             datetime(2025, 6, 15, hour + 6 if hour < 18 else 0), 72.0, 22.22,
             now, "gfs", 6, False],
        )
    for hour in [0, 12]:
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", datetime(2025, 6, 15, hour),
             datetime(2025, 6, 15, hour + 6), 71.0, 21.67,
             now, "ecmwf", 6, False],
        )
    con.execute(
        "INSERT INTO observations (station_id, observed_at, temp_f, ingested_at, obs_type) "
        "VALUES (?, ?, ?, ?, ?)",
        ["KJFK", datetime(2025, 6, 15, 12), 78.0, now, "metar"],
    )
    con.execute(
        "INSERT INTO market_ticks (market_id, city, captured_at, yes_bid, yes_ask, no_bid, no_ask, last_trade, volume, floor_strike, cap_strike) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["MKT1", "NYC", now, 0.5, 0.6, 0.4, 0.5, 0.55, 100, 70.0, 72.0],
    )
    con.close()

    assert check_gate(TEST_DB) is False


def test_gate_fails_missing_kjfk():
    """Gate should fail if no KJFK observations."""
    _seed_full_data(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("DELETE FROM observations WHERE station_id = 'KJFK'")
    con.close()
    assert check_gate(TEST_DB) is False


def test_gate_fails_null_fxx():
    """Gate should fail if any forecast row has NULL fxx."""
    _seed_full_data(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 7, 1, 12), datetime(2025, 7, 1, 14),
         80.0, 26.67, datetime(2026, 1, 1), "hrrr", None, None],
    )
    con.close()
    assert check_gate(TEST_DB) is False


def test_gate_fails_missing_gfs():
    """Gate should fail if fewer than 4 GFS run hours."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    now = datetime(2026, 1, 1)
    # Full HRRR
    for hour in range(24):
        mr = datetime(2025, 6, 15, hour)
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", mr, datetime(2025, 6, 15, (hour + 4) % 24), 75.0, 23.89,
             now, "hrrr", 4, False],
        )
    # Only 1 GFS hour
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 12), datetime(2025, 6, 15, 18),
         72.0, 22.22, now, "gfs", 6, False],
    )
    # Full ECMWF
    for hour in [0, 12]:
        con.execute(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["KNYC", datetime(2025, 6, 15, hour),
             datetime(2025, 6, 15, hour + 6), 71.0, 21.67,
             now, "ecmwf", 6, False],
        )
    con.execute(
        "INSERT INTO observations (station_id, observed_at, temp_f, ingested_at, obs_type) "
        "VALUES (?, ?, ?, ?, ?)",
        ["KJFK", datetime(2025, 6, 15, 12), 78.0, now, "metar"],
    )
    con.execute(
        "INSERT INTO market_ticks (market_id, city, captured_at, yes_bid, yes_ask, no_bid, no_ask, last_trade, volume, floor_strike, cap_strike) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["MKT1", "NYC", now, 0.5, 0.6, 0.4, 0.5, 0.55, 100, 70.0, 72.0],
    )
    con.close()
    assert check_gate(TEST_DB) is False
