import duckdb
import pytest
from datetime import datetime, date

from core.db import init_db


TEST_DB = "data/test_gfs.duckdb"


@pytest.fixture(autouse=True)
def clean_test_db():
    import os
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_backfill_gfs_creates_temp_db():
    """GFS backfill should create a temp DB and insert forecast rows."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB, read_only=True)
    count = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE model_name = 'gfs'"
    ).fetchone()[0]
    con.close()
    assert count == 0  # Empty but table exists


def test_gfs_resume_point():
    """Resume should find the latest GFS date in the temp DB."""
    from scripts.backfill_gfs_ucar import get_resume_date

    init_db(TEST_DB)

    # No data -> None
    assert get_resume_date(TEST_DB, run_hour=12) is None

    # Insert a row
    con = duckdb.connect(TEST_DB)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 12), datetime(2025, 6, 15, 13),
         75.0, 23.89, datetime(2026, 1, 1), "gfs", 1, False],
    )
    con.close()

    resume = get_resume_date(TEST_DB, run_hour=12)
    assert resume == date(2025, 6, 16)


def test_gfs_extract_inserts_with_fxx():
    """Inserted GFS rows should have fxx and is_spinup populated."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 12), datetime(2025, 6, 15, 15),
         75.0, 23.89, datetime(2026, 1, 1), "gfs", 3, False],
    )
    row = con.execute(
        "SELECT fxx, is_spinup FROM forecasts WHERE model_name = 'gfs'"
    ).fetchone()
    con.close()
    assert row == (3, False)
