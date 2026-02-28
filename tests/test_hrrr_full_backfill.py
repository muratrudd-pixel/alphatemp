import duckdb
import pytest
from datetime import datetime, date
from core.db import init_db

TEST_DB = "data/test_hrrr_full.duckdb"

@pytest.fixture(autouse=True)
def clean_test_db():
    import os
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

def test_extended_run_hours():
    """Extended runs (00z, 06z, 12z, 18z) should use fxx 1-48."""
    from scripts.backfill_hrrr_full import get_fxx_range
    assert get_fxx_range(0) == range(1, 49)
    assert get_fxx_range(6) == range(1, 49)
    assert get_fxx_range(12) == range(1, 49)
    assert get_fxx_range(18) == range(1, 49)

def test_standard_run_hours():
    """Standard runs should use fxx 1-18."""
    from scripts.backfill_hrrr_full import get_fxx_range
    assert get_fxx_range(1) == range(1, 19)
    assert get_fxx_range(7) == range(1, 19)
    assert get_fxx_range(23) == range(1, 19)

def test_spinup_flag():
    """fxx <= 3 should be flagged as spin-up."""
    from scripts.backfill_hrrr_full import is_spinup
    assert is_spinup(1) is True
    assert is_spinup(2) is True
    assert is_spinup(3) is True
    assert is_spinup(4) is False
    assert is_spinup(12) is False

def test_resume_per_run_hour(tmp_path):
    """Resume should be independent per run_hour."""
    from scripts.backfill_hrrr_full import get_resume_date
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 6), datetime(2025, 6, 15, 7),
         75.0, 23.89, datetime(2026, 1, 1), "hrrr", 1, True],
    )
    con.close()
    assert get_resume_date(db_path, run_hour=6) == date(2025, 6, 16)
    assert get_resume_date(db_path, run_hour=12) is None
