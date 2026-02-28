import duckdb
import pytest
from datetime import date, datetime
from core.db import init_db

TEST_DB = "data/test_ecmwf.duckdb"


@pytest.fixture(autouse=True)
def clean_test_db():
    import os
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_ecmwf_run_config():
    """ECMWF run hours should have correct horizon."""
    from scripts.backfill_ecmwf import get_fxx_range
    # Full runs: 0-72h
    assert get_fxx_range(0) == range(0, 73)
    assert get_fxx_range(12) == range(0, 73)
    # Short-cutoff: 0-30h
    assert get_fxx_range(6) == range(0, 31)
    assert get_fxx_range(18) == range(0, 31)


def test_ecmwf_resume(tmp_path):
    """Resume should work per run_hour."""
    from scripts.backfill_ecmwf import get_resume_date
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    assert get_resume_date(db_path, 12) is None

    con = duckdb.connect(db_path)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2024, 3, 1, 12), datetime(2024, 3, 1, 15),
         65.0, 18.33, datetime(2026, 1, 1), "ecmwf", 3, False],
    )
    con.close()
    assert get_resume_date(db_path, 12) == date(2024, 3, 2)
