import os
import duckdb
import pytest
from core.db import init_db, get_connection

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test DB before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_init_db_creates_tables():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = {t[0] for t in tables}
    assert "observations" in table_names
    assert "forecasts" in table_names
    assert "market_ticks" in table_names
    con.close()


def test_observations_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE observations").fetchall()
    col_names = {c[0] for c in cols}
    assert "station_id" in col_names
    assert "observed_at" in col_names
    assert "temp_f" in col_names
    assert "temp_c_tenth" in col_names
    assert "raw_metar" in col_names
    assert "ingested_at" in col_names
    con.close()


def test_observations_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
        VALUES ('KNYC', '2026-02-22 12:00:00', 45.0, 7.2, 'test', CURRENT_TIMESTAMP)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
            VALUES ('KNYC', '2026-02-22 12:00:00', 45.0, 7.2, 'test', CURRENT_TIMESTAMP)
        """)
    con.close()


def test_market_ticks_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE market_ticks").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["market_id", "city", "captured_at", "yes_bid", "yes_ask",
                     "no_bid", "no_ask", "last_trade", "volume"]:
        assert expected in col_names
    con.close()


def test_get_connection():
    init_db(TEST_DB)
    con = get_connection(TEST_DB)
    assert con is not None
    result = con.execute("SELECT 1").fetchone()
    assert result[0] == 1
    con.close()
