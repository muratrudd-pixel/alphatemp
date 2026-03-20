import os
import duckdb
import pytest
from core.db import init_db, get_connection, _migrate_forecasts_model_name

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


def test_forecasts_schema_updated():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE forecasts").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["station_id", "model_run", "valid_at", "temp_f", "temp_c", "ingested_at"]:
        assert expected in col_names, f"Missing column: {expected}"
    con.close()


def test_forecasts_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
        VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 45.0, 7.2, CURRENT_TIMESTAMP)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 46.0, 7.8, CURRENT_TIMESTAMP)
        """)
    con.close()


def test_forecasts_has_model_name_column():
    """model_name column exists and defaults to 'hrrr' when not specified."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    try:
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 45.0, 7.2, CURRENT_TIMESTAMP)
        """)
        row = con.execute("SELECT model_name FROM forecasts WHERE station_id = 'KNYC'").fetchone()
        assert row is not None
        assert row[0] == "hrrr"
    finally:
        con.close()


def test_forecasts_unique_constraint_includes_model_name():
    """Same (station_id, model_run, valid_at) with different model_name should succeed;
    full duplicate should raise ConstraintException."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    try:
        # First insert — hrrr
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 45.0, 7.2, CURRENT_TIMESTAMP, 'hrrr')
        """)
        # Second insert — different model_name, same composite key otherwise → should succeed
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 46.0, 7.8, CURRENT_TIMESTAMP, 'gfs')
        """)
        count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        assert count == 2

        # Third insert — full duplicate (same model_name too) → should fail
        with pytest.raises(duckdb.ConstraintException):
            con.execute("""
                INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
                VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 47.0, 8.3, CURRENT_TIMESTAMP, 'hrrr')
            """)
    finally:
        con.close()


def test_drift_signals_table_exists():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = {t[0] for t in tables}
    assert "drift_signals" in table_names
    con.close()


def test_drift_signals_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE drift_signals").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["city", "calculated_at", "model_run", "drift_score",
                     "slope_divergence", "forecast_trend", "confidence",
                     "projected_high"]:
        assert expected in col_names, f"Missing column: {expected}"
    con.close()


def test_station_bias_table_exists():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = {t[0] for t in tables}
    assert "station_bias" in table_names
    con.close()


def test_station_bias_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE station_bias").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["station_id", "calculated_at", "mean_bias", "std_error", "sample_days"]:
        assert expected in col_names, f"Missing column: {expected}"
    con.close()


def test_station_bias_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO station_bias (station_id, calculated_at, mean_bias, std_error, sample_days)
        VALUES ('KNYC', '2026-02-22 12:00:00', 0.8, 1.2, 85)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO station_bias (station_id, calculated_at, mean_bias, std_error, sample_days)
            VALUES ('KNYC', '2026-02-22 12:00:00', 0.5, 1.0, 90)
        """)
    con.close()


def test_drift_signals_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO drift_signals (city, calculated_at, model_run, drift_score,
            slope_divergence, forecast_trend, confidence, projected_high)
        VALUES ('NYC', '2026-02-22 12:00:00', '2026-02-22 06:00:00', 1.2,
            0.3, 0.5, 0.85, 31.3)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO drift_signals (city, calculated_at, model_run, drift_score,
                slope_divergence, forecast_trend, confidence, projected_high)
            VALUES ('NYC', '2026-02-22 12:00:00', '2026-02-22 06:00:00', 2.0,
                0.1, 0.2, 0.90, 32.7)
        """)
    con.close()


def test_forecast_extended_table_exists():
    """forecast_extended table is created by init_db."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = {t[0] for t in tables}
    assert "forecast_extended" in table_names
    con.close()


def test_forecast_extended_schema():
    """forecast_extended has all expected weather variable columns."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE forecast_extended").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["station_id", "model_run", "valid_at", "model_name",
                     "dewpoint_2m_f", "humidity_2m", "wind_speed_10m",
                     "wind_dir_10m", "wind_gusts_10m", "pressure_msl",
                     "cloud_cover", "precipitation", "shortwave_rad",
                     "cape", "ingested_at"]:
        assert expected in col_names, f"Missing column: {expected}"
    con.close()


def test_forecast_extended_unique_constraint():
    """UNIQUE on (station_id, model_run, valid_at, model_name)."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    # First insert
    con.execute("""
        INSERT INTO forecast_extended
        (station_id, model_run, valid_at, model_name, dewpoint_2m_f, ingested_at)
        VALUES ('KNYC', '2024-01-01 00:00', '2024-01-01 12:00', 'gfs', 35.0, CURRENT_TIMESTAMP)
    """)
    # Different model_name — should succeed
    con.execute("""
        INSERT INTO forecast_extended
        (station_id, model_run, valid_at, model_name, dewpoint_2m_f, ingested_at)
        VALUES ('KNYC', '2024-01-01 00:00', '2024-01-01 12:00', 'ecmwf', 34.0, CURRENT_TIMESTAMP)
    """)
    assert con.execute("SELECT COUNT(*) FROM forecast_extended").fetchone()[0] == 2
    # Full duplicate — should fail
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO forecast_extended
            (station_id, model_run, valid_at, model_name, dewpoint_2m_f, ingested_at)
            VALUES ('KNYC', '2024-01-01 00:00', '2024-01-01 12:00', 'gfs', 36.0, CURRENT_TIMESTAMP)
        """)
    con.close()


def test_paper_positions_table_exists():
    """paper_positions table should be created by init_db."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'paper_positions'"
    ).fetchall()
    con.close()
    assert len(tables) == 1


def test_indexes_created():
    """All performance indexes should exist after init_db."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    rows = con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    index_names = {r[0] for r in rows}
    con.close()

    expected = {
        "idx_obs_station_time",
        "idx_fcst_station_run",
        "idx_fcst_station_valid",
        "idx_fcst_model",
        "idx_drift_city_time",
        "idx_market_city_time",
        "idx_bias_station_time",
        "idx_fext_model_run",
        "idx_paper_positions_id",
        "idx_paper_positions_city_date",
        "idx_paper_positions_status",
    }
    assert expected.issubset(index_names), f"Missing indexes: {expected - index_names}"


def test_forecasts_has_fxx_column():
    """forecasts table should have fxx INTEGER column."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB, read_only=True)
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'forecasts' AND column_name = 'fxx'"
    ).fetchall()
    con.close()
    assert len(cols) == 1


def test_forecasts_has_is_spinup_column():
    """forecasts table should have is_spinup BOOLEAN column."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB, read_only=True)
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'forecasts' AND column_name = 'is_spinup'"
    ).fetchall()
    con.close()
    assert len(cols) == 1


def test_observations_has_obs_type_column():
    """observations table should have obs_type VARCHAR column."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB, read_only=True)
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'observations' AND column_name = 'obs_type'"
    ).fetchall()
    con.close()
    assert len(cols) == 1


def test_market_ticks_unique_constraint():
    """market_ticks should reject duplicate (market_id, captured_at) pairs."""
    from datetime import datetime
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    now = datetime(2026, 1, 1, 12, 0, 0)
    row = ["MKT1", "NYC", now, 0.5, 0.6, 0.4, 0.5, 0.55, 100, 70.0, 72.0]
    con.execute(
        "INSERT INTO market_ticks (market_id, city, captured_at, yes_bid, yes_ask, no_bid, no_ask, last_trade, volume, floor_strike, cap_strike) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row
    )
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO market_ticks (market_id, city, captured_at, yes_bid, yes_ask, no_bid, no_ask, last_trade, volume, floor_strike, cap_strike) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row
        )
    con.close()


def test_kjfk_in_station_coords():
    """KJFK should be in STATION_COORDS after constants update."""
    from core.constants import STATION_COORDS, CITIES
    assert "KJFK" in STATION_COORDS
    assert "KJFK" in CITIES["NYC"]["neighbors"]


def test_migrate_forecasts_model_name_from_old_schema():
    """Exercise the actual migration path: old schema -> new schema with model_name."""
    con = duckdb.connect(TEST_DB)
    try:
        # Create the OLD schema (no model_name, old UNIQUE constraint)
        con.execute("""
            CREATE TABLE forecasts (
                station_id  VARCHAR NOT NULL,
                model_run   TIMESTAMP NOT NULL,
                valid_at    TIMESTAMP NOT NULL,
                temp_f      DOUBLE,
                temp_c      DOUBLE,
                ingested_at TIMESTAMP NOT NULL,
                UNIQUE (station_id, model_run, valid_at)
            )
        """)
        # Insert a row into old schema
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 45.0, 7.2,
                    '2026-02-22 12:05:00')
        """)

        # Run the migration
        _migrate_forecasts_model_name(con)

        # Verify model_name column exists
        cols = con.execute("DESCRIBE forecasts").fetchall()
        col_names = {c[0] for c in cols}
        assert "model_name" in col_names, "model_name column missing after migration"

        # Verify row data preserved with model_name defaulted to 'hrrr'
        row = con.execute(
            "SELECT station_id, temp_f, temp_c, model_name FROM forecasts"
        ).fetchone()
        assert row is not None, "Row lost during migration"
        assert row[0] == "KNYC"
        assert row[1] == 45.0
        assert row[2] == 7.2
        assert row[3] == "hrrr"

        # Verify new UNIQUE constraint includes model_name:
        # same composite key with different model_name should succeed
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 46.0, 7.8,
                    '2026-02-22 12:05:00', 'gfs')
        """)
        assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 2

        # full duplicate (same model_name) should fail
        with pytest.raises(duckdb.ConstraintException):
            con.execute("""
                INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
                VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 47.0, 8.3,
                        '2026-02-22 12:05:00', 'hrrr')
            """)
    finally:
        con.close()
