"""AlphaTemp database initialization and connection management."""

import duckdb
from loguru import logger

DEFAULT_DB_PATH = "data/alphatemp.duckdb"


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize DuckDB with all required tables."""
    con = duckdb.connect(db_path)
    logger.info(f"Initializing database at {db_path}")

    con.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            station_id   VARCHAR NOT NULL,
            observed_at  TIMESTAMP NOT NULL,
            temp_f       DOUBLE,
            temp_c_tenth DOUBLE,
            raw_metar    VARCHAR,
            ingested_at  TIMESTAMP NOT NULL,
            UNIQUE (station_id, observed_at)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            station_id  VARCHAR NOT NULL,
            model_run   TIMESTAMP NOT NULL,
            valid_at    TIMESTAMP NOT NULL,
            temp_f      DOUBLE,
            temp_c      DOUBLE,
            ingested_at TIMESTAMP NOT NULL,
            UNIQUE (station_id, model_run, valid_at)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS market_ticks (
            market_id   VARCHAR NOT NULL,
            city        VARCHAR NOT NULL,
            captured_at TIMESTAMP NOT NULL,
            yes_bid     DOUBLE,
            yes_ask     DOUBLE,
            no_bid      DOUBLE,
            no_ask      DOUBLE,
            last_trade  DOUBLE,
            volume      INTEGER
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS drift_signals (
            city             VARCHAR NOT NULL,
            calculated_at    TIMESTAMP NOT NULL,
            model_run        TIMESTAMP NOT NULL,
            drift_score      DOUBLE,
            slope_divergence DOUBLE,
            forecast_trend   DOUBLE,
            confidence       DOUBLE,
            projected_high   DOUBLE,
            UNIQUE (city, calculated_at, model_run)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS station_bias (
            station_id   VARCHAR NOT NULL,
            calculated_at TIMESTAMP NOT NULL,
            mean_bias    DOUBLE,
            std_error    DOUBLE,
            sample_days  INTEGER,
            UNIQUE (station_id, calculated_at)
        )
    """)

    logger.info("Database tables initialized")
    con.close()


def get_connection(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection."""
    return duckdb.connect(db_path)
