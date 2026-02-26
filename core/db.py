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
            cap_strike   DOUBLE
        )
    """)

    # Migration: add floor_strike/cap_strike to existing tables missing them
    for col in ("floor_strike", "cap_strike"):
        try:
            con.execute(f"ALTER TABLE market_ticks ADD COLUMN {col} DOUBLE")
        except Exception:
            pass  # Column already exists

    # Migration: add 6-hour synoptic max/min columns to observations
    for col in ("six_hr_max_c", "six_hr_min_c"):
        try:
            con.execute(f"ALTER TABLE observations ADD COLUMN {col} DOUBLE")
        except Exception:
            pass  # Column already exists

    # Migration: add ingest_source to track which API inserted each row
    try:
        con.execute("ALTER TABLE observations ADD COLUMN ingest_source VARCHAR")
    except Exception:
        pass

    # One-time backfill: parse raw_metar for existing rows to populate 6-hour columns
    _backfill_6h_columns(con)

    # One-time backfill: tag existing rows with ingest_source based on heuristics
    _backfill_ingest_source(con)

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

    con.execute("""
        CREATE TABLE IF NOT EXISTS nws_daily (
            station_id   VARCHAR NOT NULL,
            obs_date     DATE NOT NULL,
            max_temp_f   DOUBLE,
            min_temp_f   DOUBLE,
            source       VARCHAR DEFAULT 'ACIS',
            ingested_at  TIMESTAMP NOT NULL,
            UNIQUE (station_id, obs_date)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_settlements (
            market_ticker   VARCHAR NOT NULL,
            event_ticker    VARCHAR NOT NULL,
            series_ticker   VARCHAR NOT NULL,
            city            VARCHAR NOT NULL,
            measure         VARCHAR NOT NULL,
            event_date      DATE,
            floor_strike    DOUBLE,
            cap_strike      DOUBLE,
            settled_yes     INTEGER,
            volume          INTEGER,
            close_time      TIMESTAMP,
            ingested_at     TIMESTAMP NOT NULL,
            UNIQUE(market_ticker)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_candlesticks (
            market_ticker   VARCHAR NOT NULL,
            end_period_ts   TIMESTAMP NOT NULL,
            period_minutes  INTEGER NOT NULL,
            yes_bid_close   DOUBLE,
            yes_ask_close   DOUBLE,
            price_open      DOUBLE,
            price_high      DOUBLE,
            price_low       DOUBLE,
            price_close     DOUBLE,
            volume          INTEGER,
            open_interest   INTEGER,
            UNIQUE(market_ticker, end_period_ts)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_trades (
            trade_id        VARCHAR NOT NULL,
            market_ticker   VARCHAR NOT NULL,
            yes_price       INTEGER,
            no_price        INTEGER,
            count           INTEGER,
            taker_side      VARCHAR,
            created_time    TIMESTAMP NOT NULL,
            ingested_at     TIMESTAMP NOT NULL,
            UNIQUE(trade_id)
        )
    """)

    # Indexes — accelerate the most common query patterns
    for stmt in [
        "CREATE INDEX IF NOT EXISTS idx_obs_station_time ON observations (station_id, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_fcst_station_run ON forecasts (station_id, model_run)",
        "CREATE INDEX IF NOT EXISTS idx_fcst_station_valid ON forecasts (station_id, valid_at)",
        "CREATE INDEX IF NOT EXISTS idx_drift_city_time ON drift_signals (city, calculated_at)",
        "CREATE INDEX IF NOT EXISTS idx_market_city_time ON market_ticks (city, captured_at)",
        "CREATE INDEX IF NOT EXISTS idx_bias_station_time ON station_bias (station_id, calculated_at)",
        "CREATE INDEX IF NOT EXISTS idx_nws_station_date ON nws_daily (station_id, obs_date)",
        "CREATE INDEX IF NOT EXISTS idx_ks_series ON kalshi_settlements (series_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_ks_date ON kalshi_settlements (event_date)",
        "CREATE INDEX IF NOT EXISTS idx_kc_ticker ON kalshi_candlesticks (market_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_kt_ticker ON kalshi_trades (market_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_kt_time ON kalshi_trades (created_time)",
    ]:
        con.execute(stmt)

    logger.info("Database tables initialized")
    con.close()


def _backfill_6h_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Parse raw_metar for existing rows to populate six_hr_max_c/min_c.

    Only runs once — skips if any row already has a non-NULL value.
    """
    already_done = con.execute(
        "SELECT COUNT(*) FROM observations WHERE six_hr_max_c IS NOT NULL OR six_hr_min_c IS NOT NULL"
    ).fetchone()[0]
    if already_done > 0:
        return

    from services.ingestor import parse_6h_max, parse_6h_min

    rows = con.execute(
        "SELECT rowid, raw_metar FROM observations WHERE raw_metar IS NOT NULL"
    ).fetchall()
    updated = 0
    for rowid, metar in rows:
        max_c = parse_6h_max(metar)
        min_c = parse_6h_min(metar)
        if max_c is not None or min_c is not None:
            con.execute(
                "UPDATE observations SET six_hr_max_c = ?, six_hr_min_c = ? WHERE rowid = ?",
                [max_c, min_c, rowid],
            )
            updated += 1
    if updated:
        logger.info(f"Backfilled 6-hour max/min for {updated} existing observations")


def _backfill_ingest_source(con: duckdb.DuckDBPyConnection) -> None:
    """Tag existing rows with ingest_source. Runs once."""
    already_done = con.execute(
        "SELECT COUNT(*) FROM observations WHERE ingest_source IS NOT NULL"
    ).fetchone()[0]
    if already_done > 0:
        return

    # Non-settlement stations are always Synoptic (AWC only polls 5 settlement stations)
    awc_stations = {'KNYC', 'KPHL', 'KMDW', 'KMIA', 'KLAX'}
    con.execute(
        "UPDATE observations SET ingest_source = 'synoptic' WHERE station_id NOT IN "
        + str(tuple(awc_stations))
    )
    # Settlement stations could be either — tag as 'synoptic' (first ingestor historically)
    con.execute(
        "UPDATE observations SET ingest_source = 'synoptic' WHERE ingest_source IS NULL"
    )
    count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    if count:
        logger.info(f"Backfilled ingest_source for {count} existing observations")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection."""
    return duckdb.connect(db_path)
