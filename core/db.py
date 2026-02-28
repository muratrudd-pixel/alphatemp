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
            model_name  VARCHAR NOT NULL DEFAULT 'hrrr',
            UNIQUE (station_id, model_run, valid_at, model_name)
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

    # Migration: add fxx and is_spinup to forecasts
    for col, dtype in [("fxx", "INTEGER"), ("is_spinup", "BOOLEAN DEFAULT FALSE")]:
        try:
            con.execute(f"ALTER TABLE forecasts ADD COLUMN {col} {dtype}")
        except Exception:
            pass  # Column already exists

    # Migration: add obs_type to observations (metar, dsm, micronet)
    try:
        con.execute("ALTER TABLE observations ADD COLUMN obs_type VARCHAR DEFAULT 'metar'")
    except Exception:
        pass

    # Migration: add UNIQUE constraint to market_ticks
    _migrate_market_ticks_unique(con)

    # Migration: add model_name column to forecasts (table-rebuild for UNIQUE change)
    _migrate_forecasts_model_name(con)

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

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_extended (
            station_id      VARCHAR NOT NULL,
            model_run       TIMESTAMP NOT NULL,
            valid_at        TIMESTAMP NOT NULL,
            model_name      VARCHAR NOT NULL,
            dewpoint_2m_f   DOUBLE,
            humidity_2m     DOUBLE,
            wind_speed_10m  DOUBLE,
            wind_dir_10m    DOUBLE,
            wind_gusts_10m  DOUBLE,
            pressure_msl    DOUBLE,
            cloud_cover     DOUBLE,
            precipitation   DOUBLE,
            shortwave_rad   DOUBLE,
            cape            DOUBLE,
            ingested_at     TIMESTAMP NOT NULL,
            UNIQUE (station_id, model_run, valid_at, model_name)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_positions (
            id INTEGER,
            city VARCHAR,
            event_date DATE,
            bracket_floor INTEGER,
            bracket_cap INTEGER,
            direction VARCHAR,
            model_prob DOUBLE,
            market_price DOUBLE,
            edge DOUBLE,
            entry_price DOUBLE,
            entry_time TIMESTAMP,
            exit_price DOUBLE,
            exit_time TIMESTAMP,
            settled_yes BOOLEAN,
            gross_pnl DOUBLE,
            fees DOUBLE,
            net_pnl DOUBLE,
            status VARCHAR DEFAULT 'open'
        )
    """)

    # Indexes — accelerate the most common query patterns
    for stmt in [
        "CREATE INDEX IF NOT EXISTS idx_obs_station_time ON observations (station_id, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_fcst_station_run ON forecasts (station_id, model_run)",
        "CREATE INDEX IF NOT EXISTS idx_fcst_station_valid ON forecasts (station_id, valid_at)",
        "CREATE INDEX IF NOT EXISTS idx_fcst_model ON forecasts (model_name)",
        "CREATE INDEX IF NOT EXISTS idx_drift_city_time ON drift_signals (city, calculated_at)",
        "CREATE INDEX IF NOT EXISTS idx_market_city_time ON market_ticks (city, captured_at)",
        "CREATE INDEX IF NOT EXISTS idx_bias_station_time ON station_bias (station_id, calculated_at)",
        "CREATE INDEX IF NOT EXISTS idx_nws_station_date ON nws_daily (station_id, obs_date)",
        "CREATE INDEX IF NOT EXISTS idx_ks_series ON kalshi_settlements (series_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_ks_date ON kalshi_settlements (event_date)",
        "CREATE INDEX IF NOT EXISTS idx_kc_ticker ON kalshi_candlesticks (market_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_kt_ticker ON kalshi_trades (market_ticker)",
        "CREATE INDEX IF NOT EXISTS idx_kt_time ON kalshi_trades (created_time)",
        "CREATE INDEX IF NOT EXISTS idx_fext_model_run ON forecast_extended (station_id, model_run, model_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_positions_id ON paper_positions(id)",
        "CREATE INDEX IF NOT EXISTS idx_paper_positions_city_date ON paper_positions(city, event_date)",
        "CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON paper_positions(status)",
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


def _migrate_market_ticks_unique(con: duckdb.DuckDBPyConnection) -> None:
    """Add UNIQUE constraint to market_ticks on (market_id, captured_at).

    DuckDB can't add constraints to existing tables, so we rebuild.
    Only runs if the constraint is missing.
    """
    try:
        has_unique = con.execute("""
            SELECT COUNT(*) FROM duckdb_constraints()
            WHERE table_name = 'market_ticks' AND constraint_type = 'UNIQUE'
        """).fetchone()[0]
        if has_unique > 0:
            return  # Already migrated

        logger.info("Migrating market_ticks: adding UNIQUE constraint")
        con.execute("DROP TABLE IF EXISTS market_ticks_new")

        con.execute("""
            CREATE TABLE market_ticks_new (
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
                cap_strike   DOUBLE,
                UNIQUE (market_id, captured_at)
            )
        """)
        con.execute("""
            INSERT INTO market_ticks_new
            SELECT DISTINCT ON (market_id, captured_at)
                market_id, city, captured_at, yes_bid, yes_ask,
                no_bid, no_ask, last_trade, volume, floor_strike, cap_strike
            FROM market_ticks
            ORDER BY market_id, captured_at
        """)

        old_count = con.execute("SELECT COUNT(*) FROM market_ticks").fetchone()[0]
        new_count = con.execute("SELECT COUNT(*) FROM market_ticks_new").fetchone()[0]
        dupes = old_count - new_count

        con.execute("DROP TABLE market_ticks")
        con.execute("ALTER TABLE market_ticks_new RENAME TO market_ticks")

        # Recreate indexes
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_city_time "
            "ON market_ticks (city, captured_at)"
        )

        logger.info(f"market_ticks migrated — UNIQUE added, {dupes} duplicates removed")
    except Exception:
        logger.warning(
            "Failed to migrate market_ticks for UNIQUE constraint", exc_info=True
        )


def _migrate_forecasts_model_name(con: duckdb.DuckDBPyConnection) -> None:
    """Add model_name column to forecasts and rebuild UNIQUE constraint.

    DuckDB can't drop inline UNIQUE constraints, so we rebuild the table.
    Only runs if model_name column is missing.
    """
    try:
        # Clean up any leftover from a previous failed migration
        con.execute("DROP TABLE IF EXISTS forecasts_new")

        has_col = con.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'forecasts' AND column_name = 'model_name'
        """).fetchone()[0]
        if has_col > 0:
            return  # Already migrated

        logger.info("Migrating forecasts table: adding model_name column")
        con.execute("""
            CREATE TABLE forecasts_new (
                station_id  VARCHAR NOT NULL,
                model_run   TIMESTAMP NOT NULL,
                valid_at    TIMESTAMP NOT NULL,
                temp_f      DOUBLE,
                temp_c      DOUBLE,
                ingested_at TIMESTAMP NOT NULL,
                model_name  VARCHAR NOT NULL DEFAULT 'hrrr',
                UNIQUE (station_id, model_run, valid_at, model_name)
            )
        """)
        con.execute("""
            INSERT INTO forecasts_new
            SELECT station_id, model_run, valid_at, temp_f, temp_c, ingested_at,
                   'hrrr' AS model_name
            FROM forecasts
        """)

        # Verify row counts match before destructive DROP
        old_count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        new_count = con.execute("SELECT COUNT(*) FROM forecasts_new").fetchone()[0]
        if old_count != new_count:
            con.execute("DROP TABLE forecasts_new")
            raise RuntimeError(
                f"Migration data mismatch: forecasts={old_count} vs forecasts_new={new_count}"
            )

        con.execute("DROP TABLE forecasts")
        con.execute("ALTER TABLE forecasts_new RENAME TO forecasts")
        logger.info("Forecasts table migrated — model_name column added")
    except Exception:
        logger.warning("Failed to migrate forecasts table for model_name", exc_info=True)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection."""
    return duckdb.connect(db_path)
