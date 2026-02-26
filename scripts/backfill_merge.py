"""Merge temp backfill DBs into the main alphatemp.duckdb."""

import os
import sys

os.chdir("/Users/russellrudd/Projects/alphatemp/alphatemp")
sys.path.insert(0, ".")

import duckdb
from loguru import logger
from core.db import init_db

MAIN_DB = "data/alphatemp.duckdb"

HRRR_DBS = ["data/backfill_00z.duckdb", "data/backfill_18z.duckdb"]
KALSHI_TRADES_DB = "data/backfill_kalshi_trades.duckdb"
KALSHI_CANDLES_DB = "data/backfill_kalshi_candles.duckdb"


def merge_forecasts(con, temp_path):
    """Merge HRRR forecast rows from temp DB."""
    con.execute(f"ATTACH '{temp_path}' AS src (READ_ONLY)")

    before = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]

    con.execute("""
        INSERT INTO forecasts
        SELECT s.* FROM src.forecasts s
        WHERE NOT EXISTS (
            SELECT 1 FROM forecasts m
            WHERE m.station_id = s.station_id
              AND m.model_run  = s.model_run
              AND m.valid_at   = s.valid_at
        )
    """)

    after = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    inserted = after - before
    logger.info(f"  forecasts: +{inserted:,} rows")

    con.execute("DETACH src")
    return inserted


def merge_kalshi_trades(con, temp_path):
    """Merge Kalshi trade rows from temp DB."""
    con.execute(f"ATTACH '{temp_path}' AS src (READ_ONLY)")

    before = con.execute("SELECT COUNT(*) FROM kalshi_trades").fetchone()[0]

    con.execute("""
        INSERT INTO kalshi_trades
        SELECT s.* FROM src.kalshi_trades s
        WHERE NOT EXISTS (
            SELECT 1 FROM kalshi_trades m
            WHERE m.trade_id = s.trade_id
        )
    """)

    after = con.execute("SELECT COUNT(*) FROM kalshi_trades").fetchone()[0]
    inserted = after - before
    logger.info(f"  kalshi_trades: +{inserted:,} rows")

    con.execute("DETACH src")
    return inserted


def merge_kalshi_candles(con, temp_path):
    """Merge Kalshi candlestick rows from temp DB (replaces existing)."""
    con.execute(f"ATTACH '{temp_path}' AS src (READ_ONLY)")

    # Delete existing candles and replace with wider-window data
    old_count = con.execute("SELECT COUNT(*) FROM kalshi_candlesticks").fetchone()[0]
    con.execute("DELETE FROM kalshi_candlesticks")
    logger.info(f"  Cleared {old_count:,} old candlestick rows")

    con.execute("""
        INSERT INTO kalshi_candlesticks
        SELECT s.* FROM src.kalshi_candlesticks s
    """)

    new_count = con.execute("SELECT COUNT(*) FROM kalshi_candlesticks").fetchone()[0]
    logger.info(f"  kalshi_candlesticks: {new_count:,} rows (was {old_count:,})")

    con.execute("DETACH src")
    return new_count


def main():
    # Ensure all tables exist (including kalshi_trades)
    init_db(MAIN_DB)

    con = duckdb.connect(MAIN_DB)

    # --- HRRR forecasts ---
    for temp_path in HRRR_DBS:
        if not os.path.exists(temp_path):
            logger.warning(f"{temp_path} not found, skipping")
            continue
        logger.info(f"Merging {temp_path}...")
        merge_forecasts(con, temp_path)

    # --- Kalshi trades ---
    if os.path.exists(KALSHI_TRADES_DB):
        logger.info(f"Merging {KALSHI_TRADES_DB}...")
        merge_kalshi_trades(con, KALSHI_TRADES_DB)
    else:
        logger.warning(f"{KALSHI_TRADES_DB} not found, skipping")

    # --- Kalshi candlesticks (replace with wider window) ---
    if os.path.exists(KALSHI_CANDLES_DB):
        logger.info(f"Merging {KALSHI_CANDLES_DB}...")
        merge_kalshi_candles(con, KALSHI_CANDLES_DB)
    else:
        logger.warning(f"{KALSHI_CANDLES_DB} not found, skipping")

    con.close()
    logger.info("All merges complete.")

    # Summary
    con = duckdb.connect(MAIN_DB, read_only=True)
    logger.info("=== FINAL COUNTS ===")
    for table in ['forecasts', 'kalshi_trades', 'kalshi_candlesticks', 'kalshi_settlements',
                   'observations', 'nws_daily', 'market_ticks']:
        try:
            cnt = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {table}: {cnt:,}")
        except Exception:
            pass
    con.close()

    # Clean up temp files
    all_temps = HRRR_DBS + [KALSHI_TRADES_DB, KALSHI_CANDLES_DB]
    for temp_path in all_temps:
        for f in [temp_path, temp_path + ".wal"]:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"Removed {f}")


if __name__ == "__main__":
    main()
