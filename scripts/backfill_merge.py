"""Merge temp backfill DBs into the main alphatemp.duckdb."""

import os
import sys

os.chdir("/Users/russellrudd/Projects/alphatemp/alphatemp")
sys.path.insert(0, ".")

import duckdb
from loguru import logger

MAIN_DB = "data/alphatemp.duckdb"
TEMP_DBS = ["data/backfill_00z.duckdb", "data/backfill_18z.duckdb"]


def main():
    con = duckdb.connect(MAIN_DB)

    for temp_path in TEMP_DBS:
        if not os.path.exists(temp_path):
            logger.warning(f"{temp_path} not found, skipping")
            continue

        logger.info(f"Merging {temp_path} into {MAIN_DB}...")

        # Attach temp DB and insert only rows that don't already exist
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
        logger.info(f"  Merged {inserted} rows from {temp_path}")

        con.execute("DETACH src")

    con.close()
    logger.info("Merge complete.")

    # Clean up temp files
    for temp_path in TEMP_DBS:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed {temp_path}")


if __name__ == "__main__":
    main()
