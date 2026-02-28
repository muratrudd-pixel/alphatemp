"""Backfill fxx and is_spinup for existing HRRR forecast rows.

Usage:
    python scripts/migrate_hrrr_fxx.py
    python scripts/migrate_hrrr_fxx.py --db data/alphatemp.duckdb
"""

import argparse
import duckdb
from loguru import logger
from core.db import DEFAULT_DB_PATH


def migrate(db_path=None):
    # type: (str) -> int
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    con = duckdb.connect(db_path)

    # Count rows needing migration
    need_update = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE model_name = 'hrrr' AND fxx IS NULL"
    ).fetchone()[0]

    if need_update == 0:
        logger.info("All HRRR rows already have fxx populated")
        con.close()
        return 0

    logger.info(f"Backfilling fxx/is_spinup for {need_update} HRRR rows")

    con.execute("""
        UPDATE forecasts
        SET fxx = CAST(EXTRACT(EPOCH FROM (valid_at - model_run)) / 3600 AS INTEGER),
            is_spinup = (EXTRACT(EPOCH FROM (valid_at - model_run)) / 3600) <= 3
        WHERE model_name = 'hrrr' AND fxx IS NULL
    """)

    updated = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE model_name = 'hrrr' AND fxx IS NOT NULL"
    ).fetchone()[0]
    logger.info(f"Updated {updated} rows with fxx/is_spinup")

    con.close()
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Backfill fxx/is_spinup for existing HRRR rows"
    )
    parser.add_argument("--db", default=None, help="Database path")
    args = parser.parse_args()
    migrate(db_path=args.db)


if __name__ == "__main__":
    main()
