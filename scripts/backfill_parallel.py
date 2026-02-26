"""Backfill a single HRRR run hour into a temp DB (for parallel execution).

Usage: python backfill_parallel.py <run_hour>

Writes to data/backfill_<hour>z.duckdb so multiple processes can run
concurrently without hitting DuckDB's single-writer lock.
Merge into the main DB with backfill_merge.py when done.
"""

import os
import sys
import time

os.chdir("/Users/russellrudd/Projects/alphatemp/alphatemp")
sys.path.insert(0, ".")

from loguru import logger
from core.db import init_db
from services.forecast_backfiller import backfill_forecasts

DAYS_BACK = 1900


def main():
    if len(sys.argv) < 2:
        print("Usage: python backfill_parallel.py <run_hour>")
        sys.exit(1)

    hour = int(sys.argv[1])
    db_path = f"data/backfill_{hour:02d}z.duckdb"

    # Resume from existing temp DB if it exists (backfill_forecasts is idempotent)
    if os.path.exists(db_path):
        logger.info(f"Resuming from existing {db_path}")

    init_db(db_path)

    t0 = time.time()
    logger.info(f"=== BACKFILL {hour:02d}z x {DAYS_BACK} days -> {db_path} ===")
    inserted = backfill_forecasts(days_back=DAYS_BACK, run_hour=hour, db_path=db_path)
    elapsed = (time.time() - t0) / 3600
    logger.info(f"=== {hour:02d}z COMPLETE: {inserted} rows in {elapsed:.1f}h ===")


if __name__ == "__main__":
    main()
