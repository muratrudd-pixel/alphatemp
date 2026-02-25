"""Backfill a single HRRR run hour. Usage: python backfill_single_hour.py <run_hour>"""

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
        print("Usage: python backfill_single_hour.py <run_hour>")
        sys.exit(1)

    hour = int(sys.argv[1])
    init_db()

    t0 = time.time()
    logger.info(f"=== BACKFILL {hour:02d}z x {DAYS_BACK} days ===")
    inserted = backfill_forecasts(days_back=DAYS_BACK, run_hour=hour)
    elapsed = (time.time() - t0) / 3600
    logger.info(f"=== {hour:02d}z COMPLETE: {inserted} rows in {elapsed:.1f}h ===")


if __name__ == "__main__":
    main()
