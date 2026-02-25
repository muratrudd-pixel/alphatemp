"""Run remaining HRRR backfills: 12z → 00z → 18z (sequential, DuckDB safe)."""

import os
import sys
import time

os.chdir("/Users/russellrudd/Projects/alphatemp/alphatemp")
sys.path.insert(0, ".")

from loguru import logger
from core.db import init_db
from services.forecast_backfiller import backfill_forecasts

DAYS_BACK = 1900
RUN_HOURS = [0, 6, 12, 18]

def main():
    init_db()
    start = time.time()
    logger.info(f"=== HRRR REMAINING BACKFILL: {RUN_HOURS} × {DAYS_BACK} days ===")

    for hour in RUN_HOURS:
        t0 = time.time()
        logger.info(f"--- Starting {hour:02d}z backfill ({DAYS_BACK} days) ---")
        inserted = backfill_forecasts(days_back=DAYS_BACK, run_hour=hour)
        elapsed = (time.time() - t0) / 3600
        logger.info(f"--- {hour:02d}z complete: {inserted} rows in {elapsed:.1f}h ---")

    total_h = (time.time() - start) / 3600
    logger.info(f"=== ALL HRRR BACKFILLS COMPLETE — {total_h:.1f}h total ===")

if __name__ == "__main__":
    main()
