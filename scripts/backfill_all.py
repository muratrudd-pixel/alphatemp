"""Run all backfills sequentially (DuckDB single-writer constraint).

Usage: python scripts/backfill_all.py [days_back]
Default: 1900 days (back to HRRRv4 launch Dec 2020)

Order:
  1. ACIS settlement truth (fast — already done if re-run)
  2. IEM observations (KNYC + KLGA + KEWR)
  3. Synoptic observations (last ~365 days, higher resolution)
  4. HRRR 06z forecasts (slow — ~7 hours for 1900 days)
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from core.db import init_db

load_dotenv()


def main():
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 1900
    init_db()

    logger.info(f"=== BACKFILL ALL — {days_back} days ===")

    # 1. ACIS settlement truth
    logger.info("--- Step 1/4: ACIS settlement truth ---")
    import asyncio
    from services.nws_fetcher import NWSFetcher
    nws = NWSFetcher()
    inserted = asyncio.run(nws.backfill(days_back=days_back))
    logger.info(f"ACIS done: {inserted} rows")

    # 2. IEM observations (free, covers full history)
    logger.info("--- Step 2/4: IEM observations ---")
    from scripts.backfill_iem import backfill_iem
    inserted = backfill_iem(days_back=days_back)
    logger.info(f"IEM done: {inserted} rows")

    # 3. Synoptic observations (last year only on free tier)
    logger.info("--- Step 3/4: Synoptic observations (last ~365 days) ---")
    token = os.getenv("SYNOPTIC_TOKEN")
    if token:
        from services.backfiller import backfill
        synoptic_days = min(days_back, 365)
        inserted = backfill(token=token, days_back=synoptic_days)
        logger.info(f"Synoptic done: {inserted} rows")
    else:
        logger.warning("SYNOPTIC_TOKEN not set — skipping Synoptic backfill")

    # 4. HRRR 06z forecasts
    logger.info("--- Step 4/4: HRRR 06z forecasts ---")
    from services.forecast_backfiller import backfill_forecasts
    inserted = backfill_forecasts(days_back=days_back, run_hour=6)
    logger.info(f"HRRR 06z done: {inserted} rows")

    logger.info("=== ALL BACKFILLS COMPLETE ===")


if __name__ == "__main__":
    main()
