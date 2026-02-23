"""HRRR 12z forecast backfiller — pulls 90 days of historical runs via Herbie."""

import time
from datetime import datetime, timedelta, timezone

from loguru import logger

from core.db import get_connection, init_db
from services.forecast import HRRRFetcher


def backfill_forecasts(
    days_back: int = 90,
    db_path: str = "data/alphatemp.duckdb",
    delay_seconds: float = 2.0,
) -> int:
    """Pull historical 12z HRRR runs for all settlement stations.

    One 12z run per day, forecast hours 1-18, 5 stations.
    Idempotent: skips runs that already exist in the database.

    Args:
        days_back: Number of days to look back (default 90).
        db_path: Path to DuckDB database.
        delay_seconds: Pause between HRRR runs for rate limiting.

    Returns:
        Total forecast rows inserted.
    """
    fetcher = HRRRFetcher(db_path=db_path)
    con = get_connection(db_path)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days_back)

    total_inserted = 0
    total_skipped = 0
    day_num = 0

    logger.info(
        f"Backfilling 12z HRRR runs from {start_date} to {end_date} "
        f"({days_back} days, {delay_seconds}s delay)"
    )

    current = start_date
    while current < end_date:
        day_num += 1
        pct = int(day_num / days_back * 100)
        model_run = datetime(current.year, current.month, current.day, 12, tzinfo=timezone.utc)

        # Idempotent check: skip if we already have data for this run
        existing = con.execute(
            "SELECT COUNT(*) FROM forecasts WHERE model_run = ?",
            [model_run.replace(tzinfo=None)],
        ).fetchone()[0]

        if existing > 0:
            logger.debug(f"Skipping {model_run.strftime('%Y-%m-%d %Hz')} — {existing} rows already exist")
            total_skipped += 1
            current += timedelta(days=1)
            continue

        try:
            inserted = fetcher.fetch_run(model_run)
            total_inserted += inserted
            logger.info(
                f"Day {day_num}/{days_back} ({pct}%) — "
                f"{model_run.strftime('%Y-%m-%d %Hz')}: {inserted} rows"
            )
        except Exception as e:
            logger.warning(
                f"Day {day_num}/{days_back} ({pct}%) — "
                f"{model_run.strftime('%Y-%m-%d %Hz')}: archive gap or error — {e}"
            )

        current += timedelta(days=1)
        time.sleep(delay_seconds)

    con.close()
    logger.info(
        f"Backfill complete: {total_inserted} rows inserted, "
        f"{total_skipped} days skipped (already existed)"
    )
    return total_inserted


if __name__ == "__main__":
    init_db()
    backfill_forecasts()
