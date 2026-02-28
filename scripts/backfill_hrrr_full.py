"""HRRR full 24-run backfill via AWS S3 / Herbie.

Backfills all 24 hourly HRRR runs. Extended runs (00z, 06z, 12z, 18z) fetch
fxx 1-48. Standard runs fetch fxx 1-18. Populates fxx and is_spinup columns.

Usage:
    python scripts/backfill_hrrr_full.py --run-hour 7
    python scripts/backfill_hrrr_full.py --run-hour 7 --start 2021-07-01
    python scripts/backfill_hrrr_full.py --all-hours --start 2021-07-01
"""

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS
from core.db import init_db

STATION_ID = "KNYC"
LAT, LON = STATION_COORDS[STATION_ID]

EXTENDED_HOURS = {0, 6, 12, 18}
DATA_START = date(2021, 6, 1)  # HRRR v4 archive on AWS starts ~Dec 2020


def get_fxx_range(run_hour):
    # type: (int) -> range
    """Return forecast hour range: extended (1-48) or standard (1-18)."""
    if run_hour in EXTENDED_HOURS:
        return range(1, 49)
    return range(1, 19)


def is_spinup(fxx):
    # type: (int) -> bool
    """True if forecast hour has spin-up artifacts (fxx <= 3)."""
    return fxx <= 3


def get_resume_date(db_path, run_hour):
    # type: (str, int) -> Optional[date]
    """Find latest HRRR date for this run_hour in the DB."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(model_run) FROM forecasts "
            "WHERE model_name = 'hrrr' AND EXTRACT(HOUR FROM model_run) = ?",
            [run_hour],
        ).fetchone()
        con.close()
        if result and result[0]:
            latest = result[0]
            if isinstance(latest, datetime):
                return latest.date() + timedelta(days=1)
            return latest + timedelta(days=1)
    except Exception:
        pass
    return None


def extract_nearest(msg, lat, lon):
    # type: (object, float, float) -> float
    """Extract value at nearest grid point."""
    lat_grid, lon_grid = msg.latlons()
    cos_lat = np.cos(np.radians(lat))
    dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon) * cos_lat
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(msg.values[idx])


def backfill_hrrr_hour(
    run_hour,
    start_date=None,
    end_date=None,
    db_path=None,
    delay_seconds=0.5,
):
    # type: (int, Optional[date], Optional[date], Optional[str], float) -> int
    """Backfill one HRRR run hour across all dates.
    Returns total rows inserted.
    """
    if db_path is None:
        db_path = "data/backfill_hrrr_{:02d}z.duckdb".format(run_hour)

    if start_date is None:
        start_date = DATA_START
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    init_db(db_path)

    resume = get_resume_date(db_path, run_hour)
    if resume and resume > start_date:
        logger.info("Resuming {:02d}z from {}", run_hour, resume)
        start_date = resume

    if start_date > end_date:
        logger.info("Nothing to backfill for {:02d}z", run_hour)
        return 0

    con = duckdb.connect(db_path)
    fxx_range = get_fxx_range(run_hour)
    total_inserted = 0
    total_days = (end_date - start_date).days + 1
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    logger.info(
        "Backfilling HRRR {:02d}z fxx={}-{} from {} to {} ({} days) -> {}",
        run_hour, fxx_range.start, fxx_range.stop - 1,
        start_date, end_date, total_days, db_path,
    )

    current = start_date
    day_num = 0

    while current <= end_date:
        day_num += 1
        pct = int(day_num / total_days * 100)
        model_run = datetime(current.year, current.month, current.day, run_hour)
        day_inserted = 0
        consecutive_misses = 0

        for fxx in fxx_range:
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                    priority=["aws"],
                )
                grib_path = H.download("TMP:2 m")
                grbs = pygrib.open(str(grib_path))
                msg = grbs.select(name="2 metre temperature")[0]
                consecutive_misses = 0
            except Exception:
                consecutive_misses += 1
                if consecutive_misses >= 3:
                    break  # Archive gap — stop trying this day
                continue

            valid_at = model_run + timedelta(hours=fxx)
            spinup = is_spinup(fxx)

            try:
                temp_k = extract_nearest(msg, LAT, LON)
                temp_c = round(temp_k - 273.15, 2)
                temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                    [STATION_ID, model_run, valid_at, temp_f, temp_c,
                     now_utc, fxx, spinup],
                )
                day_inserted += 1
            except duckdb.ConstraintException:
                pass
            except Exception as e:
                logger.warning("HRRR extract failed {:02d}z fxx={}: {}", run_hour, fxx, e)

            time.sleep(delay_seconds)

        total_inserted += day_inserted
        if day_num % 50 == 0 or day_inserted > 0:
            logger.info(
                "Day {}/{} ({:>3}%) — HRRR {:02d}z {}: +{} rows (total: {})",
                day_num, total_days, pct, run_hour, current,
                day_inserted, total_inserted,
            )

        current += timedelta(days=1)

    con.close()
    logger.info("HRRR {:02d}z backfill complete: {} rows", run_hour, total_inserted)
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill HRRR from AWS S3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-hour", type=int, choices=range(24),
                       metavar="0-23")
    group.add_argument("--all-hours", action="store_true",
                       help="Run all 24 hours sequentially")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s),
                        default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s),
                        default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.all_hours:
        total = 0
        for hour in range(24):
            total += backfill_hrrr_hour(
                run_hour=hour,
                start_date=args.start,
                end_date=args.end,
                db_path=args.db,
                delay_seconds=args.delay,
            )
        logger.info("All hours complete: {} total rows", total)
    else:
        backfill_hrrr_hour(
            run_hour=args.run_hour,
            start_date=args.start,
            end_date=args.end,
            db_path=args.db,
            delay_seconds=args.delay,
        )


if __name__ == "__main__":
    main()
