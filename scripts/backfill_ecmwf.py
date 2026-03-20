"""Backfill ECMWF IFS forecasts via Herbie (AWS open data archive).

Downloads GRIB files, extracts 2m temp at Central Park grid point,
writes to a temp DuckDB. Merge to main DB separately.

Usage:
    python scripts/backfill_ecmwf.py --run-hour 12
    python scripts/backfill_ecmwf.py --run-hour 0 --start 2023-06-01
"""

import argparse
import gc
import os
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

# ECMWF open data on AWS starts Jan 2023
DATA_START = date(2023, 1, 1)

# Full runs (00z, 12z): 0-72h forecast horizon
# Short-cutoff runs (06z, 18z): 0-30h forecast horizon
FULL_RUN_HOURS = {0, 12}
SHORT_RUN_HOURS = {6, 18}
VALID_RUN_HOURS = FULL_RUN_HOURS | SHORT_RUN_HOURS


def get_fxx_range(run_hour):
    # type: (int) -> range
    """Return the forecast hour range for a given ECMWF run hour."""
    if run_hour in FULL_RUN_HOURS:
        return range(0, 73)
    elif run_hour in SHORT_RUN_HOURS:
        return range(0, 31)
    else:
        raise ValueError(
            "ECMWF run_hour must be 0, 6, 12, or 18 — got {}".format(run_hour)
        )


def get_resume_date(db_path, run_hour):
    # type: (str, int) -> Optional[date]
    """Find the latest ECMWF date for this run_hour in the DB."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(model_run) FROM forecasts "
            "WHERE model_name = 'ecmwf' AND EXTRACT(HOUR FROM model_run) = ?",
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
    """Extract value at nearest grid point using cosine-weighted distance.

    ECMWF GRIB files use 0-360° longitude, so we normalize to -180/180
    before computing distance (STATION_COORDS uses negative longitudes).
    """
    lat_grid, lon_grid = msg.latlons()
    # Normalize 0-360 longitudes to -180/180 (ECMWF uses 0-360)
    lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
    cos_lat = np.cos(np.radians(lat))
    dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon) * cos_lat
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(msg.values[idx])


def backfill_ecmwf(
    run_hour=12,
    start_date=None,
    end_date=None,
    db_path=None,
    delay_seconds=1.0,
):
    # type: (int, Optional[date], Optional[date], Optional[str], float) -> int
    """Backfill ECMWF forecasts for one run hour from AWS.

    Args:
        run_hour: UTC hour (0, 6, 12, 18)
        start_date: First date to fetch
        end_date: Last date to fetch
        db_path: Temp DB path
        delay_seconds: Delay between GRIB fetches

    Returns:
        Total rows inserted.
    """
    if run_hour not in VALID_RUN_HOURS:
        raise ValueError("ECMWF run_hour must be 0, 6, 12, or 18")

    if db_path is None:
        db_path = "data/backfill_ecmwf_{:02d}z.duckdb".format(run_hour)

    if start_date is None:
        start_date = DATA_START
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    init_db(db_path)

    # Resume support
    resume = get_resume_date(db_path, run_hour)
    if resume and resume > start_date:
        logger.info("Resuming from {} (skipping already-ingested)", resume)
        start_date = resume

    if start_date > end_date:
        logger.info("Nothing to backfill")
        return 0

    fxx_range = get_fxx_range(run_hour)

    con = duckdb.connect(db_path)
    total_inserted = 0
    total_days = (end_date - start_date).days + 1
    current = start_date

    logger.info(
        "Backfilling ECMWF {:02d}z from {} to {} ({} days, fxx 0-{}) -> {}",
        run_hour, start_date, end_date, total_days, fxx_range[-1], db_path,
    )

    day_num = 0
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    while current <= end_date:
        day_num += 1
        pct = int(day_num / total_days * 100)
        model_run = datetime(current.year, current.month, current.day, run_hour)
        day_inserted = 0

        for fxx in fxx_range:
            grbs = None
            grib_path = None
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="ifs",
                    product="oper",
                    fxx=fxx,
                    priority=["aws"],
                )
                grib_path = H.download(":2t:")
                grbs = pygrib.open(str(grib_path))
                msg = grbs[1]  # single message: 2m temperature
            except Exception as e:
                logger.debug(
                    "ECMWF {:02d}z {} fxx={}: not available — {}",
                    run_hour, current, fxx, e,
                )
                if grbs is not None:
                    grbs.close()
                if grib_path is not None:
                    try:
                        os.remove(str(grib_path))
                    except OSError:
                        pass
                continue

            valid_at = model_run + timedelta(hours=fxx)

            try:
                temp_k = extract_nearest(msg, LAT, LON)
                temp_c = round(temp_k - 273.15, 2)
                temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'ecmwf', ?, FALSE)",
                    [STATION_ID, model_run, valid_at, temp_f, temp_c,
                     now_utc, fxx],
                )
                day_inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate
            except Exception as e:
                logger.warning("ECMWF extract failed fxx={}: {}", fxx, e)
            finally:
                if grbs is not None:
                    grbs.close()
                if grib_path is not None:
                    try:
                        os.remove(str(grib_path))
                    except OSError:
                        pass

            time.sleep(delay_seconds)

        total_inserted += day_inserted
        if day_inserted > 0:
            logger.info(
                "Day {}/{} ({:>3}%) — ECMWF {:02d}z {}: +{} rows (total: {})",
                day_num, total_days, pct, run_hour, current,
                day_inserted, total_inserted,
            )
        else:
            logger.debug(
                "Day {}/{} ({:>3}%) — ECMWF {:02d}z {}: archive gap",
                day_num, total_days, pct, run_hour, current,
            )

        current += timedelta(days=1)

    con.close()
    logger.info("ECMWF {:02d}z backfill complete: {} rows", run_hour, total_inserted)
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill ECMWF from AWS open data")
    parser.add_argument("--run-hour", type=int, default=12, choices=[0, 6, 12, 18])
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    backfill_ecmwf(
        run_hour=args.run_hour,
        start_date=args.start,
        end_date=args.end,
        db_path=args.db,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
