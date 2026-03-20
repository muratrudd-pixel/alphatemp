"""Backfill NBM QMD (Quantile Mapped Distribution) temperature percentiles.

Downloads GRIB files from AWS (s3://noaa-nbm-grib2-pds) via Herbie,
extracts max-temp percentile distribution at Central Park (KNYC),
writes to a temp DuckDB. Merge to main DB separately.

NBM QMD provides pre-computed 1-99% temperature percentile distributions.
The 12z f018 file contains "0-18 hour max temperature" percentiles —
this is the daily high for the current day.

Archive available from ~2020-09-29 on AWS.

Usage:
    python scripts/backfill_nbm_qmd.py
    python scripts/backfill_nbm_qmd.py --start 2023-01-01 --end 2024-12-31
    python scripts/backfill_nbm_qmd.py --delay 0.5
"""

import argparse
import gc
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS

STATION_ID = "KNYC"
LAT, LON = STATION_COORDS[STATION_ID]

# NBM QMD archive on AWS starts ~Sep 29, 2020
DATA_START = date(2020, 9, 29)

# Percentiles to extract (not all 99 — these capture the distribution shape)
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# Column names for each percentile in the DB
PCT_COLUMNS = ["pct_{:02d}".format(p) for p in PERCENTILES]

# Herbie search string: regex to grab all target percentiles in one download
# Uses non-capturing group to avoid pandas str.contains warning
SEARCH_PATTERN = (
    ":TMP:2 m above ground:0-18 hour max fcst:"
    "(?:{})% level".format("|".join(str(p) for p in PERCENTILES))
)

DB_PATH_DEFAULT = "data/backfill_nbm_qmd.duckdb"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nbm_percentiles (
    station_id VARCHAR NOT NULL,
    model_run TIMESTAMP NOT NULL,
    forecast_type VARCHAR NOT NULL,
    pct_01 REAL,
    pct_05 REAL,
    pct_10 REAL,
    pct_25 REAL,
    pct_50 REAL,
    pct_75 REAL,
    pct_90 REAL,
    pct_95 REAL,
    pct_99 REAL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (station_id, model_run, forecast_type)
)
"""


def init_nbm_db(db_path):
    # type: (str) -> None
    """Create the nbm_percentiles table if it doesn't exist."""
    con = duckdb.connect(db_path)
    con.execute(CREATE_TABLE_SQL)
    con.close()


def get_resume_date(db_path):
    # type: (str) -> Optional[date]
    """Find the latest model_run date in the DB for resume support."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(model_run) FROM nbm_percentiles "
            "WHERE station_id = ? AND forecast_type = 'max_temp'",
            [STATION_ID],
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

    NBM QMD uses Lambert Conformal projection. Longitudes are already
    in -180/180 range, but we normalize defensively just in case.
    """
    lat_grid, lon_grid = msg.latlons()
    # Normalize 0-360 longitudes to -180/180 (defensive — NBM already uses -180/180)
    lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
    cos_lat = np.cos(np.radians(lat))
    dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon) * cos_lat
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(msg.values[idx])


def extract_percentiles_for_date(model_run, delay_seconds):
    # type: (datetime, float) -> Optional[Dict[int, float]]
    """Download and extract all target percentiles for one model run.

    Downloads all percentiles in a single HTTP request using regex search,
    then parses each GRIB message by its percentileValue key.

    Returns:
        Dict mapping percentile -> temp_f, or None if data unavailable.
    """
    grib_path = None
    grbs = None

    try:
        H = Herbie(
            model_run.strftime("%Y-%m-%d %H:%M"),
            model="nbmqmd",
            fxx=18,
            product="co",
        )
        grib_path = H.download(SEARCH_PATTERN)
        grbs = pygrib.open(str(grib_path))
        msgs = grbs.read()

        if len(msgs) == 0:
            logger.debug("No messages in GRIB for {}", model_run)
            return None

        results = {}  # type: Dict[int, float]

        for msg in msgs:
            try:
                pct_val = msg["percentileValue"]
            except Exception:
                logger.warning(
                    "GRIB message missing percentileValue key for {}",
                    model_run,
                )
                continue

            if pct_val not in PERCENTILES:
                continue

            temp_k = extract_nearest(msg, LAT, LON)
            temp_f = round((temp_k - 273.15) * 9.0 / 5.0 + 32.0, 1)
            results[pct_val] = temp_f

        if len(results) != len(PERCENTILES):
            logger.warning(
                "Only got {}/{} percentiles for {} (got: {})",
                len(results), len(PERCENTILES), model_run, sorted(results.keys()),
            )

        return results if results else None

    except Exception as e:
        logger.debug("NBM QMD {} not available: {}", model_run, e)
        return None

    finally:
        if grbs is not None:
            try:
                grbs.close()
            except Exception:
                pass
        if grib_path is not None:
            try:
                os.remove(str(grib_path))
            except OSError:
                pass

        time.sleep(delay_seconds)


def backfill_nbm_qmd(
    start_date=None,
    end_date=None,
    db_path=None,
    delay_seconds=1.0,
):
    # type: (Optional[date], Optional[date], Optional[str], float) -> int
    """Backfill NBM QMD percentiles from AWS.

    Args:
        start_date: First date to fetch (default: archive start)
        end_date: Last date to fetch (default: yesterday)
        db_path: Path to temp DuckDB file
        delay_seconds: Delay between GRIB fetches

    Returns:
        Total rows inserted.
    """
    if db_path is None:
        db_path = DB_PATH_DEFAULT

    if start_date is None:
        start_date = DATA_START
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    init_nbm_db(db_path)

    # Resume support
    resume = get_resume_date(db_path)
    if resume and resume > start_date:
        logger.info("Resuming from {} (skipping already-ingested)", resume)
        start_date = resume

    if start_date > end_date:
        logger.info("Nothing to backfill")
        return 0

    con = duckdb.connect(db_path)
    total_inserted = 0
    total_days = (end_date - start_date).days + 1
    current = start_date
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    logger.info(
        "Backfilling NBM QMD 12z from {} to {} ({} days) -> {}",
        start_date, end_date, total_days, db_path,
    )

    day_num = 0

    while current <= end_date:
        day_num += 1
        pct = int(day_num / total_days * 100)
        model_run = datetime(current.year, current.month, current.day, 12)

        results = extract_percentiles_for_date(model_run, delay_seconds)

        if results:
            # Build values list matching PCT_COLUMNS order
            values = [results.get(p) for p in PERCENTILES]

            try:
                con.execute(
                    "INSERT INTO nbm_percentiles "
                    "(station_id, model_run, forecast_type, "
                    "pct_01, pct_05, pct_10, pct_25, pct_50, "
                    "pct_75, pct_90, pct_95, pct_99, ingested_at) "
                    "VALUES (?, ?, 'max_temp', "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [STATION_ID, model_run] + values + [now_utc],
                )
                total_inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate — already ingested
            except Exception as e:
                logger.warning("Insert failed for {}: {}", model_run, e)

            logger.info(
                "Day {}/{} ({:>3}%) — {} pct50={:.1f}F [{}..{}] (total: {})",
                day_num, total_days, pct, current,
                results.get(50, 0),
                results.get(1, 0), results.get(99, 0),
                total_inserted,
            )
        else:
            logger.debug(
                "Day {}/{} ({:>3}%) — {}: archive gap",
                day_num, total_days, pct, current,
            )

        # Periodic GC
        if day_num % 50 == 0:
            gc.collect()

        current += timedelta(days=1)

    con.close()
    logger.info("NBM QMD backfill complete: {} rows in {}", total_inserted, db_path)
    return total_inserted


def main():
    parser = argparse.ArgumentParser(
        description="Backfill NBM QMD temperature percentiles from AWS"
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Start date (YYYY-MM-DD), default: 2020-09-29",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="End date (YYYY-MM-DD), default: yesterday",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="DuckDB path (default: {})".format(DB_PATH_DEFAULT),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between downloads in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    backfill_nbm_qmd(
        start_date=args.start,
        end_date=args.end,
        db_path=args.db,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
