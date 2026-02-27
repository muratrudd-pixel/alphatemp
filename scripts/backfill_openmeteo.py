"""Backfill GFS and ECMWF historical forecasts from Open-Meteo API.

Fetches hourly temperature_2m for NYC (Central Park coords) and writes
to a temp DuckDB file for later merge into the main DB.

Usage:
    python scripts/backfill_openmeteo.py gfs
    python scripts/backfill_openmeteo.py ecmwf
    python scripts/backfill_openmeteo.py gfs --start 2023-01-01 --end 2023-06-30
"""

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import requests
from loguru import logger

from core.db import init_db

# Central Park coordinates
LAT = 40.7789
LON = -73.9692

# Open-Meteo model identifiers
MODEL_IDS = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs",
}

# Earliest available data per model
DATA_START_DATES = {
    "gfs": date(2021, 3, 23),
    "ecmwf": date(2017, 1, 1),
}

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
BATCH_DAYS = 60
STATION_ID = "KNYC"

# Extended weather variables for forecast_extended table
EXTENDED_VARS = [
    "dewpoint_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "pressure_msl", "cloud_cover", "precipitation",
    "shortwave_radiation", "cape",
]

# Map Open-Meteo field names -> our DB column names
_EXTENDED_COL_MAP = {
    "dewpoint_2m": "dewpoint_2m_f",
    "relative_humidity_2m": "humidity_2m",
    "wind_speed_10m": "wind_speed_10m",
    "wind_direction_10m": "wind_dir_10m",
    "wind_gusts_10m": "wind_gusts_10m",
    "pressure_msl": "pressure_msl",
    "cloud_cover": "cloud_cover",
    "precipitation": "precipitation",
    "shortwave_radiation": "shortwave_rad",
    "cape": "cape",
}


def _get_resume_date(db_path, model):
    # type: (str, str) -> Optional[date]
    """Check temp DB for the latest date already ingested for this model."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(valid_at) FROM forecasts WHERE model_name = ?",
            [model],
        ).fetchone()
        con.close()
        if result and result[0]:
            # Return the day after the latest valid_at
            latest = result[0]
            if isinstance(latest, datetime):
                return latest.date() + timedelta(days=1)
            return latest + timedelta(days=1)
    except Exception:
        pass
    return None


def _fetch_batch(model_id, start_date, end_date):
    # type: (str, date, date) -> dict
    """Fetch one batch from the Open-Meteo historical forecast API."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "models": model_id,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _insert_batch(con, model, data):
    # type: (duckdb.DuckDBPyConnection, str, dict) -> int
    """Parse API response and insert rows into forecasts table.

    Returns the number of rows inserted.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps_f = hourly.get("temperature_2m", [])

    if not times or not temps_f:
        return 0

    inserted = 0
    now = datetime.now(timezone.utc)

    for ts_str, temp_f in zip(times, temps_f):
        if temp_f is None:
            continue

        # Parse "2024-01-01T00:00" -> datetime
        valid_at = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")

        # model_run = midnight UTC of the valid_at date
        model_run = datetime(valid_at.year, valid_at.month, valid_at.day, 0, 0)

        # Convert F -> C
        temp_c = round((temp_f - 32.0) * 5.0 / 9.0, 2)

        try:
            con.execute(
                "INSERT INTO forecasts "
                "(station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [STATION_ID, model_run, valid_at, temp_f, temp_c, now, model],
            )
            inserted += 1
        except duckdb.ConstraintException:
            # Duplicate row — skip (idempotent)
            pass

    return inserted


def _fetch_batch_extended(model_id, start_date, end_date):
    # type: (str, date, date) -> dict
    """Fetch extended weather variables from Open-Meteo historical forecast API."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(EXTENDED_VARS),
        "temperature_unit": "fahrenheit",  # dewpoint in F
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "models": model_id,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _insert_batch_extended(con, model, data):
    # type: (duckdb.DuckDBPyConnection, str, dict) -> int
    """Parse extended API response and insert rows into forecast_extended.

    Returns the number of rows inserted.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return 0

    # Build per-variable arrays
    var_arrays = {}  # type: dict
    for var_name in EXTENDED_VARS:
        var_arrays[var_name] = hourly.get(var_name, [])

    inserted = 0
    now = datetime.now(timezone.utc)

    for i, ts_str in enumerate(times):
        valid_at = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")
        model_run = datetime(valid_at.year, valid_at.month, valid_at.day, 0, 0)

        # Extract values for this timestamp
        vals = {}  # type: dict
        for var_name in EXTENDED_VARS:
            arr = var_arrays[var_name]
            val = arr[i] if i < len(arr) else None
            col_name = _EXTENDED_COL_MAP[var_name]
            vals[col_name] = val

        try:
            con.execute(
                "INSERT INTO forecast_extended "
                "(station_id, model_run, valid_at, model_name, "
                "dewpoint_2m_f, humidity_2m, wind_speed_10m, wind_dir_10m, "
                "wind_gusts_10m, pressure_msl, cloud_cover, precipitation, "
                "shortwave_rad, cape, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    STATION_ID, model_run, valid_at, model,
                    vals["dewpoint_2m_f"], vals["humidity_2m"],
                    vals["wind_speed_10m"], vals["wind_dir_10m"],
                    vals["wind_gusts_10m"], vals["pressure_msl"],
                    vals["cloud_cover"], vals["precipitation"],
                    vals["shortwave_rad"], vals["cape"],
                    now,
                ],
            )
            inserted += 1
        except duckdb.ConstraintException:
            pass  # Duplicate row — skip

    return inserted


def _get_resume_date_extended(db_path, model):
    # type: (str, str) -> Optional[date]
    """Check temp DB for the latest extended date already ingested."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(valid_at) FROM forecast_extended WHERE model_name = ?",
            [model],
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


def backfill_model(model, start_date=None, end_date=None, db_path=None):
    # type: (str, Optional[date], Optional[date], Optional[str]) -> int
    """Backfill historical forecasts for one model.

    Args:
        model: "gfs" or "ecmwf"
        start_date: First date to fetch (defaults to model's earliest available)
        end_date: Last date to fetch (defaults to yesterday)
        db_path: Path to temp DuckDB file

    Returns:
        Total number of rows inserted.
    """
    if model not in MODEL_IDS:
        raise ValueError("Unknown model: {}. Use 'gfs' or 'ecmwf'.".format(model))

    if db_path is None:
        db_path = "data/backfill_{}.duckdb".format(model)

    model_id = MODEL_IDS[model]

    if start_date is None:
        start_date = DATA_START_DATES[model]
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    # Initialize the temp DB schema
    init_db(db_path)

    # Resume support: skip past already-ingested dates
    resume_date = _get_resume_date(db_path, model)
    if resume_date and resume_date > start_date:
        logger.info("Resuming from {} (skipping already-ingested data)", resume_date)
        start_date = resume_date

    if start_date > end_date:
        logger.info("Nothing to backfill — start_date {} > end_date {}", start_date, end_date)
        return 0

    con = duckdb.connect(db_path)
    total_inserted = 0
    current = start_date

    logger.info(
        "Backfilling {} ({}) from {} to {} -> {}",
        model, model_id, start_date, end_date, db_path,
    )

    try:
        while current <= end_date:
            batch_end = min(current + timedelta(days=BATCH_DAYS - 1), end_date)
            logger.info("  Fetching {} to {} ...", current, batch_end)

            try:
                data = _fetch_batch(model_id, current, batch_end)
                inserted = _insert_batch(con, model, data)
                total_inserted += inserted
                logger.info("    +{} rows (total: {})", inserted, total_inserted)
            except requests.RequestException as e:
                logger.error("    API error for {} to {}: {}", current, batch_end, e)
                time.sleep(5)
                current = batch_end + timedelta(days=1)
                continue
            except Exception as e:
                logger.error("    Unexpected error: {}", e)
                time.sleep(5)
                current = batch_end + timedelta(days=1)
                continue

            current = batch_end + timedelta(days=1)
            time.sleep(1)  # Rate limiting
    finally:
        con.close()

    logger.info("Backfill complete: {} total rows inserted for {}", total_inserted, model)
    return total_inserted


def backfill_model_extended(model, start_date=None, end_date=None, db_path=None):
    # type: (str, Optional[date], Optional[date], Optional[str]) -> int
    """Backfill extended weather variables for one model.

    Writes to forecast_extended table in a temp DuckDB file.
    """
    if model not in MODEL_IDS:
        raise ValueError("Unknown model: {}. Use 'gfs' or 'ecmwf'.".format(model))

    if db_path is None:
        db_path = "data/backfill_{}_extended.duckdb".format(model)

    model_id = MODEL_IDS[model]

    if start_date is None:
        start_date = DATA_START_DATES[model]
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    init_db(db_path)

    # Resume support
    resume_date = _get_resume_date_extended(db_path, model)
    if resume_date and resume_date > start_date:
        logger.info("Resuming extended from {} (skipping already-ingested data)", resume_date)
        start_date = resume_date

    if start_date > end_date:
        logger.info("Nothing to backfill — start_date {} > end_date {}", start_date, end_date)
        return 0

    con = duckdb.connect(db_path)
    total_inserted = 0
    current = start_date

    logger.info(
        "Backfilling extended {} ({}) from {} to {} -> {}",
        model, model_id, start_date, end_date, db_path,
    )

    try:
        while current <= end_date:
            batch_end = min(current + timedelta(days=BATCH_DAYS - 1), end_date)
            logger.info("  Fetching extended {} to {} ...", current, batch_end)

            try:
                data = _fetch_batch_extended(model_id, current, batch_end)
                inserted = _insert_batch_extended(con, model, data)
                total_inserted += inserted
                logger.info("    +{} rows (total: {})", inserted, total_inserted)
            except requests.RequestException as e:
                logger.error("    API error for {} to {}: {}", current, batch_end, e)
                time.sleep(5)
                current = batch_end + timedelta(days=1)
                continue
            except Exception as e:
                logger.error("    Unexpected error: {}", e)
                time.sleep(5)
                current = batch_end + timedelta(days=1)
                continue

            current = batch_end + timedelta(days=1)
            time.sleep(1)
    finally:
        con.close()

    logger.info("Extended backfill complete: {} total rows for {}", total_inserted, model)
    return total_inserted


def main():
    parser = argparse.ArgumentParser(
        description="Backfill GFS/ECMWF historical forecasts from Open-Meteo"
    )
    parser.add_argument(
        "model",
        choices=["gfs", "ecmwf"],
        help="Model to backfill",
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to model's earliest available.",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="End date (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to temp DuckDB file. Defaults to data/backfill_<model>.duckdb",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Backfill extended weather variables to forecast_extended table",
    )
    args = parser.parse_args()

    if args.extended:
        backfill_model_extended(
            model=args.model,
            start_date=args.start,
            end_date=args.end,
            db_path=args.db,
        )
    else:
        backfill_model(
            model=args.model,
            start_date=args.start,
            end_date=args.end,
            db_path=args.db,
        )


if __name__ == "__main__":
    main()
