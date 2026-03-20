"""Backfill IEM MOS (Model Output Statistics) data for KNYC.

Pulls NBS (NBM Short) and GFS MOS from Iowa Environmental Mesonet API.
Stores 3-hourly forecast fields (temp, dewpoint, wind, precip prob) in a
temp DuckDB for later merge.

API docs: https://mesonet.agron.iastate.edu/api/1/docs#/default/mos_api_1_mos_json_get

Usage:
    PYTHONPATH=. venv/bin/python scripts/backfill_iem_mos.py --model NBS --start 2018-11-01 --end 2026-03-10
    PYTHONPATH=. venv/bin/python scripts/backfill_iem_mos.py --model GFS --start 2003-12-01 --end 2026-03-10
"""

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import requests
from loguru import logger

STATION_ID = "KNYC"

IEM_MOS_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"

# Map CLI model name -> DB model_name
MODEL_NAME_MAP = {
    "NBS": "nbs",
    "GFS": "gfs_mos",
}

# Run hours to query per day
RUN_HOURS = [0, 12]

# Default start dates per model
MODEL_START_DATES = {
    "NBS": date(2018, 11, 1),
    "GFS": date(2003, 12, 1),
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mos_forecasts (
    station_id VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    model_run TIMESTAMP NOT NULL,
    valid_at TIMESTAMP NOT NULL,
    temp_f REAL,
    dewpoint_f REAL,
    wind_speed_kt REAL,
    precip_prob_6h REAL,
    precip_prob_12h REAL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (station_id, model_name, model_run, valid_at)
)
"""


def get_resume_date(db_path, model_name):
    # type: (str, str) -> Optional[date]
    """Find the latest model_run date for this model in the temp DB."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(model_run) FROM mos_forecasts WHERE model_name = ?",
            [model_name],
        ).fetchone()
        con.close()
        if result and result[0]:
            latest = result[0]
            if isinstance(latest, datetime):
                return latest.date()
            return latest
    except Exception:
        pass
    return None


def fetch_mos_run(model, runtime_str):
    # type: (str, str) -> List[Dict]
    """Fetch MOS data for one model run from IEM API.

    Args:
        model: IEM model code (NBS, GFS)
        runtime_str: ISO format runtime e.g. '2026-03-10T12:00:00Z'

    Returns:
        List of record dicts from the API, or empty list on failure.
    """
    params = {
        "station": STATION_ID,
        "model": model,
        "runtime": runtime_str,
    }

    try:
        resp = requests.get(IEM_MOS_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.debug("HTTP error for {} {}: {}", model, runtime_str, e)
        return []

    payload = resp.json()
    return payload.get("data", [])


def parse_record(record, model_name, now_utc):
    # type: (Dict, str, datetime) -> Optional[Tuple]
    """Parse one MOS record into a DB row tuple.

    Returns:
        Tuple of (station_id, model_name, model_run, valid_at, temp_f,
                  dewpoint_f, wind_speed_kt, precip_prob_6h, precip_prob_12h,
                  ingested_at) or None if unparseable.
    """
    try:
        runtime_utc = record.get("runtime_utc")
        ftime_utc = record.get("ftime_utc")
        if not runtime_utc or not ftime_utc:
            return None

        # Parse timestamps — IEM returns "2026-03-10T12:00:00.000"
        model_run = datetime.strptime(runtime_utc[:19], "%Y-%m-%dT%H:%M:%S")
        valid_at = datetime.strptime(ftime_utc[:19], "%Y-%m-%dT%H:%M:%S")

        station = record.get("station", STATION_ID)

        # Extract fields — these are integers/floats or None
        tmp = record.get("tmp")
        dpt = record.get("dpt")
        wsp = record.get("wsp")
        p06 = record.get("p06")
        p12 = record.get("p12")

        # Convert to float (API returns int for tmp/dpt/wsp)
        temp_f = float(tmp) if tmp is not None else None
        dewpoint_f = float(dpt) if dpt is not None else None
        wind_speed_kt = float(wsp) if wsp is not None else None
        precip_prob_6h = float(p06) if p06 is not None else None
        precip_prob_12h = float(p12) if p12 is not None else None

        return (
            station,
            model_name,
            model_run,
            valid_at,
            temp_f,
            dewpoint_f,
            wind_speed_kt,
            precip_prob_6h,
            precip_prob_12h,
            now_utc,
        )
    except Exception as e:
        logger.debug("Failed to parse record: {}", e)
        return None


def backfill_mos(
    model,          # type: str
    start_date,     # type: Optional[date]
    end_date,       # type: Optional[date]
    delay_seconds,  # type: float
):
    # type: (...) -> int
    """Backfill MOS data for one model.

    Args:
        model: IEM model code (NBS or GFS)
        start_date: First date to query
        end_date: Last date to query
        delay_seconds: Delay between API requests

    Returns:
        Total rows inserted.
    """
    model_name = MODEL_NAME_MAP[model]
    db_path = "data/backfill_mos_{}.duckdb".format(model_name)

    if start_date is None:
        start_date = MODEL_START_DATES[model]
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    # Create table
    con = duckdb.connect(db_path)
    con.execute(CREATE_TABLE_SQL)
    con.close()

    # Resume support
    resume = get_resume_date(db_path, model_name)
    if resume and resume >= start_date:
        # Resume from the day after the latest run we have
        new_start = resume + timedelta(days=1)
        logger.info(
            "Resume: latest {} run in DB is {}. Starting from {}",
            model_name, resume, new_start,
        )
        start_date = new_start

    if start_date > end_date:
        logger.info("Nothing to backfill — already up to date")
        return 0

    total_days = (end_date - start_date).days + 1
    total_requests = total_days * len(RUN_HOURS)
    total_inserted = 0
    total_empty = 0
    request_num = 0
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    logger.info(
        "Backfilling {} ({}) from {} to {} ({} days, {} requests) -> {}",
        model, model_name, start_date, end_date, total_days,
        total_requests, db_path,
    )

    con = duckdb.connect(db_path)
    current = start_date

    try:
        while current <= end_date:
            day_inserted = 0

            for run_hour in RUN_HOURS:
                request_num += 1
                runtime_dt = datetime(
                    current.year, current.month, current.day, run_hour
                )
                runtime_str = runtime_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                records = fetch_mos_run(model, runtime_str)

                if not records:
                    total_empty += 1
                    logger.debug(
                        "Req {}/{} — {} {:02d}z {}: no data",
                        request_num, total_requests, model, run_hour, current,
                    )
                    time.sleep(delay_seconds)
                    continue

                run_inserted = 0
                for rec in records:
                    row = parse_record(rec, model_name, now_utc)
                    if row is None:
                        continue
                    try:
                        con.execute(
                            "INSERT INTO mos_forecasts "
                            "(station_id, model_name, model_run, valid_at, "
                            "temp_f, dewpoint_f, wind_speed_kt, "
                            "precip_prob_6h, precip_prob_12h, ingested_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            list(row),
                        )
                        run_inserted += 1
                    except duckdb.ConstraintException:
                        pass  # Duplicate — skip

                day_inserted += run_inserted
                time.sleep(delay_seconds)

            total_inserted += day_inserted
            pct = int(request_num / total_requests * 100)

            # Log progress every 10 days or when data found
            day_num = (current - start_date).days + 1
            if day_num % 10 == 0 or day_inserted > 0:
                logger.info(
                    "Day {}/{} ({:>3}%) — {} {}: +{} rows (total: {}, empty: {})",
                    day_num, total_days, pct, model, current,
                    day_inserted, total_inserted, total_empty,
                )

            # Periodic checkpoint to flush WAL
            if day_num % 100 == 0:
                con.execute("CHECKPOINT")

            current += timedelta(days=1)

    finally:
        con.close()

    logger.info(
        "{} backfill complete: {} rows inserted, {} empty requests",
        model, total_inserted, total_empty,
    )
    return total_inserted


def main():
    parser = argparse.ArgumentParser(
        description="Backfill IEM MOS data for KNYC"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["NBS", "GFS"],
        help="MOS model to backfill",
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between API requests (default: 0.5)",
    )
    args = parser.parse_args()

    backfill_mos(
        model=args.model,
        start_date=args.start,
        end_date=args.end,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
