"""Historical ASOS data backfiller via Synoptic API."""

import math
import time
from datetime import datetime, timedelta, timezone

import duckdb
import httpx
from loguru import logger

from core.constants import CITIES
from core.db import get_connection
from services.ingestor import parse_t_group

SYNOPTIC_BASE_URL = "https://api.synopticdata.com/v2/stations/timeseries"

# Chunk size in days — Synoptic handles multi-day requests fine
CHUNK_DAYS = 7

# Pause between API calls to avoid rate limiting
REQUEST_DELAY = 0.5


def backfill(
    token: str,
    days_back: int = 730,
    db_path: str = "data/alphatemp.duckdb",
) -> int:
    """Pull historical 1-minute ASOS data for all settlement stations.

    Args:
        token: Synoptic API token
        days_back: Number of days to look back (default 730 = ~2 years)
        db_path: Path to DuckDB database

    Returns:
        Total rows inserted
    """
    # Only backfill settlement stations
    stations = [cfg["settlement"] for cfg in CITIES.values()]
    stid_str = ",".join(stations)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    total_inserted = 0
    total_chunks = math.ceil(days_back / CHUNK_DAYS)
    chunk_num = 0
    chunk_start = start

    # Quality tracking
    stations_with_data: set = set()
    total_rows_seen = 0
    tgroup_parsed = 0
    tgroup_attempted = 0

    logger.info(
        f"Backfilling {len(stations)} stations from {start.strftime('%Y-%m-%d')} "
        f"to {end.strftime('%Y-%m-%d')} ({days_back} days, {total_chunks} chunks)"
    )

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        chunk_num += 1
        pct = int(chunk_num / total_chunks * 100)

        logger.info(
            f"Chunk {chunk_num}/{total_chunks} ({pct}%) — "
            f"{chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        )

        params = {
            "stid": stid_str,
            "start": chunk_start.strftime("%Y%m%d%H%M"),
            "end": chunk_end.strftime("%Y%m%d%H%M"),
            "vars": "air_temp,metar",
            "obtimezone": "UTC",
            "token": token,
        }

        try:
            resp = httpx.get(SYNOPTIC_BASE_URL, params=params, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"API error for {chunk_start.date()}-{chunk_end.date()}: {e}")
            chunk_start = chunk_end
            time.sleep(REQUEST_DELAY)
            continue

        summary = data.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1 or "STATION" not in data:
            logger.debug(
                f"No data for {chunk_start.date()}-{chunk_end.date()}: "
                f"{summary.get('RESPONSE_MESSAGE', 'unknown')}"
            )
            chunk_start = chunk_end
            time.sleep(REQUEST_DELAY)
            continue

        rows = []
        now_ts = datetime.now(timezone.utc).replace(tzinfo=None)

        for station in data["STATION"]:
            stid = station["STID"]
            stations_with_data.add(stid)
            obs = station.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            temps = obs.get("air_temp_set_1", [])
            metars = obs.get("metar_set_1", [])

            for i, dt_str in enumerate(times):
                temp_c_val = temps[i] if i < len(temps) else None
                metar_str = metars[i] if i < len(metars) else ""

                temp_f = (
                    round(temp_c_val * 9.0 / 5.0 + 32.0, 1)
                    if temp_c_val is not None
                    else None
                )

                if metar_str:
                    tgroup_attempted += 1
                    temp_c_tenth = parse_t_group(metar_str)
                    if temp_c_tenth is not None:
                        tgroup_parsed += 1
                else:
                    temp_c_tenth = None

                # Parse observed_at to datetime for consistent TIMESTAMP type
                observed_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)

                rows.append((stid, observed_at, temp_f, temp_c_tenth, metar_str, now_ts))

        total_rows_seen += len(rows)

        if rows:
            con = get_connection(db_path)
            inserted = 0
            for row in rows:
                try:
                    con.execute(
                        """INSERT INTO observations
                           (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        list(row),
                    )
                    inserted += 1
                except duckdb.ConstraintException:
                    pass
            con.close()
            total_inserted += inserted
            logger.info(
                f"  {chunk_start.date()} → {chunk_end.date()}: "
                f"{inserted}/{len(rows)} inserted ({len(rows) - inserted} dupes)"
            )

        chunk_start = chunk_end
        time.sleep(REQUEST_DELAY)

    # Completion summary
    tgroup_rate = (
        f"{tgroup_parsed}/{tgroup_attempted} ({int(tgroup_parsed / tgroup_attempted * 100)}%)"
        if tgroup_attempted > 0
        else "N/A"
    )
    logger.info(
        f"Backfill complete: {total_inserted} rows inserted, "
        f"{total_rows_seen} total rows seen"
    )
    logger.info(
        f"  Stations with data: {len(stations_with_data)}/{len(stations)} "
        f"({', '.join(sorted(stations_with_data))})"
    )
    logger.info(f"  T-group parse rate: {tgroup_rate}")

    return total_inserted
