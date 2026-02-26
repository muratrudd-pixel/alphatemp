"""IEM ASOS historical backfiller — pulls METAR obs for KNYC + neighbors.

IEM has free, unlimited historical ASOS data. No auth required.
Pulls in monthly chunks to stay under response size limits.
"""

import math
import time
from datetime import datetime, timedelta, timezone

import duckdb
import httpx
from loguru import logger

from core.db import get_connection, init_db
from services.ingestor import parse_t_group, parse_6h_max, parse_6h_min

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# IEM uses 3-letter codes (no K prefix) for some stations
STATION_MAP = {
    "NYC": "KNYC",
    "LGA": "KLGA",
    "EWR": "KEWR",
}

CHUNK_DAYS = 30
REQUEST_DELAY = 1.0  # Be polite to IEM


def backfill_iem(
    days_back: int = 1900,
    db_path: str = "data/alphatemp.duckdb",
) -> int:
    """Pull historical ASOS observations from IEM for all stations.

    Args:
        days_back: Number of days to look back.
        db_path: Path to DuckDB database.

    Returns:
        Total rows inserted.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    total_inserted = 0
    total_dupes = 0
    total_chunks = math.ceil(days_back / CHUNK_DAYS)

    stations = list(STATION_MAP.keys())

    logger.info(
        f"IEM backfill: {len(stations)} stations, "
        f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} "
        f"({days_back} days, {total_chunks} chunks)"
    )

    for iem_code, our_id in STATION_MAP.items():
        logger.info(f"Starting station {iem_code} ({our_id})")
        chunk_start = start
        chunk_num = 0

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
            chunk_num += 1

            params = {
                "station": iem_code,
                "data": "tmpf,metar",
                "tz": "Etc/UTC",
                "format": "comma",
                "latlon": "no",
                "elev": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
                "report_type": "1,2,3,4",
                "year1": str(chunk_start.year),
                "month1": str(chunk_start.month),
                "day1": str(chunk_start.day),
                "year2": str(chunk_end.year),
                "month2": str(chunk_end.month),
                "day2": str(chunk_end.day),
            }

            try:
                resp = httpx.get(IEM_ASOS_URL, params=params, timeout=60.0)
                resp.raise_for_status()
            except Exception as e:
                logger.warning(f"IEM fetch failed for {our_id} {chunk_start.date()}: {e}")
                chunk_start = chunk_end
                time.sleep(REQUEST_DELAY)
                continue

            lines = resp.text.strip().split("\n")
            # Skip comment/header lines
            data_lines = [l for l in lines if l and not l.startswith("#") and not l.startswith("station")]

            rows = []
            now_ts = datetime.now(timezone.utc).replace(tzinfo=None)

            for line in data_lines:
                parts = line.split(",", 3)
                if len(parts) < 4:
                    continue

                _, valid_str, tmpf_str, metar_str = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else ""

                # Parse timestamp
                try:
                    observed_at = datetime.strptime(valid_str.strip(), "%Y-%m-%d %H:%M")
                except ValueError:
                    continue

                # Parse temp
                temp_f = None
                if tmpf_str.strip() not in ("M", ""):
                    try:
                        temp_f = round(float(tmpf_str.strip()), 1)
                    except ValueError:
                        pass

                # Parse METAR remarks
                metar_str = metar_str.strip()
                temp_c_tenth = None
                six_hr_max_c = None
                six_hr_min_c = None
                if metar_str:
                    temp_c_tenth = parse_t_group(metar_str)
                    six_hr_max_c = parse_6h_max(metar_str)
                    six_hr_min_c = parse_6h_min(metar_str)

                rows.append((
                    our_id, observed_at, temp_f, temp_c_tenth,
                    six_hr_max_c, six_hr_min_c, metar_str, now_ts, "iem_backfill"
                ))

            if rows:
                con = get_connection(db_path)
                inserted = 0
                dupes = 0
                for row in rows:
                    try:
                        con.execute(
                            """INSERT INTO observations
                               (station_id, observed_at, temp_f, temp_c_tenth,
                                six_hr_max_c, six_hr_min_c, raw_metar, ingested_at, ingest_source)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            list(row),
                        )
                        inserted += 1
                    except duckdb.ConstraintException:
                        dupes += 1
                con.close()
                total_inserted += inserted
                total_dupes += dupes

            pct = int(chunk_num / total_chunks * 100)
            if chunk_num % 5 == 0 or inserted > 0:
                logger.info(
                    f"  {our_id} chunk {chunk_num}/{total_chunks} ({pct}%) "
                    f"{chunk_start.date()} → {chunk_end.date()}: "
                    f"{inserted}/{len(rows)} inserted ({dupes} dupes)"
                )

            chunk_start = chunk_end
            time.sleep(REQUEST_DELAY)

    logger.info(
        f"IEM backfill complete: {total_inserted} rows inserted, "
        f"{total_dupes} duplicates skipped"
    )
    return total_inserted


if __name__ == "__main__":
    import sys
    init_db()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1900
    backfill_iem(days_back=days)
