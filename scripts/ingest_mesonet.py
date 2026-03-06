#!/usr/bin/env python3
"""Ingest NYC Mesonet station CSVs into mesonet_obs table.

Stations: BKNYRD, BXVNST, MHCHEL, QNASTO (MHLSQR excluded — all nulls)
Source CSVs: data/mesonet/{station_id}.csv

Timestamps in CSVs are local time with EDT/EST suffix.
  EDT = UTC-4  (add 4h to get UTC)
  EST = UTC-5  (add 5h to get UTC)

Stores naive UTC timestamps in DB (per project convention).

Usage:
    PYTHONPATH=. python scripts/ingest_mesonet.py
    PYTHONPATH=. python scripts/ingest_mesonet.py --db data/alphatemp.duckdb
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
from loguru import logger

STATIONS = ["BKNYRD", "BXVNST", "MHCHEL", "QNASTO"]
MESONET_DIR = Path("data/mesonet")
BATCH_SIZE = 10_000

DEFAULT_DB = "data/alphatemp.duckdb"


def parse_timestamp_to_utc(ts_str: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS EDT' or '... EST' to naive UTC datetime."""
    ts_str = ts_str.strip()
    if ts_str.endswith(" EDT"):
        naive = datetime.strptime(ts_str[:-4], "%Y-%m-%d %H:%M:%S")
        return naive + timedelta(hours=4)  # EDT = UTC-4, add 4 to get UTC
    elif ts_str.endswith(" EST"):
        naive = datetime.strptime(ts_str[:-4], "%Y-%m-%d %H:%M:%S")
        return naive + timedelta(hours=5)  # EST = UTC-5, add 5 to get UTC
    else:
        # Fallback: assume EST if no suffix
        logger.warning(f"No EDT/EST suffix on timestamp: {ts_str!r} — assuming EST")
        naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return naive + timedelta(hours=5)


def ingest_station(con: duckdb.DuckDBPyConnection, station_id: str) -> tuple:
    """Ingest one station's CSV. Returns (inserted, skipped, errors)."""
    csv_path = MESONET_DIR / f"{station_id}.csv"
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return 0, 0, 0

    inserted = 0
    skipped = 0
    errors = 0
    batch = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        # Expected columns: station, time, "temp_2m [degF]"
        for row in reader:
            ts_raw = row.get("time", "").strip()
            temp_raw = row.get("temp_2m [degF]", "").strip()

            if not ts_raw or not temp_raw:
                skipped += 1
                continue

            try:
                temp_f = float(temp_raw)
            except ValueError:
                skipped += 1
                continue

            # Sanity check: NYC temps should be in [-30, 120] °F
            if temp_f < -30 or temp_f > 120:
                logger.warning(f"{station_id} suspicious temp {temp_f}°F at {ts_raw}")
                skipped += 1
                continue

            try:
                observed_at = parse_timestamp_to_utc(ts_raw)
            except ValueError as e:
                logger.warning(f"{station_id} timestamp parse error: {ts_raw!r} — {e}")
                errors += 1
                continue

            batch.append((station_id, observed_at, temp_f))

            if len(batch) >= BATCH_SIZE:
                ins, sk = _flush_batch(con, batch)
                inserted += ins
                skipped += sk
                batch = []

    if batch:
        ins, sk = _flush_batch(con, batch)
        inserted += ins
        skipped += sk

    return inserted, skipped, errors


def _flush_batch(con: duckdb.DuckDBPyConnection, batch: list) -> tuple:
    """INSERT OR IGNORE a batch of (station_id, observed_at, temp_f) rows."""
    inserted = 0
    skipped = 0
    for station_id, observed_at, temp_f in batch:
        try:
            con.execute(
                "INSERT INTO mesonet_obs (station_id, observed_at, temp_f) VALUES (?, ?, ?)",
                [station_id, observed_at, temp_f],
            )
            inserted += 1
        except duckdb.ConstraintException:
            skipped += 1
    return inserted, skipped


def main():
    parser = argparse.ArgumentParser(description="Ingest NYC Mesonet CSVs into mesonet_obs")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to DuckDB file")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    for station_id in STATIONS:
        logger.info(f"Ingesting {station_id}...")
        ins, sk, err = ingest_station(con, station_id)
        logger.info(f"  {station_id}: inserted={ins:,}  skipped={sk:,}  errors={err:,}")
        total_inserted += ins
        total_skipped += sk
        total_errors += err

    logger.info(
        f"Done. Total: inserted={total_inserted:,}  skipped={total_skipped:,}  errors={total_errors:,}"
    )

    # Verification query
    summary = con.execute(
        "SELECT station_id, COUNT(*), MIN(observed_at), MAX(observed_at) "
        "FROM mesonet_obs GROUP BY station_id ORDER BY station_id"
    ).fetchall()
    print("\nVerification:")
    for row in summary:
        print(f"  {row[0]}: {row[1]:>7,} rows  {row[2]}  →  {row[3]}")

    con.close()


if __name__ == "__main__":
    main()
