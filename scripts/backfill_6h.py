#!/usr/bin/env python3
"""One-time migration: parse raw_metar for 6-hour synoptic max/min columns.

This is also run automatically by init_db(), but can be invoked standalone:
    python scripts/backfill_6h.py
    python scripts/backfill_6h.py --db-path data/alphatemp.duckdb
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

from core.db import get_connection
from services.ingestor import parse_6h_max, parse_6h_min


def run(db_path: str = "data/alphatemp.duckdb") -> int:
    con = get_connection(db_path)

    # Ensure columns exist
    for col in ("six_hr_max_c", "six_hr_min_c"):
        try:
            con.execute(f"ALTER TABLE observations ADD COLUMN {col} DOUBLE")
        except Exception:
            pass

    rows = con.execute(
        "SELECT rowid, raw_metar FROM observations WHERE raw_metar IS NOT NULL"
    ).fetchall()

    updated = 0
    for rowid, metar in rows:
        max_c = parse_6h_max(metar)
        min_c = parse_6h_min(metar)
        if max_c is not None or min_c is not None:
            con.execute(
                "UPDATE observations SET six_hr_max_c = ?, six_hr_min_c = ? WHERE rowid = ?",
                [max_c, min_c, rowid],
            )
            updated += 1

    con.close()
    logger.info(f"Backfilled 6-hour max/min for {updated}/{len(rows)} rows")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill 6-hour synoptic max/min")
    parser.add_argument("--db-path", default="data/alphatemp.duckdb")
    args = parser.parse_args()
    run(args.db_path)
