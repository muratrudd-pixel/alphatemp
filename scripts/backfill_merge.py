"""Merge rows from a temp backfill DB into the main DB.

Usage:
    python scripts/merge_backfill.py data/backfill_gfs_12z.duckdb
    python scripts/merge_backfill.py data/backfill_hrrr_07z.duckdb --table forecasts
"""

import argparse
import duckdb
from loguru import logger


def merge_table(main_db, temp_db, table="forecasts"):
    # type: (str, str, str) -> int
    """Copy rows from temp DB to main DB, skipping duplicates.

    Returns number of rows inserted.
    """
    main_con = duckdb.connect(main_db)

    # Attach temp DB
    main_con.execute("ATTACH '{}' AS temp_db (READ_ONLY)".format(temp_db))

    before = main_con.execute("SELECT COUNT(*) FROM {}".format(table)).fetchone()[0]

    if table == "forecasts":
        main_con.execute("""
            INSERT INTO forecasts
            SELECT t.* FROM temp_db.forecasts t
            WHERE NOT EXISTS (
                SELECT 1 FROM forecasts m
                WHERE m.station_id = t.station_id
                  AND m.model_run = t.model_run
                  AND m.valid_at = t.valid_at
                  AND m.model_name = t.model_name
            )
        """)
    elif table == "observations":
        main_con.execute("""
            INSERT INTO observations
            SELECT t.* FROM temp_db.observations t
            WHERE NOT EXISTS (
                SELECT 1 FROM observations m
                WHERE m.station_id = t.station_id
                  AND m.observed_at = t.observed_at
            )
        """)
    else:
        # Generic: try inserting all, catch constraint violations
        try:
            main_con.execute(
                "INSERT INTO {} SELECT * FROM temp_db.{}".format(table, table)
            )
        except duckdb.ConstraintException:
            logger.warning(
                "Bulk insert failed for {}, falling back to row-by-row".format(table)
            )
            rows = main_con.execute(
                "SELECT * FROM temp_db.{}".format(table)
            ).fetchall()
            for row in rows:
                try:
                    placeholders = ", ".join(["?"] * len(row))
                    main_con.execute(
                        "INSERT INTO {} VALUES ({})".format(table, placeholders),
                        list(row),
                    )
                except duckdb.ConstraintException:
                    pass

    after = main_con.execute("SELECT COUNT(*) FROM {}".format(table)).fetchone()[0]
    inserted = after - before

    main_con.execute("DETACH temp_db")
    main_con.close()

    logger.info("Merged {} rows into {} from {}".format(inserted, table, temp_db))
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Merge backfill DB into main DB")
    parser.add_argument("temp_db", help="Path to temp backfill DB")
    parser.add_argument(
        "--main-db",
        default="data/alphatemp.duckdb",
        help="Path to main DB",
    )
    parser.add_argument(
        "--table",
        default="forecasts",
        help="Table to merge (default: forecasts)",
    )
    args = parser.parse_args()

    merge_table(args.main_db, args.temp_db, args.table)


if __name__ == "__main__":
    main()
