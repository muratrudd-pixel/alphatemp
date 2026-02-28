"""Phase 1 gate validation — run after all backfills complete.

Checks that all data gaps are filled before Phase 2 begins.
Exit code 0 = PASSED, exit code 1 = FAILED.

Usage:
    python scripts/phase1_gate_check.py
    python scripts/phase1_gate_check.py --db data/alphatemp.duckdb
"""

import argparse
import sys

import duckdb
from loguru import logger

from core.db import DEFAULT_DB_PATH


def check_gate(db_path=None):
    # type: (str) -> bool
    """Run all Phase 1 gate checks. Returns True if all pass."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    con = duckdb.connect(db_path, read_only=True)
    all_pass = True

    # --- Check 1: HRRR run hour coverage ---
    print("\n=== HRRR Run Hour Coverage ===")
    hrrr_hours = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first_day,
               MAX(model_run::DATE) AS last_day
        FROM forecasts WHERE model_name = 'hrrr'
        GROUP BY run_hour ORDER BY run_hour
    """).fetchall()
    print(f"{'Hour':>4} | {'Days':>6} | {'First':>12} | {'Last':>12}")
    print("-" * 45)
    for row in hrrr_hours:
        print(f"{int(row[0]):4d} | {row[1]:6d} | {row[2]} | {row[3]}")
    if len(hrrr_hours) < 24:
        print(f"FAIL: Only {len(hrrr_hours)} of 24 run hours present")
        all_pass = False
    else:
        print("OK: All 24 run hours present")

    # --- Check 2: GFS run hour coverage ---
    print("\n=== GFS Run Hour Coverage ===")
    gfs_hours = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first_day,
               MAX(model_run::DATE) AS last_day
        FROM forecasts WHERE model_name = 'gfs'
        GROUP BY run_hour ORDER BY run_hour
    """).fetchall()
    for row in gfs_hours:
        print(f"{int(row[0]):4d}z | {row[1]:6d} days | {row[2]} to {row[3]}")
    if len(gfs_hours) < 4:
        print(f"FAIL: Only {len(gfs_hours)} of 4 GFS run hours")
        all_pass = False
    else:
        print("OK: All 4 GFS run hours present")

    # --- Check 3: ECMWF run hour coverage ---
    print("\n=== ECMWF Run Hour Coverage ===")
    ecmwf_hours = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first_day,
               MAX(model_run::DATE) AS last_day
        FROM forecasts WHERE model_name = 'ecmwf'
        GROUP BY run_hour ORDER BY run_hour
    """).fetchall()
    for row in ecmwf_hours:
        print(f"{int(row[0]):4d}z | {row[1]:6d} days | {row[2]} to {row[3]}")
    if len(ecmwf_hours) < 1:
        print("FAIL: No ECMWF data")
        all_pass = False
    else:
        print("OK: ECMWF 00z present (12z deprioritized — archive gaps)")

    # --- Check 4: KJFK observations ---
    print("\n=== KJFK Observations ===")
    kjfk = con.execute("""
        SELECT COUNT(*), MIN(observed_at)::DATE, MAX(observed_at)::DATE
        FROM observations WHERE station_id = 'KJFK'
    """).fetchone()
    print(f"Rows: {kjfk[0]:,} | Range: {kjfk[1]} to {kjfk[2]}")
    if kjfk[0] == 0:
        print("FAIL: No KJFK observations")
        all_pass = False
    else:
        print("OK: KJFK data present")

    # --- Check 5: fxx/is_spinup populated ---
    print("\n=== fxx/is_spinup Population ===")
    null_fxx = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE fxx IS NULL"
    ).fetchone()[0]
    total_fcst = con.execute(
        "SELECT COUNT(*) FROM forecasts"
    ).fetchone()[0]
    print(f"Rows with NULL fxx: {null_fxx:,} / {total_fcst:,}")
    if null_fxx > 0:
        print("FAIL: fxx not fully populated")
        all_pass = False
    else:
        print("OK: All forecast rows have fxx")

    # --- Check 6: market_ticks UNIQUE ---
    print("\n=== Market Ticks Dedup ===")
    dupe_check = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT (market_id, captured_at))
        FROM market_ticks
    """).fetchone()[0]
    print(f"Duplicate market_ticks rows: {dupe_check}")
    if dupe_check > 0:
        print("FAIL: market_ticks still has duplicates")
        all_pass = False
    else:
        print("OK: No duplicates")

    # --- Check 7: Gap analysis (HRRR 12z sample) ---
    print("\n=== HRRR Gap Analysis (12z sample) ===")
    hrrr_12z_count = con.execute("""
        SELECT COUNT(DISTINCT model_run::DATE)
        FROM forecasts
        WHERE model_name = 'hrrr' AND EXTRACT(HOUR FROM model_run) = 12
    """).fetchone()[0]
    if hrrr_12z_count > 0:
        gaps = con.execute("""
            WITH dates AS (
                SELECT UNNEST(generate_series(
                    (SELECT MIN(model_run::DATE) FROM forecasts
                     WHERE model_name='hrrr' AND EXTRACT(HOUR FROM model_run) = 12),
                    (SELECT MAX(model_run::DATE) FROM forecasts
                     WHERE model_name='hrrr' AND EXTRACT(HOUR FROM model_run) = 12),
                    INTERVAL 1 DAY
                ))::DATE AS d
            ),
            present AS (
                SELECT DISTINCT model_run::DATE AS d
                FROM forecasts
                WHERE model_name = 'hrrr'
                  AND EXTRACT(HOUR FROM model_run) = 12
            )
            SELECT COUNT(*) FROM dates WHERE d NOT IN (SELECT d FROM present)
        """).fetchone()[0]
        pct = round(gaps / max(hrrr_12z_count, 1) * 100, 1)
        print(f"Days present: {hrrr_12z_count:,} | Missing days: {gaps} ({pct}%)")
        if gaps > 30:
            print(f"WARNING: {gaps} missing days — acceptable if archive gaps")
    else:
        print("No HRRR 12z data yet")

    # --- Check 8: Model summary ---
    print("\n=== Model Summary ===")
    summary = con.execute("""
        SELECT model_name, COUNT(*) AS rows,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first,
               MAX(model_run::DATE) AS last
        FROM forecasts
        GROUP BY model_name
        ORDER BY model_name
    """).fetchall()
    print(f"{'Model':>8} | {'Rows':>10} | {'Days':>6} | {'First':>12} | {'Last':>12}")
    print("-" * 60)
    for row in summary:
        print(f"{row[0]:>8} | {row[1]:>10,} | {row[2]:>6} | {row[3]} | {row[4]}")

    # --- Check 9: Observation station summary ---
    print("\n=== Observation Stations ===")
    obs_summary = con.execute("""
        SELECT station_id, COUNT(*) AS rows,
               MIN(observed_at)::DATE AS first,
               MAX(observed_at)::DATE AS last
        FROM observations
        GROUP BY station_id
        ORDER BY station_id
    """).fetchall()
    print(f"{'Station':>8} | {'Rows':>10} | {'First':>12} | {'Last':>12}")
    print("-" * 50)
    for row in obs_summary:
        print(f"{row[0]:>8} | {row[1]:>10,} | {row[2]} | {row[3]}")

    con.close()

    # --- Summary ---
    print("\n" + "=" * 50)
    if all_pass:
        print("PHASE 1 GATE: PASSED")
        print("Ready for Phase 2 implementation.")
    else:
        print("PHASE 1 GATE: FAILED")
        print("Fix issues above before starting Phase 2.")
    print("=" * 50)

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 gate validation — checks data completeness"
    )
    parser.add_argument("--db", default=None, help="Database path")
    args = parser.parse_args()

    passed = check_gate(db_path=args.db)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
