#!/usr/bin/env python3
"""Pre-backtest data validation for the alphatemp DuckDB database.

Run this BEFORE any backtest to catch bad data early.

Usage:
    python scripts/data_audit.py
    python scripts/data_audit.py --db data/alphatemp.duckdb
"""

import argparse
import sys
from datetime import date, timedelta
from typing import Dict, List, Tuple

import duckdb
from loguru import logger

from core.db import DEFAULT_DB_PATH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _status_icon(status: str) -> str:
    if status == PASS:
        return "[PASS]"
    elif status == WARN:
        return "[WARN]"
    return "[FAIL]"


# ---------------------------------------------------------------------------
# Check 1: Row counts + date ranges
# ---------------------------------------------------------------------------

TABLE_DATE_COLS = {
    "forecasts": "valid_at",
    "observations": "observed_at",
    "mesonet_obs": "observed_at",
    "nws_daily": "obs_date",
}


def check_row_counts(con: duckdb.DuckDBPyConnection) -> str:
    _banner("Check 1: Row Counts + Date Ranges")
    status = PASS
    print(f"{'Table':<16} | {'Rows':>12} | {'Min Date':>12} | {'Max Date':>12}")
    print("-" * 62)

    for table, col in TABLE_DATE_COLS.items():
        try:
            row = con.execute(
                f"SELECT COUNT(*), MIN(CAST({col} AS DATE)), MAX(CAST({col} AS DATE)) "
                f"FROM {table}"
            ).fetchone()
            count, min_d, max_d = row
            print(f"{table:<16} | {count:>12,} | {str(min_d):>12} | {str(max_d):>12}")
            if count == 0:
                print(f"  ** {table} is EMPTY")
                status = FAIL
        except Exception as e:
            print(f"{table:<16} | ERROR: {e}")
            status = FAIL

    return status


# ---------------------------------------------------------------------------
# Check 2: Null rates
# ---------------------------------------------------------------------------

NULL_CHECKS = {
    "forecasts": "temp_f",
    "observations": "temp_f",
    "mesonet_obs": "temp_f",
    "nws_daily": "max_temp_f",
}


def check_null_rates(con: duckdb.DuckDBPyConnection) -> str:
    _banner("Check 2: Null Rates")
    status = PASS
    print(f"{'Table':<16} | {'Column':<12} | {'Total':>10} | {'Nulls':>10} | {'Rate':>7} | Status")
    print("-" * 76)

    for table, col in NULL_CHECKS.items():
        try:
            row = con.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) "
                f"FROM {table}"
            ).fetchone()
            total, nulls = row
            if total == 0:
                rate = 0.0
            else:
                rate = nulls / total * 100.0

            if rate > 20.0:
                row_status = FAIL
                status = FAIL
            elif rate > 5.0:
                row_status = WARN
                if status == PASS:
                    status = WARN
            else:
                row_status = PASS

            print(
                f"{table:<16} | {col:<12} | {total:>10,} | {nulls:>10,} | "
                f"{rate:>6.2f}% | {_status_icon(row_status)}"
            )
        except Exception as e:
            print(f"{table:<16} | ERROR: {e}")
            status = FAIL

    return status


# ---------------------------------------------------------------------------
# Check 3: Gap detection
# ---------------------------------------------------------------------------

def check_gaps(con: duckdb.DuckDBPyConnection) -> str:
    _banner("Check 3: Gap Detection")
    status = PASS

    # --- 3a: HRRR missing dates ---
    print("\n--- 3a: HRRR Missing Dates ---")
    rows = con.execute(
        "SELECT DISTINCT CAST(model_run AS DATE) FROM forecasts "
        "WHERE model_name = 'hrrr' ORDER BY 1"
    ).fetchall()

    if not rows:
        print("  No HRRR data found!")
        status = FAIL
    else:
        hrrr_dates = set(r[0] for r in rows)
        start = date(2020, 12, 12)
        end = max(hrrr_dates)
        expected = set()
        d = start
        while d <= end:
            expected.add(d)
            d += timedelta(days=1)

        missing = sorted(expected - hrrr_dates)
        print(f"  Expected range: {start} to {end} ({len(expected)} days)")
        print(f"  Present: {len(hrrr_dates)} days")
        print(f"  Missing: {len(missing)} days")

        if missing:
            # Show first 20 missing dates
            shown = missing[:20]
            for md in shown:
                print(f"    {md}")
            if len(missing) > 20:
                print(f"    ... and {len(missing) - 20} more")
            if len(missing) > 30:
                status = FAIL
                print("  [FAIL] >30 missing dates")
            else:
                if status == PASS:
                    status = WARN
                print("  [WARN] some missing dates")
        else:
            print("  [PASS] No gaps")

    # --- 3b: Mesonet gaps > 1 hour ---
    print("\n--- 3b: Mesonet Gaps > 1 Hour ---")
    gap_rows = con.execute("""
        WITH gaps AS (
            SELECT station_id, observed_at,
                   LEAD(observed_at) OVER (
                       PARTITION BY station_id ORDER BY observed_at
                   ) AS next_obs
            FROM mesonet_obs
        )
        SELECT station_id,
               observed_at AS gap_start,
               next_obs AS gap_end,
               EXTRACT(EPOCH FROM (next_obs - observed_at)) / 60.0 AS gap_minutes
        FROM gaps
        WHERE EXTRACT(EPOCH FROM (next_obs - observed_at)) / 60.0 > 60
        ORDER BY gap_minutes DESC
        LIMIT 20
    """).fetchall()

    if not gap_rows:
        print("  [PASS] No gaps > 1 hour")
    else:
        print(f"  Found {len(gap_rows)}+ gaps > 1 hour (showing top 20 by size):")
        print(f"  {'Station':<12} | {'Gap Start':<22} | {'Gap End':<22} | {'Minutes':>8}")
        print("  " + "-" * 70)
        for r in gap_rows:
            print(f"  {r[0]:<12} | {str(r[1]):<22} | {str(r[2]):<22} | {r[3]:>8.0f}")

        # Count total gaps per station
        total_gaps = con.execute("""
            WITH gaps AS (
                SELECT station_id, observed_at,
                       LEAD(observed_at) OVER (
                           PARTITION BY station_id ORDER BY observed_at
                       ) AS next_obs
                FROM mesonet_obs
            )
            SELECT station_id, COUNT(*) AS gap_count
            FROM gaps
            WHERE EXTRACT(EPOCH FROM (next_obs - observed_at)) / 60.0 > 60
            GROUP BY station_id ORDER BY gap_count DESC
        """).fetchall()
        print("\n  Gap counts by station:")
        for r in total_gaps:
            print(f"    {r[0]:<12}: {r[1]} gaps")

        if status == PASS:
            status = WARN

    return status


# ---------------------------------------------------------------------------
# Check 4: DST transition validation
# ---------------------------------------------------------------------------

def check_dst(con: duckdb.DuckDBPyConnection) -> str:
    _banner("Check 4: DST Transition Validation (mesonet_obs, UTC)")
    status = PASS

    # Spring forward 2021: 2021-03-14 02:00 EST -> 03:00 EDT (UTC: 07:00)
    # We store UTC, so there should be NO gap at spring-forward
    print("\n--- Spring Forward: 2021-03-14 ---")
    spring = con.execute("""
        SELECT observed_at
        FROM mesonet_obs
        WHERE observed_at BETWEEN '2021-03-14 05:00:00' AND '2021-03-14 10:00:00'
        ORDER BY observed_at
        LIMIT 30
    """).fetchall()

    if not spring:
        print("  No mesonet data around 2021-03-14 (may not have data this early)")
        if status == PASS:
            status = WARN
    else:
        print(f"  {len(spring)} records in UTC 05:00-10:00 window:")
        for r in spring:
            print(f"    {r[0]}")
        print("  Timestamps should be continuous (no gap) since we store UTC")

    # Fall back 2021: 2021-11-07 02:00 EDT -> 01:00 EST (UTC: 06:00)
    # We store UTC, so there should be NO duplicates
    print("\n--- Fall Back: 2021-11-07 ---")
    fall = con.execute("""
        SELECT station_id, observed_at, COUNT(*) AS cnt
        FROM mesonet_obs
        WHERE observed_at BETWEEN '2021-11-07 04:00:00' AND '2021-11-07 09:00:00'
        GROUP BY station_id, observed_at
        HAVING COUNT(*) > 1
        ORDER BY station_id, observed_at
    """).fetchall()

    if not fall:
        print("  [PASS] No duplicate UTC timestamps at fall-back")
    else:
        print("  [FAIL] Duplicate timestamps found:")
        for r in fall:
            print(f"    {r[0]} {r[1]} (count: {r[2]})")
        status = FAIL

    # Check continuity at fall-back
    fall_all = con.execute("""
        SELECT DISTINCT observed_at
        FROM mesonet_obs
        WHERE observed_at BETWEEN '2021-11-07 04:00:00' AND '2021-11-07 09:00:00'
        ORDER BY observed_at
        LIMIT 30
    """).fetchall()

    if fall_all:
        print(f"\n  {len(fall_all)} distinct timestamps in UTC 04:00-09:00 window:")
        for r in fall_all:
            print(f"    {r[0]}")

    return status


# ---------------------------------------------------------------------------
# Check 5: Spot-check physical plausibility
# ---------------------------------------------------------------------------

SPOT_CHECK_DATES = [
    date(2022, 1, 15),   # Winter
    date(2022, 4, 15),   # Spring
    date(2022, 7, 15),   # Summer
    date(2022, 10, 15),  # Fall
    date(2023, 6, 15),   # Summer (different year)
]


def check_plausibility(con: duckdb.DuckDBPyConnection) -> str:
    _banner("Check 5: Physical Plausibility Spot-Check")
    status = PASS

    print(f"{'Date':<12} | {'HRRR High':>10} | {'NWS Actual':>10} | {'Mesonet Max':>11} | Status")
    print("-" * 66)

    for d in SPOT_CHECK_DATES:
        # HRRR forecast high for that date
        hrrr_row = con.execute(
            "SELECT MAX(temp_f) FROM forecasts "
            "WHERE model_name = 'hrrr' AND CAST(valid_at AS DATE) = ?",
            [d],
        ).fetchone()
        hrrr_high = hrrr_row[0] if hrrr_row else None

        # NWS daily actual
        nws_row = con.execute(
            "SELECT max_temp_f FROM nws_daily WHERE obs_date = ?", [d]
        ).fetchone()
        nws_actual = nws_row[0] if nws_row else None

        # Mesonet daily max (any station)
        meso_row = con.execute(
            "SELECT MAX(temp_f) FROM mesonet_obs "
            "WHERE CAST(observed_at AS DATE) = ?",
            [d],
        ).fetchone()
        meso_max = meso_row[0] if meso_row else None

        # Format values
        hrrr_str = f"{hrrr_high:.1f}" if hrrr_high is not None else "N/A"
        nws_str = f"{nws_actual:.1f}" if nws_actual is not None else "N/A"
        meso_str = f"{meso_max:.1f}" if meso_max is not None else "N/A"

        # Check divergence
        values = [v for v in [hrrr_high, nws_actual, meso_max] if v is not None]
        row_status = PASS
        if len(values) >= 2:
            spread = max(values) - min(values)
            if spread > 15.0:
                row_status = FAIL
                status = FAIL

        print(
            f"{d} | {hrrr_str:>10} | {nws_str:>10} | {meso_str:>11} | "
            f"{_status_icon(row_status)}"
        )

    return status


# ---------------------------------------------------------------------------
# Check 6: GFS health post-purge
# ---------------------------------------------------------------------------

def check_gfs_health(con: duckdb.DuckDBPyConnection) -> str:
    _banner("Check 6: GFS Health Post-Purge")
    status = PASS

    rows = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(*) AS row_count,
               ROUND(AVG(temp_f), 1) AS avg_temp_f
        FROM forecasts
        WHERE model_name = 'gfs'
        GROUP BY run_hour
        ORDER BY run_hour
    """).fetchall()

    if not rows:
        print("  No GFS data found")
        return WARN

    print(f"  {'Run Hour':>8} | {'Rows':>10} | {'Avg Temp (F)':>12}")
    print("  " + "-" * 38)
    for r in rows:
        print(f"  {int(r[0]):>8} | {r[1]:>10,} | {r[2]:>12.1f}")

    # Check for non-0 run hours
    non_zero = [r for r in rows if int(r[0]) != 0]
    if non_zero:
        print(f"\n  [FAIL] Found {len(non_zero)} non-0 run hour(s) — expected only hour 0")
        status = FAIL
    else:
        print("\n  [PASS] Only hour 0 present (as expected post-purge)")

    # Check avg temp plausibility (NYC range: 20-95F)
    for r in rows:
        avg = r[2]
        if avg < 20.0 or avg > 95.0:
            print(f"  [FAIL] Run hour {int(r[0])} avg temp {avg}F outside NYC range (20-95F)")
            status = FAIL
        else:
            print(f"  [PASS] Run hour {int(r[0])} avg temp {avg}F is plausible")

    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-backtest data audit")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to DuckDB file")
    args = parser.parse_args()

    print(f"Data Audit: {args.db}")
    print(f"{'=' * 60}")

    con = duckdb.connect(args.db, read_only=True)

    results = {}  # type: Dict[str, str]
    try:
        results["1. Row Counts + Date Ranges"] = check_row_counts(con)
        results["2. Null Rates"] = check_null_rates(con)
        results["3. Gap Detection"] = check_gaps(con)
        results["4. DST Transitions"] = check_dst(con)
        results["5. Physical Plausibility"] = check_plausibility(con)
        results["6. GFS Health"] = check_gfs_health(con)
    finally:
        con.close()

    # --- Summary ---
    _banner("SUMMARY")
    overall = PASS
    for check_name, check_status in results.items():
        print(f"  {_status_icon(check_status)} {check_name}")
        if check_status == FAIL:
            overall = FAIL
        elif check_status == WARN and overall == PASS:
            overall = WARN

    print(f"\n  Overall: {_status_icon(overall)} {overall}")
    print()

    if overall == FAIL:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
