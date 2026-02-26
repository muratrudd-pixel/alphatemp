#!/usr/bin/env python3
"""Phase 2 Exploration — Feature signal detection for HRRR error prediction.

Correlates candidate features with HRRR forecast errors to determine which
(if any) are worth including in a regression model. This is the "prove it or
kill it" gate for Phase 2.

Features tested:
  1. fcst_high (MAX forecast temp) — does HRRR miss more on hot vs cold days?
  2. month — seasonal bias pattern?
  3. delta_temp (actual D-1 minus actual D-2) — regime transition effect?

Signal thresholds:
  - |Pearson r| > 0.10 for continuous features
  - F-test p < 0.05 for monthly seasonal effect

Usage:
    cd ~/Projects/alphatemp/alphatemp
    python scripts/phase2_exploration.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("/scripts", ""))

import duckdb
from loguru import logger

DB_PATH = "data/alphatemp.duckdb"
STATION_ID = "KNYC"
RUN_HOURS = [0, 6, 12, 18]


def get_error_features(con, run_hour, station_id):
    # type: (duckdb.DuckDBPyConnection, int, str) -> list
    """Pull errors with features for one run hour.

    Returns list of (error, fcst_high, month, delta_temp) tuples.
    delta_temp = actual(D-1) - actual(D-2) — uses only prior actuals.
    """
    rows = con.execute("""
        WITH daily_errors AS (
            SELECT
                n.obs_date,
                MAX(f.temp_f) - n.max_temp_f AS error,
                MAX(f.temp_f) AS fcst_high,
                EXTRACT(MONTH FROM n.obs_date) AS month,
                n.max_temp_f AS actual_high
            FROM nws_daily n
            JOIN forecasts f ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = ?
            WHERE n.station_id = ?
                AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
            ORDER BY n.obs_date
        )
        SELECT
            error,
            fcst_high,
            month,
            LAG(actual_high, 1) OVER (ORDER BY obs_date)
                - LAG(actual_high, 2) OVER (ORDER BY obs_date) AS delta_temp
        FROM daily_errors
    """, [run_hour, station_id]).fetchall()
    return rows


def pearson_r(xs, ys):
    # type: (list, list) -> float
    """Pearson correlation coefficient. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def monthly_f_test(errors, months):
    # type: (list, list) -> tuple
    """One-way ANOVA F-test for monthly grouping of errors.

    Returns (F_statistic, p_value). Uses scipy for p-value.
    """
    from scipy.stats import f as f_dist

    # Group errors by month
    groups = {}  # type: dict
    for e, m in zip(errors, months):
        groups.setdefault(int(m), []).append(e)

    # Need at least 2 groups with 2+ samples
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(valid_groups) < 2:
        return 0.0, 1.0

    # Grand mean
    all_vals = [e for vals in valid_groups.values() for e in vals]
    grand_mean = sum(all_vals) / len(all_vals)

    # Between-group and within-group sum of squares
    ss_between = sum(
        len(vals) * (sum(vals) / len(vals) - grand_mean) ** 2
        for vals in valid_groups.values()
    )
    ss_within = sum(
        sum((e - sum(vals) / len(vals)) ** 2 for e in vals)
        for vals in valid_groups.values()
    )

    k = len(valid_groups)
    n = len(all_vals)
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or ss_within == 0:
        return 0.0, 1.0

    f_stat = (ss_between / df_between) / (ss_within / df_within)
    p_value = 1.0 - f_dist.cdf(f_stat, df_between, df_within)

    return f_stat, p_value


def analyze_run_hour(con, run_hour, station_id):
    # type: (duckdb.DuckDBPyConnection, int, str) -> dict
    """Analyze feature signals for one run hour."""
    rows = get_error_features(con, run_hour, station_id)

    # Filter out rows with NULL delta_temp (first 2 dates)
    full_rows = [(e, fh, m, dt) for e, fh, m, dt in rows if dt is not None]

    errors = [r[0] for r in full_rows]
    fcst_highs = [r[1] for r in full_rows]
    months = [r[2] for r in full_rows]
    deltas = [r[3] for r in full_rows]

    # Also get errors/months for all rows (delta_temp doesn't affect these)
    all_errors = [r[0] for r in rows]
    all_months = [r[2] for r in rows]
    all_fcst_highs = [r[1] for r in rows]

    r_fcst = pearson_r(all_fcst_highs, all_errors)
    r_delta = pearson_r(deltas, errors)
    f_stat, f_pval = monthly_f_test(all_errors, all_months)

    # Monthly mean errors
    monthly_means = {}  # type: dict
    monthly_counts = {}  # type: dict
    for e, m in zip(all_errors, all_months):
        mi = int(m)
        monthly_means[mi] = monthly_means.get(mi, 0.0) + e
        monthly_counts[mi] = monthly_counts.get(mi, 0) + 1
    for mi in monthly_means:
        monthly_means[mi] /= monthly_counts[mi]

    return {
        "n_total": len(rows),
        "n_with_delta": len(full_rows),
        "r_fcst_high": r_fcst,
        "r_delta_temp": r_delta,
        "f_stat_month": f_stat,
        "f_pval_month": f_pval,
        "monthly_means": monthly_means,
        "monthly_counts": monthly_counts,
    }


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    print("\n" + "=" * 80)
    print("PHASE 2 EXPLORATION — Feature Signal Detection")
    print("=" * 80)

    signals_found = False
    all_results = {}

    for hour in RUN_HOURS:
        result = analyze_run_hour(con, hour, STATION_ID)
        all_results[hour] = result

        print(f"\n--- {hour:02d}z (n={result['n_total']}, n_with_delta={result['n_with_delta']}) ---")

        # Forecast high correlation
        r = result["r_fcst_high"]
        flag = " *** SIGNAL" if abs(r) > 0.10 else ""
        print(f"  fcst_high vs error:  r = {r:+.4f}{flag}")

        # Delta temp correlation
        r = result["r_delta_temp"]
        flag = " *** SIGNAL" if abs(r) > 0.10 else ""
        print(f"  delta_temp vs error: r = {r:+.4f}{flag}")

        # Monthly F-test
        f_stat = result["f_stat_month"]
        f_pval = result["f_pval_month"]
        flag = " *** SIGNAL" if f_pval < 0.05 else ""
        print(f"  month F-test:        F = {f_stat:.2f}, p = {f_pval:.4f}{flag}")

        # Monthly breakdown
        print(f"  Monthly mean errors:")
        for m in sorted(result["monthly_means"].keys()):
            me = result["monthly_means"][m]
            mc = result["monthly_counts"][m]
            print(f"    {m:2d}: {me:+.3f}F (n={mc})")

        # Track if any signal found
        if (abs(result["r_fcst_high"]) > 0.10 or
                abs(result["r_delta_temp"]) > 0.10 or
                result["f_pval_month"] < 0.05):
            signals_found = True

    # Summary
    print("\n" + "=" * 80)
    print("SIGNAL SUMMARY")
    print("=" * 80)

    header = f"{'Hour':>4} | {'r(fcst)':>9} | {'r(delta)':>9} | {'F(month)':>9} | {'p(month)':>9} | {'Pass?':>5}"
    print(header)
    print("-" * 65)

    for hour in RUN_HOURS:
        r = all_results[hour]
        passes = (abs(r["r_fcst_high"]) > 0.10 or
                  abs(r["r_delta_temp"]) > 0.10 or
                  r["f_pval_month"] < 0.05)
        print(
            f"{hour:02d}z  | {r['r_fcst_high']:+.4f}   | {r['r_delta_temp']:+.4f}   | "
            f"{r['f_stat_month']:8.2f}  | {r['f_pval_month']:8.4f}  | "
            f"{'YES' if passes else ' no'}"
        )

    print("-" * 65)
    if signals_found:
        print("\nVERDICT: Signal detected. Proceed to regression models.")
    else:
        print("\nVERDICT: No signal. Phase 2 killed — skip to Phase 3.")

    con.close()


if __name__ == "__main__":
    main()
