"""Diagnostics for multimodel_full OLS stacking.

Extracts OLS coefficients, seasonal performance, and residual analysis
to understand WHY multimodel_full beats HRRR OLS by 14%.

Usage:
    PYTHONPATH=. python scripts/multimodel_diagnostics.py
"""

import math
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
from scipy.linalg import lstsq

from core.db import DEFAULT_DB_PATH
from services.backtester import (
    _walk_forward_regression_data,
    _get_latest_model_fcst_highs_bulk,
    _get_delta_temp,
    _encode_month,
    _find_latest_run_hour,
    _get_forecast_high_for_model,
    WALK_FORWARD_MIN_DAYS,
)
from services.ensemble import _find_latest_run_hour


def diagnose_coefficients(db_path=DEFAULT_DB_PATH):
    """Extract OLS coefficients for multimodel_full at select run hours.

    Shows how much weight each model's forecast gets, and how that
    changes across the day.
    """
    con = duckdb.connect(db_path, read_only=True)
    eval_date = date(2025, 12, 1)  # late in dataset for max training window
    station_id = 'KNYC'

    print("=" * 80)
    print("DIAGNOSTIC 1: OLS COEFFICIENTS BY RUN HOUR (trained up to {})".format(eval_date))
    print("=" * 80)
    print("{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>6}".format(
        "Hour", "Intercept", "HRRR_high", "GFS_high", "ECMWF_high", "sin(mon)", "cos(mon)", "delta_T", "N"))
    print("-" * 80)

    for run_hour in [0, 3, 6, 9, 12, 15, 18, 21]:
        # Get HRRR training data
        training = _walk_forward_regression_data(
            con, run_hour, station_id, eval_date, model_name='hrrr',
        )
        if training is None:
            print("{:>4}z  (insufficient HRRR data)".format(run_hour))
            continue

        # Get secondary model highs
        gfs_highs = _get_latest_model_fcst_highs_bulk(
            con, 'gfs', station_id, run_hour, eval_date,
        )
        ecmwf_highs = _get_latest_model_fcst_highs_bulk(
            con, 'ecmwf', station_id, run_hour, eval_date,
        )

        # Get dates with delta_temp
        date_rows = con.execute("""
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
                    AND f.model_name = 'hrrr'
                    AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
                    AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29
                WHERE n.station_id = ?
                    AND n.obs_date < ?
                    AND n.max_temp_f IS NOT NULL
                GROUP BY n.obs_date, n.max_temp_f
                ORDER BY n.obs_date
            )
            SELECT
                obs_date, error, fcst_high, month,
                LAG(actual_high, 1) OVER (ORDER BY obs_date)
                    - LAG(actual_high, 2) OVER (ORDER BY obs_date) AS delta_temp
            FROM daily_errors
        """, [run_hour, station_id, eval_date]).fetchall()

        # Build expanded rows
        rows = []
        for row in date_rows:
            obs_d, error, hrrr_fh, m, dt = row
            if dt is None:
                continue
            gfs_fh = gfs_highs.get(obs_d, hrrr_fh)
            ecmwf_fh = ecmwf_highs.get(obs_d, hrrr_fh)
            rows.append((error, hrrr_fh, gfs_fh, ecmwf_fh, m, dt))

        if len(rows) < WALK_FORWARD_MIN_DAYS:
            print("{:>4}z  (insufficient aligned data: {} rows)".format(run_hour, len(rows)))
            continue

        # Fit OLS: [intercept, hrrr_high, gfs_high, ecmwf_high, sin_m, cos_m, delta_temp]
        n = len(rows)
        feature_indices = [0, 1, 2, 3, 4, 5]
        A = np.empty((n, 7), dtype=np.float64)
        y = np.empty(n, dtype=np.float64)

        for i, row in enumerate(rows):
            error, hrrr_h, gfs_h, ecmwf_h, month, delta = row
            sin_m, cos_m = _encode_month(month)
            A[i, 0] = 1.0
            A[i, 1] = hrrr_h
            A[i, 2] = gfs_h
            A[i, 3] = ecmwf_h
            A[i, 4] = sin_m
            A[i, 5] = cos_m
            A[i, 6] = delta
            y[i] = error

        result = lstsq(A, y)
        coeffs = result[0]

        print("{:>4}z  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}  {:>6}".format(
            run_hour, coeffs[0], coeffs[1], coeffs[2], coeffs[3],
            coeffs[4], coeffs[5], coeffs[6], n))

    print()
    con.close()


def diagnose_data_coverage(db_path=DEFAULT_DB_PATH):
    """Show how many dates have GFS/ECMWF data vs HRRR-only (imputed)."""
    con = duckdb.connect(db_path, read_only=True)
    station_id = 'KNYC'

    print("=" * 80)
    print("DIAGNOSTIC 2: SECONDARY MODEL DATA COVERAGE")
    print("=" * 80)
    print("{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}".format(
        "Hour", "HRRR days", "GFS days", "ECMWF days", "GFS %", "ECMWF %"))
    print("-" * 80)

    for run_hour in range(24):
        hrrr_count = con.execute("""
            SELECT COUNT(DISTINCT model_run::DATE) FROM forecasts
            WHERE model_name = 'hrrr' AND station_id = ?
              AND EXTRACT(HOUR FROM model_run) = ?
              AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
              AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
        """, [station_id, run_hour]).fetchone()[0]

        gfs_highs = _get_latest_model_fcst_highs_bulk(
            con, 'gfs', station_id, run_hour, date(2026, 3, 1),
        )
        ecmwf_highs = _get_latest_model_fcst_highs_bulk(
            con, 'ecmwf', station_id, run_hour, date(2026, 3, 1),
        )

        gfs_pct = 100.0 * len(gfs_highs) / hrrr_count if hrrr_count > 0 else 0
        ecmwf_pct = 100.0 * len(ecmwf_highs) / hrrr_count if hrrr_count > 0 else 0

        print("{:>4}z  {:>10}  {:>10}  {:>10}  {:>9.1f}%  {:>9.1f}%".format(
            run_hour, hrrr_count, len(gfs_highs), len(ecmwf_highs), gfs_pct, ecmwf_pct))

    print()
    con.close()


def diagnose_seasonal_brier(db_path=DEFAULT_DB_PATH):
    """Compare HRRR OLS vs multimodel_full Brier by season."""
    from services.backtester import (
        Backtester, wf_regression_full, wf_multimodel_full, RUN_HOURS,
    )

    bt = Backtester(db_path=db_path)

    # Run both models
    hrrr_result = bt.run(wf_regression_full, date(2023, 1, 1), date(2026, 2, 1), run_hours=RUN_HOURS)
    mm_result = bt.run(wf_multimodel_full, date(2023, 1, 1), date(2026, 2, 1), run_hours=RUN_HOURS)

    seasons = {
        'Winter (DJF)': [12, 1, 2],
        'Spring (MAM)': [3, 4, 5],
        'Summer (JJA)': [6, 7, 8],
        'Fall (SON)': [9, 10, 11],
    }

    print("=" * 80)
    print("DIAGNOSTIC 3: SEASONAL BRIER COMPARISON")
    print("=" * 80)
    print("{:<20}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}".format(
        "Season", "HRRR", "Multimodel", "Improve%", "HRRR n", "MM n"))
    print("-" * 80)

    for season_name, months in seasons.items():
        hrrr_season = [r for r in hrrr_result.run_results if r.settlement_date.month in months]
        mm_season = [r for r in mm_result.run_results if r.settlement_date.month in months]

        if not hrrr_season or not mm_season:
            continue

        hrrr_brier = sum(r.brier_score for r in hrrr_season) / len(hrrr_season)
        mm_brier = sum(r.brier_score for r in mm_season) / len(mm_season)
        improve = (hrrr_brier - mm_brier) / hrrr_brier * 100

        print("{:<20}  {:>10.4f}  {:>10.4f}  {:>+9.1f}%  {:>8}  {:>8}".format(
            season_name, hrrr_brier, mm_brier, improve, len(hrrr_season), len(mm_season)))

    print()


def diagnose_residuals(db_path=DEFAULT_DB_PATH):
    """Compare error distributions: HRRR OLS vs multimodel_full."""
    from services.backtester import (
        Backtester, wf_regression_full, wf_multimodel_full, RUN_HOURS,
    )

    bt = Backtester(db_path=db_path)
    hrrr_result = bt.run(wf_regression_full, date(2023, 1, 1), date(2026, 2, 1), run_hours=RUN_HOURS)
    mm_result = bt.run(wf_multimodel_full, date(2023, 1, 1), date(2026, 2, 1), run_hours=RUN_HOURS)

    # Error = predicted_bracket - actual
    hrrr_errors = [r.predicted_bracket - round(r.actual_high) for r in hrrr_result.run_results]
    mm_errors = [r.predicted_bracket - round(r.actual_high) for r in mm_result.run_results]

    print("=" * 80)
    print("DIAGNOSTIC 4: ERROR DISTRIBUTION (predicted_bracket - actual)")
    print("=" * 80)
    print("{:<20}  {:>10}  {:>10}".format("Stat", "HRRR OLS", "Multimodel"))
    print("-" * 50)

    for label, fn in [
        ("Mean error", np.mean),
        ("Median error", np.median),
        ("Std error", np.std),
        ("MAE", lambda x: np.mean(np.abs(x))),
        ("P10", lambda x: np.percentile(x, 10)),
        ("P90", lambda x: np.percentile(x, 90)),
        ("|error| <= 1", lambda x: np.mean(np.abs(x) <= 1) * 100),
        ("|error| <= 2", lambda x: np.mean(np.abs(x) <= 2) * 100),
        ("|error| <= 3", lambda x: np.mean(np.abs(x) <= 3) * 100),
    ]:
        hval = fn(hrrr_errors)
        mval = fn(mm_errors)
        if "error" in label and "<=" in label:
            print("{:<20}  {:>9.1f}%  {:>9.1f}%".format(label, hval, mval))
        else:
            print("{:<20}  {:>10.3f}  {:>10.3f}".format(label, hval, mval))
    print()


def main():
    # Run fast diagnostics first (no backtester needed)
    diagnose_coefficients()
    diagnose_data_coverage()

    # Then heavier ones that re-run the backtester
    print("\nRunning backtester for seasonal + residual diagnostics (2x ~3 min)...\n")
    diagnose_seasonal_brier()
    diagnose_residuals()


if __name__ == "__main__":
    main()
