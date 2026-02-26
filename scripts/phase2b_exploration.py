#!/usr/bin/env python3
"""Phase 2B Exploration — gate check for observation-based divergence features.

For each (run_hour, update_hour_et), computes:
  1. Phase 2 predicted bias for each historical date (expanding window)
  2. Residual = actual_error - phase2_predicted_bias
  3. Divergence features from obs truncated to update_hour
  4. Pearson r between each feature and residual

Kill condition: no (run_hour, update_hour) has |r| > 0.10 for any feature.

Usage: ./venv/bin/python scripts/phase2b_exploration.py
"""

import math
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
from scipy import stats as sp_stats

# Add project root to path
sys.path.insert(0, ".")

from services.backtester import _fit_and_predict, WALK_FORWARD_MIN_DAYS
from services.divergence import interpolate_forecast, compute_divergence_features

ET = ZoneInfo("America/New_York")
DB_PATH = "data/alphatemp.duckdb"
STATION = "KNYC"
RUN_HOURS = [0, 6, 12, 18]
UPDATE_HOURS_ET = list(range(8, 19))  # 08 through 18 ET

# Phase 2 feature indices: fcst_high + sin(month) + cos(month)
# (skip delta_temp — it was dead in Phase 2 exploration)
PHASE2_FEATURES = [0, 1, 2]


def _bulk_fetch_errors(con, run_hour, station_id):
    # type: (duckdb.DuckDBPyConnection, int, str) -> List[tuple]
    """Fetch all (obs_date, error, fcst_high, month, delta_temp) for a run_hour.

    Returns rows sorted by obs_date. delta_temp may be None for first 2 rows.
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
            obs_date,
            error,
            fcst_high,
            month,
            LAG(actual_high, 1) OVER (ORDER BY obs_date)
                - LAG(actual_high, 2) OVER (ORDER BY obs_date) AS delta_temp
        FROM daily_errors
        ORDER BY obs_date
    """, [run_hour, station_id]).fetchall()
    return rows


def _bulk_fetch_forecast_curves(con, run_hour, station_id):
    # type: (duckdb.DuckDBPyConnection, int, str) -> Dict[date, List[Tuple[float, float]]]
    """Fetch all forecast curves for a run_hour, grouped by date.

    Returns dict[obs_date] -> list of (timestamp_seconds, temp_f) sorted by valid_at.
    """
    rows = con.execute("""
        SELECT
            f.model_run::DATE as obs_date,
            f.valid_at,
            f.temp_f
        FROM forecasts f
        WHERE f.station_id = ?
            AND EXTRACT(HOUR FROM f.model_run) = ?
        ORDER BY f.model_run::DATE, f.valid_at
    """, [station_id, run_hour]).fetchall()

    curves = {}  # type: Dict[date, List[Tuple[float, float]]]
    for obs_date, valid_at, temp_f in rows:
        if obs_date not in curves:
            curves[obs_date] = []
        ts = valid_at.timestamp() if hasattr(valid_at, 'timestamp') else float(valid_at)
        curves[obs_date].append((ts, temp_f))
    return curves


def _bulk_fetch_observations(con, station_id):
    # type: (duckdb.DuckDBPyConnection, str) -> Dict[date, List[Tuple[float, float]]]
    """Fetch all observations, grouped by ET date.

    Returns dict[et_date] -> list of (timestamp_seconds, temp_f) sorted by observed_at.
    Observations are assigned to their Eastern Time date.
    """
    rows = con.execute("""
        SELECT observed_at, temp_f
        FROM observations
        WHERE station_id = ?
            AND temp_f IS NOT NULL
        ORDER BY observed_at
    """, [station_id]).fetchall()

    obs_by_date = {}  # type: Dict[date, List[Tuple[float, float]]]
    for observed_at, temp_f in rows:
        # Assign to ET date
        if hasattr(observed_at, 'replace'):
            obs_utc = observed_at.replace(tzinfo=timezone.utc)
        else:
            obs_utc = datetime.fromtimestamp(float(observed_at), tz=timezone.utc)
        et_date = obs_utc.astimezone(ET).date()
        ts = observed_at.timestamp() if hasattr(observed_at, 'timestamp') else float(observed_at)

        if et_date not in obs_by_date:
            obs_by_date[et_date] = []
        obs_by_date[et_date].append((ts, temp_f))
    return obs_by_date


def _phase2_predict_for_date(all_errors, date_idx):
    # type: (list, int) -> Optional[float]
    """Predict Phase 2 bias for date at date_idx using all prior rows.

    Returns predicted_bias or None if insufficient training data.
    """
    # Training data = rows before date_idx with non-None delta_temp
    training = []
    for i in range(date_idx):
        _, error, fcst_high, month, delta_temp = all_errors[i]
        if delta_temp is not None:
            training.append((error, fcst_high, month, delta_temp))

    if len(training) < WALK_FORWARD_MIN_DAYS:
        return None

    # Today's features
    _, _, fcst_today, month_today, delta_today = all_errors[date_idx]
    if delta_today is None:
        delta_today = 0.0  # Fallback for exploration — not critical
    features_today = (fcst_today, month_today, delta_today)

    result = _fit_and_predict(training, features_today, PHASE2_FEATURES)
    if result is None:
        return None
    return result[0]  # predicted_bias


def _update_hour_to_utc(obs_date, update_hour_et):
    # type: (date, int) -> float
    """Convert an ET update hour to UTC timestamp for a given date."""
    et_dt = datetime(obs_date.year, obs_date.month, obs_date.day,
                     update_hour_et, 0, tzinfo=ET)
    utc_dt = et_dt.astimezone(timezone.utc)
    return utc_dt.timestamp()


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    print("=" * 80)
    print("Phase 2B Exploration — Divergence Feature Signal Detection")
    print("=" * 80)

    # Pre-fetch all observations once (shared across run_hours)
    print("\nFetching observations...")
    obs_by_date = _bulk_fetch_observations(con, STATION)
    print(f"  {sum(len(v) for v in obs_by_date.values())} obs across {len(obs_by_date)} dates")

    any_signal = False

    for run_hour in RUN_HOURS:
        print(f"\n{'='*80}")
        print(f"RUN HOUR: {run_hour:02d}z")
        print(f"{'='*80}")

        # Fetch all errors + forecast curves for this run_hour
        all_errors = _bulk_fetch_errors(con, run_hour, STATION)
        curves = _bulk_fetch_forecast_curves(con, run_hour, STATION)
        print(f"  {len(all_errors)} dates with forecasts, {len(curves)} with curves")

        # Build index: obs_date -> index in all_errors
        date_to_idx = {}
        for i, (obs_date, *_) in enumerate(all_errors):
            date_to_idx[obs_date] = i

        # For each update hour, collect (residual, features) pairs
        for update_hr in UPDATE_HOURS_ET:
            residuals = []
            feat_temp = []
            feat_cumul = []
            feat_runmax = []
            feat_slope = []

            for i, (obs_date, actual_error, fcst_high, month, delta_temp) in enumerate(all_errors):
                # Need Phase 2 prediction for this date
                p2_bias = _phase2_predict_for_date(all_errors, i)
                if p2_bias is None:
                    continue

                residual = actual_error - p2_bias

                # Need forecast curve for this date
                if obs_date not in curves:
                    continue
                curve = curves[obs_date]
                fc_ts = [c[0] for c in curve]
                fc_temps = [c[1] for c in curve]

                # Need observations for this date, truncated to update hour
                if obs_date not in obs_by_date:
                    continue
                cutoff_ts = _update_hour_to_utc(obs_date, update_hr)
                day_obs = [(ts, temp) for ts, temp in obs_by_date[obs_date] if ts <= cutoff_ts]
                if len(day_obs) < 2:
                    continue

                obs_ts = [o[0] for o in day_obs]
                obs_temps = [o[1] for o in day_obs]

                # Interpolate forecast to obs times
                fcst_interp = interpolate_forecast(fc_ts, fc_temps, obs_ts)

                # Hours since first obs (for slope regression)
                t0 = obs_ts[0]
                obs_hours = [(t - t0) / 3600.0 for t in obs_ts]

                # Forecast curve temps up to update time
                fcst_up_to_t = [temp for ts, temp in curve if ts <= cutoff_ts]

                features = compute_divergence_features(
                    obs_temps, fcst_interp, obs_hours, fcst_up_to_t
                )
                if features is None:
                    continue

                residuals.append(residual)
                feat_temp.append(features["temp_divergence"])
                feat_cumul.append(features["cumulative_divergence"])
                feat_runmax.append(features["running_max_divergence"])
                feat_slope.append(features["slope_divergence"])

            # Compute correlations
            n = len(residuals)
            if n < 30:
                print(f"  {update_hr:02d} ET: n={n} (too few samples, skipping)")
                continue

            res_arr = np.array(residuals)
            results = []
            for name, feat in [
                ("temp_div", feat_temp),
                ("cumul_div", feat_cumul),
                ("runmax_div", feat_runmax),
                ("slope_div", feat_slope),
            ]:
                arr = np.array(feat)
                r, p = sp_stats.pearsonr(arr, res_arr)
                results.append((name, r, p))
                if abs(r) > 0.10:
                    any_signal = True

            # Combined R² (all 4 features)
            X = np.column_stack([feat_temp, feat_cumul, feat_runmax, feat_slope])
            X = np.column_stack([np.ones(n), X])  # Add intercept
            try:
                coeffs, res, _, _ = np.linalg.lstsq(X, res_arr, rcond=None)
                y_hat = X @ coeffs
                ss_res = np.sum((res_arr - y_hat) ** 2)
                ss_tot = np.sum((res_arr - np.mean(res_arr)) ** 2)
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except Exception:
                r_squared = 0.0

            print(f"  {update_hr:02d} ET: n={n:4d}  ", end="")
            for name, r, p in results:
                sig = "*" if abs(r) > 0.10 else " "
                print(f"{name}={r:+.3f}{sig}  ", end="")
            print(f"R²={r_squared:.3f}")

    con.close()

    print(f"\n{'='*80}")
    if any_signal:
        print("GATE: PASS — at least one feature has |r| > 0.10")
        print("Proceed with Phase 2B implementation.")
    else:
        print("GATE: FAIL — no feature has |r| > 0.10")
        print("Phase 2B killed. Observation divergence doesn't predict residuals.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
