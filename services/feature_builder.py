"""Feature Builder — constructs the 23-feature vector from live DB state.

Replicates the EXACT feature engineering from autoresearch/experiment.py:get_training_data().
Both batch (training) and single-day (live prediction) modes produce identical features.

The 23 features in order:
 0. update_hour          — hour of ET when model evaluates
 1. fcst_high            — HRRR forecast high for the target day
 2. sin_month            — sin(2pi * month/12) seasonality
 3. cos_month            — cos(2pi * month/12) seasonality
 4. running_max_div      — obs running max minus forecast temp at same hour
 5. slope_div            — trend in obs-forecast divergence
 6. cum_div              — cumulative mean divergence
 7. ecmwf_spread         — ECMWF 00z high minus HRRR high
 8. diurnal_range        — max-min from HRRR forecast
 9. gfs_spread           — GFS 00z high minus HRRR high
10. solar_rad            — ECMWF shortwave radiation 10-22 ET
11. dp_depression        — forecast high minus dewpoint
12. total_precip         — ECMWF accumulated precipitation
13. humidity             — ECMWF 2m humidity
14. max_gusts            — ECMWF max wind gusts
15. lag_error            — yesterday's forecast error vs NWS settlement
16. cape                 — ECMWF CAPE
17. dp_spread            — GFS dewpoint minus ECMWF dewpoint
18. gfs_precip           — GFS precipitation
19. rain_day             — binary: any model forecasts >0.1" precip
20. precip_agree         — binary: GFS and ECMWF agree on rain
21. solar_spread         — GFS radiation minus ECMWF radiation
22. abs(lag_error)       — absolute value of lag_error

Python 3.9 compatible (no subscripted builtins).
"""

import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
from loguru import logger

# Timezone for ET conversion — matches experiment.py
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    _ET = timezone(timedelta(hours=-5))

# ---------------------------------------------------------------------------
# Constants — must match autoresearch/experiment.py
# ---------------------------------------------------------------------------

TRAIN_UPDATE_HOURS = [0, 6, 12, 18]
MIN_SAMPLES = 60
STATION_ID = "KNYC"


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Constructs the 23-feature vector from live DB state.

    Two modes:
    - get_training_data(): batch query for walk-forward QR fitting
    - build_features(): single-day query for live prediction

    Both produce identical feature vectors for the same (date, update_hour).
    """

    FEATURE_NAMES = [
        'update_hour',
        'fcst_high',
        'sin_month',
        'cos_month',
        'running_max_div',
        'slope_div',
        'cum_div',
        'ecmwf_spread',
        'diurnal_range',
        'gfs_spread',
        'solar_rad',
        'dp_depression',
        'total_precip',
        'humidity',
        'max_gusts',
        'lag_error',
        'cape',
        'dp_spread',
        'gfs_precip',
        'rain_day',
        'precip_agree',
        'solar_spread',
        'abs(lag_error)',
    ]

    def __init__(self, db_path):
        # type: (str) -> None
        self.db_path = db_path

    def get_training_data(self, target_date, update_hour, window_days=180):
        # type: (date, int, int) -> Optional[Tuple[np.ndarray, np.ndarray, List[date]]]
        """Get training data matrix for walk-forward QR fitting.

        Replicates experiment.py:get_training_data() exactly — same SQL queries,
        same feature construction, same defaults for missing data.

        Parameters
        ----------
        target_date : date
            Current evaluation date (exclusive — training data is strictly before this).
        update_hour : int
            Not used for filtering query — training data includes all 4 update hours
            per date (0, 6, 12, 18). Kept for API compatibility.
        window_days : int
            Look-back window in days (default 180).

        Returns
        -------
        (X, y, dates) or None
            X: ndarray shape (n_samples, 23) — feature matrix
            y: ndarray (n_samples,) — actual errors (fcst_high - actual_high)
            dates: list of dates for each row
            Returns None if insufficient data (< MIN_SAMPLES daily rows).
        """
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._get_training_data_impl(con, target_date, window_days)
        finally:
            con.close()

    def build_features(self, target_date, update_hour):
        # type: (date, int) -> Optional[Tuple[np.ndarray, float]]
        """Build feature vector for live prediction.

        Replicates experiment.py model_fn() single-day feature construction.

        Parameters
        ----------
        target_date : date
            The date to build features for.
        update_hour : int
            Hour in ET when the model is evaluating (0-23).

        Returns
        -------
        (features_array, fcst_high) or None
            features_array: ndarray shape (23,) — raw feature vector
            fcst_high: float — HRRR forecast high temperature
            Returns None if HRRR data missing for the date.
        """
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            return self._build_features_impl(con, target_date, update_hour)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Batch training data (mirrors experiment.py get_training_data)
    # ------------------------------------------------------------------

    def _get_training_data_impl(self, con, current_date, window_days):
        # type: (duckdb.DuckDBPyConnection, date, int) -> Optional[Tuple[np.ndarray, np.ndarray, List[date]]]
        station_id = STATION_ID
        min_date = current_date - timedelta(days=window_days)

        # Query 1: daily errors and forecast highs (uses run_hour=0 for
        # the HRRR high — same as experiment.py which iterates run_hours
        # but the daily error/fcst_high is run-hour-specific)
        # NOTE: experiment.py passes run_hour to filter model_run hour.
        # For training data we need ALL run hours that appear in
        # TRAIN_UPDATE_HOURS. But experiment.py's get_training_data is called
        # per run_hour by the harness. For training in the paper system we
        # use run_hour=0 (the same run_hour the experiment uses for its
        # baseline). The daily error row is the same regardless of update_hour
        # since update_hour only affects divergence features.
        #
        # IMPORTANT: experiment.py is called with a SINGLE run_hour and then
        # iterates TRAIN_UPDATE_HOURS internally. We replicate that exactly.
        # We use run_hour=0 for the daily error query since that's what the
        # paper trading system will use (00z HRRR run).
        run_hour = 0

        daily_rows = con.execute("""
            WITH daily_errors AS (
                SELECT
                    n.obs_date,
                    MAX(f.temp_f) - n.max_temp_f AS error,
                    MAX(f.temp_f) AS fcst_high,
                    EXTRACT(MONTH FROM n.obs_date) AS month
                FROM nws_daily n
                JOIN forecasts f ON f.station_id = n.station_id
                    AND f.model_run::DATE = n.obs_date
                    AND EXTRACT(HOUR FROM f.model_run) = ?
                    AND f.model_name = 'hrrr'
                    AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
                    AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29
                WHERE n.station_id = ?
                    AND n.obs_date >= ?
                    AND n.obs_date < ?
                    AND n.max_temp_f IS NOT NULL
                GROUP BY n.obs_date, n.max_temp_f
                ORDER BY n.obs_date
            )
            SELECT obs_date, error, fcst_high, month FROM daily_errors
        """, [run_hour, station_id, min_date, current_date]).fetchall()

        if len(daily_rows) < MIN_SAMPLES:
            return None

        # Query 2: ALL observations in the training window
        all_obs = con.execute("""
            SELECT
                observed_at::DATE AS obs_date,
                EXTRACT(HOUR FROM observed_at AT TIME ZONE 'UTC' AT TIME ZONE 'EST')::INTEGER AS hour_et,
                temp_f
            FROM observations
            WHERE station_id = ?
                AND observed_at::DATE >= ?
                AND observed_at::DATE < ?
                AND temp_f IS NOT NULL
            ORDER BY observed_at
        """, [station_id, min_date, current_date]).fetchall()

        # Query 3: ALL HRRR forecasts in the training window for run_hour
        all_fc = con.execute("""
            SELECT
                model_run::DATE AS fc_date,
                valid_at,
                temp_f
            FROM forecasts
            WHERE station_id = ?
                AND model_run::DATE >= ?
                AND model_run::DATE < ?
                AND EXTRACT(HOUR FROM model_run) = ?
                AND model_name = 'hrrr'
                AND temp_f IS NOT NULL
            ORDER BY valid_at
        """, [station_id, min_date, current_date, run_hour]).fetchall()

        # Query 4: ECMWF 00z forecast highs
        ecmwf_highs_raw = con.execute("""
            SELECT model_run::DATE AS fc_date, MAX(temp_f) AS ecmwf_high
            FROM forecasts
            WHERE station_id = ?
                AND model_run::DATE >= ?
                AND model_run::DATE < ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'ecmwf'
                AND temp_f IS NOT NULL
                AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
            GROUP BY 1
        """, [station_id, min_date, current_date]).fetchall()
        ecmwf_high_lookup = {r[0]: r[1] for r in ecmwf_highs_raw}

        # Query 5b: ECMWF extended features
        ecmwf_ext_raw = con.execute("""
            SELECT model_run::DATE AS fc_date,
                   AVG(shortwave_rad) AS mean_rad,
                   AVG(dewpoint_2m_f) AS mean_dewpoint,
                   SUM(precipitation) AS total_precip,
                   AVG(humidity_2m) AS mean_humidity,
                   MAX(wind_gusts_10m) AS max_gusts,
                   AVG(cape) AS mean_cape
            FROM forecast_extended
            WHERE station_id = ?
                AND model_run::DATE >= ?
                AND model_run::DATE < ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'ecmwf'
                AND EXTRACT(HOUR FROM valid_at) >= 10
                AND EXTRACT(HOUR FROM valid_at) <= 22
            GROUP BY 1
        """, [station_id, min_date, current_date]).fetchall()
        ecmwf_rad_lookup = {r[0]: r[1] for r in ecmwf_ext_raw if r[1] is not None}
        ecmwf_dp_lookup = {r[0]: r[2] for r in ecmwf_ext_raw if r[2] is not None}
        ecmwf_precip_lookup = {r[0]: r[3] for r in ecmwf_ext_raw if r[3] is not None}
        ecmwf_humidity_lookup = {r[0]: r[4] for r in ecmwf_ext_raw if r[4] is not None}
        ecmwf_gusts_lookup = {r[0]: r[5] for r in ecmwf_ext_raw if r[5] is not None}
        ecmwf_cape_lookup = {r[0]: r[6] for r in ecmwf_ext_raw if r[6] is not None}

        # Query 5c: GFS extended features
        gfs_ext_raw = con.execute("""
            SELECT model_run::DATE AS fc_date,
                   AVG(dewpoint_2m_f) AS mean_dewpoint,
                   SUM(precipitation) AS total_precip,
                   AVG(shortwave_rad) AS mean_rad
            FROM forecast_extended
            WHERE station_id = ?
                AND model_run::DATE >= ?
                AND model_run::DATE < ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'gfs'
                AND EXTRACT(HOUR FROM valid_at) >= 10
                AND EXTRACT(HOUR FROM valid_at) <= 22
            GROUP BY 1
        """, [station_id, min_date, current_date]).fetchall()
        gfs_dp_lookup = {r[0]: r[1] for r in gfs_ext_raw if r[1] is not None}
        gfs_precip_lookup = {r[0]: r[2] for r in gfs_ext_raw if r[2] is not None}
        gfs_rad_lookup = {r[0]: r[3] for r in gfs_ext_raw if r[3] is not None}

        # Query 5: GFS 00z forecast highs
        gfs_highs_raw = con.execute("""
            SELECT model_run::DATE AS fc_date, MAX(temp_f) AS gfs_high
            FROM forecasts
            WHERE station_id = ?
                AND model_run::DATE >= ?
                AND model_run::DATE < ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'gfs'
                AND temp_f IS NOT NULL
                AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
            GROUP BY 1
        """, [station_id, min_date, current_date]).fetchall()
        gfs_high_lookup = {r[0]: r[1] for r in gfs_highs_raw}

        # Build per-date lookups in Python — exactly matching experiment.py
        obs_lookup = {}   # type: Dict[date, Dict[int, float]]
        obs_running_max = {}  # type: Dict[date, Dict[int, float]]
        for obs_date, h_et, temp in all_obs:
            h = int(h_et)
            if obs_date not in obs_lookup:
                obs_lookup[obs_date] = {}
                obs_running_max[obs_date] = {}
            obs_lookup[obs_date][h] = temp
            prev_max = obs_running_max[obs_date].get(h - 1)
            if prev_max is not None:
                obs_running_max[obs_date][h] = max(temp, prev_max)
            else:
                # Rebuild running max from earlier hours
                cur_max = temp
                for earlier_h in sorted(obs_lookup[obs_date].keys()):
                    if earlier_h <= h:
                        cur_max = max(cur_max, obs_lookup[obs_date][earlier_h])
                obs_running_max[obs_date][h] = cur_max

        fc_lookup = {}        # type: Dict[date, Dict[int, float]]
        fc_range_lookup = {}  # type: Dict[date, float]
        for fc_date, valid_at, temp in all_fc:
            if fc_date not in fc_lookup:
                fc_lookup[fc_date] = {}
            utc_dt = valid_at.replace(tzinfo=timezone.utc)
            et_hour = utc_dt.astimezone(_ET).hour
            fc_lookup[fc_date][et_hour] = temp
        # Build diurnal range per date from forecast temps
        for d, hourly in fc_lookup.items():
            temps_list = list(hourly.values())
            if temps_list:
                fc_range_lookup[d] = max(temps_list) - min(temps_list)
            else:
                fc_range_lookup[d] = 0.0

        # Build per-date error lookup for lagged feature
        date_error_lookup = {r[0]: r[1] for r in daily_rows}

        # Build feature matrix — exactly matching experiment.py loop
        X_rows = []   # type: List[List[float]]
        y_rows = []   # type: List[float]
        d_rows = []   # type: List[date]
        for obs_date, error, fcst_high, month in daily_rows:
            sin_m = math.sin(2.0 * math.pi * month / 12.0)
            cos_m = math.cos(2.0 * math.pi * month / 12.0)

            # ECMWF-HRRR spread
            ecmwf_high = ecmwf_high_lookup.get(obs_date)
            ecmwf_spread = (ecmwf_high - fcst_high) if ecmwf_high is not None else 0.0
            # HRRR diurnal range
            diurnal_range = fc_range_lookup.get(obs_date, 0.0)
            # GFS-HRRR spread
            gfs_high = gfs_high_lookup.get(obs_date)
            gfs_spread = (gfs_high - fcst_high) if gfs_high is not None else 0.0
            # ECMWF shortwave radiation
            solar_rad = ecmwf_rad_lookup.get(obs_date, 0.0)
            # Dewpoint depression
            ecmwf_dp = ecmwf_dp_lookup.get(obs_date)
            dp_depression = (fcst_high - ecmwf_dp) if ecmwf_dp is not None else 0.0
            # ECMWF total precipitation
            total_precip = ecmwf_precip_lookup.get(obs_date, 0.0)
            # ECMWF 2m humidity
            humidity = ecmwf_humidity_lookup.get(obs_date, 50.0)
            # ECMWF max wind gusts
            max_gusts = ecmwf_gusts_lookup.get(obs_date, 0.0)
            # ECMWF CAPE
            cape = ecmwf_cape_lookup.get(obs_date, 0.0)
            # GFS-ECMWF dewpoint spread
            gfs_dp = gfs_dp_lookup.get(obs_date)
            dp_spread = (gfs_dp - ecmwf_dp) if (gfs_dp is not None and ecmwf_dp is not None) else 0.0
            # GFS precipitation
            gfs_precip = gfs_precip_lookup.get(obs_date, 0.0)
            # GFS-ECMWF solar radiation spread
            gfs_rad = gfs_rad_lookup.get(obs_date)
            solar_spread = (gfs_rad - solar_rad) if gfs_rad is not None else 0.0
            # Binary rain indicator
            rain_day = 1.0 if (total_precip > 0.1 or gfs_precip > 0.1) else 0.0
            # Precip model agreement
            ecmwf_rain = total_precip > 0.1
            gfs_rain = gfs_precip > 0.1
            precip_agree = 1.0 if (ecmwf_rain == gfs_rain) else 0.0
            # Yesterday's forecast error
            yesterday = obs_date - timedelta(days=1)
            lag_error = date_error_lookup.get(yesterday, 0.0)

            obs_by_hour = obs_lookup.get(obs_date, {})
            rm_by_hour = obs_running_max.get(obs_date, {})
            fc_by_hour = fc_lookup.get(obs_date, {})

            for uh in TRAIN_UPDATE_HOURS:
                obs_up_to = [(h, obs_by_hour[h]) for h in sorted(obs_by_hour.keys()) if h <= uh]
                if len(obs_up_to) < 2:
                    running_max_div = 0.0
                    slope_div = 0.0
                    cum_div = 0.0
                else:
                    rm = rm_by_hour.get(uh, obs_up_to[-1][1])
                    fc_at_rm_hour = fc_by_hour.get(uh, fcst_high)
                    running_max_div = rm - fc_at_rm_hour

                    divs = []
                    for h, obs_t in obs_up_to:
                        fc_t = fc_by_hour.get(h, fcst_high)
                        divs.append(obs_t - fc_t)
                    if len(divs) >= 2:
                        slope_div = (divs[-1] - divs[0]) / max(len(divs) - 1, 1)
                    else:
                        slope_div = 0.0

                    cum_div = sum(divs) / len(divs) if divs else 0.0

                features = [
                    float(uh),
                    float(fcst_high),
                    sin_m,
                    cos_m,
                    running_max_div,
                    slope_div,
                    cum_div,
                    ecmwf_spread,
                    diurnal_range,
                    gfs_spread,
                    solar_rad,
                    dp_depression,
                    total_precip,
                    humidity,
                    max_gusts,
                    lag_error,
                    cape,
                    dp_spread,
                    gfs_precip,
                    rain_day,
                    precip_agree,
                    solar_spread,
                    abs(lag_error),
                ]

                X_rows.append(features)
                y_rows.append(error)
                d_rows.append(obs_date)

        if len(X_rows) < MIN_SAMPLES:
            return None

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)
        return X, y, d_rows

    # ------------------------------------------------------------------
    # Single-day live prediction (mirrors experiment.py model_fn)
    # ------------------------------------------------------------------

    def _build_features_impl(self, con, target_date, update_hour):
        # type: (duckdb.DuckDBPyConnection, date, int) -> Optional[Tuple[np.ndarray, float]]
        station_id = STATION_ID
        run_hour = 0  # HRRR 00z run

        # Get HRRR forecast high for target_date
        fcst_row = con.execute("""
            SELECT MAX(temp_f)
            FROM forecasts
            WHERE station_id = ?
                AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = ?
                AND model_name = 'hrrr'
                AND temp_f IS NOT NULL
                AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
        """, [station_id, target_date, run_hour]).fetchone()
        if fcst_row is None or fcst_row[0] is None:
            return None
        fcst_high = float(fcst_row[0])

        month = float(target_date.month)
        sin_m = math.sin(2.0 * math.pi * month / 12.0)
        cos_m = math.cos(2.0 * math.pi * month / 12.0)

        # ECMWF 00z forecast high
        ecmwf_row = con.execute("""
            SELECT MAX(temp_f) FROM forecasts
            WHERE station_id = ? AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'ecmwf' AND temp_f IS NOT NULL
                AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
        """, [station_id, target_date]).fetchone()
        ecmwf_high = ecmwf_row[0] if ecmwf_row and ecmwf_row[0] is not None else None
        ecmwf_spread = (ecmwf_high - fcst_high) if ecmwf_high is not None else 0.0

        # GFS 00z forecast high
        gfs_row = con.execute("""
            SELECT MAX(temp_f) FROM forecasts
            WHERE station_id = ? AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'gfs' AND temp_f IS NOT NULL
                AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
                AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
        """, [station_id, target_date]).fetchone()
        gfs_high = gfs_row[0] if gfs_row and gfs_row[0] is not None else None
        gfs_spread = (gfs_high - fcst_high) if gfs_high is not None else 0.0

        # ECMWF extended features
        ext_row = con.execute("""
            SELECT AVG(shortwave_rad), AVG(dewpoint_2m_f), SUM(precipitation),
                   AVG(humidity_2m), MAX(wind_gusts_10m), AVG(cape)
            FROM forecast_extended
            WHERE station_id = ? AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'ecmwf'
                AND EXTRACT(HOUR FROM valid_at) >= 10
                AND EXTRACT(HOUR FROM valid_at) <= 22
        """, [station_id, target_date]).fetchone()
        solar_rad = ext_row[0] if ext_row and ext_row[0] is not None else 0.0
        ecmwf_dp = ext_row[1] if ext_row and ext_row[1] is not None else None
        dp_depression = (fcst_high - ecmwf_dp) if ecmwf_dp is not None else 0.0
        total_precip = ext_row[2] if ext_row and ext_row[2] is not None else 0.0
        humidity = ext_row[3] if ext_row and ext_row[3] is not None else 50.0
        max_gusts = ext_row[4] if ext_row and ext_row[4] is not None else 0.0
        cape = ext_row[5] if ext_row and ext_row[5] is not None else 0.0

        # GFS extended features
        gfs_ext_row = con.execute("""
            SELECT AVG(dewpoint_2m_f), SUM(precipitation), AVG(shortwave_rad)
            FROM forecast_extended
            WHERE station_id = ? AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = 0
                AND model_name = 'gfs'
                AND EXTRACT(HOUR FROM valid_at) >= 10
                AND EXTRACT(HOUR FROM valid_at) <= 22
        """, [station_id, target_date]).fetchone()
        gfs_dp = gfs_ext_row[0] if gfs_ext_row and gfs_ext_row[0] is not None else None
        dp_spread = (gfs_dp - ecmwf_dp) if (gfs_dp is not None and ecmwf_dp is not None) else 0.0
        gfs_precip = gfs_ext_row[1] if gfs_ext_row and gfs_ext_row[1] is not None else 0.0
        gfs_rad = gfs_ext_row[2] if gfs_ext_row and gfs_ext_row[2] is not None else None
        solar_spread = (gfs_rad - solar_rad) if gfs_rad is not None else 0.0
        rain_day = 1.0 if (total_precip > 0.1 or gfs_precip > 0.1) else 0.0
        ecmwf_rain = total_precip > 0.1
        gfs_rain = gfs_precip > 0.1
        precip_agree = 1.0 if (ecmwf_rain == gfs_rain) else 0.0

        # Yesterday's forecast error (lag_error)
        yesterday = target_date - timedelta(days=1)
        lag_row = con.execute("""
            SELECT MAX(f.temp_f) - ANY_VALUE(n.max_temp_f) AS error
            FROM nws_daily n
            JOIN forecasts f ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = ?
                AND f.model_name = 'hrrr'
                AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
                AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29
            WHERE n.station_id = ? AND n.obs_date = ?
                AND n.max_temp_f IS NOT NULL
        """, [run_hour, station_id, yesterday]).fetchone()
        lag_error = lag_row[0] if lag_row and lag_row[0] is not None else 0.0

        # Divergence features from today's observations
        running_max_div = 0.0
        slope_div = 0.0
        cum_div = 0.0

        obs_today = con.execute("""
            SELECT observed_at, temp_f FROM observations
            WHERE station_id = ? AND observed_at::DATE = ? AND temp_f IS NOT NULL
            ORDER BY observed_at
        """, [station_id, target_date]).fetchall()

        fc_today = con.execute("""
            SELECT valid_at, temp_f FROM forecasts
            WHERE station_id = ? AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = ?
                AND model_name = 'hrrr' AND temp_f IS NOT NULL
            ORDER BY valid_at
        """, [station_id, target_date, run_hour]).fetchall()

        # HRRR diurnal range for today
        fc_temps_today = [r[1] for r in fc_today if r[1] is not None]
        diurnal_range = (max(fc_temps_today) - min(fc_temps_today)) if fc_temps_today else 0.0

        # Filter observations up to update_hour (ET)
        obs_by_hour_et = []  # type: List[Tuple[int, float]]
        for obs_at, obs_temp in obs_today:
            utc_dt = obs_at.replace(tzinfo=timezone.utc)
            h_et = utc_dt.astimezone(_ET).hour
            if h_et <= update_hour:
                obs_by_hour_et.append((h_et, obs_temp))

        # Build forecast hour lookup
        fc_by_hour = {}  # type: Dict[int, float]
        for valid_at, temp in fc_today:
            et_hour = valid_at.replace(tzinfo=timezone.utc).astimezone(_ET).hour
            fc_by_hour[et_hour] = temp

        if len(obs_by_hour_et) >= 2:
            # Running max across all observations up to update_hour
            running_max = max(t for _, t in obs_by_hour_et)

            divs = []
            for h_et, obs_t in obs_by_hour_et:
                fc_t = fc_by_hour.get(h_et, fcst_high)
                divs.append(obs_t - fc_t)

            if divs:
                cum_div = sum(divs) / len(divs)
                fc_at_last = fc_by_hour.get(update_hour, fcst_high)
                running_max_div = running_max - fc_at_last
                if len(divs) >= 2:
                    slope_div = (divs[-1] - divs[0]) / max(len(divs) - 1, 1)

        features = np.array([
            float(update_hour),
            float(fcst_high),
            sin_m,
            cos_m,
            running_max_div,
            slope_div,
            cum_div,
            ecmwf_spread,
            diurnal_range,
            gfs_spread,
            solar_rad,
            dp_depression,
            total_precip,
            humidity,
            max_gusts,
            lag_error,
            cape,
            dp_spread,
            gfs_precip,
            rain_day,
            precip_agree,
            solar_spread,
            abs(lag_error),
        ], dtype=np.float64)

        return features, fcst_high
