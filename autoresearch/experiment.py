"""AlphaTemp Autoresearch — Experiment File

THIS IS THE FILE THE AI AGENT MODIFIES.

Contract:
- Export model_fn(provider, ref_time) -> Optional[Dict[int, float]]
- Return 1°F integer bracket probabilities, or None if can't forecast
- Only use training data from BEFORE the evaluation date (walk-forward)
- Python 3.9 compatible
"""

import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

from services.data_provider import BacktestDataProvider

# ── Description (updated by the agent each experiment) ──────────────────────
DESCRIPTION = "Upper tail lambda_upper=0.19 with new lower tail"

# ── Hyperparameters ─────────────────────────────────────────────────────────
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TRAIN_WINDOW_DAYS = 180
TRAIN_UPDATE_HOURS = [0, 6, 12, 18]
MIN_SAMPLES = 90
MIN_BRACKET_PROB = 0.0001

# Timezone for ET conversion
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone as _tz
    _ET = timezone(timedelta(hours=-5))

# Module-level caches (reset between harness runs via importlib)
_coeff_cache = {}  # type: Dict[Tuple[int, str], Optional[List[np.ndarray]]]


# ── Quantile Regression Solver ──────────────────────────────────────────────

def fit_quantile_regression(X, y, tau):
    # type: (np.ndarray, np.ndarray, float) -> Optional[np.ndarray]
    """Fit linear quantile regression via LP. Returns coefficients or None."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    if n < MIN_SAMPLES:
        return None

    ones = np.ones((n, 1), dtype=np.float64)
    X_aug = np.hstack([ones, X])
    k = p + 1

    c = np.concatenate([
        np.zeros(k),
        tau * np.ones(n),
        (1 - tau) * np.ones(n),
    ])

    A_eq = sparse.hstack([
        sparse.csc_matrix(X_aug),
        sparse.eye(n),
        -sparse.eye(n),
    ], format='csc')
    b_eq = y

    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result.success:
        return None
    return result.x[:k]


# ── CDF Construction ────────────────────────────────────────────────────────

def build_bracket_probs(temp_quantiles, tau_values, running_max=None, radius=15):
    # type: (List[float], List[float], Optional[float], int) -> Dict[int, float]
    """Build 1°F bracket probabilities from predicted temperature quantiles.

    Uses piecewise-linear CDF with exponential decay tails.
    """
    pairs = sorted(zip(temp_quantiles, tau_values))
    temps = [p[0] for p in pairs]
    taus = [p[1] for p in pairs]
    for i in range(1, len(taus)):
        if taus[i] < taus[i - 1]:
            taus[i] = taus[i - 1]

    q05, q95 = temps[0], temps[-1]
    tau_lo, tau_hi = taus[0], taus[-1]

    q50_idx = min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5))
    q50 = temps[q50_idx]

    spread_upper = max(q95 - q50, 0.5)
    spread_lower = max(q50 - q05, 0.5)
    lambda_upper = 0.19 / spread_upper
    lambda_lower = 0.36 / spread_lower

    def cdf(t):
        # type: (float) -> float
        if running_max is not None and t < running_max:
            return 0.0
        if t <= q05:
            return tau_lo * math.exp(-lambda_lower * (q05 - t))
        if t >= q95:
            return 1.0 - (1.0 - tau_hi) * math.exp(-lambda_upper * (t - q95))
        for i in range(len(temps) - 1):
            if temps[i] <= t <= temps[i + 1]:
                if temps[i + 1] == temps[i]:
                    return taus[i + 1]
                frac = (t - temps[i]) / (temps[i + 1] - temps[i])
                return taus[i] + frac * (taus[i + 1] - taus[i])
        return taus[-1]

    center_int = round(temps[q50_idx])
    probs = {}  # type: Dict[int, float]
    for k in range(center_int - radius, center_int + radius + 1):
        p = cdf(k + 0.5) - cdf(k - 0.5)
        if p > MIN_BRACKET_PROB:
            probs[k] = p

    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    return probs


# ── Feature Engineering (batch queries) ────────────────────────────────────

def get_training_data(con, run_hour, station_id, current_date):
    # type: (...) -> Optional[Tuple[np.ndarray, np.ndarray]]
    """Build walk-forward training data using batch queries.

    Three SQL queries total (not N+1). Returns (X_features, y_errors).
    """
    min_date = current_date - timedelta(days=TRAIN_WINDOW_DAYS)

    # Query 1: daily errors and forecast highs
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

    dates_in_window = [r[0] for r in daily_rows]

    # Query 2: ALL observations in the training window (batch)
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

    # Query 3: ALL forecasts in the training window for this run_hour (batch)
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

    # Query 4: ECMWF 00z forecast highs (multi-model consensus)
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

    # Query 5b: ECMWF extended features (radiation + dewpoint) in single query
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

    # Query 5c: GFS extended features (dewpoint for multi-model moisture comparison)
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

    # Query 5: GFS 00z forecast highs (only 00z is clean)
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

    # Build per-date lookups in Python
    obs_lookup = {}  # type: Dict[date, Dict[int, float]]
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

    fc_lookup = {}  # type: Dict[date, Dict[int, float]]
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

    # Build feature matrix
    X_rows = []
    y_rows = []
    for obs_date, error, fcst_high, month in daily_rows:
        sin_m = math.sin(2.0 * math.pi * month / 12.0)
        cos_m = math.cos(2.0 * math.pi * month / 12.0)

        # ECMWF-HRRR spread: how much ECMWF disagrees with HRRR
        ecmwf_high = ecmwf_high_lookup.get(obs_date)
        ecmwf_spread = (ecmwf_high - fcst_high) if ecmwf_high is not None else 0.0
        # HRRR diurnal range (uncertainty proxy)
        diurnal_range = fc_range_lookup.get(obs_date, 0.0)
        # GFS-HRRR spread
        gfs_high = gfs_high_lookup.get(obs_date)
        gfs_spread = (gfs_high - fcst_high) if gfs_high is not None else 0.0
        # ECMWF shortwave radiation
        solar_rad = ecmwf_rad_lookup.get(obs_date, 0.0)
        # Dewpoint depression (forecast high - dewpoint: dryness indicator)
        ecmwf_dp = ecmwf_dp_lookup.get(obs_date)
        dp_depression = (fcst_high - ecmwf_dp) if ecmwf_dp is not None else 0.0
        # ECMWF total precipitation (rain caps high temps)
        total_precip = ecmwf_precip_lookup.get(obs_date, 0.0)
        # ECMWF 2m humidity (moisture affects heating efficiency)
        humidity = ecmwf_humidity_lookup.get(obs_date, 50.0)
        # ECMWF max wind gusts (convective mixing indicator)
        max_gusts = ecmwf_gusts_lookup.get(obs_date, 0.0)
        # ECMWF CAPE (convective instability)
        cape = ecmwf_cape_lookup.get(obs_date, 0.0)
        # GFS-ECMWF dewpoint spread (moisture model disagreement)
        gfs_dp = gfs_dp_lookup.get(obs_date)
        dp_spread = (gfs_dp - ecmwf_dp) if (gfs_dp is not None and ecmwf_dp is not None) else 0.0
        # GFS precipitation (multi-model rain consensus)
        gfs_precip = gfs_precip_lookup.get(obs_date, 0.0)
        # GFS-ECMWF solar radiation spread (cloud/radiation disagreement)
        gfs_rad = gfs_rad_lookup.get(obs_date)
        solar_spread = (gfs_rad - solar_rad) if gfs_rad is not None else 0.0
        # Binary rain indicator (any model forecasts rain)
        rain_day = 1.0 if (total_precip > 0.1 or gfs_precip > 0.1) else 0.0
        # Precip model agreement (1=agree, 0=disagree on rain)
        ecmwf_rain = total_precip > 0.1
        gfs_rain = gfs_precip > 0.1
        precip_agree = 1.0 if (ecmwf_rain == gfs_rain) else 0.0
        # Yesterday's forecast error (error persistence)
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

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)
    return X, y


# ── Model Function ──────────────────────────────────────────────────────────

def model_fn(provider, ref_time):
    # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
    """Predict 1°F bracket probabilities for the given forecast scenario.

    This is the function the harness calls. Do not change the signature.
    """
    con = provider._shared_con
    station_id = provider.station_id
    run_hour = provider.model_run.hour
    current_date = provider.model_run.date()

    fcst_high = provider.get_forecast_high(station_id)
    if fcst_high is None:
        return None

    # Cache coefficients by (run_hour, date) — training data and LP fits
    # don't change between update hours for the same (run_hour, date)
    cache_key = (run_hour, str(current_date))
    if cache_key not in _coeff_cache:
        result = get_training_data(con, run_hour, station_id, current_date)
        if result is None:
            _coeff_cache[cache_key] = None
        else:
            X_train, y_train = result
            coefficients = []  # type: List[np.ndarray]
            failed = False
            for tau in QUANTILES:
                coeffs = fit_quantile_regression(X_train, y_train, tau)
                if coeffs is None:
                    _coeff_cache[cache_key] = None
                    failed = True
                    break
                coefficients.append(coeffs)
            if not failed:
                _coeff_cache[cache_key] = coefficients

    coefficients = _coeff_cache[cache_key]
    if coefficients is None:
        return None

    # Build today's feature vector (fast — just 3 queries for current date)
    ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
    update_hour_et = ref_utc.astimezone(_ET).hour
    cutoff_ts = ref_utc.timestamp()

    month = float(current_date.month)
    sin_m = math.sin(2.0 * math.pi * month / 12.0)
    cos_m = math.cos(2.0 * math.pi * month / 12.0)

    # ECMWF 00z forecast high for today
    ecmwf_row = con.execute("""
        SELECT MAX(temp_f) FROM forecasts
        WHERE station_id = ? AND model_run::DATE = ?
            AND EXTRACT(HOUR FROM model_run) = 0
            AND model_name = 'ecmwf' AND temp_f IS NOT NULL
            AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
            AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
    """, [station_id, current_date]).fetchone()
    ecmwf_high = ecmwf_row[0] if ecmwf_row and ecmwf_row[0] is not None else None
    ecmwf_spread = (ecmwf_high - fcst_high) if ecmwf_high is not None else 0.0

    # GFS 00z forecast high for today
    gfs_row = con.execute("""
        SELECT MAX(temp_f) FROM forecasts
        WHERE station_id = ? AND model_run::DATE = ?
            AND EXTRACT(HOUR FROM model_run) = 0
            AND model_name = 'gfs' AND temp_f IS NOT NULL
            AND (EXTRACT(HOUR FROM model_run) + fxx) >= 5
            AND (EXTRACT(HOUR FROM model_run) + fxx) < 29
    """, [station_id, current_date]).fetchone()
    gfs_high = gfs_row[0] if gfs_row and gfs_row[0] is not None else None
    gfs_spread = (gfs_high - fcst_high) if gfs_high is not None else 0.0

    # ECMWF extended features for today (radiation + dewpoint + precip + humidity + gusts)
    ext_row = con.execute("""
        SELECT AVG(shortwave_rad), AVG(dewpoint_2m_f), SUM(precipitation),
               AVG(humidity_2m), MAX(wind_gusts_10m), AVG(cape)
        FROM forecast_extended
        WHERE station_id = ? AND model_run::DATE = ?
            AND EXTRACT(HOUR FROM model_run) = 0
            AND model_name = 'ecmwf'
            AND EXTRACT(HOUR FROM valid_at) >= 10
            AND EXTRACT(HOUR FROM valid_at) <= 22
    """, [station_id, current_date]).fetchone()
    solar_rad = ext_row[0] if ext_row and ext_row[0] is not None else 0.0
    ecmwf_dp = ext_row[1] if ext_row and ext_row[1] is not None else None
    dp_depression = (fcst_high - ecmwf_dp) if ecmwf_dp is not None else 0.0
    total_precip = ext_row[2] if ext_row and ext_row[2] is not None else 0.0
    humidity = ext_row[3] if ext_row and ext_row[3] is not None else 50.0
    max_gusts = ext_row[4] if ext_row and ext_row[4] is not None else 0.0
    cape = ext_row[5] if ext_row and ext_row[5] is not None else 0.0

    # GFS extended features for today
    gfs_ext_row = con.execute("""
        SELECT AVG(dewpoint_2m_f), SUM(precipitation), AVG(shortwave_rad) FROM forecast_extended
        WHERE station_id = ? AND model_run::DATE = ?
            AND EXTRACT(HOUR FROM model_run) = 0
            AND model_name = 'gfs'
            AND EXTRACT(HOUR FROM valid_at) >= 10
            AND EXTRACT(HOUR FROM valid_at) <= 22
    """, [station_id, current_date]).fetchone()
    gfs_dp = gfs_ext_row[0] if gfs_ext_row and gfs_ext_row[0] is not None else None
    dp_spread = (gfs_dp - ecmwf_dp) if (gfs_dp is not None and ecmwf_dp is not None) else 0.0
    gfs_precip = gfs_ext_row[1] if gfs_ext_row and gfs_ext_row[1] is not None else 0.0
    gfs_rad = gfs_ext_row[2] if gfs_ext_row and gfs_ext_row[2] is not None else None
    solar_spread = (gfs_rad - solar_rad) if gfs_rad is not None else 0.0
    rain_day = 1.0 if (total_precip > 0.1 or gfs_precip > 0.1) else 0.0
    ecmwf_rain = total_precip > 0.1
    gfs_rain = gfs_precip > 0.1
    precip_agree = 1.0 if (ecmwf_rain == gfs_rain) else 0.0

    # Yesterday's forecast error (error persistence)
    yesterday = current_date - timedelta(days=1)
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

    running_max_div = 0.0
    slope_div = 0.0
    cum_div = 0.0
    running_max = None

    obs_today = con.execute("""
        SELECT observed_at, temp_f FROM observations
        WHERE station_id = ? AND observed_at::DATE = ? AND temp_f IS NOT NULL
        ORDER BY observed_at
    """, [station_id, current_date]).fetchall()

    fc_today = con.execute("""
        SELECT valid_at, temp_f FROM forecasts
        WHERE station_id = ? AND model_run::DATE = ?
            AND EXTRACT(HOUR FROM model_run) = ?
            AND model_name = 'hrrr' AND temp_f IS NOT NULL
        ORDER BY valid_at
    """, [station_id, current_date, run_hour]).fetchall()

    # HRRR diurnal range for today
    fc_temps_today = [r[1] for r in fc_today if r[1] is not None]
    diurnal_range = (max(fc_temps_today) - min(fc_temps_today)) if fc_temps_today else 0.0

    obs_truncated = [(o[0], o[1]) for o in obs_today
                     if o[0].replace(tzinfo=timezone.utc).timestamp() <= cutoff_ts]

    if len(obs_truncated) >= 2:
        running_max = max(t for _, t in obs_truncated)

        fc_by_hour = {}  # type: Dict[int, float]
        for valid_at, temp in fc_today:
            et_hour = valid_at.replace(tzinfo=timezone.utc).astimezone(_ET).hour
            fc_by_hour[et_hour] = temp

        divs = []
        for obs_at, obs_t in obs_truncated:
            h_et = obs_at.replace(tzinfo=timezone.utc).astimezone(_ET).hour
            fc_t = fc_by_hour.get(h_et, fcst_high)
            divs.append(obs_t - fc_t)

        if divs:
            cum_div = sum(divs) / len(divs)
            fc_at_last = fc_by_hour.get(update_hour_et, fcst_high)
            running_max_div = running_max - fc_at_last
            if len(divs) >= 2:
                slope_div = (divs[-1] - divs[0]) / max(len(divs) - 1, 1)

    features_today = np.array([
        float(update_hour_et),
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
    ])

    x_row = np.concatenate([[1.0], features_today])
    error_quantiles = [float(np.dot(c, x_row)) for c in coefficients]

    temp_quantiles = [fcst_high - eq for eq in error_quantiles]
    inverted_taus = [1.0 - tau for tau in QUANTILES]

    return build_bracket_probs(temp_quantiles, inverted_taus, running_max=running_max)
