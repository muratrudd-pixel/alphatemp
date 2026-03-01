"""XGBoost bias correction and quantile regression for Phase 2.

Two modes:
1. Point estimate: standard regression for bias correction
2. Quantile regression: 7 quantiles for direct uncertainty estimation

Features:
    fcst_high, sin_month, cos_month, delta_temp, ensemble_spread
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("xgboost not installed — XGBoost candidate disabled")

from scipy.stats import norm

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
WALK_FORWARD_MIN_DAYS = 90

# Hyperparameters (fixed — no CV inside walk-forward loop)
_XGB_PARAMS = {
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "verbosity": 0,
}
_N_ROUNDS = 200


def fit_xgb_bias(X, y, n_rounds=_N_ROUNDS):
    # type: (np.ndarray, np.ndarray, int) -> object
    """Fit XGBoost regression model for bias correction.

    Args:
        X: Feature matrix (n, p)
        y: Target values (actual high temps) (n,)
        n_rounds: Number of boosting rounds

    Returns:
        Trained xgb.Booster
    """
    if xgb is None:
        raise ImportError("xgboost required for XGBoost candidate")

    dtrain = xgb.DMatrix(X, label=y)
    params = dict(_XGB_PARAMS)
    params["objective"] = "reg:squarederror"
    model = xgb.train(params, dtrain, num_boost_round=n_rounds)
    return model


def fit_xgb_quantiles(X, y, n_rounds=_N_ROUNDS):
    # type: (np.ndarray, np.ndarray, int) -> List[object]
    """Fit XGBoost quantile regression models for each quantile.

    Returns list of 7 trained models (one per quantile).
    """
    if xgb is None:
        raise ImportError("xgboost required for XGBoost candidate")

    models = []
    for q in QUANTILES:
        dtrain = xgb.DMatrix(X, label=y)
        params = dict(_XGB_PARAMS)
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = q
        model = xgb.train(params, dtrain, num_boost_round=n_rounds)
        models.append(model)

    return models


def predict_quantile_brackets(models, X_single):
    # type: (List[object], np.ndarray) -> Dict[int, float]
    """Predict bracket probabilities from quantile models.

    Uses piecewise-linear CDF between quantile points,
    exponential decay for tails beyond q05/q95.

    Args:
        models: List of 7 trained quantile models
        X_single: Feature vector for a single prediction (1, p)

    Returns:
        Dict[int, float] -- 1-deg-F bracket probabilities
    """
    if xgb is None:
        raise ImportError("xgboost required")

    dtest = xgb.DMatrix(X_single)
    quantile_values = [float(m.predict(dtest)[0]) for m in models]

    # Enforce monotonicity (forward-scan correction)
    for i in range(1, len(quantile_values)):
        if quantile_values[i] < quantile_values[i - 1]:
            quantile_values[i] = quantile_values[i - 1] + 0.01

    q05, q10, q25, q50, q75, q90, q95 = quantile_values

    # Build piecewise-linear CDF
    quantile_points = list(zip(QUANTILES, quantile_values))

    def cdf(t):
        """Piecewise-linear CDF with exponential tails."""
        if t <= q05:
            # Lower tail: exponential decay
            lam = 1.0 / max(q50 - q05, 0.5)
            return 0.05 * math.exp(-lam * (q05 - t))
        elif t >= q95:
            # Upper tail: exponential decay
            lam = 1.0 / max(q95 - q50, 0.5)
            return 1.0 - 0.05 * math.exp(-lam * (t - q95))
        else:
            # Interior: piecewise linear
            for i in range(len(quantile_points) - 1):
                tau_lo, val_lo = quantile_points[i]
                tau_hi, val_hi = quantile_points[i + 1]
                if val_lo <= t <= val_hi:
                    if val_hi - val_lo < 0.01:
                        return (tau_lo + tau_hi) / 2
                    frac = (t - val_lo) / (val_hi - val_lo)
                    return tau_lo + frac * (tau_hi - tau_lo)
            return 0.5  # Fallback

    # Generate bracket probabilities
    center = int(round(q50))
    bracket_probs = {}
    for temp in range(center - 15, center + 16):
        p = cdf(temp + 0.5) - cdf(temp - 0.5)
        if p > 1e-6:
            bracket_probs[temp] = max(p, 0.0)

    # Normalize
    total = sum(bracket_probs.values())
    if total > 0:
        bracket_probs = {k: v / total for k, v in bracket_probs.items()}

    return bracket_probs


# ---------------------------------------------------------------------------
# Walk-forward model function for backtester integration
# ---------------------------------------------------------------------------

def _get_training_data(con, run_hour, forecast_date):
    # type: (object, int, object) -> Optional[Tuple[np.ndarray, np.ndarray]]
    """Get training features + targets from gold views.

    Returns (X_train, y_train) or None if insufficient data.
    Features: fcst_high, sin_month, cos_month, delta_temp, spread
    where delta_temp = actual(D-1) - actual(D-2) from nws_daily.
    """
    training = con.execute("""
        WITH ordered AS (
            SELECT
                g.forecast_date,
                g.fcst_high,
                g.sin_month,
                g.cos_month,
                COALESCE(m.ensemble_spread, 0.0) AS spread,
                n.max_temp_f,
                LAG(n.max_temp_f, 1) OVER (ORDER BY g.forecast_date)
                    - LAG(n.max_temp_f, 2) OVER (ORDER BY g.forecast_date)
                    AS delta_temp
            FROM gold_hrrr_bias_features g
            JOIN nws_daily n ON g.forecast_date = n.obs_date AND n.station_id = 'KNYC'
            LEFT JOIN gold_multi_model_features m
                ON g.forecast_date = m.forecast_date AND g.run_hour = m.run_hour
            WHERE g.run_hour = ?
              AND g.forecast_date < ?
            ORDER BY g.forecast_date
        )
        SELECT fcst_high, sin_month, cos_month, delta_temp, spread, max_temp_f
        FROM ordered
        WHERE delta_temp IS NOT NULL
    """, [run_hour, forecast_date]).fetchall()

    if len(training) < WALK_FORWARD_MIN_DAYS:
        return None

    X = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in training])
    y = np.array([r[5] for r in training])
    return (X, y)


def xgboost_model_fn(provider, ref_time):
    # type: (object, object) -> Optional[Dict[int, float]]
    """XGBoost walk-forward model function for Backtester."""
    if xgb is None:
        return None

    from services.data_provider import BacktestDataProvider
    if not isinstance(provider, BacktestDataProvider):
        return None

    fcst_high = provider.get_forecast_high('KNYC')
    if fcst_high is None:
        return None

    model_run = provider.model_run
    forecast_date = model_run.date() if hasattr(model_run, 'date') else model_run
    run_hour = model_run.hour if hasattr(model_run, 'hour') else 0

    con = provider._shared_con

    # Get training data
    data = _get_training_data(con, run_hour, forecast_date)
    if data is None:
        return None
    X_train, y_train = data

    # Get current ensemble spread
    row = con.execute(
        "SELECT ensemble_spread FROM gold_multi_model_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    spread = row[0] if row and row[0] is not None else 0.0

    # Get delta_temp from nws_daily (actual(D-1) - actual(D-2))
    delta_rows = con.execute("""
        SELECT max_temp_f FROM nws_daily
        WHERE station_id = 'KNYC' AND obs_date < ?
          AND max_temp_f IS NOT NULL
        ORDER BY obs_date DESC LIMIT 2
    """, [forecast_date]).fetchall()
    delta_temp = (delta_rows[0][0] - delta_rows[1][0]) if len(delta_rows) >= 2 else 0.0

    # Build feature vector for prediction
    month = forecast_date.month
    sin_m = math.sin(2 * math.pi * month / 12.0)
    cos_m = math.cos(2 * math.pi * month / 12.0)

    X_pred = np.array([[fcst_high, sin_m, cos_m, delta_temp, spread]])

    # Fit quantile models and predict
    models = fit_xgb_quantiles(X_train, y_train, n_rounds=_N_ROUNDS)
    bracket_probs = predict_quantile_brackets(models, X_pred)

    return bracket_probs


# Expose raw for Phase 2B integration
def _xgb_raw(provider, ref_time):
    # type: (object, object) -> Optional[Tuple[float, float]]
    """Return (center, std) estimate from XGBoost quantiles."""
    probs = xgboost_model_fn(provider, ref_time)
    if probs is None:
        return None
    # Estimate center and std from bracket probabilities
    temps = sorted(probs.keys())
    center = sum(t * p for t, p in probs.items())
    variance = sum(p * (t - center) ** 2 for t, p in probs.items())
    std = max(math.sqrt(variance), 0.3)
    return (center, std)


xgboost_model_fn.raw = _xgb_raw  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Extended features variant: base features + weather variables from
# forecast_extended (dewpoint, humidity, wind, pressure, cloud, precip,
# radiation, dewpoint depression). These failed under OLS — the hypothesis
# is that XGBoost can capture nonlinear interactions OLS can't.
# ---------------------------------------------------------------------------

def _get_training_data_extended(con, run_hour, forecast_date):
    # type: (object, int, object) -> Optional[Tuple[np.ndarray, np.ndarray]]
    """Get training features + targets including extended weather variables.

    Returns (X_train, y_train) or None if insufficient data.
    Features: fcst_high, sin_month, cos_month, delta_temp, spread,
              mean_dewpoint, mean_humidity, max_wind, mean_pressure,
              mean_cloud, total_precip, mean_radiation, dewpoint_depression
    """
    training = con.execute("""
        WITH ordered AS (
            SELECT
                g.forecast_date,
                g.fcst_high,
                g.sin_month,
                g.cos_month,
                COALESCE(m.ensemble_spread, 0.0) AS spread,
                n.max_temp_f,
                LAG(n.max_temp_f, 1) OVER (ORDER BY g.forecast_date)
                    - LAG(n.max_temp_f, 2) OVER (ORDER BY g.forecast_date)
                    AS delta_temp,
                AVG(fe.dewpoint_2m_f) AS mean_dewpoint,
                AVG(fe.humidity_2m) AS mean_humidity,
                MAX(fe.wind_speed_10m) AS max_wind,
                AVG(fe.pressure_msl) AS mean_pressure,
                AVG(fe.cloud_cover) AS mean_cloud,
                COALESCE(SUM(fe.precipitation), 0) AS total_precip,
                AVG(fe.shortwave_rad) AS mean_radiation,
                AVG(f.temp_f) - AVG(fe.dewpoint_2m_f) AS dewpoint_depression
            FROM gold_hrrr_bias_features g
            JOIN nws_daily n ON g.forecast_date = n.obs_date AND n.station_id = 'KNYC'
            LEFT JOIN gold_multi_model_features m
                ON g.forecast_date = m.forecast_date AND g.run_hour = m.run_hour
            LEFT JOIN forecasts f
                ON f.station_id = 'KNYC'
                AND f.model_run::DATE = g.forecast_date
                AND EXTRACT(HOUR FROM f.model_run) = g.run_hour
                AND f.model_name = 'hrrr'
                AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
                AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29
            LEFT JOIN forecast_extended fe
                ON fe.station_id = f.station_id
                AND fe.model_run = f.model_run
                AND fe.valid_at = f.valid_at
                AND fe.model_name = f.model_name
            WHERE g.run_hour = ?
              AND g.forecast_date < ?
            GROUP BY g.forecast_date, g.fcst_high, g.sin_month, g.cos_month,
                     spread, n.max_temp_f
            ORDER BY g.forecast_date
        )
        SELECT fcst_high, sin_month, cos_month, delta_temp, spread,
               COALESCE(mean_dewpoint, 0), COALESCE(mean_humidity, 0),
               COALESCE(max_wind, 0), COALESCE(mean_pressure, 0),
               COALESCE(mean_cloud, 0), COALESCE(total_precip, 0),
               COALESCE(mean_radiation, 0), COALESCE(dewpoint_depression, 0),
               max_temp_f
        FROM ordered
        WHERE delta_temp IS NOT NULL
    """, [run_hour, forecast_date]).fetchall()

    if len(training) < WALK_FORWARD_MIN_DAYS:
        return None

    # 13 features: base (5) + extended (8)
    X = np.array([[r[i] for i in range(13)] for r in training])
    y = np.array([r[13] for r in training])
    return (X, y)


def xgboost_extended_model_fn(provider, ref_time):
    # type: (object, object) -> Optional[Dict[int, float]]
    """XGBoost walk-forward model with extended weather features."""
    if xgb is None:
        return None

    from services.data_provider import BacktestDataProvider
    if not isinstance(provider, BacktestDataProvider):
        return None

    fcst_high = provider.get_forecast_high('KNYC')
    if fcst_high is None:
        return None

    model_run = provider.model_run
    forecast_date = model_run.date() if hasattr(model_run, 'date') else model_run
    run_hour = model_run.hour if hasattr(model_run, 'hour') else 0

    con = provider._shared_con

    data = _get_training_data_extended(con, run_hour, forecast_date)
    if data is None:
        return None
    X_train, y_train = data

    # Get current-day features
    row = con.execute(
        "SELECT ensemble_spread FROM gold_multi_model_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    spread = row[0] if row and row[0] is not None else 0.0

    delta_rows = con.execute("""
        SELECT max_temp_f FROM nws_daily
        WHERE station_id = 'KNYC' AND obs_date < ?
          AND max_temp_f IS NOT NULL
        ORDER BY obs_date DESC LIMIT 2
    """, [forecast_date]).fetchall()
    delta_temp = (delta_rows[0][0] - delta_rows[1][0]) if len(delta_rows) >= 2 else 0.0

    # Get extended features for today (settlement-day fxx only)
    ext_row = con.execute("""
        SELECT
            AVG(fe.dewpoint_2m_f),
            AVG(fe.humidity_2m),
            MAX(fe.wind_speed_10m),
            AVG(fe.pressure_msl),
            AVG(fe.cloud_cover),
            COALESCE(SUM(fe.precipitation), 0),
            AVG(fe.shortwave_rad),
            AVG(f.temp_f) - AVG(fe.dewpoint_2m_f)
        FROM forecasts f
        JOIN forecast_extended fe
            ON fe.station_id = f.station_id
            AND fe.model_run = f.model_run
            AND fe.valid_at = f.valid_at
            AND fe.model_name = f.model_name
        WHERE f.station_id = 'KNYC'
            AND f.model_run::DATE = ?
            AND EXTRACT(HOUR FROM f.model_run) = ?
            AND f.model_name = 'hrrr'
            AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
            AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29
    """, [forecast_date, run_hour]).fetchone()

    if ext_row is None or ext_row[0] is None:
        # Fall back to base model if no extended features available
        return xgboost_model_fn(provider, ref_time)

    ext_features = tuple(v if v is not None else 0.0 for v in ext_row)

    month = forecast_date.month
    sin_m = math.sin(2 * math.pi * month / 12.0)
    cos_m = math.cos(2 * math.pi * month / 12.0)

    X_pred = np.array([[fcst_high, sin_m, cos_m, delta_temp, spread] + list(ext_features)])

    models = fit_xgb_quantiles(X_train, y_train, n_rounds=_N_ROUNDS)
    bracket_probs = predict_quantile_brackets(models, X_pred)

    return bracket_probs


xgboost_extended_model_fn.raw = _xgb_raw  # type: ignore[attr-defined]
