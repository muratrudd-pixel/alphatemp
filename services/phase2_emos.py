"""EMOS (Ensemble Model Output Statistics) bias correction.

Gneiting et al. 2005 — Non-Homogeneous Gaussian Regression.
Jointly estimates mean and variance from ensemble output:
    mu    = a + b * fcst_high
    sigma = sqrt(max(c + d * spread^2, 0.09))

Trained via CRPS minimization (scipy.optimize.minimize).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import norm

WALK_FORWARD_MIN_DAYS = 90

# Warm-start caches: weather biases evolve slowly, so yesterday's coefficients
# are a near-perfect starting point for today's optimization.
_warm_start_cache = {}  # type: Dict[Tuple[int, object], Tuple[float, float, float, float]]
_fitted_cache = {}  # type: Dict[Tuple[int, object], Tuple[float, float, float, float]]


def _crps_gaussian(mu, sigma, obs):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> float
    """Mean CRPS for Gaussian predictions vs observations.

    CRPS(N(mu, sigma), y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma
    """
    z = (obs - mu) / sigma
    crps_vals = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )
    return float(np.mean(crps_vals))


def fit_emos(fcst, spread, actual, x0=None):
    # type: (np.ndarray, np.ndarray, np.ndarray, Optional[Tuple]) -> Tuple[float, float, float, float]
    """Fit EMOS coefficients (a, b, c, d) via CRPS minimization.

    Args:
        fcst: Forecast high temperatures (n,)
        spread: Ensemble spread values (n,)
        actual: Observed high temperatures (n,)
        x0: Initial guess for (a, b, c, d). If None, computed from data.

    Returns:
        (a, b, c, d) tuple of fitted coefficients.
    """
    fcst = np.asarray(fcst, dtype=float)
    spread = np.asarray(spread, dtype=float)
    actual = np.asarray(actual, dtype=float)

    def objective(params):
        a, b, c, d = params
        mu = a + b * fcst
        var = c + d * spread ** 2
        var = np.maximum(var, 0.09)  # Floor at 0.3^2
        sigma = np.sqrt(var)
        return _crps_gaussian(mu, sigma, actual)

    # Initial guess: no bias, unit slope, empirical variance, no spread effect
    if x0 is None:
        residuals = actual - fcst
        x0 = (float(np.mean(residuals)), 1.0, float(np.var(residuals)), 0.1)

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
    )

    a, b, c, d = result.x
    return (float(a), float(b), float(c), float(d))


def predict_emos(coeffs, fcst_high, ensemble_spread):
    # type: (Tuple[float, float, float, float], float, float) -> Tuple[float, float]
    """Predict center and std using fitted EMOS coefficients.

    Args:
        coeffs: (a, b, c, d) from fit_emos
        fcst_high: Single forecast high temperature
        ensemble_spread: Ensemble spread for this prediction

    Returns:
        (center, std) tuple.
    """
    a, b, c, d = coeffs
    center = a + b * fcst_high
    var = c + d * ensemble_spread ** 2
    var = max(var, 0.09)  # Floor at 0.3^2
    std = math.sqrt(var)
    return (float(center), float(std))


# ---------------------------------------------------------------------------
# Walk-forward model function for backtester integration
# ---------------------------------------------------------------------------

def emos_model_fn(provider, ref_time):
    # type: (object, object) -> Optional[Dict[int, float]]
    """EMOS walk-forward model function compatible with Backtester.

    Reads forecast high from provider, queries gold_multi_model_features
    for ensemble spread, fits EMOS on expanding window, returns bracket probs.
    """
    from services.data_provider import BacktestDataProvider

    if not isinstance(provider, BacktestDataProvider):
        return None

    # Get forecast high
    fcst_high = provider.get_forecast_high('KNYC')
    if fcst_high is None:
        return None

    # Get ensemble spread from gold view
    model_run = provider.model_run
    forecast_date = model_run.date() if hasattr(model_run, 'date') else model_run
    run_hour = model_run.hour if hasattr(model_run, 'hour') else 0

    con = provider._shared_con
    row = con.execute(
        "SELECT ensemble_spread FROM gold_multi_model_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    ensemble_spread = row[0] if row and row[0] is not None else 0.0

    # Check fitted cache first — avoids refitting for repeated calls
    cache_key = (run_hour, forecast_date)
    if cache_key in _fitted_cache:
        coeffs = _fitted_cache[cache_key]
    else:
        # Get training data: all prior dates
        training = con.execute("""
            SELECT g.fcst_high, COALESCE(m.ensemble_spread, 0.0), n.max_temp_f
            FROM gold_hrrr_bias_features g
            JOIN nws_daily n ON g.forecast_date = n.obs_date AND n.station_id = 'KNYC'
            LEFT JOIN gold_multi_model_features m
                ON g.forecast_date = m.forecast_date AND g.run_hour = m.run_hour
            WHERE g.run_hour = ?
              AND g.forecast_date < ?
            ORDER BY g.forecast_date
        """, [run_hour, forecast_date]).fetchall()

        if len(training) < WALK_FORWARD_MIN_DAYS:
            return None

        fcst_arr = np.array([r[0] for r in training])
        spread_arr = np.array([r[1] if r[1] is not None else 0.0 for r in training])
        actual_arr = np.array([r[2] for r in training])

        # Warm-start: use previous day's coefficients as initial guess
        warm_key = (run_hour, forecast_date)
        x0 = _warm_start_cache.get((run_hour,), None)

        coeffs = fit_emos(fcst_arr, spread_arr, actual_arr, x0=x0)

        # Cache for warm-start and repeated calls
        _warm_start_cache[(run_hour,)] = coeffs
        _fitted_cache[cache_key] = coeffs

    center, std = predict_emos(coeffs, fcst_high, ensemble_spread)

    # Generate bracket probabilities (1-deg-F brackets around center)
    bracket_probs = {}
    for temp in range(int(center) - 15, int(center) + 16):
        p = norm.cdf(temp + 0.5, center, std) - norm.cdf(temp - 0.5, center, std)
        if p > 1e-6:
            bracket_probs[temp] = p

    # Normalize
    total = sum(bracket_probs.values())
    if total > 0:
        bracket_probs = {k: v / total for k, v in bracket_probs.items()}

    return bracket_probs


# Expose raw (center, std) for Phase 2B integration
def _emos_raw(provider, ref_time):
    # type: (object, object) -> Optional[Tuple[float, float]]
    """Return (center, std) or None — for ensemble/Phase 2B use."""
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
    row = con.execute(
        "SELECT ensemble_spread FROM gold_multi_model_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    ensemble_spread = row[0] if row and row[0] is not None else 0.0

    cache_key = (run_hour, forecast_date)
    if cache_key not in _fitted_cache:
        # Force a full prediction to populate the cache
        result = emos_model_fn(provider, ref_time)
        if result is None:
            return None

    if cache_key not in _fitted_cache:
        return None

    coeffs = _fitted_cache[cache_key]
    return predict_emos(coeffs, fcst_high, ensemble_spread)


emos_model_fn.raw = _emos_raw  # type: ignore[attr-defined]
