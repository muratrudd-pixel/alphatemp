"""Quantile regression model for Phase 3 — Dynamic Uncertainty.

Replaces Gaussian CDF with piecewise-linear CDF built from predicted
error quantiles. Each quantile τ gets its own linear regression
minimizing pinball loss, solved as a linear program.

Technique reference: Koenker & Bassett (1978), "Regression Quantiles"
LP formulation: minimize τ·u + (1-τ)·v  s.t.  Xβ + u - v = y, u,v ≥ 0

Tail treatment: exponential decay past q05/q95 (Gemini directive) instead
of flat clamping. Prevents probability mass stacking at extreme brackets.

Physical floor: when running_max is provided, CDF is hard-clamped to 0
below the observed running max (daily high cannot go below what's already
been observed).
"""

import math
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
_MIN_BRACKET_PROB = 0.0001
_QR_MIN_SAMPLES = 90  # same walk-forward minimum as Phase 2


# ---------------------------------------------------------------------------
# Pinball loss (diagnostic / evaluation only — fitting uses LP)
# ---------------------------------------------------------------------------

def pinball_loss(tau, actual, predicted):
    # type: (float, ..., ...) -> ...
    """Pinball (check) loss for quantile regression.

    If actual > predicted: loss = tau * |error|  (penalizes underprediction)
    If actual < predicted: loss = (1-tau) * |error| (penalizes overprediction)

    Works with scalars or numpy arrays.
    """
    error = np.asarray(actual) - np.asarray(predicted)
    return np.where(error >= 0, tau * error, (tau - 1.0) * error)


# ---------------------------------------------------------------------------
# Linear quantile regression via LP
# ---------------------------------------------------------------------------

def fit_quantile_regression(X, y, tau, min_samples=_QR_MIN_SAMPLES):
    # type: (np.ndarray, np.ndarray, float, int) -> Optional[np.ndarray]
    """Fit linear quantile regression for a single quantile τ.

    Solves the LP formulation:
        minimize  τ·1ᵀu + (1-τ)·1ᵀv
        s.t.      [X|1]β + u - v = y
                  u, v ≥ 0

    where u = max(0, y - Xβ) and v = max(0, Xβ - y) are the positive
    and negative residual parts.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (WITHOUT intercept column — added internally).
    y : ndarray of shape (n_samples,)
        Target values (actual errors).
    tau : float in (0, 1)
        Quantile level.
    min_samples : int
        Minimum training rows required. Returns None if insufficient.

    Returns
    -------
    ndarray of shape (1 + n_features,) or None
        Coefficients [intercept, feat0, feat1, ...]. None if insufficient data
        or solver fails.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    if n < min_samples:
        return None

    # Add intercept column
    ones = np.ones((n, 1), dtype=np.float64)
    X_aug = np.hstack([ones, X])  # shape (n, p+1)
    k = p + 1  # number of coefficients (intercept + features)

    # LP decision variables: [β (k), u (n), v (n)]
    # Objective: minimize τ·sum(u) + (1-τ)·sum(v)
    c = np.concatenate([
        np.zeros(k),           # β coefficients: no direct cost
        tau * np.ones(n),      # u: positive residual penalty
        (1 - tau) * np.ones(n) # v: negative residual penalty
    ])

    # Equality constraint: X_aug·β + u - v = y
    # Use sparse matrices to avoid O(n²) memory from dense identity blocks
    A_eq = sparse.hstack([
        sparse.csc_matrix(X_aug),  # β part: (n, k) dense -> sparse
        sparse.eye(n),             # u part (+1): O(n) not O(n²)
        -sparse.eye(n),            # v part (-1): O(n) not O(n²)
    ], format='csc')
    b_eq = y

    # Bounds: β is unbounded, u >= 0, v >= 0
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not result.success:
        return None

    return result.x[:k]  # intercept + feature coefficients


# ---------------------------------------------------------------------------
# Piecewise-linear CDF from predicted quantiles — exponential decay tails
# ---------------------------------------------------------------------------

def build_piecewise_cdf(temp_quantiles, tau_values=QUANTILES, running_max=None):
    # type: (List[float], List[float], Optional[float]) -> Callable
    """Build a piecewise-linear CDF from predicted temperature quantiles.

    Parameters
    ----------
    temp_quantiles : list of float
        Predicted temperature values at each quantile position.
        Must be same length as tau_values.
    tau_values : list of float
        Quantile positions (cumulative probabilities), default QUANTILES.
    running_max : float or None
        If provided, the observed running max temperature so far today.
        CDF is hard-clamped to 0 below this value (physical floor:
        daily high cannot be below what's already been observed).

    Returns
    -------
    callable
        CDF function: float -> float, returns P(temp <= t).

    Notes
    -----
    - Crossed quantiles are auto-sorted (monotonicity enforced).
    - Tails use exponential decay past q05/q95 instead of flat clamping.
      This prevents probability mass from stacking at extreme brackets.
    - When running_max is provided, CDF(t) = 0 for all t < running_max.
    """
    assert len(temp_quantiles) == len(tau_values)

    # Pair and sort by temperature to enforce monotonicity
    pairs = sorted(zip(temp_quantiles, tau_values))
    temps = [p[0] for p in pairs]
    taus = [p[1] for p in pairs]

    # Enforce tau monotonicity (cumulative max) — crossed quantiles
    # can produce non-monotonic taus after sorting by temperature
    for i in range(1, len(taus)):
        if taus[i] < taus[i - 1]:
            taus[i] = taus[i - 1]

    # Extract key quantiles for exponential tail computation
    q05 = temps[0]   # lowest quantile temperature
    q95 = temps[-1]  # highest quantile temperature
    tau_lo = taus[0]  # lowest tau (typically 0.05)
    tau_hi = taus[-1]  # highest tau (typically 0.95)

    # Find median (or closest to 0.5) for decay rate computation
    q50_idx = min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5))
    q50 = temps[q50_idx]

    # Exponential decay rates from IQR spread
    # Guard against degenerate cases (very small spread)
    spread_upper = max(q95 - q50, 0.5)  # at least 0.5°F
    spread_lower = max(q50 - q05, 0.5)  # at least 0.5°F
    lambda_upper = 1.0 / spread_upper
    lambda_lower = 1.0 / spread_lower

    def cdf(t):
        # type: (float) -> float

        # Physical floor: daily high can't be below running_max
        if running_max is not None and t < running_max:
            return 0.0

        # Lower tail: exponential decay below q05
        if t <= q05:
            # P(T <= t) = tau_lo * exp(-lambda_lower * (q05 - t))
            return tau_lo * math.exp(-lambda_lower * (q05 - t))

        # Upper tail: exponential decay above q95
        if t >= q95:
            # P(T <= t) = 1 - (1 - tau_hi) * exp(-lambda_upper * (t - q95))
            return 1.0 - (1.0 - tau_hi) * math.exp(-lambda_upper * (t - q95))

        # Interior: linear interpolation between quantile points
        for i in range(len(temps) - 1):
            if temps[i] <= t <= temps[i + 1]:
                if temps[i + 1] == temps[i]:
                    return taus[i + 1]
                frac = (t - temps[i]) / (temps[i + 1] - temps[i])
                return taus[i] + frac * (taus[i + 1] - taus[i])

        return taus[-1]  # fallback

    return cdf


def bracket_probs_from_quantiles(temp_quantiles, tau_values=QUANTILES,
                                 running_max=None, radius=15):
    # type: (List[float], List[float], Optional[float], int) -> Dict[int, float]
    """Convert predicted temperature quantiles to 1°F bracket probabilities.

    Parameters
    ----------
    temp_quantiles : list of float
        Predicted temperature values at each quantile position.
    tau_values : list of float
        Quantile positions (cumulative probabilities).
    running_max : float or None
        Observed running max temp (physical floor for CDF).
    radius : int
        Number of brackets above/below the median to evaluate.

    Returns
    -------
    dict of {int: float}
        1°F integer bracket -> probability. Renormalized to sum to 1.0.
    """
    cdf = build_piecewise_cdf(temp_quantiles, tau_values, running_max=running_max)

    # Center on the median (tau=0.5 point, or midpoint of temps)
    pairs = sorted(zip(temp_quantiles, tau_values))
    median_temp = pairs[len(pairs) // 2][0]
    center_int = round(median_temp)

    probs = {}  # type: Dict[int, float]
    for k in range(center_int - radius, center_int + radius + 1):
        p = cdf(k + 0.5) - cdf(k - 0.5)
        if p > _MIN_BRACKET_PROB:
            probs[k] = p

    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    return probs


# ---------------------------------------------------------------------------
# End-to-end: fit all quantiles → predict → bracket probabilities
# ---------------------------------------------------------------------------

def predict_quantiles(X_train, y_train, features_today, fcst_high,
                      running_max=None, tau_values=QUANTILES,
                      min_samples=_QR_MIN_SAMPLES):
    # type: (np.ndarray, np.ndarray, np.ndarray, float, Optional[float], List[float], int) -> Optional[Dict[int, float]]
    """Fit 7 quantile regressions and return bracket probabilities.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Training feature matrix (no intercept — added internally).
    y_train : ndarray of shape (n_samples,)
        Training target: actual_error = fcst_high - nws_settlement_high.
    features_today : ndarray of shape (n_features,)
        Today's feature vector for prediction.
    fcst_high : float
        Today's forecast high temperature.
    running_max : float or None
        Observed running max temp today (physical floor).
    tau_values : list of float
        Quantile positions to fit.
    min_samples : int
        Minimum training rows for each QR fit.

    Returns
    -------
    dict of {int: float} or None
        1°F bracket probabilities, or None if any quantile fit fails.
    """
    features_today = np.asarray(features_today, dtype=np.float64)

    # Fit each quantile independently
    error_quantiles = []  # type: List[float]
    for tau in tau_values:
        coeffs = fit_quantile_regression(X_train, y_train, tau,
                                         min_samples=min_samples)
        if coeffs is None:
            return None
        # Predict: intercept + features · coefficients
        x_row = np.concatenate([[1.0], features_today])
        q_error = float(np.dot(coeffs, x_row))
        error_quantiles.append(q_error)

    # Convert error quantiles to temperature quantiles
    # temp = fcst_high - error is a DECREASING transform, so quantile
    # ordering inverts: tau_error=0.05 → tau_temp=0.95 (low error = high temp)
    temp_quantiles = [fcst_high - eq for eq in error_quantiles]
    inverted_taus = [1.0 - tau for tau in tau_values]

    return bracket_probs_from_quantiles(temp_quantiles, inverted_taus,
                                        running_max=running_max)


# ---------------------------------------------------------------------------
# QR-specific diagnostics
# ---------------------------------------------------------------------------

def quantile_coverage(actuals, predicted_quantiles, tau_values=QUANTILES):
    # type: (np.ndarray, List[List[float]], List[float]) -> List[float]
    """Compute empirical coverage for each predicted quantile.

    For each tau, what fraction of actuals fall below the predicted quantile?
    Perfect calibration: coverage[i] ~ tau_values[i].

    Parameters
    ----------
    actuals : array-like of shape (n_samples,)
        Actual observed values.
    predicted_quantiles : list of lists, shape (n_quantiles, n_samples)
        predicted_quantiles[i][j] = predicted tau_i quantile for sample j.
    tau_values : list of float
        Quantile positions.

    Returns
    -------
    list of float
        Empirical coverage for each quantile.
    """
    actuals = np.asarray(actuals)
    coverages = []
    for i in range(len(tau_values)):
        preds = np.asarray(predicted_quantiles[i])
        below = np.mean(actuals <= preds)
        coverages.append(float(below))
    return coverages


def mean_pinball_loss(actuals, predicted_quantiles, tau_values=QUANTILES):
    # type: (np.ndarray, List[List[float]], List[float]) -> float
    """Mean pinball loss averaged across all quantiles.

    Parameters
    ----------
    actuals : array-like of shape (n_samples,)
    predicted_quantiles : list of lists, shape (n_quantiles, n_samples)
    tau_values : list of float

    Returns
    -------
    float
        Mean pinball loss across all quantiles and samples.
    """
    actuals = np.asarray(actuals)
    total_loss = 0.0
    for i, tau in enumerate(tau_values):
        preds = np.asarray(predicted_quantiles[i])
        losses = pinball_loss(tau, actuals, preds)
        total_loss += float(np.mean(losses))
    return total_loss / len(tau_values)
