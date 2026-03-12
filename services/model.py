"""Quantile Regression Model — extracted from autoresearch/experiment.py.

Encapsulates the LP-based quantile regression fitting and piecewise-linear
CDF bracket probability prediction into a reusable QRModel class.

Math is identical to experiment.py: same LP formulation, same CDF construction,
same constants, same feature standardization.

Python 3.9 compatible (no subscripted builtins).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Constants — must match autoresearch/experiment.py exactly
# ---------------------------------------------------------------------------

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
WINDOW = 180
MIN_SAMPLES = 60
LAMBDA_UPPER_FACTOR = 0.29
LAMBDA_LOWER_FACTOR = 0.36
RADIUS = 18
MIN_BRACKET_PROB = 0.0001


# ---------------------------------------------------------------------------
# QRModel
# ---------------------------------------------------------------------------

class QRModel:
    """Quantile regression model with caching for walk-forward backtesting.

    Usage:
        model = QRModel()
        coefficients = model.fit(X_train, y_train, run_hour=0, date_key="2026-03-11")
        probs = model.predict_bracket_probs(features, fcst_high=75.0, run_hour=0, date_key="2026-03-11")
    """

    def __init__(self):
        # type: () -> None
        self._coeff_cache = {}   # type: Dict[Tuple[int, str], Optional[List[np.ndarray]]]
        self._scale_cache = {}   # type: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]

    def fit(self, X_train, y_train, run_hour, date_key):
        # type: (np.ndarray, np.ndarray, int, str) -> Optional[List[np.ndarray]]
        """Fit 7 quantile regressions on standardized features.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Training feature matrix (no intercept — added internally).
        y_train : ndarray of shape (n_samples,)
            Training target: forecast error (fcst_high - actual).
        run_hour : int
            HRRR model run hour (0, 6, 12, 18).
        date_key : str
            Date string for cache key (e.g. "2026-03-11").

        Returns
        -------
        list of 7 ndarray or None
            Coefficient arrays (one per quantile), or None if insufficient
            data or any LP solve fails. Each array has length 1 + n_features.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        cache_key = (run_hour, date_key)

        # Standardize: zero mean, unit variance
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0)
        feat_std[feat_std < 1e-10] = 1.0  # guard against zero-variance features
        X_scaled = (X_train - feat_mean) / feat_std

        self._scale_cache[cache_key] = (feat_mean, feat_std)

        # Fit one LP per quantile
        coefficients = []  # type: List[np.ndarray]
        for tau in QUANTILES:
            coeffs = self._fit_single_quantile(X_scaled, y_train, tau)
            if coeffs is None:
                self._coeff_cache[cache_key] = None
                return None
            coefficients.append(coeffs)

        self._coeff_cache[cache_key] = coefficients
        return coefficients

    def predict_bracket_probs(self, features, fcst_high, run_hour, date_key,
                              running_max=None):
        # type: (np.ndarray, float, int, str, Optional[float]) -> Optional[Dict[int, float]]
        """Predict bracket probabilities from cached model.

        Parameters
        ----------
        features : ndarray of shape (n_features,)
            Today's raw feature vector (will be standardized using cached params).
        fcst_high : float
            Today's HRRR forecast high temperature.
        run_hour : int
            HRRR model run hour.
        date_key : str
            Date string matching the fit() call.
        running_max : float or None
            Observed running max temperature today (physical CDF floor).

        Returns
        -------
        dict of {int: float} or None
            1-degree-F bracket probabilities normalized to sum=1.0,
            or None if no coefficients are cached for this key.
        """
        cache_key = (run_hour, date_key)

        coefficients = self._coeff_cache.get(cache_key)
        if coefficients is None:
            return None

        scale_params = self._scale_cache.get(cache_key)
        if scale_params is None:
            return None

        feat_mean, feat_std = scale_params
        features = np.asarray(features, dtype=np.float64)
        features_scaled = (features - feat_mean) / feat_std

        # Predict error quantiles: intercept + scaled features · coefficients
        x_row = np.concatenate([[1.0], features_scaled])
        error_quantiles = [float(np.dot(c, x_row)) for c in coefficients]

        # Convert error quantiles to temperature quantiles
        # temp = fcst_high - error  (decreasing transform inverts quantile ordering)
        temp_quantiles = [fcst_high - eq for eq in error_quantiles]
        inverted_taus = [1.0 - tau for tau in QUANTILES]

        return self._build_bracket_probs(temp_quantiles, inverted_taus,
                                         running_max=running_max)

    @staticmethod
    def _fit_single_quantile(X, y, tau):
        # type: (np.ndarray, np.ndarray, float) -> Optional[np.ndarray]
        """Fit linear quantile regression for a single quantile tau via LP.

        Solves:
            minimize  tau * 1^T u + (1-tau) * 1^T v
            s.t.      [1|X] beta + u - v = y
                      u, v >= 0

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (already standardized, no intercept — added here).
        y : ndarray of shape (n_samples,)
            Target values.
        tau : float in (0, 1)
            Quantile level.

        Returns
        -------
        ndarray of shape (1 + n_features,) or None
            Coefficients [intercept, feat0, ...]. None if insufficient data
            or solver fails.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        if n < MIN_SAMPLES:
            return None

        # Add intercept column
        ones = np.ones((n, 1), dtype=np.float64)
        X_aug = np.hstack([ones, X])
        k = p + 1  # intercept + features

        # LP objective: [0...0 | tau...tau | (1-tau)...(1-tau)]
        c = np.concatenate([
            np.zeros(k),
            tau * np.ones(n),
            (1 - tau) * np.ones(n),
        ])

        # Equality constraint: X_aug * beta + u - v = y
        A_eq = sparse.hstack([
            sparse.csc_matrix(X_aug),
            sparse.eye(n),
            -sparse.eye(n),
        ], format='csc')
        b_eq = y

        # Bounds: beta unbounded, u >= 0, v >= 0
        bounds = [(None, None)] * k + [(0, None)] * (2 * n)
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result.success:
            return None
        return result.x[:k]

    @staticmethod
    def _build_cdf(temp_quantiles, tau_values, running_max=None):
        # type: (List[float], List[float], Optional[float]) -> object
        """Build piecewise-linear CDF with exponential tails.

        Returns a callable cdf(t) -> float.
        """
        pairs = sorted(zip(temp_quantiles, tau_values))
        temps = [p[0] for p in pairs]
        taus = [p[1] for p in pairs]

        # Enforce tau monotonicity
        for i in range(1, len(taus)):
            if taus[i] < taus[i - 1]:
                taus[i] = taus[i - 1]

        q05, q95 = temps[0], temps[-1]
        tau_lo, tau_hi = taus[0], taus[-1]

        # Find median (closest to 0.5)
        q50_idx = min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5))
        q50 = temps[q50_idx]

        # Exponential decay rates
        spread_upper = max(q95 - q50, 0.5)
        spread_lower = max(q50 - q05, 0.5)
        lambda_upper = LAMBDA_UPPER_FACTOR / spread_upper
        lambda_lower = LAMBDA_LOWER_FACTOR / spread_lower

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

        return cdf

    @staticmethod
    def _build_bracket_probs(temp_quantiles, tau_values, running_max=None):
        # type: (List[float], List[float], Optional[float]) -> Dict[int, float]
        """Convert temperature quantiles to 1-degree-F bracket probabilities.

        Builds CDF, evaluates P(bracket) = CDF(k+0.5) - CDF(k-0.5) for
        center +/- RADIUS brackets, filters by MIN_BRACKET_PROB, normalizes.
        """
        # Need temps sorted for center calculation
        pairs = sorted(zip(temp_quantiles, tau_values))
        taus_sorted = [p[1] for p in pairs]
        temps_sorted = [p[0] for p in pairs]

        # Enforce monotonicity to find q50 index correctly
        taus_mono = list(taus_sorted)
        for i in range(1, len(taus_mono)):
            if taus_mono[i] < taus_mono[i - 1]:
                taus_mono[i] = taus_mono[i - 1]

        q50_idx = min(range(len(taus_mono)), key=lambda i: abs(taus_mono[i] - 0.5))
        center_int = round(temps_sorted[q50_idx])

        cdf = QRModel._build_cdf(temp_quantiles, tau_values, running_max=running_max)

        probs = {}  # type: Dict[int, float]
        for k in range(center_int - RADIUS, center_int + RADIUS + 1):
            p = cdf(k + 0.5) - cdf(k - 0.5)
            if p > MIN_BRACKET_PROB:
                probs[k] = p

        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total, 4) for k, v in probs.items()}
        return probs
