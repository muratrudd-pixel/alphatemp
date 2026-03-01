"""Ensemble combination -- Gaussian mixture over multiple model predictions.

Combines bias-corrected Gaussian distributions from multiple weather models
into bracket probabilities via a weighted mixture distribution.
"""
import itertools
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm


def combine_mixture_brackets(
    predictions,  # type: List[Tuple[float, float]]
    weights,      # type: List[float]
    radius=15,    # type: int
):
    # type: (...) -> Dict[int, float]
    """Compute bracket probabilities from a weighted Gaussian mixture.

    Parameters
    ----------
    predictions : list of (mu, sigma)
        Each model's bias-corrected center and residual std.
    weights : list of float
        Per-model weights (must sum to ~1.0).
    radius : int
        Half-width of bracket range around the weighted mean center.

    Returns
    -------
    Dict mapping integer temperature to probability.
    """
    if not predictions:
        return {}

    # Weighted center for bracket range
    center = sum(w * mu for (mu, _), w in zip(predictions, weights))
    center_int = round(center)

    probs = {}  # type: Dict[int, float]
    for k in range(center_int - radius, center_int + radius + 1):
        p = 0.0
        for (mu, sigma), w in zip(predictions, weights):
            if sigma <= 0:
                continue
            p += w * (norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma))
        if p > 0.0001:
            probs[k] = round(p, 4)

    # Renormalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    return probs


def make_ensemble_model_fn(model_fns, weights=None):
    # type: (List, Optional[List[float]]) -> callable
    """Create an ensemble ModelFn from multiple model functions.

    Each model_fn must have a .raw attribute that returns Optional[Tuple[float, float]].
    The ensemble calls .raw on each, collects (center, std) pairs,
    and combines them via Gaussian mixture.

    Parameters
    ----------
    model_fns : list of ModelFn
        Each must have a .raw(provider, ref_time) -> Optional[(center, std)]
    weights : list of float or None
        Per-model weights. None = equal weights.
    """
    n = len(model_fns)
    if weights is None:
        weights = [1.0 / n] * n

    def ensemble_fn(provider, ref_time):
        predictions = []
        active_weights = []

        for mfn, w in zip(model_fns, weights):
            params = mfn.raw(provider, ref_time)
            if params is not None:
                predictions.append(params)
                active_weights.append(w)

        if not predictions:
            return None

        # Renormalize weights for active models only
        total_w = sum(active_weights)
        active_weights = [w / total_w for w in active_weights]

        return combine_mixture_brackets(predictions, active_weights)

    ensemble_fn.__name__ = "ensemble"
    return ensemble_fn


# ---------------------------------------------------------------------------
# Latest-run ensemble with age-based decay weights
# ---------------------------------------------------------------------------

class _ProviderProxy:
    """Lightweight stand-in for BacktestDataProvider with a different model_run."""

    def __init__(self, con, station_id, model_run):
        self._shared_con = con
        self.station_id = station_id
        self.model_run = model_run


def _find_latest_run_hour(con, model_name, station_id, settlement_date, max_hour):
    # type: (...) -> Optional[int]
    """Find most recent run_hour for this model on settlement_date, at or before max_hour."""
    row = con.execute("""
        SELECT MAX(EXTRACT(HOUR FROM model_run)::INTEGER)
        FROM forecasts
        WHERE model_name = ? AND station_id = ?
          AND model_run::DATE = ?
          AND EXTRACT(HOUR FROM model_run) <= ?
    """, [model_name, station_id, settlement_date, max_hour]).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def make_latest_run_ensemble_fn(model_fns_with_names, decay_lambda=0.1):
    # type: (List[Tuple], float) -> callable
    """Create ensemble that uses the latest available run from each model.

    model_fns_with_names: list of (model_fn, model_name) pairs
        Each model_fn must have .raw(provider, ref_time) -> Optional[(center, std)]
    decay_lambda: exponential decay rate for age-based weighting
    """
    def ensemble_fn(provider, ref_time):
        con = provider._shared_con
        station_id = provider.station_id
        settlement_date = provider.model_run.date()
        current_hour = provider.model_run.hour

        predictions = []
        weights = []

        for mfn, mname in model_fns_with_names:
            latest_hour = _find_latest_run_hour(
                con, mname, station_id, settlement_date, current_hour,
            )
            if latest_hour is None:
                continue

            age_hours = current_hour - latest_hour
            proxy = _ProviderProxy(
                con, station_id,
                datetime(
                    settlement_date.year, settlement_date.month,
                    settlement_date.day, latest_hour,
                ),
            )

            params = mfn.raw(proxy, ref_time)
            if params is None:
                continue

            predictions.append(params)
            weights.append(math.exp(-decay_lambda * age_hours))

        if not predictions:
            return None

        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        return combine_mixture_brackets(predictions, weights)

    ensemble_fn.__name__ = "ensemble_latest_run"
    return ensemble_fn


# ---------------------------------------------------------------------------
# Level 2: Walk-forward learned mixture weights via grid search
# ---------------------------------------------------------------------------

# Module-level cache for precomputed raw predictions.
# Keyed by (run_hour, model_names_tuple). Cleared between backtester runs.
_learned_weight_cache = {}  # type: Dict


def clear_learned_weight_cache():
    """Clear the precomputed prediction cache (call between backtester runs)."""
    _learned_weight_cache.clear()


def _compute_brier_for_weights(weights, model_bracket_probs, actuals):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> float
    """Vectorized Brier score for a set of mixture weights.

    model_bracket_probs: shape (n_dates, n_models, n_brackets)
    actuals: shape (n_dates, n_brackets) — one-hot outcome indicators
    weights: shape (n_models,)

    Returns mean Brier score across all dates.
    """
    # Weighted mixture: (n_dates, n_brackets)
    mixture = np.einsum('dmb,m->db', model_bracket_probs, weights)
    # Brier = sum_k (p_k - o_k)^2 per date, then mean
    brier_per_date = np.sum((mixture - actuals) ** 2, axis=1)
    return float(np.mean(brier_per_date))


def _generate_simplex_weights(n_models, step=0.05):
    # type: (int, float) -> List[List[float]]
    """Generate all weight combinations on the simplex with given step size.

    Each weight >= 0 and weights sum to 1.0.
    For n_models=3, step=0.05: ~231 combinations.
    """
    n_steps = int(round(1.0 / step))
    combos = []
    # Generate all integer partitions of n_steps into n_models parts
    if n_models == 1:
        return [[1.0]]

    def _recurse(remaining, depth, current):
        if depth == n_models - 1:
            current.append(remaining * step)
            combos.append(list(current))
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i * step)
            _recurse(remaining - i, depth + 1, current)
            current.pop()

    _recurse(n_steps, 0, [])
    return combos


def _grid_search_simplex_weights(model_bracket_probs, actuals, n_models, step=0.05):
    # type: (np.ndarray, np.ndarray, int, float) -> Tuple[List[float], float]
    """Grid search over weight simplex to minimize Brier score.

    model_bracket_probs: shape (n_dates, n_models, n_brackets)
    actuals: shape (n_dates, n_brackets) — one-hot outcome indicators

    Returns (best_weights, best_brier).
    """
    weight_combos = _generate_simplex_weights(n_models, step)

    best_brier = float('inf')
    best_weights = [1.0 / n_models] * n_models

    for wts in weight_combos:
        w = np.array(wts, dtype=np.float64)
        brier = _compute_brier_for_weights(w, model_bracket_probs, actuals)
        if brier < best_brier:
            best_brier = brier
            best_weights = wts

    return best_weights, best_brier


def make_learned_weight_ensemble_fn(model_fns_with_names, lookback_days=365, reoptimize_every=30):
    # type: (List[Tuple], int, int) -> callable
    """Ensemble with walk-forward learned per-hour weights.

    At each eval point: optimize weights on prior lookback_days of cached
    predictions, apply to today's mixture.

    Weights are re-optimized every `reoptimize_every` days for performance
    (avoids N^2 grid search on every date).

    model_fns_with_names: list of (model_fn, model_name) pairs
    lookback_days: how many prior days to use for weight optimization
    reoptimize_every: re-optimize weights every N days (default 30)
    """
    n_models = len(model_fns_with_names)
    model_names_tuple = tuple(name for _, name in model_fns_with_names)

    def ensemble_fn(provider, ref_time):
        con = provider._shared_con
        station_id = provider.station_id
        settlement_date = provider.model_run.date()
        current_hour = provider.model_run.hour

        # Step 1: Get today's predictions from each model via latest-run logic
        today_predictions = []
        today_active = []

        for mfn, mname in model_fns_with_names:
            latest_hour = _find_latest_run_hour(
                con, mname, station_id, settlement_date, current_hour,
            )
            if latest_hour is None:
                today_active.append(False)
                today_predictions.append(None)
                continue

            proxy = _ProviderProxy(
                con, station_id,
                datetime(
                    settlement_date.year, settlement_date.month,
                    settlement_date.day, latest_hour,
                ),
            )
            params = mfn.raw(proxy, ref_time)
            today_active.append(params is not None)
            today_predictions.append(params)

        active_indices = [i for i in range(n_models) if today_active[i]]
        if not active_indices:
            return None

        # If only one model active, just use it directly
        if len(active_indices) == 1:
            idx = active_indices[0]
            return combine_mixture_brackets(
                [today_predictions[idx]], [1.0],
            )

        # Step 2: Get historical predictions for weight optimization
        # Fetch NWS actuals for the lookback window
        lookback_start = settlement_date - timedelta(days=lookback_days)
        actuals_rows = con.execute("""
            SELECT obs_date, max_temp_f
            FROM nws_daily
            WHERE station_id = ? AND obs_date >= ? AND obs_date < ?
              AND max_temp_f IS NOT NULL
            ORDER BY obs_date
        """, [station_id, lookback_start, settlement_date]).fetchall()

        if len(actuals_rows) < 90:
            # Not enough history — fall back to equal weights
            active_preds = [today_predictions[i] for i in active_indices]
            eq_w = [1.0 / len(active_indices)] * len(active_indices)
            return combine_mixture_brackets(active_preds, eq_w)

        # For each historical date, get raw predictions from each active model
        # using the same latest-run logic
        radius = 15
        n_brackets = 2 * radius + 1

        hist_model_brackets = []  # list of (date, bracket_probs_per_model)
        hist_actuals = []

        for obs_d, actual_high in actuals_rows:
            actual_int = round(actual_high)

            model_params_for_date = []
            all_have_data = True

            for idx in active_indices:
                mfn, mname = model_fns_with_names[idx]
                latest_hour = _find_latest_run_hour(
                    con, mname, station_id, obs_d, current_hour,
                )
                if latest_hour is None:
                    all_have_data = False
                    break

                proxy = _ProviderProxy(
                    con, station_id,
                    datetime(obs_d.year, obs_d.month, obs_d.day, latest_hour),
                )
                params = mfn.raw(proxy, ref_time)
                if params is None:
                    all_have_data = False
                    break
                model_params_for_date.append(params)

            if not all_have_data:
                continue

            # Convert each model's (center, std) to bracket probabilities
            # Use a fixed bracket range centered on the mean of models
            centers = [p[0] for p in model_params_for_date]
            mean_center = sum(centers) / len(centers)
            center_int = round(mean_center)
            bracket_range = list(range(center_int - radius, center_int + radius + 1))

            per_model_probs = np.zeros((len(active_indices), n_brackets), dtype=np.float64)
            for m_idx, (mu, sigma) in enumerate(model_params_for_date):
                if sigma <= 0:
                    continue
                for b_idx, k in enumerate(bracket_range):
                    per_model_probs[m_idx, b_idx] = (
                        norm.cdf((k + 0.5 - mu) / sigma) -
                        norm.cdf((k - 0.5 - mu) / sigma)
                    )
                # Normalize
                row_total = per_model_probs[m_idx].sum()
                if row_total > 0:
                    per_model_probs[m_idx] /= row_total

            # One-hot actual outcome
            actual_onehot = np.zeros(n_brackets, dtype=np.float64)
            if actual_int in bracket_range:
                actual_onehot[bracket_range.index(actual_int)] = 1.0
            else:
                # Actual outside bracket range — assign to nearest edge
                if actual_int < bracket_range[0]:
                    actual_onehot[0] = 1.0
                else:
                    actual_onehot[-1] = 1.0

            hist_model_brackets.append(per_model_probs)
            hist_actuals.append(actual_onehot)

        if len(hist_model_brackets) < 90:
            # Not enough aligned history — equal weights
            active_preds = [today_predictions[i] for i in active_indices]
            eq_w = [1.0 / len(active_indices)] * len(active_indices)
            return combine_mixture_brackets(active_preds, eq_w)

        # Step 3: Grid search for optimal weights
        model_probs_arr = np.array(hist_model_brackets)  # (n_dates, n_active, n_brackets)
        actuals_arr = np.array(hist_actuals)               # (n_dates, n_brackets)

        best_weights, _ = _grid_search_simplex_weights(
            model_probs_arr, actuals_arr, len(active_indices), step=0.05,
        )

        # Step 4: Apply learned weights to today's predictions
        active_preds = [today_predictions[i] for i in active_indices]
        return combine_mixture_brackets(active_preds, best_weights)

    ensemble_fn.__name__ = "ensemble_learned_weights"
    return ensemble_fn
