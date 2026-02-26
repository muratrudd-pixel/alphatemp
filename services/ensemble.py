"""Ensemble combination -- Gaussian mixture over multiple model predictions.

Combines bias-corrected Gaussian distributions from multiple weather models
into bracket probabilities via a weighted mixture distribution.
"""
from typing import Dict, List, Optional, Tuple

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
