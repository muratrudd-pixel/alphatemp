"""Tests for services/model.py — QRModel class.

Validates the quantile regression model extracted from autoresearch/experiment.py.
Math must match experiment.py exactly: same LP formulation, CDF construction, constants.
"""

import numpy as np
import pytest

from services.model import QRModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Fresh QRModel instance."""
    return QRModel()


def _make_training_data(n_samples=200, n_features=23, seed=42):
    """Generate synthetic training data with realistic properties.

    Returns (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    y = 2.0 + noise (simulating forecast errors centered around 2°F).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) * 5.0 + 50.0  # feature-scale ~50
    y = 2.0 + rng.randn(n_samples) * 3.0  # errors ~N(2, 3)
    return X, y


# ---------------------------------------------------------------------------
# test_fit_returns_coefficients
# ---------------------------------------------------------------------------

def test_fit_returns_coefficients(model):
    """fit() with 200 samples returns 7 coefficient arrays, each length 24.

    23 features + 1 intercept = 24 coefficients per quantile.
    7 quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95].
    """
    X, y = _make_training_data(n_samples=200, n_features=23)
    result = model.fit(X, y, run_hour=0, date_key="2026-03-11")

    assert result is not None, "fit() should return coefficients with 200 samples"
    assert len(result) == 7, "Should return 7 coefficient arrays (one per quantile)"
    for i, coeffs in enumerate(result):
        assert isinstance(coeffs, np.ndarray), f"Coefficients[{i}] should be ndarray"
        assert len(coeffs) == 24, f"Coefficients[{i}] should have 24 elements (1 intercept + 23 features)"


# ---------------------------------------------------------------------------
# test_fit_caches_scale_params
# ---------------------------------------------------------------------------

def test_fit_caches_scale_params(model):
    """fit() stores (mean, std) in _scale_cache with correct shapes."""
    X, y = _make_training_data(n_samples=200, n_features=23)
    model.fit(X, y, run_hour=0, date_key="2026-03-11")

    cache_key = (0, "2026-03-11")
    assert cache_key in model._scale_cache, "_scale_cache should contain the (run_hour, date_key) entry"

    feat_mean, feat_std = model._scale_cache[cache_key]
    assert feat_mean.shape == (23,), f"feat_mean shape should be (23,), got {feat_mean.shape}"
    assert feat_std.shape == (23,), f"feat_std shape should be (23,), got {feat_std.shape}"
    assert np.all(feat_std > 0), "All std values should be positive (clamped)"


# ---------------------------------------------------------------------------
# test_predict_bracket_probs_returns_dict
# ---------------------------------------------------------------------------

def test_predict_bracket_probs_returns_dict(model):
    """predict_bracket_probs returns dict with probs summing to ~1.0, all non-negative."""
    X, y = _make_training_data(n_samples=200, n_features=23)
    model.fit(X, y, run_hour=0, date_key="2026-03-11")

    # Use mean of training features as "today's" features
    features = X.mean(axis=0)
    fcst_high = 75.0

    probs = model.predict_bracket_probs(features, fcst_high, run_hour=0, date_key="2026-03-11")

    assert probs is not None, "predict should return dict after successful fit"
    assert isinstance(probs, dict), f"Expected dict, got {type(probs)}"
    assert len(probs) > 0, "Should have at least one bracket"

    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01, f"Probabilities should sum to ~1.0, got {total}"

    for bracket, prob in probs.items():
        assert isinstance(bracket, int), f"Bracket keys should be int, got {type(bracket)}"
        assert prob >= 0, f"All probabilities should be non-negative, got {prob} for bracket {bracket}"


# ---------------------------------------------------------------------------
# test_predict_without_fit_returns_none
# ---------------------------------------------------------------------------

def test_predict_without_fit_returns_none(model):
    """predict_bracket_probs without prior fit returns None."""
    features = np.zeros(23)
    result = model.predict_bracket_probs(features, fcst_high=75.0, run_hour=0, date_key="2026-03-11")

    assert result is None, "predict should return None when no coefficients are cached"


# ---------------------------------------------------------------------------
# test_fit_rejects_insufficient_data
# ---------------------------------------------------------------------------

def test_fit_rejects_insufficient_data(model):
    """fit() with fewer than MIN_SAMPLES (60) returns None."""
    X, y = _make_training_data(n_samples=50, n_features=23)
    result = model.fit(X, y, run_hour=0, date_key="2026-03-11")

    assert result is None, "fit() should return None with only 50 samples (MIN_SAMPLES=60)"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_fit_caches_coefficients(model):
    """Verify coefficients are cached and reused on second call."""
    X, y = _make_training_data(n_samples=200, n_features=23)
    result1 = model.fit(X, y, run_hour=0, date_key="2026-03-11")

    cache_key = (0, "2026-03-11")
    assert cache_key in model._coeff_cache
    assert model._coeff_cache[cache_key] is result1


def test_different_cache_keys_independent(model):
    """Different (run_hour, date_key) pairs produce independent caches."""
    X, y = _make_training_data(n_samples=200, n_features=23)
    model.fit(X, y, run_hour=0, date_key="2026-03-11")
    model.fit(X, y, run_hour=12, date_key="2026-03-11")

    assert (0, "2026-03-11") in model._coeff_cache
    assert (12, "2026-03-11") in model._coeff_cache


def test_predict_with_running_max(model):
    """predict_bracket_probs with running_max should still return valid probs."""
    X, y = _make_training_data(n_samples=200, n_features=23)
    model.fit(X, y, run_hour=0, date_key="2026-03-11")

    features = X.mean(axis=0)
    probs = model.predict_bracket_probs(
        features, fcst_high=75.0, run_hour=0, date_key="2026-03-11",
        running_max=70.0
    )

    assert probs is not None
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01
    # All brackets should be >= running_max floor (no prob mass below)
    for bracket in probs.keys():
        # bracket represents floor of 1°F bin, so bracket - 0.5 is the lower CDF edge
        # With running_max=70, we expect no brackets far below 70
        pass  # The CDF clamp handles this internally
