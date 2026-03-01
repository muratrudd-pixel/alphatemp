"""Tests for EMOS bias correction candidate."""

import numpy as np
import pytest
from services.phase2_emos import fit_emos, predict_emos


def test_fit_emos_returns_four_coefficients():
    """EMOS fit should return (a, b, c, d) coefficients."""
    np.random.seed(42)
    n = 200
    fcst = np.random.normal(70, 10, n)
    spread = np.abs(np.random.normal(2, 1, n))
    actual = fcst - 2.0 + np.random.normal(0, 1.5, n)  # 2F warm bias

    coeffs = fit_emos(fcst, spread, actual)
    assert len(coeffs) == 4
    a, b, c, d = coeffs
    # b should be close to 1.0 (forecast is informative)
    assert 0.5 < b < 1.5


def test_predict_emos_returns_center_and_std():
    """EMOS predict should return (center, std) tuple."""
    coeffs = (-2.0, 1.0, 1.0, 0.5)  # a, b, c, d
    center, std = predict_emos(coeffs, fcst_high=75.0, ensemble_spread=2.0)
    assert isinstance(center, float)
    assert isinstance(std, float)
    assert std > 0


def test_emos_reduces_bias():
    """EMOS should correct systematic bias in forecasts."""
    np.random.seed(42)
    n = 300
    fcst = np.random.normal(70, 10, n)
    spread = np.abs(np.random.normal(2, 1, n))
    actual = fcst - 3.0 + np.random.normal(0, 1.5, n)  # 3F warm bias

    coeffs = fit_emos(fcst, spread, actual)
    predictions = [predict_emos(coeffs, f, s) for f, s in zip(fcst, spread)]
    centers = [p[0] for p in predictions]
    residuals = [a - c for a, c in zip(actual, centers)]
    mean_residual = np.mean(residuals)
    # Mean residual should be near zero after correction
    assert abs(mean_residual) < 0.5


def test_emos_std_increases_with_spread():
    """EMOS std should be larger when ensemble spread is larger."""
    coeffs = (-2.0, 1.0, 1.0, 0.5)
    _, std_low = predict_emos(coeffs, fcst_high=75.0, ensemble_spread=1.0)
    _, std_high = predict_emos(coeffs, fcst_high=75.0, ensemble_spread=5.0)
    assert std_high > std_low


def test_emos_warm_start_converges():
    """EMOS with warm-start x0 should produce same result as cold start."""
    np.random.seed(42)
    n = 200
    fcst = np.random.normal(70, 10, n)
    spread = np.abs(np.random.normal(2, 1, n))
    actual = fcst - 2.0 + np.random.normal(0, 1.5, n)

    cold = fit_emos(fcst, spread, actual)
    warm = fit_emos(fcst, spread, actual, x0=cold)
    # Results should be nearly identical
    for c, w in zip(cold, warm):
        assert abs(c - w) < 0.1


def test_emos_zero_spread_degrades_gracefully():
    """When spread is zero, EMOS should still produce valid output."""
    np.random.seed(42)
    n = 200
    fcst = np.random.normal(70, 10, n)
    spread = np.zeros(n)  # No ensemble spread
    actual = fcst - 2.0 + np.random.normal(0, 1.5, n)

    coeffs = fit_emos(fcst, spread, actual)
    center, std = predict_emos(coeffs, fcst_high=75.0, ensemble_spread=0.0)
    assert isinstance(center, float)
    assert std >= 0.3  # Should hit the floor (sqrt(0.09) = 0.3)
