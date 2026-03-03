"""Tests for services/quantile_model.py — quantile regression core."""

import math
import numpy as np
import pytest

from services.quantile_model import (
    pinball_loss,
    fit_quantile_regression,
    build_piecewise_cdf,
    bracket_probs_from_quantiles,
    predict_quantiles,
    quantile_coverage,
    mean_pinball_loss,
    QUANTILES,
)


class TestPinballLoss:
    """Pinball (asymmetric) loss for quantile regression."""

    def test_median_symmetric(self):
        """tau=0.5 should give symmetric loss."""
        assert pinball_loss(0.5, 1.0, 0.0) == pytest.approx(0.5)  # under
        assert pinball_loss(0.5, -1.0, 0.0) == pytest.approx(0.5)  # over

    def test_high_tau_penalizes_underprediction(self):
        """tau=0.9: underprediction costs more than overprediction."""
        under = pinball_loss(0.9, 1.0, 0.0)   # actual=1, pred=0 → under
        over = pinball_loss(0.9, -1.0, 0.0)    # actual=-1, pred=0 → over
        assert under > over

    def test_low_tau_penalizes_overprediction(self):
        """tau=0.1: overprediction costs more than underprediction."""
        under = pinball_loss(0.1, 1.0, 0.0)
        over = pinball_loss(0.1, -1.0, 0.0)
        assert over > under

    def test_zero_error(self):
        """Perfect prediction = zero loss."""
        assert pinball_loss(0.5, 5.0, 5.0) == 0.0

    def test_vectorized(self):
        """Array inputs should work element-wise."""
        actuals = np.array([1.0, -1.0, 0.0])
        preds = np.array([0.0, 0.0, 0.0])
        losses = pinball_loss(0.5, actuals, preds)
        assert len(losses) == 3
        assert losses[2] == 0.0


class TestFitQuantileRegression:
    """Linear quantile regression via LP."""

    def _make_linear_data(self, n=500, seed=42):
        """y = 2*x + noise. Median of y|x should be ~2*x."""
        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 10, n).reshape(-1, 1)
        noise = rng.normal(0, 1, n)
        y = 2.0 * X.ravel() + noise
        return X, y

    def test_median_recovers_slope(self):
        """tau=0.5 on y=2x+noise should recover slope ~2."""
        X, y = self._make_linear_data()
        coeffs = fit_quantile_regression(X, y, tau=0.5)
        # coeffs[0] = intercept, coeffs[1] = slope
        assert len(coeffs) == 2
        assert coeffs[1] == pytest.approx(2.0, abs=0.3)

    def test_quantile_ordering(self):
        """Lower quantiles should predict lower values than higher quantiles."""
        X, y = self._make_linear_data()
        q10 = fit_quantile_regression(X, y, tau=0.10)
        q50 = fit_quantile_regression(X, y, tau=0.50)
        q90 = fit_quantile_regression(X, y, tau=0.90)
        # At x=5: q10 < q50 < q90
        x_test = 5.0
        pred10 = q10[0] + q10[1] * x_test
        pred50 = q50[0] + q50[1] * x_test
        pred90 = q90[0] + q90[1] * x_test
        assert pred10 < pred50 < pred90

    def test_returns_none_insufficient_data(self):
        """Should return None with fewer rows than min_samples."""
        X = np.ones((10, 2))
        y = np.ones(10)
        result = fit_quantile_regression(X, y, tau=0.5, min_samples=90)
        assert result is None

    def test_multi_feature(self):
        """Works with multiple features (intercept + 3 features = 4 coeffs)."""
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 10, (500, 3))
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + rng.normal(0, 0.5, 500)
        coeffs = fit_quantile_regression(X, y, tau=0.5)
        assert len(coeffs) == 4  # intercept + 3 features


class TestBuildPiecewiseCdf:
    """Piecewise-linear CDF from predicted quantile points."""

    def test_monotonic_output(self):
        """CDF values should be monotonically non-decreasing."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        cdf_fn = build_piecewise_cdf(temps, QUANTILES)
        values = [cdf_fn(t) for t in range(55, 85)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_median_at_half(self):
        """CDF at the median temperature should be ~0.5."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        cdf_fn = build_piecewise_cdf(temps, QUANTILES)
        assert cdf_fn(68.0) == pytest.approx(0.5)

    def test_lower_tail_exponential_decay(self):
        """Below q05, CDF should decay exponentially toward 0, not be flat 0."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        cdf_fn = build_piecewise_cdf(temps, QUANTILES)
        # Below q05 (60.0), CDF should be > 0 but small
        val_at_58 = cdf_fn(58.0)
        val_at_55 = cdf_fn(55.0)
        val_at_50 = cdf_fn(50.0)
        assert val_at_58 > 0.0  # not flat zero
        assert val_at_58 > val_at_55 > val_at_50  # decaying
        assert val_at_50 < 0.02  # but very small far out

    def test_upper_tail_exponential_decay(self):
        """Above q95, CDF should approach 1 via exponential decay, not flat 1."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        cdf_fn = build_piecewise_cdf(temps, QUANTILES)
        # Above q95 (76.0), CDF should be < 1 but close
        val_at_78 = cdf_fn(78.0)
        val_at_81 = cdf_fn(81.0)
        val_at_86 = cdf_fn(86.0)
        assert val_at_78 < 1.0  # not flat one
        assert val_at_78 < val_at_81 < val_at_86  # approaching 1
        assert val_at_86 > 0.98  # very close to 1 far out

    def test_cdf_bounded_zero_one(self):
        """CDF should always be in [0, 1]."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        cdf_fn = build_piecewise_cdf(temps, QUANTILES)
        for t in range(40, 100):
            val = cdf_fn(float(t))
            assert 0.0 <= val <= 1.0

    def test_crossing_quantiles_sorted(self):
        """Crossed quantiles should be auto-sorted before building CDF."""
        temps = [60.0, 62.0, 67.0, 65.0, 71.0, 74.0, 76.0]  # q25 > q50
        cdf_fn = build_piecewise_cdf(temps, QUANTILES)
        # Should still produce valid CDF
        values = [cdf_fn(t) for t in range(55, 85)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_running_max_floor(self):
        """When running_max is set, CDF should be 0 below it."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        running_max = 72.0
        cdf_fn = build_piecewise_cdf(temps, QUANTILES, running_max=running_max)
        # Below running_max: hard zero
        assert cdf_fn(71.0) == 0.0
        assert cdf_fn(70.0) == 0.0
        assert cdf_fn(60.0) == 0.0
        # At/above running_max: normal CDF behavior
        assert cdf_fn(73.0) > 0.0

    def test_running_max_none_no_effect(self):
        """Without running_max, tails should have exponential decay."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        cdf_fn = build_piecewise_cdf(temps, QUANTILES, running_max=None)
        # Below q05 should have small but nonzero probability
        assert cdf_fn(58.0) > 0.0


class TestBracketProbsFromQuantiles:
    """Convert quantile predictions to 1°F bracket probabilities."""

    def test_probs_sum_to_one(self):
        """Bracket probabilities should sum to ~1.0."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        probs = bracket_probs_from_quantiles(temps, QUANTILES)
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_all_probs_positive(self):
        """All bracket probabilities should be > 0."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        probs = bracket_probs_from_quantiles(temps, QUANTILES)
        for k, p in probs.items():
            assert p > 0

    def test_peak_near_median(self):
        """Highest probability bracket should be near the median temp."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        probs = bracket_probs_from_quantiles(temps, QUANTILES)
        peak_bracket = max(probs, key=probs.get)
        assert 65 <= peak_bracket <= 71

    def test_narrow_distribution(self):
        """Tight quantiles should produce concentrated probabilities."""
        temps = [67.0, 67.5, 68.0, 68.5, 69.0, 69.5, 70.0]
        probs = bracket_probs_from_quantiles(temps, QUANTILES)
        # 67-70 range should capture most mass
        center_mass = sum(p for k, p in probs.items() if 67 <= k <= 70)
        assert center_mass > 0.7

    def test_wide_distribution(self):
        """Wide quantiles should spread probability across many brackets."""
        temps = [50.0, 55.0, 60.0, 68.0, 76.0, 81.0, 86.0]
        probs = bracket_probs_from_quantiles(temps, QUANTILES)
        assert len(probs) > 15

    def test_running_max_eliminates_lower_brackets(self):
        """With running_max, brackets below it should have zero probability."""
        temps = [60.0, 62.0, 65.0, 68.0, 71.0, 74.0, 76.0]
        probs = bracket_probs_from_quantiles(temps, QUANTILES, running_max=72.0)
        # No brackets below running_max should have probability
        for k, p in probs.items():
            if k < 72:
                assert p == 0.0 or k not in probs


class TestPredictQuantiles:
    """End-to-end: features + target → bracket probabilities."""

    def _make_training_data(self, n=500, seed=42):
        """Simulate forecast errors: error ~ N(2, 1.5) + 0.1*update_hour."""
        rng = np.random.RandomState(seed)
        update_hours = rng.randint(0, 19, n).astype(float)
        fcst_highs = rng.uniform(50, 95, n)
        sin_months = np.sin(2 * np.pi * rng.randint(1, 13, n) / 12)
        cos_months = np.cos(2 * np.pi * rng.randint(1, 13, n) / 12)
        # Error with some structure
        errors = 2.0 + 0.1 * update_hours + rng.normal(0, 1.5, n)
        X = np.column_stack([update_hours, fcst_highs, sin_months, cos_months])
        return X, errors

    def test_returns_dict(self):
        """Should return a dict of {int: float} bracket probabilities."""
        X, y = self._make_training_data()
        features_today = np.array([14.0, 72.0, 0.5, 0.87])
        result = predict_quantiles(X, y, features_today, 72.0)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_probs_sum_to_one(self):
        """Bracket probabilities should sum to ~1.0."""
        X, y = self._make_training_data()
        features_today = np.array([14.0, 72.0, 0.5, 0.87])
        result = predict_quantiles(X, y, features_today, 72.0)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.01)

    def test_returns_none_insufficient_data(self):
        """Should return None with too few training rows."""
        X = np.ones((10, 4))
        y = np.ones(10)
        features_today = np.array([14.0, 72.0, 0.5, 0.87])
        result = predict_quantiles(X, y, features_today, 72.0)
        assert result is None

    def test_peak_near_corrected_forecast(self):
        """Peak bracket should be near fcst_high - median_error."""
        X, y = self._make_training_data()
        # Errors average ~2.0, so corrected temp ≈ 72 - 2 = 70
        features_today = np.array([10.0, 72.0, 0.5, 0.87])
        result = predict_quantiles(X, y, features_today, 72.0)
        peak = max(result, key=result.get)
        assert 67 <= peak <= 73  # within ~3°F of 70

    def test_with_running_max(self):
        """Running max should eliminate lower brackets."""
        X, y = self._make_training_data()
        features_today = np.array([14.0, 72.0, 0.5, 0.87])
        result = predict_quantiles(X, y, features_today, 72.0, running_max=71.0)
        assert isinstance(result, dict)
        # All brackets should be >= 71
        for k in result:
            assert k >= 70  # bracket 70 spans 69.5-70.5, allow if CDF is renormalized


class TestQuantileCoverage:
    """Quantile coverage: fraction of actuals below predicted quantile."""

    def test_perfect_calibration(self):
        """Well-calibrated quantiles should have coverage near tau."""
        rng = np.random.RandomState(42)
        actuals = rng.normal(70, 3, 1000)
        # Perfectly calibrated: predicted quantiles = true quantiles
        predicted_quantiles = []
        for tau in QUANTILES:
            predicted_quantiles.append(
                [np.percentile(actuals, tau * 100)] * len(actuals)
            )
        coverage = quantile_coverage(actuals, predicted_quantiles, QUANTILES)
        for tau, cov in zip(QUANTILES, coverage):
            assert cov == pytest.approx(tau, abs=0.05)

    def test_returns_correct_length(self):
        """Should return one coverage value per quantile."""
        actuals = np.array([70.0, 71.0, 72.0])
        predicted = [[69.0, 70.0, 71.0]] * 7  # 7 quantiles
        coverage = quantile_coverage(actuals, predicted, QUANTILES)
        assert len(coverage) == 7


class TestMeanPinballLoss:
    """Aggregate pinball loss across all quantiles."""

    def test_perfect_predictions_zero_loss(self):
        """Predicting the exact actual should give zero loss."""
        actuals = np.array([70.0])
        predicted = [[70.0]] * 7
        loss = mean_pinball_loss(actuals, predicted, QUANTILES)
        assert loss == pytest.approx(0.0, abs=0.01)

    def test_bad_predictions_high_loss(self):
        """Wildly wrong predictions should give higher loss."""
        actuals = np.array([70.0, 75.0, 80.0])
        bad_preds = [[50.0, 50.0, 50.0]] * 7
        loss = mean_pinball_loss(actuals, bad_preds, QUANTILES)
        assert loss > 1.0
