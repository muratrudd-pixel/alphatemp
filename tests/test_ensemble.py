"""Test Gaussian mixture ensemble combination."""
import pytest
from services.ensemble import combine_mixture_brackets, make_ensemble_model_fn


def test_equal_weight_two_identical_models():
    """Two identical Gaussians should produce same brackets as single."""
    single = combine_mixture_brackets([(70.0, 3.0)], [1.0])
    double = combine_mixture_brackets([(70.0, 3.0), (70.0, 3.0)], [0.5, 0.5])
    # Peak should be same
    assert max(single, key=single.get) == max(double, key=double.get)
    # Total should be ~1.0
    assert abs(sum(double.values()) - 1.0) < 0.01


def test_mixture_wider_than_components():
    """Two models disagreeing should produce wider distribution."""
    single = combine_mixture_brackets([(70.0, 2.5)], [1.0])
    mixture = combine_mixture_brackets([(68.0, 2.5), (72.0, 2.5)], [0.5, 0.5])
    # Mixture should have more brackets with non-trivial probability
    assert len(mixture) >= len(single)


def test_three_model_equal_weights():
    """HRRR + GFS + ECMWF equal weight ensemble."""
    predictions = [(70.0, 2.5), (68.0, 3.0), (71.0, 2.8)]
    weights = [1/3, 1/3, 1/3]
    brackets = combine_mixture_brackets(predictions, weights)
    assert abs(sum(brackets.values()) - 1.0) < 0.01
    peak = max(brackets, key=brackets.get)
    assert 68 <= peak <= 71


def test_empty_predictions():
    """No predictions should return empty dict."""
    assert combine_mixture_brackets([], []) == {}


def test_single_model_matches_gaussian():
    """Single model should match a plain Gaussian."""
    from scipy.stats import norm
    brackets = combine_mixture_brackets([(70.0, 3.0)], [1.0])
    # P(69.5 < X < 70.5) for N(70, 3)
    expected_70 = norm.cdf(0.5/3.0) - norm.cdf(-0.5/3.0)
    # Should be close after renormalization
    assert abs(brackets[70] - expected_70) < 0.02


def test_weights_matter():
    """Heavier weight on one model should shift the peak."""
    heavy_left = combine_mixture_brackets([(65.0, 2.0), (75.0, 2.0)], [0.9, 0.1])
    heavy_right = combine_mixture_brackets([(65.0, 2.0), (75.0, 2.0)], [0.1, 0.9])
    peak_left = max(heavy_left, key=heavy_left.get)
    peak_right = max(heavy_right, key=heavy_right.get)
    assert peak_left < peak_right


def test_make_ensemble_model_fn():
    """Ensemble wraps model .raw calls correctly."""
    # Create fake model functions with .raw attribute
    def fake_model_1(provider, ref_time):
        return {70: 1.0}
    fake_model_1.raw = lambda provider, ref_time: (70.0, 3.0)

    def fake_model_2(provider, ref_time):
        return {68: 1.0}
    fake_model_2.raw = lambda provider, ref_time: (68.0, 2.5)

    ensemble = make_ensemble_model_fn([fake_model_1, fake_model_2])
    result = ensemble(None, None)  # provider/ref_time not used by fakes
    assert result is not None
    assert abs(sum(result.values()) - 1.0) < 0.01
    peak = max(result, key=result.get)
    assert 68 <= peak <= 70


def test_ensemble_handles_none_from_model():
    """If a model returns None from .raw, skip it."""
    def model_ok(provider, ref_time):
        return {70: 1.0}
    model_ok.raw = lambda provider, ref_time: (70.0, 3.0)

    def model_fail(provider, ref_time):
        return None
    model_fail.raw = lambda provider, ref_time: None

    ensemble = make_ensemble_model_fn([model_ok, model_fail])
    result = ensemble(None, None)
    assert result is not None  # Should still work with one model
    assert 70 == max(result, key=result.get)
