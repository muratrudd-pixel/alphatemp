# Phase 3.7: Dynamic Uncertainty — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fixed residual std with a predicted std that collapses as evidence accumulates through the day.

**Architecture:** Parallel variance regression at Phase 2B layer. For each evaluation, after fitting the mean OLS, compute its residuals, winsorize, square, log-transform, and fit a second OLS predicting log(variance) from confidence features. Each NWP model gets its own variance regression. No existing code is modified — new functions only.

**Tech Stack:** Python 3.9, NumPy, SciPy (lstsq), DuckDB, pytest

**Key files to understand before starting:**
- `services/backtester.py:1206-1239` — `_fit_and_predict_phase2b()` (the function we're augmenting)
- `services/backtester.py:1242-1339` — `_make_phase2b_model()` (factory we're cloning for dynamic variant)
- `services/backtester.py:904-950` — `_ensure_level2()` (training data cache)
- `services/backtester.py:700-710` — `_PHASE2B_FEATURE_KEYS` (divergence feature indices)
- `services/ensemble.py:11-53` — `combine_mixture_brackets()` (already uses per-model std)
- `services/neighbor_obs.py:105-144` — `compute_peak_signal()` (Tier 1 variance feature)

---

### Task 1: Variance Model — Utility Functions

**Files:**
- Create: `services/variance_model.py`
- Create: `tests/test_variance_model.py`

**Step 1: Write failing tests**

```python
# tests/test_variance_model.py
import pytest
import math
from services.variance_model import winsorize, hours_until_sunset


class TestWinsorize:
    def test_clips_outliers(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        result = winsorize(values, lower_pct=5, upper_pct=95)
        # 100.0 should be clipped to ~95th percentile
        assert max(result) < 100.0
        assert len(result) == len(values)

    def test_preserves_inliers(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = winsorize(values, lower_pct=2, upper_pct=98)
        assert result == pytest.approx(values, abs=0.1)

    def test_symmetric_clipping(self):
        values = [-50.0, 1.0, 2.0, 3.0, 4.0, 50.0]
        result = winsorize(values, lower_pct=10, upper_pct=90)
        assert min(result) > -50.0
        assert max(result) < 50.0

    def test_empty_returns_empty(self):
        assert winsorize([], 2, 98) == []


class TestHoursUntilSunset:
    def test_summer_afternoon(self):
        # June 21, 3 PM ET, NYC (40.78N) — sunset ~8:30 PM
        result = hours_until_sunset(6, 21, 15, latitude=40.78)
        assert 4.5 < result < 6.5

    def test_winter_afternoon(self):
        # Dec 21, 3 PM ET, NYC — sunset ~4:30 PM
        result = hours_until_sunset(12, 21, 15, latitude=40.78)
        assert 0.5 < result < 2.5

    def test_after_sunset_returns_zero(self):
        # Dec 21, 6 PM ET — well past sunset
        result = hours_until_sunset(12, 21, 18, latitude=40.78)
        assert result == 0.0

    def test_morning_returns_many_hours(self):
        # June 21, 6 AM ET — sunrise just happened, sunset ~14.5h away
        result = hours_until_sunset(6, 21, 6, latitude=40.78)
        assert result > 12.0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_variance_model.py -v`
Expected: FAIL — module not found

**Step 3: Implement**

```python
# services/variance_model.py
"""Dynamic uncertainty model for Phase 3.7.

Provides variance regression utilities: winsorization, log-variance OLS,
solar position calculations, and calibration metrics.
"""
import math
from typing import List, Tuple, Optional

import numpy as np


def winsorize(values, lower_pct=2, upper_pct=98):
    # type: (List[float], int, int) -> List[float]
    """Clip values to [lower_pct, upper_pct] percentiles.

    Prevents outlier explosion when squaring residuals for variance
    regression. A single 10F miss creates residual^2=100 that permanently
    inflates predicted variance in an expanding window.
    """
    if not values:
        return []
    arr = np.array(values)
    lo = np.percentile(arr, lower_pct)
    hi = np.percentile(arr, upper_pct)
    return np.clip(arr, lo, hi).tolist()


def hours_until_sunset(month, day, hour_et, latitude=40.78):
    # type: (int, int, int, float) -> float
    """Approximate hours until sunset for a given date and hour (ET).

    Uses simplified solar declination and hour angle calculation.
    Returns 0.0 if already past sunset.

    Args:
        month: 1-12
        day: 1-31
        hour_et: 0-23 Eastern Time
        latitude: degrees north (default 40.78 = Central Park)
    """
    # Day of year (approximate)
    doy = (month - 1) * 30.44 + day

    # Solar declination (radians)
    decl = math.radians(23.44) * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))

    # Hour angle at sunset
    lat_rad = math.radians(latitude)
    cos_ha = -math.tan(lat_rad) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))  # clamp for polar edge cases
    ha_sunset = math.degrees(math.acos(cos_ha))

    # Sunset hour in ET (solar noon ~ 12:00 ET, rough approximation)
    sunset_et = 12.0 + ha_sunset / 15.0

    remaining = sunset_et - hour_et
    return max(0.0, remaining)
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_variance_model.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add services/variance_model.py tests/test_variance_model.py
git commit -m "feat(phase3.7): add variance model utilities — winsorize and hours_until_sunset"
```

---

### Task 2: Variance Model — Core OLS Functions

**Files:**
- Modify: `services/variance_model.py`
- Modify: `tests/test_variance_model.py`

**Step 1: Write failing tests**

```python
# Add to tests/test_variance_model.py
from services.variance_model import fit_variance_ols, predict_std

class TestFitVarianceOLS:
    def test_returns_coefficients(self):
        # 20 training points: residuals with known variance pattern
        np.random.seed(42)
        residuals = np.random.normal(0, 2.0, 100).tolist()
        # Variance feature: just update_hour (0-18, repeated)
        var_features = [[float(i % 19)] for i in range(100)]
        coeffs = fit_variance_ols(residuals, var_features, [0])
        assert coeffs is not None
        assert len(coeffs) == 2  # intercept + 1 feature

    def test_returns_none_below_min_days(self):
        residuals = [1.0, 2.0]
        var_features = [[0.0], [1.0]]
        coeffs = fit_variance_ols(residuals, var_features, [0], min_days=90)
        assert coeffs is None

    def test_winsorizes_before_squaring(self):
        # One massive outlier — should be clipped
        residuals = [1.0] * 99 + [50.0]
        var_features = [[float(i)] for i in range(100)]
        coeffs = fit_variance_ols(residuals, var_features, [0])
        assert coeffs is not None


class TestPredictStd:
    def test_returns_positive_std(self):
        coeffs = np.array([1.0, -0.05])  # intercept + slope
        features_today = [10.0]  # update_hour = 10
        std = predict_std(coeffs, features_today, [0])
        assert std > 0

    def test_floored_at_0_3(self):
        # Coefficients that would predict very low variance
        coeffs = np.array([-10.0, 0.0])
        features_today = [18.0]
        std = predict_std(coeffs, features_today, [0])
        assert std == pytest.approx(0.3)

    def test_capped_at_5_0(self):
        # Coefficients that would predict enormous variance
        coeffs = np.array([20.0, 0.0])
        features_today = [0.0]
        std = predict_std(coeffs, features_today, [0])
        assert std == pytest.approx(5.0)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_variance_model.py::TestFitVarianceOLS -v`
Expected: FAIL — function not defined

**Step 3: Implement**

```python
# Add to services/variance_model.py
from scipy.linalg import lstsq

# Minimum training window (same as mean regression)
_VARIANCE_MIN_DAYS = 90
_STD_FLOOR = 0.3
_STD_CAP = 5.0


def fit_variance_ols(residuals, var_features, feature_indices, min_days=_VARIANCE_MIN_DAYS):
    # type: (List[float], List[List[float]], List[int], int) -> Optional[np.ndarray]
    """Fit OLS on log(winsorized_residual^2 + 1e-6) ~ variance features.

    Args:
        residuals: Mean regression residuals (one per training date).
        var_features: Variance feature vectors for each training date.
        feature_indices: Which features to use from var_features.
        min_days: Minimum training window.

    Returns:
        OLS coefficients or None if insufficient data.
    """
    n = len(residuals)
    if n < min_days:
        return None

    # Winsorize before squaring
    clipped = winsorize(residuals, lower_pct=2, upper_pct=98)

    # Target: log(residual^2 + epsilon)
    y = np.array([math.log(r * r + 1e-6) for r in clipped])

    # Build feature matrix
    n_feats = len(feature_indices)
    A = np.empty((n, 1 + n_feats), dtype=np.float64)
    for i in range(n):
        A[i, 0] = 1.0
        for j, idx in enumerate(feature_indices):
            A[i, 1 + j] = var_features[i][idx]

    result = lstsq(A, y)
    return result[0]


def predict_std(var_coeffs, features_today, feature_indices):
    # type: (np.ndarray, List[float], List[int]) -> float
    """Predict std from variance regression coefficients.

    Returns sqrt(exp(prediction)), floored at 0.3F and capped at 5.0F.
    """
    x = np.array([1.0] + [features_today[idx] for idx in feature_indices])
    log_var = float(np.dot(var_coeffs, x))

    # Clamp log_var to avoid overflow/underflow
    log_var = max(-4.0, min(6.0, log_var))  # exp(6) ~ 403, sqrt ~ 20; exp(-4) ~ 0.018, sqrt ~ 0.13

    predicted_std = math.sqrt(math.exp(log_var))
    return max(_STD_FLOOR, min(_STD_CAP, predicted_std))
```

**Step 4: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_variance_model.py -v`
Expected: PASS (all 14 tests)

**Step 5: Commit**

```bash
git add services/variance_model.py tests/test_variance_model.py
git commit -m "feat(phase3.7): add variance OLS fit and predict functions"
```

---

### Task 3: Variance Model — Evaluation Metrics

**Files:**
- Modify: `services/variance_model.py`
- Modify: `tests/test_variance_model.py`

**Step 1: Write failing tests**

```python
# Add to tests/test_variance_model.py
from services.variance_model import compute_ece, compute_log_loss, brier_decomposition


class TestComputeECE:
    def test_perfect_calibration(self):
        # When predicted probs match empirical frequency, ECE ~ 0
        predicted = [0.5] * 100
        actual = [1] * 50 + [0] * 50
        ece = compute_ece(predicted, actual, n_bins=10)
        assert ece < 0.05

    def test_overconfident_high_ece(self):
        # Always predicting 0.9 but only 50% hit rate
        predicted = [0.9] * 100
        actual = [1] * 50 + [0] * 50
        ece = compute_ece(predicted, actual, n_bins=10)
        assert ece > 0.3

    def test_returns_between_0_and_1(self):
        predicted = [0.3, 0.5, 0.7, 0.9]
        actual = [0, 1, 1, 1]
        ece = compute_ece(predicted, actual, n_bins=5)
        assert 0.0 <= ece <= 1.0


class TestComputeLogLoss:
    def test_perfect_predictions(self):
        predicted = [0.99, 0.01]
        actual = [1, 0]
        ll = compute_log_loss(predicted, actual)
        assert ll < 0.05

    def test_bad_predictions_high_loss(self):
        predicted = [0.01, 0.99]
        actual = [1, 0]
        ll = compute_log_loss(predicted, actual)
        assert ll > 3.0

    def test_clamps_extreme_probs(self):
        # Should not return inf even with 0 or 1 predictions
        predicted = [0.0, 1.0]
        actual = [1, 0]
        ll = compute_log_loss(predicted, actual)
        assert math.isfinite(ll)


class TestBrierDecomposition:
    def test_returns_three_components(self):
        predicted = [0.3, 0.5, 0.7, 0.9]
        actual = [0, 1, 1, 1]
        rel, res, unc = brier_decomposition(predicted, actual, n_bins=4)
        assert all(math.isfinite(v) for v in (rel, res, unc))

    def test_reliability_near_zero_for_calibrated(self):
        # 50% predictions with 50% hit rate
        predicted = [0.5] * 100
        actual = [1] * 50 + [0] * 50
        rel, _, _ = brier_decomposition(predicted, actual, n_bins=10)
        assert rel < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_variance_model.py::TestComputeECE -v`
Expected: FAIL

**Step 3: Implement**

```python
# Add to services/variance_model.py

def compute_ece(predicted, actual, n_bins=10):
    # type: (List[float], List[int], int) -> float
    """Expected Calibration Error.

    Bins predictions into n_bins equal-width bins, computes
    |mean_predicted - mean_actual| per bin, weighted by bin size.
    """
    if not predicted:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    actuals_in_bin = [[] for _ in range(n_bins)]

    for p, a in zip(predicted, actual):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(p)
        actuals_in_bin[idx].append(a)

    n_total = len(predicted)
    ece = 0.0
    for i in range(n_bins):
        if not bins[i]:
            continue
        mean_pred = sum(bins[i]) / len(bins[i])
        mean_actual = sum(actuals_in_bin[i]) / len(actuals_in_bin[i])
        ece += len(bins[i]) / n_total * abs(mean_pred - mean_actual)

    return ece


def compute_log_loss(predicted, actual):
    # type: (List[float], List[int]) -> float
    """Mean log-loss (cross-entropy). Clamps predictions to [1e-7, 1-1e-7]."""
    if not predicted:
        return 0.0
    eps = 1e-7
    total = 0.0
    for p, a in zip(predicted, actual):
        p_clamped = max(eps, min(1.0 - eps, p))
        if a == 1:
            total -= math.log(p_clamped)
        else:
            total -= math.log(1.0 - p_clamped)
    return total / len(predicted)


def brier_decomposition(predicted, actual, n_bins=10):
    # type: (List[float], List[int], int) -> Tuple[float, float, float]
    """Decompose Brier score into Reliability, Resolution, Uncertainty.

    Brier = Reliability - Resolution + Uncertainty
    Perfect calibration: Reliability -> 0
    Good discrimination: Resolution -> high

    Returns (reliability, resolution, uncertainty).
    """
    if not predicted:
        return (0.0, 0.0, 0.0)

    n = len(predicted)
    base_rate = sum(actual) / n
    uncertainty = base_rate * (1.0 - base_rate)

    bins = [[] for _ in range(n_bins)]
    actuals_in_bin = [[] for _ in range(n_bins)]

    for p, a in zip(predicted, actual):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(p)
        actuals_in_bin[idx].append(a)

    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        if not bins[i]:
            continue
        n_k = len(bins[i])
        mean_pred = sum(bins[i]) / n_k
        mean_actual = sum(actuals_in_bin[i]) / n_k
        reliability += n_k / n * (mean_pred - mean_actual) ** 2
        resolution += n_k / n * (mean_actual - base_rate) ** 2

    return (reliability, resolution, uncertainty)
```

**Step 4: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_variance_model.py -v`
Expected: PASS (all 22 tests)

**Step 5: Commit**

```bash
git add services/variance_model.py tests/test_variance_model.py
git commit -m "feat(phase3.7): add calibration metrics — ECE, log-loss, Brier decomposition"
```

---

### Task 4: Dynamic Model Factory in Backtester

**Files:**
- Modify: `services/backtester.py` (add new function, do NOT modify existing functions)
- Modify: `tests/test_phase2b_models.py`

**Context:** The existing `_make_phase2b_model()` at line 1242 is the template. We're creating a new factory `_make_phase2b_dynamic_model()` that clones its logic but replaces the fixed residual_std with a predicted std from variance regression.

**Step 1: Write failing tests**

```python
# Add to tests/test_phase2b_models.py
from services.backtester import wf_phase2b_dynamic

class TestDynamicStdModel:
    def test_returns_valid_probs(self, test_db):
        probs = _run_model(wf_phase2b_dynamic, test_db)
        if probs is not None:
            assert all(0 <= p <= 1 for p in probs.values())
            assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_raw_returns_tuple(self, test_db):
        result = _run_raw_model(wf_phase2b_dynamic, test_db)
        if result is not None:
            center, std = result
            assert isinstance(center, float)
            assert isinstance(std, float)
            assert 0.3 <= std <= 5.0  # within floor/cap
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_phase2b_models.py::TestDynamicStdModel -v`
Expected: FAIL — wf_phase2b_dynamic not found

**Step 3: Implement**

Add to `services/backtester.py` (after `_make_phase2b_model` definition, around line 1340):

```python
from services.variance_model import fit_variance_ols, predict_std, hours_until_sunset

# Variance feature index mapping:
#   0 = update_hour_et, 1 = slope_divergence, 2 = cumulative_divergence,
#   3 = running_max_divergence, 4 = fcst_high, 5 = hours_until_sunset
_VARIANCE_FEATURE_KEYS = [
    "update_hour", "slope_divergence", "cumulative_divergence",
    "running_max_divergence", "fcst_high", "hours_until_sunset",
]


def _make_phase2b_dynamic_model(name, mean_feature_indices, var_feature_indices, model_name='hrrr'):
    # type: (str, list, list, str) -> ModelFn
    """Factory: Phase 2B with dynamic std from variance regression.

    Same mean regression as _make_phase2b_model. Adds parallel variance
    regression predicting log(residual^2) from confidence features.
    """
    def _raw(provider, ref_time):
        # type: (BacktestDataProvider, datetime) -> Optional[Tuple[float, float]]
        con = provider._shared_con
        station_id = provider.station_id
        run_hour = provider.model_run.hour
        current_date = provider.model_run.date()

        fcst_high = provider.get_forecast_high(station_id)
        if fcst_high is None:
            return None

        # Phase 2 prediction (from cache)
        l1 = _ensure_level1(con, run_hour, station_id, model_name=model_name)
        p2_pred = l1["p2_preds"].get(current_date)
        if p2_pred is None:
            return None
        phase2_bias, p2_std = p2_pred

        # Determine update hour
        ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
        update_hour_et = ref_utc.astimezone(_ET).hour
        cutoff_ts = ref_utc.timestamp()

        # Today's divergence features
        curves = l1["curves"]
        if current_date not in curves or len(curves[current_date]) < 2:
            center = fcst_high - phase2_bias
            return (center, p2_std)

        curve = curves[current_date]
        obs_by_date = l1["obs_by_date"]
        day_obs = obs_by_date.get(current_date, [])
        truncated = [(ts, temp) for ts, temp in day_obs if ts <= cutoff_ts]
        if len(truncated) < 2:
            center = fcst_high - phase2_bias
            return (center, p2_std)

        obs_ts = [o[0] for o in truncated]
        obs_temps = [o[1] for o in truncated]
        fc_ts = [c[0] for c in curve]
        fc_temps = [c[1] for c in curve]

        fcst_interp = interpolate_forecast(fc_ts, fc_temps, obs_ts)
        t0 = obs_ts[0]
        obs_hours = [(t - t0) / 3600.0 for t in obs_ts]
        fcst_up_to_t = [temp for ts, temp in curve if ts <= cutoff_ts]

        today_feats = compute_divergence_features(
            obs_temps, fcst_interp, obs_hours, fcst_up_to_t
        )
        if today_feats is None:
            center = fcst_high - phase2_bias
            return (center, p2_std)

        today_mean_vec = [today_feats[k] for k in _PHASE2B_FEATURE_KEYS]

        # Today's variance features
        sunset_hrs = hours_until_sunset(
            current_date.month, current_date.day, update_hour_et
        )
        today_var_vec = [
            float(update_hour_et),
            today_feats["slope_divergence"],
            today_feats["cumulative_divergence"],
            today_feats["running_max_divergence"],
            float(fcst_high),
            sunset_hrs,
        ]

        # Training data (filtered to dates < current_date)
        all_training = _ensure_level2(con, run_hour, station_id, update_hour_et, model_name=model_name)
        filtered = [
            (d, residual, feat_vec)
            for d, residual, feat_vec in all_training
            if d < current_date
        ]

        if len(filtered) < WALK_FORWARD_MIN_DAYS:
            center = fcst_high - phase2_bias
            return (center, p2_std)

        # Mean regression (same as standard Phase 2B)
        mean_rows = [(residual, feat_vec) for d, residual, feat_vec in filtered]
        p2b_result = _fit_and_predict_phase2b(mean_rows, today_mean_vec, mean_feature_indices)
        if p2b_result is None:
            center = fcst_high - phase2_bias
            return (center, p2_std)

        phase2b_residual, _ = p2b_result  # discard fixed std

        # Variance regression
        # Recompute mean regression residuals for variance training
        n = len(mean_rows)
        n_mf = len(mean_feature_indices)
        A = np.empty((n, 1 + n_mf), dtype=np.float64)
        y = np.empty(n, dtype=np.float64)
        for i, (res, fv) in enumerate(mean_rows):
            A[i, 0] = 1.0
            for j, idx in enumerate(mean_feature_indices):
                A[i, 1 + j] = fv[idx]
            y[i] = res
        mean_coeffs = lstsq(A, y)[0]
        mean_residuals = (y - A @ mean_coeffs).tolist()

        # Build variance feature vectors for training dates
        var_features_train = []
        for d, residual, feat_vec in filtered:
            sunset_h = hours_until_sunset(d.month, d.day, update_hour_et)
            # Get fcst_high for this training date from L1 cache
            p2_train = l1["p2_preds"].get(d)
            train_fcst_high = 0.0
            if p2_train is not None:
                # Approximate: fcst_high ~ center + phase2_bias (reverse the correction)
                train_fcst_high = p2_train[0] + p2_train[1]  # Not exact, use stored value
            # Actually we need the raw fcst_high. Use a simpler proxy: stored in curves
            # For now, approximate with the max of the forecast curve for this date
            train_curve = curves.get(d, [])
            if train_curve:
                train_fcst_high = max(t for _, t in train_curve)

            var_features_train.append([
                float(update_hour_et),
                feat_vec[3],  # slope_divergence
                feat_vec[1],  # cumulative_divergence
                feat_vec[2],  # running_max_divergence
                float(train_fcst_high),
                sunset_h,
            ])

        var_coeffs = fit_variance_ols(mean_residuals, var_features_train, var_feature_indices)
        if var_coeffs is None:
            center = fcst_high - (phase2_bias + phase2b_residual)
            return (center, p2b_result[1])  # fall back to fixed std

        dynamic_std = predict_std(var_coeffs, today_var_vec, var_feature_indices)
        center = fcst_high - (phase2_bias + phase2b_residual)
        return (center, dynamic_std)

    def model_fn(provider, ref_time):
        params = _raw(provider, ref_time)
        if params is None:
            return None
        center, residual_std = params
        return _compute_bracket_probs_from_dist(norm(0, 1), center, residual_std)

    model_fn.__name__ = name
    model_fn.__doc__ = "Phase 2B dynamic std: {}".format(name)
    model_fn.raw = _raw
    return model_fn


# Pre-built instances for ablation
# Variance feature indices: 0=update_hour, 1=slope, 2=cumul, 3=runmax, 4=fcst_high, 5=sunset
wf_phase2b_dynamic = _make_phase2b_dynamic_model(
    "wf_phase2b_dynamic",
    mean_feature_indices=[0, 1, 2, 3],  # all 4 divergence features for mean
    var_feature_indices=[0, 1, 2, 3, 4, 5],  # all 6 variance features
)
```

**Step 4: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python -m pytest tests/test_phase2b_models.py -v`
Expected: PASS (all existing tests + 2 new)

**Step 5: Commit**

```bash
git add services/backtester.py tests/test_phase2b_models.py
git commit -m "feat(phase3.7): add dynamic std model factory with variance regression"
```

---

### Task 5: Ablation Analysis Script

**Files:**
- Create: `scripts/phase37_variance_analysis.py`

**Step 1: Write the analysis script**

The script should:
1. Run baseline (static std Phase 2B) and dynamic variants across 0-18 ET
2. For each variant, report: Brier score, ECE, log-loss, Brier decomposition
3. Plot std trajectory by hour (avg predicted std across all dates for each update hour)
4. Gate check: >2% Brier improvement AND lower ECE

```python
#!/usr/bin/env python
"""Phase 3.7: Dynamic Uncertainty — Variance Feature Ablation Analysis.

Tests variance features individually and in combination.
Reports: Brier, ECE, log-loss, Brier decomposition, std trajectory.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import date
from services.backtester import (
    Backtester,
    _make_phase2b_model,
    _make_phase2b_dynamic_model,
)
from services.variance_model import compute_ece, compute_log_loss, brier_decomposition

# Date range
START = date(2021, 6, 1)
END = date(2026, 2, 25)
RUN_HOURS = [0, 6, 12, 18]
UPDATE_HOURS = list(range(0, 19))

DB_PATH = "data/alphatemp.duckdb"

# Ablation variants
# Variance feature indices: 0=update_hour, 1=slope, 2=cumul, 3=runmax, 4=fcst_high, 5=sunset
VARIANTS = {
    "baseline": None,  # static std (existing Phase 2B)
    "var_hour":      [0],           # update_hour only
    "var_slope":     [1],           # slope_divergence only
    "var_cumul":     [2],           # cumulative_divergence only
    "var_runmax":    [3],           # running_max_divergence only
    "var_fcst":      [4],           # fcst_high only
    "var_sunset":    [5],           # hours_until_sunset only
    "var_hour_slope":     [0, 1],   # update_hour + slope
    "var_hour_slope_sunset": [0, 1, 5],  # hour + slope + sunset
    "var_all":       [0, 1, 2, 3, 4, 5],  # all 6
}

MEAN_FEATURES = [0, 1, 2, 3]  # all 4 divergence features for mean regression


def build_models():
    """Build baseline and dynamic variant model functions."""
    models = {}

    # Baseline: standard Phase 2B with static std
    models["baseline"] = _make_phase2b_model(
        "baseline", MEAN_FEATURES, model_name="hrrr"
    )

    # Dynamic variants
    for vname, var_indices in VARIANTS.items():
        if var_indices is None:
            continue
        models[vname] = _make_phase2b_dynamic_model(
            vname, MEAN_FEATURES, var_indices, model_name="hrrr"
        )

    return models


def main():
    models = build_models()
    variant_names = list(VARIANTS.keys())

    print("=" * 90)
    print("Phase 3.7: Dynamic Uncertainty — Variance Feature Ablation")
    print("=" * 90)
    print(f"  Date range:    {START} to {END}")
    print(f"  Run hours:     {RUN_HOURS}")
    print(f"  Update hours:  0-18 ET")
    print(f"  Variants:      {len(variant_names)}")
    print()

    results = {}
    total_start = time.time()

    for vname in variant_names:
        print(f"[{variant_names.index(vname)+1}/{len(variant_names)}] Running {vname}...")
        model_fn = models[vname]

        bt = Backtester(
            db_path=DB_PATH,
            model_fn=model_fn,
            start_date=START,
            end_date=END,
            run_hours=RUN_HOURS,
            update_hours=UPDATE_HOURS,
        )
        t0 = time.time()
        result = bt.run()
        elapsed = time.time() - t0
        results[vname] = result
        print(f"  Done: mean_brier={result.mean_brier:.4f}  ({elapsed:.0f}s)")
        print()

    total_elapsed = time.time() - total_start
    print(f"All variants complete. Total time: {total_elapsed:.0f}s")

    # --- Section 1: Per-hour Brier ---
    print()
    print("=" * 90)
    print("SECTION 1: Per-Update-Hour Brier Scores")
    print("=" * 90)
    header = f"{'Hr ET':>6}"
    for vname in variant_names:
        header += f" {vname[:10]:>10}"
    print(header)
    print("-" * len(header))

    for h in UPDATE_HOURS:
        row = f"{h:>6}"
        for vname in variant_names:
            brier = results[vname].by_hour.get(h, {}).get("brier", float("nan"))
            row += f" {brier:>10.4f}"
        print(row)

    # --- Section 2: Improvement vs baseline ---
    print()
    print("=" * 90)
    print("SECTION 2: Improvement vs Baseline (%)")
    print("=" * 90)
    baseline_by_hour = results["baseline"].by_hour

    for h in UPDATE_HOURS:
        row = f"{h:>6}"
        base_b = baseline_by_hour.get(h, {}).get("brier", 0)
        for vname in variant_names:
            if vname == "baseline":
                continue
            var_b = results[vname].by_hour.get(h, {}).get("brier", 0)
            if base_b > 0:
                pct = (base_b - var_b) / base_b * 100
                row += f" {pct:>+9.2f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # --- Section 3: Gate check ---
    print()
    print("=" * 90)
    print("SECTION 3: Gate Check (>2% Brier AND lower ECE)")
    print("=" * 90)

    base_mean = results["baseline"].mean_brier
    print(f"{'Variant':<25} {'Brier':>8} {'Improv':>8} {'Verdict':>10}")
    print("-" * 55)
    for vname in variant_names:
        brier = results[vname].mean_brier
        if vname == "baseline":
            print(f"{vname:<25} {brier:>8.4f} {'---':>8} {'BASE':>10}")
        else:
            improv = (base_mean - brier) / base_mean * 100
            verdict = "PASS" if improv > 2.0 else ("DISCUSS" if improv > 1.0 else "KILL")
            print(f"{vname:<25} {brier:>8.4f} {improv:>+7.2f}% {verdict:>10}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python scripts/phase37_variance_analysis.py`
Expected: Runs all variants, outputs tables. ~30-45 min total.

**Step 3: Commit**

```bash
git add scripts/phase37_variance_analysis.py
git commit -m "feat(phase3.7): add variance feature ablation analysis script"
```

---

### Task 6: Run Analysis and Evaluate Results

**Step 1:** Read the analysis output

**Step 2:** For each variant, evaluate against dual gate:
- Brier improvement >2% vs baseline?
- ECE improved?

**Step 3:** If any variants pass:
- Extend to GFS/ECMWF (add `model_name` parameter)
- Test through ensemble
- Update MASTER-PLAN.md with results

**Step 4:** If no variants pass:
- Document what was tried
- Consider: quantile regression escape hatch? Different feature engineering?
- Update MASTER-PLAN.md with KILL verdict

**Step 5:** Commit results and update HANDOFF.md

---

## Notes for Implementer

**Critical context:**
- Python 3.9 — use `typing.List`, `typing.Dict`, NOT `list[...]`, `dict[...]`
- Tests run from `~/Projects/alphatemp/alphatemp/` (NOT the outer repo)
- Test runner: `../venv/bin/python -m pytest tests/... -v`
- Git commits from `~/Projects/alphatemp/alphatemp/` (it's a submodule)
- The DB is at `data/alphatemp.duckdb` (relative path from the inner repo)
- `_fit_and_predict_phase2b()` at line 1206 — do NOT modify this function, call it from the new factory
- All existing model factories and tests must continue to pass unchanged
- `WALK_FORWARD_MIN_DAYS = 90` (line 204) — same minimum for variance regression
- Existing `_ensure_level2()` cache returns `(obs_date, residual, feat_vector)` tuples
- Forecast curves are in L1 cache: `l1["curves"][date]` = list of `(timestamp, temp_f)`

**Variance feature mapping (used in factory and ablation script):**
```
Index 0: update_hour_et     (0-18, from ref_time)
Index 1: slope_divergence   (mean_feat_vec[3])
Index 2: cumulative_divergence (mean_feat_vec[1])
Index 3: running_max_divergence (mean_feat_vec[2])
Index 4: fcst_high          (from provider.get_forecast_high())
Index 5: hours_until_sunset (computed from date + hour)
```
