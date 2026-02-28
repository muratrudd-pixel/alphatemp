# Phase 2: Bias Correction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find the best bias correction method for 24 HRRR runs + multi-model ensemble, beating the existing OLS baseline (Brier 0.7979 on 4 runs).

**Architecture:** Three candidates (OLS, EMOS, XGBoost QR) evaluated head-to-head on the same walk-forward expanding window. All use the existing `Backtester` framework. Gold feature views provide pre-computed inputs. Strategy backtester runs at each gate for P&L diagnostics (not a kill signal at Phase 2).

**Tech Stack:** Python 3.9, DuckDB, scipy (OLS/EMOS), xgboost (Candidate C), numpy, loguru, pytest

**Design doc:** `docs/plans/2026-02-27-rebuild-design.md` (Phase 2 section)

**Prerequisite:** Phase 1 gate PASSED — all 24 HRRR runs, GFS 06z/12z/18z, ECMWF 00z ingested and merged.

---

## Task Dependencies

```
Task 1 (Expand RUN_HOURS) ─┬── Task 2 (OLS 24-run baseline)
                            ├── Task 3 (EMOS candidate)
                            ├── Task 4 (XGBoost QR candidate)
                            └── Task 5 (Spin-up ablation)

Task 2, 3, 4, 5 all complete ── Task 6 (Head-to-head comparison)
Task 6 ── Task 7 (Multi-model ensemble)
Task 7 ── Task 8 (Strategy backtester P&L diagnostic)
Task 8 ── Task 9 (Phase 2 gate evaluation)
```

Tasks 2-5 can run in parallel after Task 1.

---

### Task 1: Expand Backtester to All 24 HRRR Run Hours

**Files:**
- Modify: `services/backtester.py:35`
- Create: `tests/test_backtester_24h.py`

**Context:** The existing `RUN_HOURS = [0, 6, 12, 18]` evaluates only 4 of 24 HRRR runs. We now have all 24 hours in the database. The backtester must support evaluating all 24 (or any subset). GFS/ECMWF still run at [0, 6, 12, 18] so model-specific hour filtering is needed.

**Step 1: Write failing test**

```python
# tests/test_backtester_24h.py

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db
from services.backtester import Backtester, RUN_HOURS


def test_run_hours_includes_all_24():
    """RUN_HOURS should list all 24 HRRR run hours."""
    assert len(RUN_HOURS) == 24
    assert RUN_HOURS == list(range(24))


def test_backtester_accepts_run_hours_param(tmp_path):
    """Backtester.run() should accept a run_hours parameter."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    bt = Backtester(db_path=db_path)
    # Should not raise — just verify the parameter is accepted
    result = bt.run(
        model_fn=lambda prov, ref: None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        run_hours=[0, 12],
    )
    assert result is not None


def test_backtester_filters_by_run_hours(tmp_path):
    """Backtester should only evaluate specified run_hours."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)

    # Seed forecasts at multiple run hours
    con = duckdb.connect(db_path)
    for hour in [0, 6, 12, 18]:
        model_run = datetime(2024, 6, 15, hour)
        for fxx in range(1, 19):
            valid_at = model_run + timedelta(hours=fxx)
            con.execute(
                "INSERT INTO forecasts (station_id, model_run, valid_at, "
                "temp_f, temp_c, ingested_at, model_name, fxx, is_spinup) "
                "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                ['KNYC', model_run, valid_at, 75.0, 23.9,
                 datetime(2024, 6, 15), fxx, fxx <= 3],
            )
    # Seed NWS daily
    con.execute(
        "INSERT INTO nws_daily (station_id, obs_date, max_temp_f, ingested_at) "
        "VALUES ('KNYC', '2024-06-15', 78.0, '2024-06-16')"
    )
    con.close()

    bt = Backtester(db_path=db_path)
    # Evaluate only 0z and 12z
    result = bt.run(
        model_fn=lambda prov, ref: {k: 1/31 for k in range(60, 91)},
        start_date=date(2024, 6, 15),
        end_date=date(2024, 6, 15),
        run_hours=[0, 12],
    )
    # Should have results for exactly 2 run hours, not 4
    hours_seen = set()
    for r in result.results:
        hours_seen.add(r.run_hour)
    assert hours_seen == {0, 12}
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_backtester_24h.py -v`
Expected: FAIL — `RUN_HOURS` has 4 elements, `run_hours` param not accepted

**Step 3: Implement**

In `services/backtester.py`:

1. Change `RUN_HOURS = [0, 6, 12, 18]` to `RUN_HOURS = list(range(24))`
2. Add `run_hours` parameter to `Backtester.run()` (default `None` = use all)
3. Inside the date×hour loop, skip hours not in `run_hours` if specified

```python
# Line 35
RUN_HOURS = list(range(24))

# In Backtester.run(), add parameter:
def run(self, model_fn, start_date, end_date, run_hours=None, ...):
    hours = run_hours if run_hours is not None else RUN_HOURS
    for current_date in date_range:
        for run_hour in hours:
            ...
```

**Step 4: Run tests**

Run: `PYTHONPATH=. pytest tests/test_backtester_24h.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services/backtester.py tests/test_backtester_24h.py
git commit -m "feat(phase2): expand backtester to all 24 HRRR run hours"
```

---

### Task 2: OLS Baseline on 24 HRRR Runs (Candidate A)

**Files:**
- Modify: `services/backtester.py` (update `_fit_and_predict` to read from gold views)
- Create: `tests/test_phase2_ols_24h.py`
- Create: `scripts/run_phase2_ablation.py`

**Context:** The existing OLS regression uses `fcst_high + sin/cos month + delta_temp`. It trained on 4 run hours. We need to re-run it on all 24 hours and record the new baseline Brier. The `gold_hrrr_bias_features` view pre-computes these features.

**Step 1: Write the ablation runner script**

```python
# scripts/run_phase2_ablation.py
"""Run Phase 2 bias correction ablation across all candidates.

Usage:
    python scripts/run_phase2_ablation.py --candidate ols
    python scripts/run_phase2_ablation.py --candidate ols --run-hours 0,6,12,18
    python scripts/run_phase2_ablation.py --candidate all
"""

import argparse
import json
from datetime import date
from typing import List, Optional

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.backtester import Backtester


def run_ablation(
    candidate: str,
    db_path: str = DEFAULT_DB_PATH,
    run_hours: Optional[List[int]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    """Run a single ablation candidate and print results."""
    bt = Backtester(db_path=db_path)

    # Import the appropriate model function
    if candidate == "ols":
        from services.backtester import wf_regression_full as model_fn
    elif candidate == "ols_no_spinup":
        from services.backtester import wf_regression_full_no_spinup as model_fn
    elif candidate == "emos":
        from services.phase2_emos import emos_model_fn as model_fn
    elif candidate == "xgboost":
        from services.phase2_xgboost import xgboost_model_fn as model_fn
    else:
        raise ValueError(f"Unknown candidate: {candidate}")

    result = bt.run(
        model_fn=model_fn,
        start_date=start_date or date(2023, 1, 1),
        end_date=end_date or date(2026, 2, 1),
        run_hours=run_hours,
    )

    logger.info("Candidate: {}", candidate)
    logger.info("Run hours: {}", run_hours or "all 24")
    logger.info("Mean Brier: {:.4f}", result.mean_brier)
    logger.info("Top-1 hit: {:.1%}", result.top1_hit_rate)
    logger.info("Top-2 hit: {:.1%}", result.top2_hit_rate)
    logger.info("N results: {}", len(result.results))

    # Per-hour breakdown
    logger.info("Per-hour Brier:")
    for hour, metrics in sorted(result.per_hour.items()):
        logger.info("  {:02d}z: {:.4f} (n={})", hour, metrics['mean_brier'], metrics['n'])

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2 ablation runner")
    parser.add_argument("--candidate", required=True,
                        choices=["ols", "ols_no_spinup", "emos", "xgboost", "all"])
    parser.add_argument("--run-hours", default=None,
                        help="Comma-separated run hours (default: all 24)")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    run_hours = None
    if args.run_hours:
        run_hours = [int(h) for h in args.run_hours.split(",")]

    candidates = ["ols", "ols_no_spinup", "emos", "xgboost"] if args.candidate == "all" else [args.candidate]

    for cand in candidates:
        try:
            run_ablation(cand, args.db, run_hours, args.start, args.end)
        except Exception as e:
            logger.error("Candidate {} failed: {}", cand, e)


if __name__ == "__main__":
    main()
```

**Step 2: Write test for OLS 24-hour baseline**

```python
# tests/test_phase2_ols_24h.py

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db


def _seed_24h_data(db_path, n_days=120):
    """Seed 120 days × 24 hours of HRRR forecasts + NWS daily."""
    con = duckdb.connect(db_path)
    base = date(2024, 1, 1)
    for d in range(n_days):
        current = base + timedelta(days=d)
        actual_high = 50 + 30 * (d % 365) / 365  # Seasonal pattern
        for hour in range(24):
            model_run = datetime(current.year, current.month, current.day, hour)
            bias = 2.0 + 0.5 * (hour / 24)  # Slight hour-dependent bias
            for fxx in range(1, 19):
                valid_at = model_run + timedelta(hours=fxx)
                temp_f = actual_high + bias + (fxx * 0.1)
                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                    ['KNYC', model_run, valid_at, temp_f,
                     round((temp_f - 32) * 5/9, 2),
                     datetime.now(), fxx, fxx <= 3],
                )
        con.execute(
            "INSERT INTO nws_daily "
            "(station_id, obs_date, max_temp_f, ingested_at) "
            "VALUES ('KNYC', ?, ?, ?)",
            [current, actual_high, datetime.now()],
        )
    con.close()


def test_ols_24h_produces_results(tmp_path):
    """OLS on 24 run hours should produce results for all hours."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    _seed_24h_data(db_path)

    from services.backtester import Backtester, wf_regression_full
    bt = Backtester(db_path=db_path)
    result = bt.run(
        model_fn=wf_regression_full,
        start_date=date(2024, 4, 1),  # After 90-day warm-up
        end_date=date(2024, 4, 30),
        run_hours=list(range(24)),
    )
    assert len(result.results) > 0
    assert result.mean_brier < 1.0  # Sanity — not perfect but not broken
    # Should have results for multiple run hours
    hours_seen = {r.run_hour for r in result.results}
    assert len(hours_seen) >= 20  # Allow some hours to skip if no data
```

**Step 3: Run test, implement any necessary changes, run again**

Run: `PYTHONPATH=. pytest tests/test_phase2_ols_24h.py -v`

**Step 4: Run actual OLS ablation on real data**

Run: `PYTHONPATH=. python scripts/run_phase2_ablation.py --candidate ols`

Record the output — this is the new baseline Brier to beat.

**Step 5: Commit**

```bash
git add scripts/run_phase2_ablation.py tests/test_phase2_ols_24h.py
git commit -m "feat(phase2): OLS baseline ablation on 24 HRRR run hours"
```

---

### Task 3: EMOS Candidate (Candidate B)

**Files:**
- Create: `services/phase2_emos.py`
- Create: `tests/test_phase2_emos.py`

**Context:** EMOS (Ensemble Model Output Statistics) jointly estimates mean and variance. The key formula:

```
mu = a + b * fcst_high
sigma^2 = c + d * ensemble_spread^2
```

Where `ensemble_spread` is the std across available model forecasts (HRRR, GFS, ECMWF) for the same date. The `gold_multi_model_features` view provides `ensemble_spread` pre-computed.

**EMOS advantage:** Produces calibrated mean AND variance in one step — may solve the Phase 3 static-std problem without needing a separate uncertainty layer.

**Step 1: Write failing tests**

```python
# tests/test_phase2_emos.py

import numpy as np
import pytest
from services.phase2_emos import fit_emos, predict_emos


def test_fit_emos_returns_four_coefficients():
    """EMOS fit should return (a, b, c, d) coefficients."""
    np.random.seed(42)
    n = 200
    fcst = np.random.normal(70, 10, n)
    spread = np.abs(np.random.normal(2, 1, n))
    actual = fcst - 2.0 + np.random.normal(0, 1.5, n)  # 2°F warm bias

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
    actual = fcst - 3.0 + np.random.normal(0, 1.5, n)  # 3°F warm bias

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
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_phase2_emos.py -v`
Expected: FAIL — `services/phase2_emos` does not exist

**Step 3: Implement EMOS**

```python
# services/phase2_emos.py
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


def fit_emos(fcst, spread, actual):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> Tuple[float, float, float, float]
    """Fit EMOS coefficients (a, b, c, d) via CRPS minimization.

    Args:
        fcst: Forecast high temperatures (n,)
        spread: Ensemble spread values (n,)
        actual: Observed high temperatures (n,)

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
    residuals = actual - fcst
    x0 = [float(np.mean(residuals)), 1.0, float(np.var(residuals)), 0.1]

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
```

**Step 4: Write the walk-forward EMOS model function**

Add to `services/phase2_emos.py`:

```python
# Walk-forward model function for backtester integration
WALK_FORWARD_MIN_DAYS = 90


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

    con = provider._con
    row = con.execute(
        "SELECT ensemble_spread FROM gold_multi_model_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    ensemble_spread = row[0] if row and row[0] is not None else 2.0  # Default spread

    # Get training data: all prior dates
    training = con.execute("""
        SELECT g.fcst_high, m.ensemble_spread, n.max_temp_f
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
    spread_arr = np.array([r[1] if r[1] is not None else 2.0 for r in training])
    actual_arr = np.array([r[2] for r in training])

    # Fit EMOS
    coeffs = fit_emos(fcst_arr, spread_arr, actual_arr)
    center, std = predict_emos(coeffs, fcst_high, ensemble_spread)

    # Generate bracket probabilities (1°F brackets around center)
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
```

**Step 5: Run tests**

Run: `PYTHONPATH=. pytest tests/test_phase2_emos.py -v`
Expected: PASS

**Step 6: Run EMOS ablation**

Run: `PYTHONPATH=. python scripts/run_phase2_ablation.py --candidate emos`

**Step 7: Commit**

```bash
git add services/phase2_emos.py tests/test_phase2_emos.py
git commit -m "feat(phase2): add EMOS bias correction candidate"
```

---

### Task 4: XGBoost Quantile Regression (Candidate C)

**Files:**
- Create: `services/phase2_xgboost.py`
- Create: `tests/test_phase2_xgboost.py`

**Context:** XGBoost gradient-boosted regressor on all available features. Run in two modes: (1) standard regression for point estimate, (2) quantile regression with pinball loss for direct uncertainty estimation. The key risk is overfitting — mandatory 5-fold time-series CV.

**Extended features sub-ablation (Gemini #2 insight):** Extended weather features (dewpoint, humidity, wind, pressure, cloud cover, precipitation, radiation, CAPE) were killed under OLS in Phase 3 because they failed the 2% Brier gate. Gemini noted these features likely failed due to OLS's linear constraints, not lack of signal. Tree-based models can express conditional interactions (e.g., "wind only cools if off the ocean") that OLS cannot. Run XGBoost in two variants: (1) base features only (fcst_high, sin/cos month, delta_temp, spread), (2) base + extended features from `forecast_extended` table. If extended features improve Brier under XGBoost, they have signal — OLS just couldn't use it.

**Step 1: Write failing tests**

```python
# tests/test_phase2_xgboost.py

import numpy as np
import pytest

try:
    import xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

pytestmark = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")


def test_xgb_bias_fit_returns_model():
    """XGBoost fit should return a trained model."""
    from services.phase2_xgboost import fit_xgb_bias
    np.random.seed(42)
    n = 300
    X = np.column_stack([
        np.random.normal(70, 10, n),   # fcst_high
        np.sin(2 * np.pi * np.random.randint(1, 13, n) / 12),  # sin_month
        np.cos(2 * np.pi * np.random.randint(1, 13, n) / 12),  # cos_month
        np.random.normal(0, 2, n),     # delta_temp
    ])
    y = X[:, 0] - 2.0 + np.random.normal(0, 1.5, n)  # actual with bias

    model = fit_xgb_bias(X, y)
    assert model is not None
    preds = model.predict(xgboost.DMatrix(X))
    assert len(preds) == n


def test_xgb_bias_reduces_error():
    """XGBoost predictions should have lower RMSE than raw forecast."""
    from services.phase2_xgboost import fit_xgb_bias
    np.random.seed(42)
    n = 500
    fcst = np.random.normal(70, 10, n)
    actual = fcst - 3.0 + np.random.normal(0, 1.5, n)
    X = np.column_stack([
        fcst,
        np.sin(2 * np.pi * np.random.randint(1, 13, n) / 12),
        np.cos(2 * np.pi * np.random.randint(1, 13, n) / 12),
        np.random.normal(0, 2, n),
    ])

    model = fit_xgb_bias(X, actual)
    preds = model.predict(xgboost.DMatrix(X))

    rmse_raw = np.sqrt(np.mean((actual - fcst) ** 2))
    rmse_xgb = np.sqrt(np.mean((actual - preds) ** 2))
    # XGBoost should do at least as well as raw
    assert rmse_xgb <= rmse_raw + 0.5


def test_xgb_quantile_brackets():
    """XGBoost quantile regression should produce valid bracket probabilities."""
    from services.phase2_xgboost import fit_xgb_quantiles, predict_quantile_brackets
    np.random.seed(42)
    n = 500
    fcst = np.random.normal(70, 10, n)
    actual = fcst - 2.0 + np.random.normal(0, 2, n)
    X = np.column_stack([
        fcst,
        np.sin(2 * np.pi * np.random.randint(1, 13, n) / 12),
        np.cos(2 * np.pi * np.random.randint(1, 13, n) / 12),
        np.random.normal(0, 2, n),
    ])

    models = fit_xgb_quantiles(X, actual)
    assert len(models) == 7  # 7 quantiles

    # Predict brackets for a single point
    x_single = X[0:1]
    probs = predict_quantile_brackets(models, x_single)
    assert isinstance(probs, dict)
    assert abs(sum(probs.values()) - 1.0) < 0.01
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=. pytest tests/test_phase2_xgboost.py -v`

**Step 3: Implement XGBoost module**

```python
# services/phase2_xgboost.py
"""XGBoost bias correction and quantile regression for Phase 2.

Two modes:
1. Point estimate: standard regression for bias correction
2. Quantile regression: 7 quantiles for direct uncertainty estimation

Features:
    fcst_high, sin_month, cos_month, delta_temp, run_hour,
    ensemble_spread, fxx (optional), is_spinup (optional)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("xgboost not installed — XGBoost candidate disabled")

from scipy.stats import norm

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
WALK_FORWARD_MIN_DAYS = 90


def fit_xgb_bias(X, y, n_rounds=200):
    # type: (np.ndarray, np.ndarray, int) -> object
    """Fit XGBoost regression model for bias correction.

    Args:
        X: Feature matrix (n, p)
        y: Target values (actual high temps) (n,)
        n_rounds: Number of boosting rounds

    Returns:
        Trained xgb.Booster
    """
    if xgb is None:
        raise ImportError("xgboost required for XGBoost candidate")

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "verbosity": 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=n_rounds)
    return model


def fit_xgb_quantiles(X, y, n_rounds=200):
    # type: (np.ndarray, np.ndarray, int) -> List[object]
    """Fit XGBoost quantile regression models for each quantile.

    Returns list of 7 trained models (one per quantile).
    """
    if xgb is None:
        raise ImportError("xgboost required for XGBoost candidate")

    models = []
    for q in QUANTILES:
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": q,
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=n_rounds)
        models.append(model)

    return models


def predict_quantile_brackets(models, X_single):
    # type: (List[object], np.ndarray) -> Dict[int, float]
    """Predict bracket probabilities from quantile models.

    Uses piecewise-linear CDF between quantile points,
    exponential decay for tails beyond q05/q95.

    Args:
        models: List of 7 trained quantile models
        X_single: Feature vector for a single prediction (1, p)

    Returns:
        Dict[int, float] — 1°F bracket probabilities
    """
    if xgb is None:
        raise ImportError("xgboost required")

    dtest = xgb.DMatrix(X_single)
    quantile_values = [float(m.predict(dtest)[0]) for m in models]

    # Ensure monotonicity
    for i in range(1, len(quantile_values)):
        if quantile_values[i] < quantile_values[i - 1]:
            quantile_values[i] = quantile_values[i - 1] + 0.01

    q05, q10, q25, q50, q75, q90, q95 = quantile_values

    # Build piecewise-linear CDF
    quantile_points = list(zip(QUANTILES, quantile_values))

    def cdf(t):
        """Piecewise-linear CDF with exponential tails."""
        if t <= q05:
            # Lower tail: exponential decay
            lam = 1.0 / max(q50 - q05, 0.5)
            return 0.05 * math.exp(-lam * (q05 - t))
        elif t >= q95:
            # Upper tail: exponential decay
            lam = 1.0 / max(q95 - q50, 0.5)
            return 1.0 - 0.05 * math.exp(-lam * (t - q95))
        else:
            # Interior: piecewise linear
            for i in range(len(quantile_points) - 1):
                tau_lo, val_lo = quantile_points[i]
                tau_hi, val_hi = quantile_points[i + 1]
                if val_lo <= t <= val_hi:
                    if val_hi - val_lo < 0.01:
                        return (tau_lo + tau_hi) / 2
                    frac = (t - val_lo) / (val_hi - val_lo)
                    return tau_lo + frac * (tau_hi - tau_lo)
            return 0.5  # Fallback

    # Generate bracket probabilities
    center = int(round(q50))
    bracket_probs = {}
    for temp in range(center - 15, center + 16):
        p = cdf(temp + 0.5) - cdf(temp - 0.5)
        if p > 1e-6:
            bracket_probs[temp] = max(p, 0.0)

    # Normalize
    total = sum(bracket_probs.values())
    if total > 0:
        bracket_probs = {k: v / total for k, v in bracket_probs.items()}

    return bracket_probs
```

**Step 4: Add walk-forward model function** (append to `services/phase2_xgboost.py`):

```python
def xgboost_model_fn(provider, ref_time):
    # type: (object, object) -> Optional[Dict[int, float]]
    """XGBoost walk-forward model function for Backtester."""
    if xgb is None:
        return None

    from services.data_provider import BacktestDataProvider
    if not isinstance(provider, BacktestDataProvider):
        return None

    fcst_high = provider.get_forecast_high('KNYC')
    if fcst_high is None:
        return None

    model_run = provider.model_run
    forecast_date = model_run.date() if hasattr(model_run, 'date') else model_run
    run_hour = model_run.hour if hasattr(model_run, 'hour') else 0

    con = provider._con

    # Get training data from gold views
    training = con.execute("""
        SELECT g.fcst_high, g.sin_month, g.cos_month, g.delta_temp,
               COALESCE(m.ensemble_spread, 2.0) AS spread,
               n.max_temp_f
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

    X_train = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in training])
    y_train = np.array([r[5] for r in training])

    # Get current ensemble spread
    row = con.execute(
        "SELECT ensemble_spread FROM gold_multi_model_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    spread = row[0] if row and row[0] is not None else 2.0

    # Build feature vector for prediction
    month = forecast_date.month
    sin_m = math.sin(2 * math.pi * month / 12.0)
    cos_m = math.cos(2 * math.pi * month / 12.0)
    # delta_temp from gold view
    delta_row = con.execute(
        "SELECT delta_temp FROM gold_hrrr_bias_features "
        "WHERE forecast_date = ? AND run_hour = ?",
        [forecast_date, run_hour],
    ).fetchone()
    delta_temp = delta_row[0] if delta_row and delta_row[0] is not None else 0.0

    X_pred = np.array([[fcst_high, sin_m, cos_m, delta_temp, spread]])

    # Fit quantile models and predict
    models = fit_xgb_quantiles(X_train, y_train, n_rounds=150)
    bracket_probs = predict_quantile_brackets(models, X_pred)

    return bracket_probs
```

**Step 5: Run tests**

Run: `PYTHONPATH=. pytest tests/test_phase2_xgboost.py -v`

**Step 6: Commit**

```bash
git add services/phase2_xgboost.py tests/test_phase2_xgboost.py
git commit -m "feat(phase2): add XGBoost quantile regression candidate"
```

---

### Task 5: Spin-Up Ablation

**Files:**
- Modify: `services/backtester.py`
- Create: `tests/test_spinup_ablation.py`

**Context:** HRRR fxx 1-3 have spin-up artifacts. Test whether excluding `is_spinup = TRUE` rows from the forecast high calculation improves bias correction. This is a sub-test of the OLS candidate — same model, different feature filtering.

**Step 1: Write test**

```python
# tests/test_spinup_ablation.py

import duckdb
import pytest
from datetime import date, datetime, timedelta
from core.db import init_db


def test_spinup_exclusion_changes_fcst_high(tmp_path):
    """Excluding spin-up fxx should change the forecast high."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)

    model_run = datetime(2024, 6, 15, 12)
    # Insert spin-up hours with higher temps (simulating artifacts)
    for fxx in range(1, 19):
        temp = 80.0 if fxx <= 3 else 75.0  # Spin-up artificially warm
        con.execute(
            "INSERT INTO forecasts "
            "(station_id, model_run, valid_at, temp_f, temp_c, "
            "ingested_at, model_name, fxx, is_spinup) "
            "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
            ['KNYC', model_run, model_run + timedelta(hours=fxx),
             temp, round((temp - 32) * 5/9, 2),
             datetime.now(), fxx, fxx <= 3],
        )
    con.close()

    con = duckdb.connect(db_path, read_only=True)
    # With spin-up
    high_all = con.execute(
        "SELECT MAX(temp_f) FROM forecasts "
        "WHERE station_id = 'KNYC' AND model_run = ?",
        [model_run],
    ).fetchone()[0]

    # Without spin-up
    high_no_spinup = con.execute(
        "SELECT MAX(temp_f) FROM forecasts "
        "WHERE station_id = 'KNYC' AND model_run = ? AND is_spinup = FALSE",
        [model_run],
    ).fetchone()[0]

    con.close()

    assert high_all == 80.0  # Includes spin-up
    assert high_no_spinup == 75.0  # Excludes spin-up
    assert high_all != high_no_spinup
```

**Step 2: Implement spin-up filtered model function**

In `services/backtester.py`, add a variant of `wf_regression_full` that filters `is_spinup = FALSE` when computing `fcst_high`. The simplest approach: add a `gold_hrrr_bias_features_no_spinup` view to `core/db.py`, or pass a filter parameter to the existing feature query.

Recommended: Add the view to `core/db.py`:

```sql
CREATE VIEW IF NOT EXISTS gold_hrrr_bias_features_no_spinup AS
SELECT
    model_run::DATE AS forecast_date,
    EXTRACT(HOUR FROM model_run)::INTEGER AS run_hour,
    MAX(temp_f) AS fcst_high,
    SIN(2 * PI() * EXTRACT(MONTH FROM model_run::DATE) / 12.0) AS sin_month,
    COS(2 * PI() * EXTRACT(MONTH FROM model_run::DATE) / 12.0) AS cos_month,
    MAX(temp_f) - MIN(temp_f) AS delta_temp,
    COUNT(*) AS n_fxx,
    MIN(fxx) AS min_fxx,
    MAX(fxx) AS max_fxx,
    FALSE AS has_spinup
FROM forecasts
WHERE model_name = 'hrrr'
  AND station_id = 'KNYC'
  AND fxx IS NOT NULL
  AND is_spinup = FALSE
GROUP BY model_run::DATE, EXTRACT(HOUR FROM model_run)
```

Then create `wf_regression_full_no_spinup` that reads from this view instead.

**Step 3: Run ablation**

Run: `PYTHONPATH=. python scripts/run_phase2_ablation.py --candidate ols_no_spinup`

Compare Brier with `--candidate ols`. If Brier improves, adopt spin-up exclusion globally.

**Step 4: Commit**

```bash
git add core/db.py services/backtester.py tests/test_spinup_ablation.py
git commit -m "feat(phase2): spin-up ablation (exclude fxx <= 3)"
```

---

### Task 6: Head-to-Head Comparison

**Files:**
- Create: `scripts/phase2_compare.py`

**Context:** All three candidates have run. Now compare them head-to-head on the same evaluation set.

**Step 1: Write comparison script**

```python
# scripts/phase2_compare.py
"""Head-to-head comparison of Phase 2 bias correction candidates.

Runs all candidates on identical date ranges and compares:
- Mean Brier score
- Per-hour Brier breakdown
- Top-1 and Top-2 hit rates
- Calibration (ECE)

Usage:
    python scripts/phase2_compare.py
    python scripts/phase2_compare.py --start 2023-06-01 --end 2025-12-31
"""

import argparse
from datetime import date
from typing import Dict

from loguru import logger

from core.db import DEFAULT_DB_PATH
from services.backtester import Backtester


def compare_candidates(db_path=DEFAULT_DB_PATH, start=None, end=None):
    """Run all candidates and compare."""
    start = start or date(2023, 1, 1)
    end = end or date(2026, 2, 1)

    bt = Backtester(db_path=db_path)

    candidates = {}

    # Candidate A: OLS (all fxx)
    from services.backtester import wf_regression_full
    logger.info("Running Candidate A: OLS (all fxx)...")
    candidates["OLS"] = bt.run(wf_regression_full, start, end)

    # Candidate A': OLS (no spin-up)
    from services.backtester import wf_regression_full_no_spinup
    logger.info("Running Candidate A': OLS (no spin-up)...")
    candidates["OLS_no_spinup"] = bt.run(wf_regression_full_no_spinup, start, end)

    # Candidate B: EMOS
    try:
        from services.phase2_emos import emos_model_fn
        logger.info("Running Candidate B: EMOS...")
        candidates["EMOS"] = bt.run(emos_model_fn, start, end)
    except Exception as e:
        logger.error("EMOS failed: {}", e)

    # Candidate C: XGBoost QR
    try:
        from services.phase2_xgboost import xgboost_model_fn
        logger.info("Running Candidate C: XGBoost QR...")
        candidates["XGBoost_QR"] = bt.run(xgboost_model_fn, start, end)
    except Exception as e:
        logger.error("XGBoost failed: {}", e)

    # Print comparison table
    print("\n" + "=" * 70)
    print("PHASE 2 HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"{'Candidate':<20} {'Brier':>8} {'Top-1':>8} {'Top-2':>8} {'N':>6}")
    print("-" * 70)

    best_brier = 1.0
    best_name = ""
    for name, result in sorted(candidates.items(), key=lambda x: x[1].mean_brier):
        marker = ""
        if result.mean_brier < best_brier:
            best_brier = result.mean_brier
            best_name = name
        print(f"{name:<20} {result.mean_brier:>8.4f} "
              f"{result.top1_hit_rate:>7.1%} "
              f"{result.top2_hit_rate:>7.1%} "
              f"{len(result.results):>6}")

    print("-" * 70)
    print(f"CHAMPION: {best_name} (Brier {best_brier:.4f})")

    # Per-hour breakdown for champion
    champion = candidates[best_name]
    print(f"\n{best_name} per-hour Brier:")
    for hour, metrics in sorted(champion.per_hour.items()):
        print(f"  {hour:02d}z: {metrics['mean_brier']:.4f} (n={metrics['n']})")

    # Gate check
    baseline_brier = candidates.get("OLS", candidates[list(candidates.keys())[0]]).mean_brier
    improvement = (baseline_brier - best_brier) / baseline_brier * 100
    print(f"\nImprovement over OLS: {improvement:.1f}%")
    if improvement >= 2.0:
        print("PHASE 2 GATE: PASSED (>= 2% Brier improvement)")
    else:
        print("PHASE 2 GATE: PENDING (< 2% improvement — investigate)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    compare_candidates(args.db, args.start, args.end)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/phase2_compare.py
git commit -m "feat(phase2): head-to-head comparison script"
```

---

### Task 7: Multi-Model Ensemble Integration

**Files:**
- Modify: `services/ensemble.py` (if needed)
- Create: `tests/test_phase2_ensemble.py`

**Context:** After the single-model champion is chosen, test whether a multi-model ensemble (HRRR + GFS + ECMWF) improves Brier. The existing `services/ensemble.py` has `combine_mixture_brackets` — use it to combine the champion's predictions from each model.

**Step 1: Write test**

```python
# tests/test_phase2_ensemble.py

from services.ensemble import combine_mixture_brackets


def test_ensemble_combines_three_models():
    """Ensemble should combine HRRR, GFS, ECMWF predictions."""
    # Three models with slightly different centers
    predictions = [
        (72.0, 2.5),  # HRRR
        (73.0, 3.0),  # GFS
        (71.5, 2.8),  # ECMWF
    ]
    weights = [0.5, 0.25, 0.25]

    probs = combine_mixture_brackets(predictions, weights, radius=15)
    assert isinstance(probs, dict)
    assert abs(sum(probs.values()) - 1.0) < 0.01
    # Peak should be near weighted center (~72.1)
    peak_temp = max(probs, key=probs.get)
    assert 71 <= peak_temp <= 73
```

**Step 2: Run ensemble ablation**

Using the existing ensemble infrastructure, test the champion bias model applied to each NWP model, then combined:

```python
# In scripts/run_phase2_ablation.py or scripts/phase2_compare.py
# Add ensemble variant that runs champion on HRRR, GFS, ECMWF separately
# then combines via Gaussian mixture
```

**Step 3: Commit**

```bash
git add tests/test_phase2_ensemble.py
git commit -m "feat(phase2): multi-model ensemble ablation"
```

---

### Task 8: Strategy Backtester P&L Diagnostic

**Files:**
- Create: `scripts/phase2_pnl_diagnostic.py`

**Context:** Run the strategy backtester against the Phase 2 champion. P&L is NOT a kill signal here — the static std problem from Phase 3 still exists. We're tracking trajectory: "Is P&L trending up compared to the -44.7% baseline?"

**Step 1: Write P&L diagnostic script**

```python
# scripts/phase2_pnl_diagnostic.py
"""Phase 2 P&L diagnostic — tracks trajectory, not a kill signal.

Usage:
    python scripts/phase2_pnl_diagnostic.py
"""

from datetime import date
from loguru import logger
from core.db import DEFAULT_DB_PATH
from services.strategy_backtester import (
    BacktestConfig,
    run_strategy_backtest,
)


def run_pnl_diagnostic(db_path=DEFAULT_DB_PATH):
    """Run strategy backtest with Phase 2 champion."""
    config = BacktestConfig(
        starting_capital=100.0,
        burn_in_days=90,
        fixed_bet_size=1,
        min_displacement=0.12,
    )

    # Import the Phase 2 champion model function
    # (Update this import after Task 6 identifies the winner)
    from services.backtester import wf_regression_full as champion_fn

    logger.info("Running Phase 2 P&L diagnostic...")
    result = run_strategy_backtest(
        model_fn=champion_fn,
        config=config,
        start_date=date(2023, 6, 1),
        end_date=date(2026, 2, 1),
        db_path=db_path,
    )

    print("\n" + "=" * 50)
    print("PHASE 2 P&L DIAGNOSTIC")
    print("=" * 50)
    print(f"Total P&L:    ${result.total_pnl:.2f}")
    print(f"Win Rate:     {result.win_rate:.1%}")
    print(f"Trades:       {result.n_trades}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print(f"Final Capital: ${result.final_capital:.2f}")
    print(f"Return:       {(result.final_capital / 100 - 1) * 100:.1f}%")
    print()
    print("Baseline: -44.7% return, 4.4% win rate")
    print(f"Trajectory: {'IMPROVING' if result.total_pnl > -44.70 else 'WORSE'}")
    print("=" * 50)
    print()
    print("NOTE: P&L is diagnostic only at Phase 2.")
    print("Static std problem persists until Phase 3.")


if __name__ == "__main__":
    run_pnl_diagnostic()
```

**Step 2: Commit**

```bash
git add scripts/phase2_pnl_diagnostic.py
git commit -m "feat(phase2): P&L diagnostic script (trajectory tracking)"
```

---

### Task 9: Phase 2 Gate Evaluation

**Files:**
- Create: `scripts/phase2_gate_check.py`

**Context:** Final gate evaluation. Criteria:
- Primary: Brier > 2% improvement over OLS baseline
- Diagnostic: P&L trajectory (is it trending up?)
- Kill condition: If no candidate beats OLS by 2%, OLS remains champion

**Step 1: Write gate check script**

```python
# scripts/phase2_gate_check.py
"""Phase 2 gate evaluation.

Checks:
1. Champion Brier vs OLS baseline — must improve >= 2%
2. P&L trajectory — diagnostic only (not a kill signal)
3. Per-hour consistency — champion should not regress on any hour

Exit code 0 = PASSED, exit code 1 = FAILED.
"""

import sys
from datetime import date
from loguru import logger
from core.db import DEFAULT_DB_PATH


def check_phase2_gate(db_path=DEFAULT_DB_PATH):
    """Run Phase 2 gate checks."""
    all_pass = True

    # Check 1: Run head-to-head comparison
    print("=== Phase 2 Gate Check ===")
    print()

    from scripts.phase2_compare import compare_candidates
    # This prints the comparison and identifies the champion

    # Check 2: Verify >= 2% Brier improvement
    # (Implemented in phase2_compare.py)

    # Check 3: P&L trajectory
    print()
    from scripts.phase2_pnl_diagnostic import run_pnl_diagnostic
    run_pnl_diagnostic(db_path)

    return all_pass


if __name__ == "__main__":
    passed = check_phase2_gate()
    sys.exit(0 if passed else 1)
```

**Step 2: Commit**

```bash
git add scripts/phase2_gate_check.py
git commit -m "feat(phase2): gate evaluation script"
```

---

## Post-Gate: What Happens Next

If Phase 2 gate passes:
1. Update `CLAUDE.md` with the champion model name
2. Record decision in `memory/decisions.md`
3. Proceed to Phase 3 (Uncertainty and Calibration) — the static std problem

If Phase 2 gate fails (no candidate beats OLS by 2%):
1. OLS remains champion — it's still the best we have
2. Investigate whether the 24-run expansion itself caused regression
3. Consider: is the data quality from backfill sufficient?
4. Proceed to Phase 3 anyway with OLS — the Brier gate is secondary to the dynamic uncertainty problem

## Dependencies to Install

Before running any Phase 2 tasks:

```bash
pip install xgboost  # For Candidate C
# scipy already installed (for OLS/EMOS)
# numpy already installed
```
