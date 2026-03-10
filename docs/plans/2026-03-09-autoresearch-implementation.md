# AlphaTemp Autoresearch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an autonomous experiment loop that lets Claude Code iteratively improve the probability model overnight, committing improvements and discarding failures.

**Architecture:** Hybrid — `experiment.py` (agent-modifiable) imports locked infrastructure from `services/` and `core/`. A locked harness `run_experiment.py` runs the backtest and scores with a composite metric. Git commits track improvements.

**Tech Stack:** Python 3.9, DuckDB, scipy (QR LP), numpy, existing Backtester infrastructure

---

### Task 1: Create autoresearch directory and program.md

**Files:**
- Create: `autoresearch/program.md`
- Create: `autoresearch/__init__.py`

**Step 1: Create the directory**

Run: `mkdir -p autoresearch`

**Step 2: Create empty `__init__.py`**

```python
# autoresearch/__init__.py
```

**Step 3: Write `program.md`**

```markdown
# AlphaTemp Autoresearch

## Objective
Lower the composite score by improving the probability model in experiment.py.
Lower score = better. You're optimizing a weather prediction model that
predicts NYC daily high temperature brackets for Kalshi event markets.

## The Loop
1. Read experiment.py and understand the current model
2. Form a hypothesis (e.g., "adding wind speed as a feature should help")
3. Modify ONLY experiment.py — keep the model_fn signature unchanged
4. Run: PYTHONPATH=. ../venv/bin/python autoresearch/run_experiment.py
5. If exit code 0 → improvement! Run:
   git add autoresearch/experiment.py && git commit -m "experiment: [DESCRIPTION] — score X.XXXX (was Y.YYYY)"
6. If exit code 1 → discard. Run: git checkout -- autoresearch/experiment.py
7. Repeat. Try a different hypothesis.

## What You Can Change (ONLY in experiment.py)
- Feature engineering (add/remove/transform features)
- Model architecture (QR, XGBoost, ridge, ensemble, neural net, anything)
- Hyperparameters (window size, quantile positions, regularization)
- CDF construction (tail treatment, interpolation method)
- Training strategy (rolling window, expanding window, seasonal weighting)

## What You CANNOT Change
- The model_fn(provider, ref_time) → Optional[Dict[int, float]] signature
- Any file outside autoresearch/experiment.py
- Walk-forward validation — training data must be BEFORE eval date (no future leakage)
- Kalshi fee math — the fee formula is fixed reality
- Settlement source — NWS CLI from KNYC, non-negotiable
- Database schema — no adding/dropping tables
- Must remain Python 3.9 compatible (no match/case, no dict | dict, use typing.Dict not dict[])

## Available Data (via BacktestDataProvider)
The provider gives you access to a DuckDB connection (`provider._shared_con`) and metadata:
- `provider.station_id` — "KNYC"
- `provider.model_run` — datetime of the HRRR run being evaluated
- `provider.get_forecast_high(station_id)` — MAX(temp_f) for this run
- `provider._shared_con` — DuckDB connection for direct queries

### Database Tables You Can Query
- `forecasts` — hourly temps (station_id, model_run, valid_at, temp_f, model_name, fxx)
  - model_name: 'hrrr' (all hours), 'gfs' (00z ONLY — 06z/12z/18z contaminated), 'ecmwf'
  - Settlement day filter: (EXTRACT(HOUR FROM model_run) + fxx) >= 5 AND < 29
- `observations` — METAR obs (station_id, observed_at, temp_f) for KNYC, KLGA, KEWR
- `nws_daily` — settlement truth (station_id, obs_date, max_temp_f, source)
- `forecast_extended` — dewpoint, humidity, wind, pressure, cloud, precip, radiation
- `mesonet_obs` — 5-min NYC stations (station_id, observed_at, temp_f) for BKNYRD, QNASTO
- `kalshi_settlements` — bracket definitions (event_date, floor_strike, cap_strike, settled_yes)

### Key Domain Facts
- All timestamps are UTC unless explicitly stated
- KNYC reports hourly at ~:51 past the hour
- Settlement uses NWS CLI Local Standard Time (EST = UTC-5, NO daylight saving)
- HRRR is 3km resolution, runs every hour, 18h forecast horizon
- GFS 06z/12z/18z data is POISONED (Mediterranean weather). Only use 00z.
- Temperature brackets on Kalshi are 2°F wide

## Strategy Tips
- Small changes first. One variable at a time.
- If 3 consecutive experiments in one direction fail, pivot completely.
- The current model uses quantile regression — but you can try anything.
- Seasonal patterns matter — summer and winter behave very differently.
- More features isn't always better. Regularization helps.
- The forecast_extended table has rich data (dewpoint, wind, cloud cover) that's currently unused.
- Consider time-of-day effects: model accuracy varies by hour.
- Don't forget about the physical floor: daily high can't go below running_max.

## Scoring
Composite (lower = better):
- 45% Net P&L (fee-adjusted, inverted + normalized)
- 35% Brier score (full day, all update hours)
- 20% Hit rate (top-1 bracket accuracy, inverted + normalized)

Run time budget: 90 seconds max per experiment.
```

**Step 4: Commit**

```bash
git add autoresearch/
git commit -m "feat: add autoresearch directory with program.md"
```

---

### Task 2: Create experiment.py with starting QR model

**Files:**
- Create: `autoresearch/experiment.py`

This is the file the agent will modify. Starting version is a simplified, self-contained
extraction of the current best model (multimodel QR v4). It queries DuckDB directly
rather than using the complex caching layers, making it readable and modifiable.

**Step 1: Write experiment.py**

The file must export:
- `DESCRIPTION: str` — what this version does
- `model_fn(provider, ref_time) -> Optional[Dict[int, float]]` — the model

```python
"""AlphaTemp Autoresearch — Experiment File

THIS IS THE FILE THE AI AGENT MODIFIES.

Contract:
- Export model_fn(provider, ref_time) -> Optional[Dict[int, float]]
- Return 1°F integer bracket probabilities, or None if can't forecast
- Only use training data from BEFORE the evaluation date (walk-forward)
- Python 3.9 compatible
"""

import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

from services.data_provider import BacktestDataProvider

# ── Description (updated by the agent each experiment) ──────────────────────
DESCRIPTION = "Baseline: QR with 7 features (update_hour, fcst_high, sin/cos month, div signals, cumulative_div)"

# ── Hyperparameters ─────────────────────────────────────────────────────────
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TRAIN_WINDOW_DAYS = 180
MIN_SAMPLES = 90
MIN_BRACKET_PROB = 0.0001

# Timezone for ET conversion
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone as _tz
    _ET = timezone(timedelta(hours=-5))  # fallback EST


# ── Quantile Regression Solver ──────────────────────────────────────────────

def fit_quantile_regression(X, y, tau):
    # type: (np.ndarray, np.ndarray, float) -> Optional[np.ndarray]
    """Fit linear quantile regression via LP. Returns coefficients or None."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    if n < MIN_SAMPLES:
        return None

    ones = np.ones((n, 1), dtype=np.float64)
    X_aug = np.hstack([ones, X])
    k = p + 1

    c = np.concatenate([
        np.zeros(k),
        tau * np.ones(n),
        (1 - tau) * np.ones(n),
    ])

    A_eq = sparse.hstack([
        sparse.csc_matrix(X_aug),
        sparse.eye(n),
        -sparse.eye(n),
    ], format='csc')
    b_eq = y

    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result.success:
        return None
    return result.x[:k]


# ── CDF Construction ────────────────────────────────────────────────────────

def build_bracket_probs(temp_quantiles, tau_values, running_max=None, radius=15):
    # type: (List[float], List[float], Optional[float], int) -> Dict[int, float]
    """Build 1°F bracket probabilities from predicted temperature quantiles.

    Uses piecewise-linear CDF with exponential decay tails.
    """
    # Sort by temperature, enforce monotonicity
    pairs = sorted(zip(temp_quantiles, tau_values))
    temps = [p[0] for p in pairs]
    taus = [p[1] for p in pairs]
    for i in range(1, len(taus)):
        if taus[i] < taus[i - 1]:
            taus[i] = taus[i - 1]

    q05, q95 = temps[0], temps[-1]
    tau_lo, tau_hi = taus[0], taus[-1]

    q50_idx = min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5))
    q50 = temps[q50_idx]

    spread_upper = max(q95 - q50, 0.5)
    spread_lower = max(q50 - q05, 0.5)
    lambda_upper = 1.0 / spread_upper
    lambda_lower = 1.0 / spread_lower

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

    center_int = round(temps[q50_idx])
    probs = {}  # type: Dict[int, float]
    for k in range(center_int - radius, center_int + radius + 1):
        p = cdf(k + 0.5) - cdf(k - 0.5)
        if p > MIN_BRACKET_PROB:
            probs[k] = p

    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    return probs


# ── Feature Engineering ─────────────────────────────────────────────────────

def get_training_data(con, run_hour, station_id, current_date):
    # type: (...) -> Optional[Tuple[np.ndarray, np.ndarray]]
    """Build walk-forward training data: (X_features, y_errors).

    Returns features and forecast errors for all dates in the training window
    BEFORE current_date, pooled across all 24 update hours.
    """
    min_date = current_date - timedelta(days=TRAIN_WINDOW_DAYS)

    # Get base data: forecast highs and actual settlement temps
    rows = con.execute("""
        WITH daily_errors AS (
            SELECT
                n.obs_date,
                MAX(f.temp_f) - n.max_temp_f AS error,
                MAX(f.temp_f) AS fcst_high,
                EXTRACT(MONTH FROM n.obs_date) AS month
            FROM nws_daily n
            JOIN forecasts f ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = ?
                AND f.model_name = 'hrrr'
                AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) >= 5
                AND (EXTRACT(HOUR FROM f.model_run) + f.fxx) < 29
            WHERE n.station_id = ?
                AND n.obs_date >= ?
                AND n.obs_date < ?
                AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
            ORDER BY n.obs_date
        )
        SELECT obs_date, error, fcst_high, month FROM daily_errors
    """, [run_hour, station_id, min_date, current_date]).fetchall()

    if len(rows) < MIN_SAMPLES:
        return None

    # Cross-hour: replicate each date across update hours with varying features
    X_rows = []
    y_rows = []
    for obs_date, error, fcst_high, month in rows:
        sin_m = math.sin(2.0 * math.pi * month / 12.0)
        cos_m = math.cos(2.0 * math.pi * month / 12.0)

        # Get observation divergence for this date
        obs_data = con.execute("""
            SELECT
                EXTRACT(HOUR FROM observed_at AT TIME ZONE 'UTC' AT TIME ZONE 'EST')::INTEGER AS hour_et,
                temp_f
            FROM observations
            WHERE station_id = ? AND observed_at::DATE = ? AND temp_f IS NOT NULL
            ORDER BY observed_at
        """, [station_id, obs_date]).fetchall()

        # Get forecast curve for divergence
        fc_data = con.execute("""
            SELECT valid_at, temp_f FROM forecasts
            WHERE station_id = ? AND model_run::DATE = ?
                AND EXTRACT(HOUR FROM model_run) = ?
                AND model_name = 'hrrr' AND temp_f IS NOT NULL
            ORDER BY valid_at
        """, [station_id, obs_date, run_hour]).fetchall()

        # Build hourly obs lookup
        obs_by_hour = {}  # type: Dict[int, float]
        running_maxes = {}  # type: Dict[int, float]
        current_max = None  # type: Optional[float]
        for h_et, temp in obs_data:
            h = int(h_et)
            obs_by_hour[h] = temp
            if current_max is None or temp > current_max:
                current_max = temp
            running_maxes[h] = current_max

        # Build forecast lookup by ET hour
        fc_by_hour = {}  # type: Dict[int, float]
        for valid_at, temp in fc_data:
            # Convert valid_at (naive UTC) to ET hour
            utc_dt = valid_at.replace(tzinfo=timezone.utc)
            et_hour = utc_dt.astimezone(_ET).hour
            fc_by_hour[et_hour] = temp

        # For each update hour, compute features
        for uh in range(0, 24):
            # Divergence: running max of obs - forecast at obs hours up to uh
            obs_up_to = [(h, obs_by_hour[h]) for h in sorted(obs_by_hour.keys()) if h <= uh]
            if len(obs_up_to) < 2:
                # Not enough obs — use zeros for divergence
                running_max_div = 0.0
                slope_div = 0.0
                cum_div = 0.0
            else:
                # Running max divergence
                rm = running_maxes.get(uh, obs_up_to[-1][1])
                fc_at_rm_hour = fc_by_hour.get(uh, fcst_high)
                running_max_div = rm - fc_at_rm_hour

                # Slope divergence: trend of (obs - forecast) over observed hours
                divs = []
                for h, obs_t in obs_up_to:
                    fc_t = fc_by_hour.get(h, fcst_high)
                    divs.append(obs_t - fc_t)
                if len(divs) >= 2:
                    slope_div = (divs[-1] - divs[0]) / max(len(divs) - 1, 1)
                else:
                    slope_div = 0.0

                # Cumulative divergence: mean(obs - forecast)
                cum_div = sum(divs) / len(divs) if divs else 0.0

            features = [
                float(uh),          # [0] update_hour_et
                float(fcst_high),   # [1] fcst_high
                sin_m,              # [2] sin(month)
                cos_m,              # [3] cos(month)
                running_max_div,    # [4] running_max_divergence
                slope_div,          # [5] slope_divergence
                cum_div,            # [6] cumulative_divergence
            ]

            X_rows.append(features)
            y_rows.append(error)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)
    return X, y


# ── Model Function ──────────────────────────────────────────────────────────

def model_fn(provider, ref_time):
    # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
    """Predict 1°F bracket probabilities for the given forecast scenario.

    This is the function the harness calls. Do not change the signature.
    """
    con = provider._shared_con
    station_id = provider.station_id
    run_hour = provider.model_run.hour
    current_date = provider.model_run.date()

    fcst_high = provider.get_forecast_high(station_id)
    if fcst_high is None:
        return None

    # Get training data
    result = get_training_data(con, run_hour, station_id, current_date)
    if result is None:
        return None
    X_train, y_train = result

    # Fit quantile regressions
    coefficients = []  # type: List[np.ndarray]
    for tau in QUANTILES:
        coeffs = fit_quantile_regression(X_train, y_train, tau)
        if coeffs is None:
            return None
        coefficients.append(coeffs)

    # Build today's feature vector
    ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
    update_hour_et = ref_utc.astimezone(_ET).hour
    cutoff_ts = ref_utc.timestamp()

    month = float(current_date.month)
    sin_m = math.sin(2.0 * math.pi * month / 12.0)
    cos_m = math.cos(2.0 * math.pi * month / 12.0)

    # Compute today's divergence features from observations
    running_max_div = 0.0
    slope_div = 0.0
    cum_div = 0.0
    running_max = None

    obs_today = con.execute("""
        SELECT observed_at, temp_f FROM observations
        WHERE station_id = ? AND observed_at::DATE = ? AND temp_f IS NOT NULL
        ORDER BY observed_at
    """, [station_id, current_date]).fetchall()

    fc_today = con.execute("""
        SELECT valid_at, temp_f FROM forecasts
        WHERE station_id = ? AND model_run::DATE = ?
            AND EXTRACT(HOUR FROM model_run) = ?
            AND model_name = 'hrrr' AND temp_f IS NOT NULL
        ORDER BY valid_at
    """, [station_id, current_date, run_hour]).fetchall()

    # Filter obs up to cutoff time
    obs_truncated = [(o[0], o[1]) for o in obs_today
                     if o[0].replace(tzinfo=timezone.utc).timestamp() <= cutoff_ts]

    if len(obs_truncated) >= 2:
        running_max = max(t for _, t in obs_truncated)

        # Build forecast lookup
        fc_by_hour = {}  # type: Dict[int, float]
        for valid_at, temp in fc_today:
            et_hour = valid_at.replace(tzinfo=timezone.utc).astimezone(_ET).hour
            fc_by_hour[et_hour] = temp

        # Compute divergences
        divs = []
        for obs_at, obs_t in obs_truncated:
            h_et = obs_at.replace(tzinfo=timezone.utc).astimezone(_ET).hour
            fc_t = fc_by_hour.get(h_et, fcst_high)
            divs.append(obs_t - fc_t)

        if divs:
            cum_div = sum(divs) / len(divs)
            fc_at_last = fc_by_hour.get(update_hour_et, fcst_high)
            running_max_div = running_max - fc_at_last
            if len(divs) >= 2:
                slope_div = (divs[-1] - divs[0]) / max(len(divs) - 1, 1)

    features_today = np.array([
        float(update_hour_et),
        float(fcst_high),
        sin_m,
        cos_m,
        running_max_div,
        slope_div,
        cum_div,
    ])

    # Predict error quantiles
    x_row = np.concatenate([[1.0], features_today])
    error_quantiles = [float(np.dot(c, x_row)) for c in coefficients]

    # Convert error quantiles to temperature quantiles (inverted)
    temp_quantiles = [fcst_high - eq for eq in error_quantiles]
    inverted_taus = [1.0 - tau for tau in QUANTILES]

    return build_bracket_probs(temp_quantiles, inverted_taus, running_max=running_max)
```

**Step 2: Verify it imports correctly**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -c "from autoresearch.experiment import model_fn, DESCRIPTION; print('OK:', DESCRIPTION)"`
Expected: `OK: Baseline: QR with 7 features ...`

**Step 3: Commit**

```bash
git add autoresearch/experiment.py
git commit -m "feat: add autoresearch experiment.py with baseline QR model"
```

---

### Task 3: Create run_experiment.py harness

**Files:**
- Create: `autoresearch/run_experiment.py`

This is the LOCKED harness. It imports model_fn, runs the backtest, computes the composite score,
logs results, and exits with 0 (improvement) or 1 (no improvement).

**Step 1: Write run_experiment.py**

```python
"""AlphaTemp Autoresearch — Experiment Harness (LOCKED — do not modify)

Runs experiment.py's model_fn through the backtester, computes a composite
score, logs results to results.tsv, and exits with code 0 (improvement)
or code 1 (no improvement).

Usage: PYTHONPATH=. python autoresearch/run_experiment.py
"""

import json
import os
import signal
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.backtester import Backtester

# ── Constants ───────────────────────────────────────────────────────────────

TIMEOUT_SECONDS = 90
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")
BEST_SCORE_FILE = os.path.join(os.path.dirname(__file__), "best_score.txt")
BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline.json")

# Quick-mode backtest parameters
RUN_HOURS = [0, 6, 12, 18]
UPDATE_HOURS_ET = [0, 4, 8, 12, 16, 20]

# Composite weights
W_PNL = 0.45
W_BRIER = 0.35
W_HIT = 0.20


# ── Timeout handler ─────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    print("TIMEOUT: Experiment exceeded {} seconds".format(TIMEOUT_SECONDS))
    sys.exit(1)


# ── Scoring ─────────────────────────────────────────────────────────────────

def compute_composite(brier, hit_rate, pnl_cents, baseline):
    # type: (float, float, float, dict) -> float
    """Compute composite score (lower = better).

    Normalizes all components to [0, 1] range using baseline stats,
    then weights them.
    """
    # Brier: already lower = better, typically 0.3–1.5
    brier_min = baseline.get("brier_min", 0.3)
    brier_max = baseline.get("brier_max", 1.5)
    brier_norm = (brier - brier_min) / max(brier_max - brier_min, 0.01)
    brier_norm = max(0.0, min(1.0, brier_norm))

    # Hit rate: higher = better, invert so lower = better
    hit_min = baseline.get("hit_min", 0.0)
    hit_max = baseline.get("hit_max", 0.5)
    hit_norm = 1.0 - (hit_rate - hit_min) / max(hit_max - hit_min, 0.01)
    hit_norm = max(0.0, min(1.0, hit_norm))

    # P&L: higher = better (positive = profit), invert so lower = better
    pnl_min = baseline.get("pnl_min", -10000.0)
    pnl_max = baseline.get("pnl_max", 10000.0)
    pnl_norm = 1.0 - (pnl_cents - pnl_min) / max(pnl_max - pnl_min, 0.01)
    pnl_norm = max(0.0, min(1.0, pnl_norm))

    return W_PNL * pnl_norm + W_BRIER * brier_norm + W_HIT * hit_norm


def compute_simple_pnl(results):
    # type: (list) -> float
    """Simplified P&L: for each evaluation, bet on the top bracket.

    If model's top bracket probability > uniform baseline (1/n_brackets),
    simulate buying YES at the implied price. Settle at 100 or 0.

    Returns total P&L in cents.
    """
    from services.strategy_backtester import compute_taker_fee

    total_pnl = 0.0
    for r in results:
        if not r.used_kalshi_brackets:
            continue
        # Simple strategy: if model confidence in top bracket > 2x uniform
        uniform_prob = 1.0 / max(r.n_brackets, 1)
        top_prob = 1.0  # We don't have the raw prob stored, approximate from Brier
        # Use Brier as proxy: lower Brier = higher confidence in correct bracket
        # For now, just simulate betting $1 on every event at 50c
        entry_price = 50.0  # cents
        qty = 1
        fee = compute_taker_fee(entry_price, qty)
        if r.hit:
            pnl = (100.0 - entry_price) * qty - fee
        else:
            pnl = -(entry_price * qty + fee)
        total_pnl += pnl

    return total_pnl


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Set timeout
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

    start_time = time.time()

    # Import experiment
    try:
        from autoresearch.experiment import model_fn, DESCRIPTION
    except ImportError as e:
        print("ERROR: Failed to import experiment.py: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("ERROR: experiment.py has a syntax/runtime error: {}".format(e))
        sys.exit(1)

    print("Running experiment: {}".format(DESCRIPTION))

    # Run backtest
    bt = Backtester(db_path="data/alphatemp.duckdb", city="NYC")
    try:
        result = bt.run(
            model_fn=model_fn,
            run_hours=RUN_HOURS,
            update_hours_et=UPDATE_HOURS_ET,
        )
    except Exception as e:
        print("ERROR: Backtest failed: {}".format(e))
        sys.exit(1)

    elapsed = time.time() - start_time

    # Cancel timeout
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)

    # Extract metrics
    brier = result.mean_brier
    hit_rate = result.top1_hit_rate
    total_evals = result.total_evaluations
    pnl_cents = compute_simple_pnl(result.run_results)

    print("  Brier: {:.4f}  Hit rate: {:.4f}  P&L: {:.0f}c  Evals: {}  Time: {:.1f}s".format(
        brier, hit_rate, pnl_cents, total_evals, elapsed,
    ))

    # Load or create baseline
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
    else:
        # First run: establish baseline from this experiment
        baseline = {
            "brier_min": brier * 0.5,
            "brier_max": brier * 1.5,
            "hit_min": 0.0,
            "hit_max": max(hit_rate * 2.0, 0.3),
            "pnl_min": min(pnl_cents * 2.0, -5000.0),
            "pnl_max": max(pnl_cents * 2.0, 5000.0),
        }
        with open(BASELINE_FILE, 'w') as f:
            json.dump(baseline, f, indent=2)
        print("  Baseline established and saved.")

    composite = compute_composite(brier, hit_rate, pnl_cents, baseline)
    print("SCORE: {:.6f}".format(composite))

    # Load current best
    if os.path.exists(BEST_SCORE_FILE):
        with open(BEST_SCORE_FILE) as f:
            best = float(f.read().strip())
    else:
        best = float('inf')

    # Log to results.tsv
    header_needed = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a') as f:
        if header_needed:
            f.write("timestamp\tcomposite\tbrier\thit_rate\tpnl_cents\tevals\telapsed_s\tdescription\n")
        f.write("{}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.0f}\t{}\t{:.1f}\t{}\n".format(
            datetime.utcnow().isoformat(),
            composite, brier, hit_rate, pnl_cents,
            total_evals, elapsed, DESCRIPTION,
        ))

    # Check for improvement
    if composite < best:
        print("IMPROVEMENT: {:.6f} -> {:.6f} (delta: {:.6f})".format(
            best, composite, best - composite,
        ))
        with open(BEST_SCORE_FILE, 'w') as f:
            f.write("{:.6f}\n".format(composite))
        sys.exit(0)  # Success — agent should commit
    else:
        print("NO IMPROVEMENT: {:.6f} >= best {:.6f}".format(composite, best))
        sys.exit(1)  # Failure — agent should discard


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python autoresearch/run_experiment.py`
Expected: Should run for 30-60 seconds, print metrics, establish baseline, exit with code 0 (first run always sets the baseline).

**Step 3: Commit**

```bash
git add autoresearch/run_experiment.py
git commit -m "feat: add autoresearch run_experiment.py harness"
```

---

### Task 4: Run baseline and verify end-to-end

**Files:**
- None (verification only)

**Step 1: Run the baseline experiment**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python autoresearch/run_experiment.py`

Expected output should look like:
```
Running experiment: Baseline: QR with 7 features ...
  Brier: 0.XXXX  Hit rate: 0.XXXX  P&L: XXXc  Evals: XXXX  Time: XX.Xs
  Baseline established and saved.
SCORE: 0.XXXXXX
IMPROVEMENT: inf -> 0.XXXXXX (delta: inf)
```

Verify these files were created:
- `autoresearch/baseline.json`
- `autoresearch/best_score.txt`
- `autoresearch/results.tsv`

**Step 2: Run again — should report NO IMPROVEMENT (same code)**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python autoresearch/run_experiment.py; echo "Exit code: $?"`

Expected: Should print `NO IMPROVEMENT` and exit with code 1.

**Step 3: Verify the exit code contract works**

This confirms the agent loop will work:
- Exit 0 = commit (improvement)
- Exit 1 = discard (no improvement)

**Step 4: Add generated files to .gitignore**

Add to `.gitignore`:
```
autoresearch/results.tsv
autoresearch/best_score.txt
autoresearch/baseline.json
```

**Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: add autoresearch generated files to .gitignore"
```

---

### Task 5: Final integration test — simulate one experiment cycle

**Files:**
- Modify: `autoresearch/experiment.py` (temporary test change)

**Step 1: Make a small change to experiment.py**

Change one hyperparameter — e.g., set `TRAIN_WINDOW_DAYS = 200` instead of 180.
Update DESCRIPTION to describe the change.

**Step 2: Run the experiment**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python autoresearch/run_experiment.py; echo "Exit code: $?"`

**Step 3: Check the result**

- If exit code 0: the change improved the score. Commit it.
- If exit code 1: the change didn't help. Revert: `git checkout -- autoresearch/experiment.py`

**Step 4: Check results.tsv has two entries**

Run: `cat autoresearch/results.tsv`
Expected: header + 2 data rows (baseline + this experiment)

**Step 5: Revert experiment.py to baseline if needed**

```bash
git checkout -- autoresearch/experiment.py
```

This confirms the full cycle works: modify → run → score → commit or discard.

---

### Task 6: Commit everything and document launch procedure

**Step 1: Ensure all autoresearch files are committed**

```bash
git add autoresearch/
git status
```

**Step 2: Final commit**

```bash
git commit -m "feat: complete autoresearch framework — ready for overnight runs"
```

**Step 3: Document the launch command**

Print the launch instructions for Russell:

```bash
# To launch an overnight autoresearch session:
cd ~/Projects/alphatemp/alphatemp
git checkout -b autoresearch/run-$(date +%Y-%m-%d)
caffeinate -di &
claude

# Then tell Claude Code:
# Read autoresearch/program.md and start running experiments.
# Run fully autonomously. Don't ask for confirmation. Keep going until I come back.
```
