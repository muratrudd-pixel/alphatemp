"""Backtester — replay model functions against historical data and score.

Iterates settlement dates × HRRR run hours, feeds historical data through
any model function via BacktestDataProvider, scores against NWS settlement
using Kalshi's standard 2°F bracket structure.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import duckdb
import numpy as np
from loguru import logger
from scipy.linalg import lstsq
from scipy.stats import norm, t as student_t

from core.constants import CITIES
from core.db import get_connection
from services.data_provider import (
    BacktestDataProvider,
    HRRR_AVAILABILITY_LAG_HOURS,
    _strip_tz,
)


# ---------------------------------------------------------------------------
# Type alias for pluggable model functions.
# (BacktestDataProvider, ref_time) -> Optional[Dict[int, float]]
# Returns 1°F integer bracket probabilities, or None if it can't forecast.
# ---------------------------------------------------------------------------
ModelFn = Callable[[BacktestDataProvider, datetime], Optional[Dict[int, float]]]

RUN_HOURS = [0, 6, 12, 18]


# ---------------------------------------------------------------------------
# Kalshi bracket helpers
# ---------------------------------------------------------------------------

@dataclass
class KalshiBracket:
    """One bracket from a Kalshi settlement event."""
    floor_strike: Optional[float]  # None for lower tail
    cap_strike: Optional[float]    # None for upper tail
    settled_yes: int               # 1 if this bracket settled YES

    def contains(self, temp: int) -> bool:
        """Check if an integer temperature falls in this bracket.

        Convention (verified against KXHIGHNY settlement data):
        - Lower tail: temp < cap_strike  (strictly less than)
        - Interior:   floor_strike <= temp <= cap_strike  (inclusive both)
        - Upper tail:  temp > floor_strike  (strictly greater than)
        """
        if self.floor_strike is None:
            # Lower tail
            return temp < self.cap_strike
        if self.cap_strike is None:
            # Upper tail
            return temp > self.floor_strike
        # Interior bracket
        return self.floor_strike <= temp <= self.cap_strike


def map_probs_to_kalshi_brackets(
    bracket_probs_1f: Dict[int, float],
    kalshi_brackets: List[KalshiBracket],
) -> List[float]:
    """Map 1°F integer bracket probabilities to Kalshi's 2°F bracket structure.

    Sums the model's probability mass for each integer temperature that falls
    into each Kalshi bracket. Returns a list of probabilities aligned with
    the kalshi_brackets list.
    """
    mapped = []
    for kb in kalshi_brackets:
        p = sum(prob for temp, prob in bracket_probs_1f.items() if kb.contains(temp))
        mapped.append(p)

    # Normalize — the model's 1°F probs may not cover all Kalshi brackets exactly
    total = sum(mapped)
    if total > 0 and abs(total - 1.0) > 0.01:
        mapped = [p / total for p in mapped]

    return mapped


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def compute_brier_score(
    mapped_probs: List[float],
    kalshi_brackets: List[KalshiBracket],
) -> float:
    """Multi-category Brier score for Kalshi bracket predictions.

    BS = sum_k (p_k - o_k)^2
    where o_k = 1 if bracket k settled YES, 0 otherwise.

    Lower is better. Perfect = 0.0.
    """
    n = len(mapped_probs)
    if n == 0:
        return 2.0  # Worst possible

    total = 0.0
    for i, kb in enumerate(kalshi_brackets):
        outcome = 1.0 if kb.settled_yes else 0.0
        total += (mapped_probs[i] - outcome) ** 2

    return total


def compute_brier_score_1f(
    bracket_probs: Dict[int, float],
    actual_high: float,
) -> float:
    """Brier score using 1°F integer brackets (fallback for dates without Kalshi data).

    BS = sum_k (p_k - o_k)^2 where o_k = 1 if k == round(actual_high).
    """
    actual_bracket = round(actual_high)
    total = 0.0
    for k, p in bracket_probs.items():
        outcome = 1.0 if k == actual_bracket else 0.0
        total += (p - outcome) ** 2
    # Penalize if actual bracket not in model's output
    if actual_bracket not in bracket_probs:
        total += 1.0
    return total


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single model_run evaluation."""
    settlement_date: date
    run_hour: int
    station_id: str
    forecast_high: Optional[float]
    actual_high: float
    brier_score: float
    predicted_bracket: int     # argmax of 1°F bracket probs
    hit: bool                  # predicted_bracket == round(actual_high)
    used_kalshi_brackets: bool  # True if scored with 2°F Kalshi brackets
    n_brackets: int            # Number of brackets scored against
    update_hour_et: Optional[int] = None  # Phase 2B: ET hour of intraday update


@dataclass
class BacktestResult:
    """Aggregate results from a full backtest run."""
    run_results: List[RunResult]
    mean_brier: float
    top1_hit_rate: float
    top2_hit_rate: float
    total_days: int
    total_evaluations: int
    skipped: int
    kalshi_scored: int         # How many used real Kalshi bracket scoring
    fallback_scored: int       # How many used 1°F fallback scoring
    by_run_hour: Dict[int, dict]


# ---------------------------------------------------------------------------
# Dummy model for Phase 0 gate
# ---------------------------------------------------------------------------

def uniform_model(provider: BacktestDataProvider, ref_time: datetime) -> Optional[Dict[int, float]]:
    """Dummy model: uniform distribution over 31 brackets around forecast high."""
    fcst_high = provider.get_forecast_high(provider.station_id)
    if fcst_high is None:
        return None
    center = round(fcst_high)
    radius = 15
    n = 2 * radius + 1
    return {k: 1.0 / n for k in range(center - radius, center + radius + 1)}


def bias_corrected_model(provider: BacktestDataProvider, ref_time: datetime) -> Optional[Dict[int, float]]:
    """Phase 1 model: bias-corrected Gaussian via ProbabilityEngine.

    Same code path as live — just different data source.
    """
    from services.probability import ProbabilityEngine
    engine = ProbabilityEngine(data_provider=provider)
    forecast = engine.calculate_city("NYC", ref_time=ref_time)
    if forecast is None:
        return None
    return forecast.bracket_probs


# ---------------------------------------------------------------------------
# Walk-forward bias-corrected models (Phase 1)
# ---------------------------------------------------------------------------

# Minimum number of prior dates required before we trust the bias estimate
WALK_FORWARD_MIN_DAYS = 90


def _compute_bracket_probs_from_dist(dist, center, std, radius=15):
    # type: (object, float, float, int) -> Dict[int, float]
    """Compute P(high = k°F) for 1°F integer brackets using any scipy dist.

    dist must support .cdf(x) — e.g. norm(0,1) or t(df).
    """
    center_int = round(center)
    probs = {}
    for k in range(center_int - radius, center_int + radius + 1):
        p = dist.cdf((k + 0.5 - center) / std) - dist.cdf((k - 0.5 - center) / std)
        if p > 0.0001:
            probs[k] = round(p, 4)

    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    return probs


def _walk_forward_bias_query(con, run_hour, station_id, current_date, model_name='hrrr'):
    # type: (duckdb.DuckDBPyConnection, int, str, date, str) -> Optional[tuple]
    """Expanding-window bias stats from all dates BEFORE current_date for one run_hour.

    Returns (mean_bias, std_error, n, excess_kurtosis) or None.
    Error = MAX(forecast_high) - nws_daily.max_temp_f (NWS settlement truth).
    """
    row = con.execute("""
        SELECT
            AVG(error) as mean_bias,
            STDDEV(error) as std_error,
            COUNT(*) as n,
            (AVG(POWER(error - sub.global_mean, 4)) /
             POWER(GREATEST(STDDEV(error), 0.01), 4)) - 3.0 as excess_kurtosis
        FROM (
            SELECT
                MAX(f.temp_f) - n.max_temp_f as error,
                AVG(MAX(f.temp_f) - n.max_temp_f) OVER () as global_mean
            FROM nws_daily n
            JOIN forecasts f ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = ?
                AND f.model_name = ?
            WHERE n.station_id = ?
                AND n.obs_date < ?
                AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
        ) sub
    """, [run_hour, model_name, station_id, current_date]).fetchone()

    if not row or row[2] is None or row[2] < WALK_FORWARD_MIN_DAYS:
        return None
    return row  # (mean_bias, std_error, n, excess_kurtosis)


def walk_forward_model(provider, ref_time, model_name='hrrr'):
    # type: (BacktestDataProvider, datetime, str) -> Optional[Dict[int, float]]
    """Walk-forward bias-corrected Gaussian using per-run-hour stats.

    For each evaluation:
    1. Computes expanding-window mean bias and std from ALL prior dates
       (strict walk-forward: obs_date < current_date, no peeking)
    2. Uses NWS settlement truth (nws_daily.max_temp_f), not observations
    3. Per-run-hour: 00z, 06z, 12z, 18z each get their own bias/std
    4. No drift, stability, or convergence factors — isolate the bias correction effect
    """
    con = provider._shared_con
    station_id = provider.station_id
    run_hour = provider.model_run.hour
    current_date = provider.model_run.date()

    fcst_high = provider.get_forecast_high(station_id)
    if fcst_high is None:
        return None

    stats = _walk_forward_bias_query(con, run_hour, station_id, current_date, model_name=model_name)
    if stats is None:
        return None

    mean_bias, std_error = stats[0], stats[1]
    std = max(0.3, std_error if std_error else 2.0)
    center = fcst_high - mean_bias

    return _compute_bracket_probs_from_dist(norm(0, 1), center, std)


def walk_forward_t_model(provider, ref_time, model_name='hrrr'):
    # type: (BacktestDataProvider, datetime, str) -> Optional[Dict[int, float]]
    """Walk-forward bias-corrected Student-t for heavy tails.

    Same as walk_forward_model but uses Student-t distribution instead of
    Gaussian. Heavier tails put more probability mass on extreme outcomes,
    which should reduce Brier score when forecast errors are leptokurtic.

    Degrees of freedom estimated from excess kurtosis:
      df = 6/kurtosis + 4  (for kurtosis > 0)
      Clamped to [3, 30] — below 3 the variance is infinite, above 30 is ~Gaussian.
    """
    con = provider._shared_con
    station_id = provider.station_id
    run_hour = provider.model_run.hour
    current_date = provider.model_run.date()

    fcst_high = provider.get_forecast_high(station_id)
    if fcst_high is None:
        return None

    stats = _walk_forward_bias_query(con, run_hour, station_id, current_date, model_name=model_name)
    if stats is None:
        return None

    mean_bias, std_error, n, excess_kurtosis = stats
    std = max(0.3, std_error if std_error else 2.0)
    center = fcst_high - mean_bias

    # Estimate df from excess kurtosis (kurtosis of t-dist = 6/(df-4) for df>4)
    if excess_kurtosis is not None and excess_kurtosis > 0:
        df = 6.0 / excess_kurtosis + 4.0
        df = max(3.0, min(30.0, df))
    else:
        df = 30.0  # Near-Gaussian when kurtosis <= 0

    return _compute_bracket_probs_from_dist(student_t(df), center, std)


# ---------------------------------------------------------------------------
# Walk-forward REGRESSION models (Phase 2)
# ---------------------------------------------------------------------------

def _walk_forward_regression_data(con, run_hour, station_id, current_date, model_name='hrrr'):
    # type: (duckdb.DuckDBPyConnection, int, str, date, str) -> Optional[list]
    """Expanding-window training data: (error, fcst_high, month, delta_temp) from prior dates.

    delta_temp = actual(D-1) - actual(D-2) — only uses prior actuals.
    Returns None if fewer than WALK_FORWARD_MIN_DAYS complete rows.
    """
    rows = con.execute("""
        WITH daily_errors AS (
            SELECT
                n.obs_date,
                MAX(f.temp_f) - n.max_temp_f AS error,
                MAX(f.temp_f) AS fcst_high,
                EXTRACT(MONTH FROM n.obs_date) AS month,
                n.max_temp_f AS actual_high
            FROM nws_daily n
            JOIN forecasts f ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = ?
                AND f.model_name = ?
            WHERE n.station_id = ?
                AND n.obs_date < ?
                AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
            ORDER BY n.obs_date
        )
        SELECT
            error,
            fcst_high,
            month,
            LAG(actual_high, 1) OVER (ORDER BY obs_date)
                - LAG(actual_high, 2) OVER (ORDER BY obs_date) AS delta_temp
        FROM daily_errors
    """, [run_hour, model_name, station_id, current_date]).fetchall()

    # Filter out rows where delta_temp is NULL (first 2 dates in the window)
    complete = [(e, fh, m, dt) for e, fh, m, dt in rows if dt is not None]
    if len(complete) < WALK_FORWARD_MIN_DAYS:
        return None
    return complete


def _encode_month(month):
    # type: (float,) -> Tuple[float, float]
    """Encode month as (sin, cos) pair for smooth seasonal capture."""
    angle = 2.0 * math.pi * month / 12.0
    return (math.sin(angle), math.cos(angle))


def _fit_and_predict(rows, features_today, feature_indices):
    # type: (list, tuple, list) -> Optional[Tuple[float, float]]
    """Manual OLS via scipy.linalg.lstsq.

    rows: list of (error, fcst_high, month, delta_temp)
    features_today: (fcst_high, month, delta_temp) for the prediction date
    feature_indices: which columns to use from the feature set:
        0 = fcst_high, 1 = sin(month), 2 = cos(month), 3 = delta_temp

    Returns (predicted_bias, residual_std) or None on failure.
    """
    n = len(rows)
    if n < WALK_FORWARD_MIN_DAYS:
        return None

    # Build feature matrix: each row = [1 (intercept), selected features...]
    n_features = len(feature_indices)
    A = np.empty((n, 1 + n_features), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    for i, (error, fcst_high, month, delta_temp) in enumerate(rows):
        sin_m, cos_m = _encode_month(month)
        all_features = [fcst_high, sin_m, cos_m, delta_temp]
        A[i, 0] = 1.0  # intercept
        for j, idx in enumerate(feature_indices):
            A[i, 1 + j] = all_features[idx]
        y[i] = error

    # Solve via least squares
    result = lstsq(A, y)
    coeffs = result[0]

    # Predict for today
    fcst_today, month_today, delta_today = features_today
    sin_m, cos_m = _encode_month(month_today)
    all_today = [fcst_today, sin_m, cos_m, delta_today]
    x_today = np.array([1.0] + [all_today[idx] for idx in feature_indices])
    predicted_bias = float(np.dot(coeffs, x_today))

    # Residual std from training data
    residuals = y - A @ coeffs
    residual_std = float(np.std(residuals, ddof=1 + n_features))

    return (predicted_bias, max(0.3, residual_std))


def _get_delta_temp(con, station_id, current_date):
    # type: (duckdb.DuckDBPyConnection, str, date) -> Optional[float]
    """Get actual(D-1) - actual(D-2) using only data BEFORE current_date.

    D-1 = the day before current_date, D-2 = two days before.
    Both must exist in nws_daily.
    """
    rows = con.execute("""
        SELECT max_temp_f
        FROM nws_daily
        WHERE station_id = ?
            AND obs_date < ?
            AND max_temp_f IS NOT NULL
        ORDER BY obs_date DESC
        LIMIT 2
    """, [station_id, current_date]).fetchall()

    if len(rows) < 2:
        return None
    # rows[0] = D-1, rows[1] = D-2 (descending order)
    return rows[0][0] - rows[1][0]


# Feature index mapping for _fit_and_predict:
#   0 = fcst_high, 1 = sin(month), 2 = cos(month), 3 = delta_temp

_FEATURE_SETS = {
    "wf_regression_full":  [0, 1, 2, 3],       # fcst_high + month + delta
    "wf_regression_fcst":  [0],                  # fcst_high only
    "wf_regression_month": [1, 2],               # month (sin/cos) only
    "wf_regression_delta": [3],                   # delta_temp only
}


def _make_regression_model(name, feature_indices, model_name='hrrr'):
    # type: (str, list, str) -> ModelFn
    """Factory: create a walk-forward regression ModelFn for a given feature subset."""

    def model_fn(provider, ref_time):
        # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
        con = provider._shared_con
        station_id = provider.station_id
        run_hour = provider.model_run.hour
        current_date = provider.model_run.date()

        fcst_high = provider.get_forecast_high(station_id)
        if fcst_high is None:
            return None

        # Get training data (expanding window, all prior dates)
        training = _walk_forward_regression_data(con, run_hour, station_id, current_date, model_name=model_name)
        if training is None:
            return None

        # Get today's delta_temp
        delta_temp = _get_delta_temp(con, station_id, current_date)
        if delta_temp is None:
            return None

        month = float(current_date.month)
        features_today = (fcst_high, month, delta_temp)

        result = _fit_and_predict(training, features_today, feature_indices)
        if result is None:
            return None

        predicted_bias, residual_std = result
        center = fcst_high - predicted_bias
        return _compute_bracket_probs_from_dist(norm(0, 1), center, residual_std)

    model_fn.__name__ = name
    model_fn.__doc__ = "Walk-forward regression: {}".format(name)
    return model_fn


# Pre-built regression model functions (HRRR — default)
wf_regression_full = _make_regression_model("wf_regression_full", _FEATURE_SETS["wf_regression_full"])
wf_regression_fcst = _make_regression_model("wf_regression_fcst", _FEATURE_SETS["wf_regression_fcst"])
wf_regression_month = _make_regression_model("wf_regression_month", _FEATURE_SETS["wf_regression_month"])
wf_regression_delta = _make_regression_model("wf_regression_delta", _FEATURE_SETS["wf_regression_delta"])

# --- GFS regression models (Phase 3) ---
wf_regression_full_gfs = _make_regression_model("wf_regression_full_gfs", _FEATURE_SETS["wf_regression_full"], model_name="gfs")
wf_regression_fcst_gfs = _make_regression_model("wf_regression_fcst_gfs", _FEATURE_SETS["wf_regression_fcst"], model_name="gfs")
wf_regression_month_gfs = _make_regression_model("wf_regression_month_gfs", _FEATURE_SETS["wf_regression_month"], model_name="gfs")
wf_regression_delta_gfs = _make_regression_model("wf_regression_delta_gfs", _FEATURE_SETS["wf_regression_delta"], model_name="gfs")

# --- ECMWF regression models (Phase 3) ---
wf_regression_full_ecmwf = _make_regression_model("wf_regression_full_ecmwf", _FEATURE_SETS["wf_regression_full"], model_name="ecmwf")
wf_regression_fcst_ecmwf = _make_regression_model("wf_regression_fcst_ecmwf", _FEATURE_SETS["wf_regression_fcst"], model_name="ecmwf")
wf_regression_month_ecmwf = _make_regression_model("wf_regression_month_ecmwf", _FEATURE_SETS["wf_regression_month"], model_name="ecmwf")
wf_regression_delta_ecmwf = _make_regression_model("wf_regression_delta_ecmwf", _FEATURE_SETS["wf_regression_delta"], model_name="ecmwf")


# ---------------------------------------------------------------------------
# Walk-forward PHASE 2B models — observation-based residual correction
# ---------------------------------------------------------------------------

from zoneinfo import ZoneInfo
from services.divergence import (
    interpolate_forecast,
    compute_divergence_features,
)

_ET = ZoneInfo("America/New_York")

# Phase 2B divergence feature index mapping:
#   0 = temp_divergence, 1 = cumulative_divergence,
#   2 = running_max_divergence, 3 = slope_divergence
_PHASE2B_FEATURE_KEYS = [
    "temp_divergence",
    "cumulative_divergence",
    "running_max_divergence",
    "slope_divergence",
]

_PHASE2B_FEATURE_SETS = {
    "wf_phase2b_full":    [0, 1, 2, 3],  # all 4 features
    "wf_phase2b_core":    [0, 1, 2, 3],  # same as full (reserved for synoptic variant)
    "wf_phase2b_instant": [0],            # temp_divergence only
    "wf_phase2b_cumul":   [1],            # cumulative only
    "wf_phase2b_runmax":  [2],            # running max only
    "wf_phase2b_slope":   [3],            # slope only
}

# Phase 2 feature set used inside Phase 2B (fcst_high + sin/cos month)
_P2_FEATURES_FOR_P2B = [0, 1, 2]

# ---------------------------------------------------------------------------
# Two-level cache: avoids redundant SQL + Phase 2 expanding-window fits.
#
# Level 1: (con_id, run_hour, station_id)
#   -> bulk data + Phase 2 predictions for ALL dates (once per run_hour)
#
# Level 2: (con_id, run_hour, station_id, update_hour_et)
#   -> Phase 2B training rows: (date, residual, [feat0..3]) per update hour
# ---------------------------------------------------------------------------
_p2b_level1 = {}  # type: Dict  # level 1 cache
_p2b_level2 = {}  # type: Dict  # level 2 cache


def _compute_features_for_date(curve, day_obs, cutoff_ts):
    # type: (List[Tuple[float, float]], List[Tuple[float, float]], float) -> Optional[Dict[str, float]]
    """Compute divergence features for one date at one update time."""
    truncated = [(ts, temp) for ts, temp in day_obs if ts <= cutoff_ts]
    if len(truncated) < 2:
        return None

    obs_ts = [o[0] for o in truncated]
    obs_temps = [o[1] for o in truncated]
    fc_ts = [c[0] for c in curve]
    fc_temps = [c[1] for c in curve]

    fcst_interp = interpolate_forecast(fc_ts, fc_temps, obs_ts)
    t0 = obs_ts[0]
    obs_hours = [(t - t0) / 3600.0 for t in obs_ts]
    fcst_up_to_t = [temp for ts, temp in curve if ts <= cutoff_ts]

    return compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_up_to_t)


def _ensure_level1(con, run_hour, station_id, model_name='hrrr'):
    # type: (duckdb.DuckDBPyConnection, int, str, str) -> dict
    """Populate level-1 cache: bulk data + Phase 2 predictions for all dates.

    Called once per (connection, run_hour, model_name). Returns cached dict with keys:
      curves, obs_by_date, errors_list, p2_preds
    """
    key = (id(con), run_hour, station_id, model_name)
    if key in _p2b_level1:
        return _p2b_level1[key]

    # Clear stale entries from old connections
    stale = [k for k in _p2b_level1 if k[0] != id(con)]
    for k in stale:
        del _p2b_level1[k]
    stale2 = [k for k in _p2b_level2 if k[0] != id(con)]
    for k in stale2:
        del _p2b_level2[k]

    logger.info(f"Phase 2B: bulk-fetching data for run_hour={run_hour:02d}z...")

    # --- Forecast curves ---
    curve_rows = con.execute("""
        SELECT f.model_run::DATE as obs_date, f.valid_at, f.temp_f
        FROM forecasts f
        WHERE f.station_id = ? AND EXTRACT(HOUR FROM f.model_run) = ?
            AND f.model_name = ?
        ORDER BY f.model_run::DATE, f.valid_at
    """, [station_id, run_hour, model_name]).fetchall()

    curves = {}  # type: Dict[date, List[Tuple[float, float]]]
    for obs_date, valid_at, temp_f in curve_rows:
        if obs_date not in curves:
            curves[obs_date] = []
        ts = valid_at.timestamp() if hasattr(valid_at, 'timestamp') else float(valid_at)
        curves[obs_date].append((ts, temp_f))

    # --- Observations ---
    obs_rows = con.execute("""
        SELECT observed_at, temp_f
        FROM observations
        WHERE station_id = ? AND temp_f IS NOT NULL
        ORDER BY observed_at
    """, [station_id]).fetchall()

    obs_by_date = {}  # type: Dict[date, List[Tuple[float, float]]]
    for observed_at, temp_f in obs_rows:
        obs_utc = observed_at.replace(tzinfo=timezone.utc) if hasattr(observed_at, 'replace') \
            else datetime.fromtimestamp(float(observed_at), tz=timezone.utc)
        et_date = obs_utc.astimezone(_ET).date()
        ts = observed_at.timestamp() if hasattr(observed_at, 'timestamp') else float(observed_at)
        if et_date not in obs_by_date:
            obs_by_date[et_date] = []
        obs_by_date[et_date].append((ts, temp_f))

    # --- Errors + Phase 2 features ---
    error_rows = con.execute("""
        WITH daily_errors AS (
            SELECT
                n.obs_date,
                MAX(f.temp_f) - n.max_temp_f AS error,
                MAX(f.temp_f) AS fcst_high,
                EXTRACT(MONTH FROM n.obs_date) AS month,
                n.max_temp_f AS actual_high
            FROM nws_daily n
            JOIN forecasts f ON f.station_id = n.station_id
                AND f.model_run::DATE = n.obs_date
                AND EXTRACT(HOUR FROM f.model_run) = ?
                AND f.model_name = ?
            WHERE n.station_id = ? AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
            ORDER BY n.obs_date
        )
        SELECT
            obs_date, error, fcst_high, month,
            LAG(actual_high, 1) OVER (ORDER BY obs_date)
                - LAG(actual_high, 2) OVER (ORDER BY obs_date) AS delta_temp
        FROM daily_errors
        ORDER BY obs_date
    """, [run_hour, model_name, station_id]).fetchall()

    errors_list = [(d, e, fh, m, dt) for d, e, fh, m, dt in error_rows]

    # --- Phase 2 expanding-window predictions for ALL dates ---
    # Compute once, reuse across all update_hours
    p2_preds = {}  # type: Dict[date, Tuple[float, float]]  # date -> (predicted_bias, p2_std)
    phase2_training = []  # type: list
    for obs_date, actual_error, fcst_high, month, delta_temp in errors_list:
        # Predict BEFORE adding this date to training (walk-forward)
        if len(phase2_training) >= WALK_FORWARD_MIN_DAYS:
            features_today = (fcst_high, month, delta_temp if delta_temp is not None else 0.0)
            p2_result = _fit_and_predict(phase2_training, features_today, _P2_FEATURES_FOR_P2B)
            if p2_result is not None:
                p2_preds[obs_date] = p2_result  # (predicted_bias, p2_std)

        # Then add to training
        if delta_temp is not None:
            phase2_training.append((actual_error, fcst_high, month, delta_temp))

    result = {
        "curves": curves,
        "obs_by_date": obs_by_date,
        "errors_list": errors_list,
        "p2_preds": p2_preds,
    }
    _p2b_level1[key] = result
    logger.info(
        f"Phase 2B: {len(curves)} curves, {len(obs_by_date)} obs-dates, "
        f"{len(p2_preds)} Phase 2 predictions cached for {run_hour:02d}z"
    )
    return result


def _ensure_level2(con, run_hour, station_id, update_hour_et, model_name='hrrr'):
    # type: (duckdb.DuckDBPyConnection, int, str, int, str) -> List[Tuple[date, float, List[float]]]
    """Populate level-2 cache: Phase 2B training rows for one update_hour.

    Returns list of (date, residual, [feat0..3]) sorted by date.
    For a given current_date D, filter to entries where date < D.
    """
    key = (id(con), run_hour, station_id, update_hour_et, model_name)
    if key in _p2b_level2:
        return _p2b_level2[key]

    l1 = _ensure_level1(con, run_hour, station_id, model_name=model_name)
    curves = l1["curves"]
    obs_by_date = l1["obs_by_date"]
    errors_list = l1["errors_list"]
    p2_preds = l1["p2_preds"]

    training = []  # type: List[Tuple[date, float, List[float]]]
    for obs_date, actual_error, fcst_high, month, delta_temp in errors_list:
        if obs_date not in p2_preds:
            continue
        predicted_bias, _ = p2_preds[obs_date]
        residual = actual_error - predicted_bias

        if obs_date not in curves or obs_date not in obs_by_date:
            continue

        cutoff_ts = datetime(
            obs_date.year, obs_date.month, obs_date.day,
            update_hour_et, 0, tzinfo=_ET,
        ).astimezone(timezone.utc).timestamp()

        feats = _compute_features_for_date(curves[obs_date], obs_by_date[obs_date], cutoff_ts)
        if feats is None:
            continue

        feat_vector = [feats[k] for k in _PHASE2B_FEATURE_KEYS]
        training.append((obs_date, residual, feat_vector))

    _p2b_level2[key] = training
    return training


def _fit_and_predict_phase2b(training_rows, features_today, feature_indices):
    # type: (list, List[float], List[int]) -> Optional[Tuple[float, float]]
    """OLS regression on Phase 2B training data.

    training_rows: list of (residual, [feat0..3])
    features_today: divergence feature values for current eval
    feature_indices: which of the 4 features to use

    Returns (predicted_residual, residual_std) or None.
    """
    n = len(training_rows)
    if n < WALK_FORWARD_MIN_DAYS:
        return None

    n_feats = len(feature_indices)
    A = np.empty((n, 1 + n_feats), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    for i, (residual, feat_vec) in enumerate(training_rows):
        A[i, 0] = 1.0
        for j, idx in enumerate(feature_indices):
            A[i, 1 + j] = feat_vec[idx]
        y[i] = residual

    result = lstsq(A, y)
    coeffs = result[0]

    x_today = np.array([1.0] + [features_today[idx] for idx in feature_indices])
    predicted_residual = float(np.dot(coeffs, x_today))

    residuals = y - A @ coeffs
    residual_std = float(np.std(residuals, ddof=1 + n_feats))

    return (predicted_residual, max(0.3, residual_std))


def _make_phase2b_model(name, feature_indices, model_name='hrrr'):
    # type: (str, list, str) -> ModelFn
    """Factory: Phase 2B walk-forward residual regression model.

    Phase 2B predicts what Phase 2 couldn't — the day-specific deviation
    visible in real-time observations vs the raw HRRR curve.

    Final: center = fcst_high - (Phase2_bias + Phase2B_residual)

    Performance: bulk data + Phase 2 predictions cached at module level.
    Per-evaluation cost is one OLS fit on the filtered training matrix.
    """
    def model_fn(provider, ref_time):
        # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
        con = provider._shared_con
        station_id = provider.station_id
        run_hour = provider.model_run.hour
        current_date = provider.model_run.date()

        fcst_high = provider.get_forecast_high(station_id)
        if fcst_high is None:
            return None

        # --- Phase 2 prediction (from cache) ---
        l1 = _ensure_level1(con, run_hour, station_id, model_name=model_name)
        p2_pred = l1["p2_preds"].get(current_date)
        if p2_pred is None:
            return None
        phase2_bias, p2_std = p2_pred

        # --- Determine update hour from ref_time ---
        ref_utc = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
        update_hour_et = ref_utc.astimezone(_ET).hour
        cutoff_ts = ref_utc.timestamp()

        # --- Today's divergence features ---
        curves = l1["curves"]
        if current_date not in curves or len(curves[current_date]) < 2:
            center = fcst_high - phase2_bias
            return _compute_bracket_probs_from_dist(norm(0, 1), center, p2_std)

        curve = curves[current_date]
        obs_by_date = l1["obs_by_date"]
        day_obs = obs_by_date.get(current_date, [])
        truncated = [(ts, temp) for ts, temp in day_obs if ts <= cutoff_ts]
        if len(truncated) < 2:
            center = fcst_high - phase2_bias
            return _compute_bracket_probs_from_dist(norm(0, 1), center, p2_std)

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
            return _compute_bracket_probs_from_dist(norm(0, 1), center, p2_std)

        today_feat_vec = [today_feats[k] for k in _PHASE2B_FEATURE_KEYS]

        # --- Phase 2B training (from cache, filtered to dates < current_date) ---
        all_training = _ensure_level2(con, run_hour, station_id, update_hour_et, model_name=model_name)
        filtered = [
            (residual, feat_vec)
            for d, residual, feat_vec in all_training
            if d < current_date
        ]

        p2b_result = _fit_and_predict_phase2b(filtered, today_feat_vec, feature_indices)
        if p2b_result is None:
            center = fcst_high - phase2_bias
            return _compute_bracket_probs_from_dist(norm(0, 1), center, p2_std)

        phase2b_residual, p2b_std = p2b_result
        center = fcst_high - (phase2_bias + phase2b_residual)
        return _compute_bracket_probs_from_dist(norm(0, 1), center, p2b_std)

    model_fn.__name__ = name
    model_fn.__doc__ = "Phase 2B residual regression: {}".format(name)
    return model_fn


# Pre-built Phase 2B model functions (HRRR — default)
wf_phase2b_full = _make_phase2b_model("wf_phase2b_full", _PHASE2B_FEATURE_SETS["wf_phase2b_full"])
wf_phase2b_core = _make_phase2b_model("wf_phase2b_core", _PHASE2B_FEATURE_SETS["wf_phase2b_core"])
wf_phase2b_instant = _make_phase2b_model("wf_phase2b_instant", _PHASE2B_FEATURE_SETS["wf_phase2b_instant"])
wf_phase2b_cumul = _make_phase2b_model("wf_phase2b_cumul", _PHASE2B_FEATURE_SETS["wf_phase2b_cumul"])
wf_phase2b_runmax = _make_phase2b_model("wf_phase2b_runmax", _PHASE2B_FEATURE_SETS["wf_phase2b_runmax"])
wf_phase2b_slope = _make_phase2b_model("wf_phase2b_slope", _PHASE2B_FEATURE_SETS["wf_phase2b_slope"])

# --- GFS Phase 2B models (Phase 3) ---
wf_phase2b_full_gfs = _make_phase2b_model("wf_phase2b_full_gfs", _PHASE2B_FEATURE_SETS["wf_phase2b_full"], model_name="gfs")
wf_phase2b_core_gfs = _make_phase2b_model("wf_phase2b_core_gfs", _PHASE2B_FEATURE_SETS["wf_phase2b_core"], model_name="gfs")
wf_phase2b_instant_gfs = _make_phase2b_model("wf_phase2b_instant_gfs", _PHASE2B_FEATURE_SETS["wf_phase2b_instant"], model_name="gfs")
wf_phase2b_cumul_gfs = _make_phase2b_model("wf_phase2b_cumul_gfs", _PHASE2B_FEATURE_SETS["wf_phase2b_cumul"], model_name="gfs")
wf_phase2b_runmax_gfs = _make_phase2b_model("wf_phase2b_runmax_gfs", _PHASE2B_FEATURE_SETS["wf_phase2b_runmax"], model_name="gfs")
wf_phase2b_slope_gfs = _make_phase2b_model("wf_phase2b_slope_gfs", _PHASE2B_FEATURE_SETS["wf_phase2b_slope"], model_name="gfs")

# --- ECMWF Phase 2B models (Phase 3) ---
wf_phase2b_full_ecmwf = _make_phase2b_model("wf_phase2b_full_ecmwf", _PHASE2B_FEATURE_SETS["wf_phase2b_full"], model_name="ecmwf")
wf_phase2b_core_ecmwf = _make_phase2b_model("wf_phase2b_core_ecmwf", _PHASE2B_FEATURE_SETS["wf_phase2b_core"], model_name="ecmwf")
wf_phase2b_instant_ecmwf = _make_phase2b_model("wf_phase2b_instant_ecmwf", _PHASE2B_FEATURE_SETS["wf_phase2b_instant"], model_name="ecmwf")
wf_phase2b_cumul_ecmwf = _make_phase2b_model("wf_phase2b_cumul_ecmwf", _PHASE2B_FEATURE_SETS["wf_phase2b_cumul"], model_name="ecmwf")
wf_phase2b_runmax_ecmwf = _make_phase2b_model("wf_phase2b_runmax_ecmwf", _PHASE2B_FEATURE_SETS["wf_phase2b_runmax"], model_name="ecmwf")
wf_phase2b_slope_ecmwf = _make_phase2b_model("wf_phase2b_slope_ecmwf", _PHASE2B_FEATURE_SETS["wf_phase2b_slope"], model_name="ecmwf")


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Iterates historical dates, feeds data through a model function, scores results.

    For each settlement date in nws_daily:
      For each HRRR run hour (00z, 06z, 12z, 18z):
        1. Construct a BacktestDataProvider for (station_id, model_run, ref_time)
        2. Call model_fn(provider, ref_time) -> bracket_probs (1°F integers)
        3. Map to Kalshi 2°F brackets if available for that date
        4. Compute Brier score against settlement outcome
    """

    def __init__(
        self,
        db_path: str = "data/alphatemp.duckdb",
        city: str = "NYC",
    ):
        self.db_path = db_path
        self.city = city
        self.station_id = CITIES[city]["settlement"]

    def _get_settlement_dates(
        self,
        con: duckdb.DuckDBPyConnection,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Tuple[date, float]]:
        """Get (obs_date, max_temp_f) from nws_daily within range."""
        query = """
            SELECT obs_date, max_temp_f
            FROM nws_daily
            WHERE station_id = ?
            AND max_temp_f IS NOT NULL
        """
        params = [self.station_id]  # type: list

        if start_date:
            query += " AND obs_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND obs_date <= ?"
            params.append(end_date)

        query += " ORDER BY obs_date"
        return [(r[0], r[1]) for r in con.execute(query, params).fetchall()]

    def _get_kalshi_brackets(
        self,
        con: duckdb.DuckDBPyConnection,
        settlement_date: date,
    ) -> Optional[List[KalshiBracket]]:
        """Get bracket definitions from kalshi_settlements for this date."""
        rows = con.execute(
            """SELECT floor_strike, cap_strike, settled_yes
               FROM kalshi_settlements
               WHERE event_date = ?
               ORDER BY floor_strike NULLS FIRST""",
            [settlement_date],
        ).fetchall()

        if not rows or len(rows) < 2:
            return None

        # Filter out brackets with both strikes NULL (unparsed ticker data)
        brackets = [
            KalshiBracket(floor_strike=r[0], cap_strike=r[1], settled_yes=r[2])
            for r in rows
            if r[0] is not None or r[1] is not None
        ]
        return brackets if len(brackets) >= 2 else None

    def _has_forecast_data(
        self,
        con: duckdb.DuckDBPyConnection,
        model_run: datetime,
        model_name: str = 'hrrr',
    ) -> bool:
        """Check if forecast data exists for this model_run and model_name."""
        row = con.execute(
            "SELECT COUNT(*) FROM forecasts WHERE station_id = ? AND model_run = ? AND model_name = ?",
            [self.station_id, model_run, model_name],
        ).fetchone()
        return row[0] > 0

    def run(
        self,
        model_fn: ModelFn,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        run_hours: Optional[List[int]] = None,
        update_hours_et: Optional[List[int]] = None,
    ) -> BacktestResult:
        """Execute the backtest and return scored results.

        Parameters
        ----------
        update_hours_et : list of int, optional
            Eastern Time hours for intraday updates (Phase 2B).
            When provided, each (date, run_hour) is evaluated at every
            update hour, with ref_time set to that ET hour converted to UTC.
            When None (default): Phase 1/2 behavior (ref_time = model_run + 2h).
        """
        if run_hours is None:
            run_hours = RUN_HOURS

        con = duckdb.connect(self.db_path, read_only=True)

        try:
            settlement_dates = self._get_settlement_dates(con, start_date, end_date)
            n_updates = len(update_hours_et) if update_hours_et else 1
            logger.info(
                "Backtesting {} settlement dates, run hours: {}, "
                "update hours: {}".format(
                    len(settlement_dates), run_hours,
                    update_hours_et or "default (model_run+2h)",
                )
            )

            results = []  # type: List[RunResult]
            skipped = 0
            kalshi_scored = 0
            fallback_scored = 0

            kalshi_cache = {}  # type: Dict[date, Optional[List[KalshiBracket]]]

            for i, (obs_date, actual_high) in enumerate(settlement_dates):
                if obs_date not in kalshi_cache:
                    kalshi_cache[obs_date] = self._get_kalshi_brackets(con, obs_date)
                kalshi_brackets = kalshi_cache[obs_date]

                for hour in run_hours:
                    model_run_utc = datetime(
                        obs_date.year, obs_date.month, obs_date.day,
                        hour, 0, tzinfo=timezone.utc,
                    )
                    model_run_naive = _strip_tz(model_run_utc)

                    if not self._has_forecast_data(con, model_run_naive):
                        skipped += n_updates
                        continue

                    # Determine ref_times to evaluate
                    if update_hours_et is None:
                        # Phase 1/2 default: single eval at model_run + lag
                        ref_times_and_labels = [
                            (model_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS), None)
                        ]
                    else:
                        ref_times_and_labels = []
                        for uhr in update_hours_et:
                            from zoneinfo import ZoneInfo
                            _et_tz = ZoneInfo("America/New_York")
                            ref_et = datetime(
                                obs_date.year, obs_date.month, obs_date.day,
                                uhr, 0, tzinfo=_et_tz,
                            )
                            ref_utc = ref_et.astimezone(timezone.utc)
                            # Skip if forecast wouldn't be available yet
                            earliest = model_run_utc + timedelta(
                                hours=HRRR_AVAILABILITY_LAG_HOURS,
                            )
                            if ref_utc < earliest:
                                skipped += 1
                                continue
                            ref_times_and_labels.append((ref_utc, uhr))

                    for ref_time, update_label in ref_times_and_labels:
                        provider = BacktestDataProvider(
                            db_path=self.db_path,
                            station_id=self.station_id,
                            model_run=model_run_utc,
                            ref_time=ref_time,
                            connection=con,
                        )

                        bracket_probs = model_fn(provider, ref_time)
                        if bracket_probs is None:
                            skipped += 1
                            continue

                        if kalshi_brackets:
                            mapped = map_probs_to_kalshi_brackets(
                                bracket_probs, kalshi_brackets,
                            )
                            brier = compute_brier_score(mapped, kalshi_brackets)
                            used_kalshi = True
                            n_brackets = len(kalshi_brackets)
                            kalshi_scored += 1
                        else:
                            brier = compute_brier_score_1f(bracket_probs, actual_high)
                            used_kalshi = False
                            n_brackets = len(bracket_probs)
                            fallback_scored += 1

                        predicted_bracket = max(bracket_probs, key=bracket_probs.get)
                        hit = predicted_bracket == round(actual_high)
                        fcst_high = provider.get_forecast_high(self.station_id)

                        results.append(RunResult(
                            settlement_date=obs_date,
                            run_hour=hour,
                            station_id=self.station_id,
                            forecast_high=fcst_high,
                            actual_high=actual_high,
                            brier_score=brier,
                            predicted_bracket=predicted_bracket,
                            hit=hit,
                            used_kalshi_brackets=used_kalshi,
                            n_brackets=n_brackets,
                            update_hour_et=update_label,
                        ))

                if (i + 1) % 200 == 0:
                    logger.info(f"  Processed {i + 1}/{len(settlement_dates)} days...")

            return self._aggregate(
                results, len(settlement_dates), skipped, kalshi_scored, fallback_scored
            )
        finally:
            con.close()

    def _aggregate(
        self,
        results: List[RunResult],
        total_days: int,
        skipped: int,
        kalshi_scored: int,
        fallback_scored: int,
    ) -> BacktestResult:
        """Compute aggregate metrics from individual run results."""
        if not results:
            return BacktestResult(
                run_results=results,
                mean_brier=2.0,
                top1_hit_rate=0.0,
                top2_hit_rate=0.0,
                total_days=total_days,
                total_evaluations=0,
                skipped=skipped,
                kalshi_scored=0,
                fallback_scored=0,
                by_run_hour={},
            )

        mean_brier = sum(r.brier_score for r in results) / len(results)
        top1_hits = sum(1 for r in results if r.hit)
        top1_rate = top1_hits / len(results)
        top2_rate = top1_rate  # Placeholder

        # Per-run-hour breakdown
        by_hour = {}  # type: Dict[int, dict]
        for hour in RUN_HOURS:
            hour_results = [r for r in results if r.run_hour == hour]
            if hour_results:
                by_hour[hour] = {
                    "count": len(hour_results),
                    "mean_brier": round(
                        sum(r.brier_score for r in hour_results) / len(hour_results), 4
                    ),
                    "top1_hit_rate": round(
                        sum(1 for r in hour_results if r.hit) / len(hour_results), 4
                    ),
                }

        # Per-update-hour breakdown (Phase 2B)
        update_hours_seen = sorted(set(
            r.update_hour_et for r in results if r.update_hour_et is not None
        ))
        by_update_hour = {}  # type: Dict[int, dict]
        for uhr in update_hours_seen:
            uhr_results = [r for r in results if r.update_hour_et == uhr]
            if uhr_results:
                by_update_hour[uhr] = {
                    "count": len(uhr_results),
                    "mean_brier": round(
                        sum(r.brier_score for r in uhr_results) / len(uhr_results), 4
                    ),
                    "top1_hit_rate": round(
                        sum(1 for r in uhr_results if r.hit) / len(uhr_results), 4
                    ),
                }

        result = BacktestResult(
            run_results=results,
            mean_brier=round(mean_brier, 4),
            top1_hit_rate=round(top1_rate, 4),
            top2_hit_rate=round(top2_rate, 4),
            total_days=total_days,
            total_evaluations=len(results),
            skipped=skipped,
            kalshi_scored=kalshi_scored,
            fallback_scored=fallback_scored,
            by_run_hour=by_hour,
        )
        # Attach update-hour breakdown as extra attribute (avoids changing dataclass)
        result.by_update_hour = by_update_hour  # type: ignore[attr-defined]
        return result

    def print_summary(self, result: BacktestResult) -> None:
        """Print a human-readable summary of backtest results."""
        logger.info("=" * 60)
        logger.info("Backtest Summary")
        logger.info("=" * 60)
        logger.info(
            f"Days: {result.total_days} | Evaluations: {result.total_evaluations} "
            f"| Skipped: {result.skipped}"
        )
        logger.info(
            f"Scoring: {result.kalshi_scored} Kalshi 2°F, "
            f"{result.fallback_scored} fallback 1°F"
        )
        logger.info(f"Mean Brier Score: {result.mean_brier:.4f}")
        logger.info(f"Top-1 Hit Rate:   {result.top1_hit_rate:.1%}")
        if result.by_run_hour:
            logger.info("-" * 60)
            logger.info("Per Run Hour:")
            for hour, stats in sorted(result.by_run_hour.items()):
                logger.info(
                    f"  {hour:02d}z: n={stats['count']}, "
                    f"brier={stats['mean_brier']:.4f}, "
                    f"top1={stats['top1_hit_rate']:.1%}"
                )
        logger.info("=" * 60)
