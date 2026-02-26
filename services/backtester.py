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


def _walk_forward_bias_query(con, run_hour, station_id, current_date):
    # type: (duckdb.DuckDBPyConnection, int, str, date) -> Optional[tuple]
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
            WHERE n.station_id = ?
                AND n.obs_date < ?
                AND n.max_temp_f IS NOT NULL
            GROUP BY n.obs_date, n.max_temp_f
        ) sub
    """, [run_hour, station_id, current_date]).fetchone()

    if not row or row[2] is None or row[2] < WALK_FORWARD_MIN_DAYS:
        return None
    return row  # (mean_bias, std_error, n, excess_kurtosis)


def walk_forward_model(provider, ref_time):
    # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
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

    stats = _walk_forward_bias_query(con, run_hour, station_id, current_date)
    if stats is None:
        return None

    mean_bias, std_error = stats[0], stats[1]
    std = max(0.3, std_error if std_error else 2.0)
    center = fcst_high - mean_bias

    return _compute_bracket_probs_from_dist(norm(0, 1), center, std)


def walk_forward_t_model(provider, ref_time):
    # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
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

    stats = _walk_forward_bias_query(con, run_hour, station_id, current_date)
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

def _walk_forward_regression_data(con, run_hour, station_id, current_date):
    # type: (duckdb.DuckDBPyConnection, int, str, date) -> Optional[list]
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
    """, [run_hour, station_id, current_date]).fetchall()

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


def _make_regression_model(name, feature_indices):
    # type: (str, list) -> ModelFn
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
        training = _walk_forward_regression_data(con, run_hour, station_id, current_date)
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


# Pre-built regression model functions
wf_regression_full = _make_regression_model("wf_regression_full", _FEATURE_SETS["wf_regression_full"])
wf_regression_fcst = _make_regression_model("wf_regression_fcst", _FEATURE_SETS["wf_regression_fcst"])
wf_regression_month = _make_regression_model("wf_regression_month", _FEATURE_SETS["wf_regression_month"])
wf_regression_delta = _make_regression_model("wf_regression_delta", _FEATURE_SETS["wf_regression_delta"])


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
    ) -> bool:
        """Check if forecast data exists for this model_run."""
        row = con.execute(
            "SELECT COUNT(*) FROM forecasts WHERE station_id = ? AND model_run = ?",
            [self.station_id, model_run],
        ).fetchone()
        return row[0] > 0

    def run(
        self,
        model_fn: ModelFn,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        run_hours: Optional[List[int]] = None,
    ) -> BacktestResult:
        """Execute the backtest and return scored results.

        Uses a single persistent read-only connection shared across all
        BacktestDataProvider instances to minimize DuckDB overhead.
        """
        if run_hours is None:
            run_hours = RUN_HOURS

        con = duckdb.connect(self.db_path, read_only=True)

        try:
            settlement_dates = self._get_settlement_dates(con, start_date, end_date)
            logger.info(
                f"Backtesting {len(settlement_dates)} settlement dates, "
                f"run hours: {run_hours}"
            )

            results = []  # type: List[RunResult]
            skipped = 0
            kalshi_scored = 0
            fallback_scored = 0

            # Pre-fetch Kalshi brackets for all dates (batch is faster than per-date)
            kalshi_cache = {}  # type: Dict[date, Optional[List[KalshiBracket]]]

            for i, (obs_date, actual_high) in enumerate(settlement_dates):
                # Fetch Kalshi brackets for this date (cached per date)
                if obs_date not in kalshi_cache:
                    kalshi_cache[obs_date] = self._get_kalshi_brackets(con, obs_date)
                kalshi_brackets = kalshi_cache[obs_date]

                for hour in run_hours:
                    model_run_utc = datetime(
                        obs_date.year, obs_date.month, obs_date.day,
                        hour, 0, tzinfo=timezone.utc,
                    )
                    # Strip tz for DuckDB queries (TIMESTAMP columns are naive UTC)
                    model_run_naive = _strip_tz(model_run_utc)

                    if not self._has_forecast_data(con, model_run_naive):
                        skipped += 1
                        continue

                    ref_time = model_run_utc + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)

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

                    # Score with Kalshi 2°F brackets if available, else 1°F fallback
                    if kalshi_brackets:
                        mapped = map_probs_to_kalshi_brackets(bracket_probs, kalshi_brackets)
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
                    ))

                # Progress logging every 200 days
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

        # Top-2: not applicable to Kalshi bracket scoring (only meaningful for 1°F)
        # Keep it based on the 1°F bracket_probs argmax vs actual
        top2_rate = top1_rate  # Placeholder — real top-2 needs the full probs

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

        return BacktestResult(
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
