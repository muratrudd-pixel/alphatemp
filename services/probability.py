"""Probability Engine — narrowing Gaussian distributions for temperature bracket pricing."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger
from scipy.stats import norm

from core.constants import CITIES, STATION_COORDS, POLL_INTERVAL_SECONDS
from core.db import get_connection
from services.bias_model import BiasModel, StationBias


@dataclass
class CityForecast:
    """Probability distribution for a city's daily high temperature."""
    city: str
    center: float           # Bias-adjusted expected high
    std: float              # Current uncertainty (narrows through the day)
    interval_90: tuple      # 90% confidence interval (low, high)
    bracket_probs: Dict[int, float]  # P(high = k°F) for each integer bracket


class ProbabilityEngine:
    """Produces narrowing Gaussian distributions per city per cycle.

    Center = forecast_high - historical_bias + same_day_drift
    Std = historical_std_error * time_factor * stability_factor * convergence_factor
    """

    CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path
        self._bias_cache: Dict[str, StationBias] = {}
        self._cache_loaded_at: Optional[datetime] = None

    def load_bias_cache(self) -> None:
        """Load the latest historical bias stats for all stations."""
        con = get_connection(self.db_path)
        for station_id in STATION_COORDS:
            row = con.execute(
                """SELECT mean_bias, std_error, sample_days
                   FROM station_bias
                   WHERE station_id = ?
                   ORDER BY calculated_at DESC LIMIT 1""",
                [station_id],
            ).fetchone()
            if row:
                self._bias_cache[station_id] = StationBias(
                    station_id=station_id,
                    mean_bias=row[0],
                    std_error=row[1],
                    sample_days=row[2],
                )
        con.close()
        self._cache_loaded_at = datetime.now(timezone.utc)
        logger.info(f"Loaded bias cache for {len(self._bias_cache)} stations")

    def _get_forecast_high(self, con, station_id: str) -> Optional[float]:
        """Get the forecasted high from the latest model run."""
        row = con.execute(
            """SELECT MAX(f.temp_f) FROM forecasts f
               WHERE f.station_id = ?
               AND f.model_run = (
                   SELECT MAX(model_run) FROM forecasts WHERE station_id = ?
               )""",
            [station_id, station_id],
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def _get_drift_score(self, con, city: str) -> float:
        """Get the latest drift score for a city."""
        row = con.execute(
            """SELECT drift_score FROM drift_signals
               WHERE city = ?
               ORDER BY calculated_at DESC LIMIT 1""",
            [city],
        ).fetchone()
        return row[0] if row else 0.0

    def _compute_time_factor(self, ref_time: datetime) -> float:
        """Time-based uncertainty reduction: 1.0 at 8am ET -> 0.3 by 3pm ET.

        More hours remaining = more uncertainty.
        Uses zoneinfo for correct DST handling.
        """
        from core.timezone import utc_to_et_hour
        et_hour = utc_to_et_hour(ref_time)
        if et_hour <= 8:
            return 1.0
        elif et_hour >= 15:
            return 0.3
        else:
            # Linear interpolation from 1.0 at 8am to 0.3 at 3pm
            return 1.0 - 0.7 * (et_hour - 8) / 7.0

    def _compute_stability_factor(self, con, city: str) -> float:
        """Low drift variance across recent signals = tighter distribution."""
        rows = con.execute(
            """SELECT drift_score FROM drift_signals
               WHERE city = ?
               ORDER BY calculated_at DESC LIMIT 10""",
            [city],
        ).fetchall()

        if len(rows) < 2:
            return 1.0

        scores = [r[0] for r in rows]
        mean_d = sum(scores) / len(scores)
        variance = sum((s - mean_d) ** 2 for s in scores) / len(scores)

        # Low variance -> factor near 0.5, high variance -> factor near 1.5
        return max(0.5, min(1.5, 0.5 + variance))

    def _compute_convergence_factor(self, con, station_id: str) -> float:
        """Successive HRRR runs agreeing = tighter distribution."""
        rows = con.execute(
            """SELECT model_run, MAX(temp_f) as fcst_high
               FROM forecasts
               WHERE station_id = ?
               GROUP BY model_run
               ORDER BY model_run DESC
               LIMIT 3""",
            [station_id],
        ).fetchall()

        if len(rows) < 2:
            return 1.0

        highs = [r[1] for r in rows if r[1] is not None]
        if len(highs) < 2:
            return 1.0

        # Spread of recent forecast highs
        spread = max(highs) - min(highs)

        # Tight convergence (< 1°F spread) -> 0.6, wide spread (> 4°F) -> 1.3
        if spread < 1.0:
            return 0.6
        elif spread > 4.0:
            return 1.3
        else:
            return 0.6 + 0.7 * (spread - 1.0) / 3.0

    def _compute_bracket_probs(self, center: float, std: float, radius: int = 15) -> Dict[int, float]:
        """Compute P(high = k°F) for integer brackets around the center.

        P(bracket=k) = Phi((k+0.5 - center)/std) - Phi((k-0.5 - center)/std)

        Radius of 15 ensures coverage across the Kalshi bracket range even when
        the model center and market center disagree.
        """
        if std <= 0:
            # Degenerate: all probability on nearest integer
            nearest = round(center)
            return {nearest: 1.0}

        center_int = round(center)
        probs = {}

        for k in range(center_int - radius, center_int + radius + 1):
            p = norm.cdf((k + 0.5 - center) / std) - norm.cdf((k - 0.5 - center) / std)
            if p > 0.0001:  # Skip negligible brackets
                probs[k] = round(p, 4)

        # Normalize to ensure probs sum to ~1.0 (rounding correction)
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total, 4) for k, v in probs.items()}

        return probs

    def _refresh_cache_if_stale(self) -> None:
        """Reload bias cache if older than TTL."""
        if self._cache_loaded_at is None:
            self.load_bias_cache()
            return
        age = (datetime.now(timezone.utc) - self._cache_loaded_at).total_seconds()
        if age >= self.CACHE_TTL_SECONDS:
            logger.info("Bias cache stale — refreshing")
            self.load_bias_cache()

    def calculate_city(self, city: str, ref_time: Optional[datetime] = None) -> Optional[CityForecast]:
        """Produce a probability distribution for one city."""
        self._refresh_cache_if_stale()

        if ref_time is None:
            ref_time = datetime.now(timezone.utc)

        station_id = CITIES[city]["settlement"]
        con = get_connection(self.db_path)

        # Get forecast high
        fcst_high = self._get_forecast_high(con, station_id)
        if fcst_high is None:
            con.close()
            return None

        # Get historical bias
        bias = self._bias_cache.get(station_id)
        historical_bias = bias.mean_bias if bias else 0.0
        historical_std = bias.std_error if bias else 2.0  # Conservative default

        # Get same-day drift
        drift = self._get_drift_score(con, city)

        # Center: adjust forecast by historical bias and current drift
        center = round(fcst_high - historical_bias + drift, 1)

        # Std: historical error scaled by time, stability, and convergence
        time_factor = self._compute_time_factor(ref_time)
        stability_factor = self._compute_stability_factor(con, city)
        convergence_factor = self._compute_convergence_factor(con, station_id)

        std = round(max(0.3, historical_std * time_factor * stability_factor * convergence_factor), 2)

        con.close()

        # 90% confidence interval
        z90 = 1.645
        interval_90 = (round(center - z90 * std, 1), round(center + z90 * std, 1))

        # Bracket probabilities
        bracket_probs = self._compute_bracket_probs(center, std)

        return CityForecast(
            city=city,
            center=center,
            std=std,
            interval_90=interval_90,
            bracket_probs=bracket_probs,
        )

    def calculate_all(self, ref_time: Optional[datetime] = None) -> Dict[str, CityForecast]:
        """Produce distributions for all cities."""
        results = {}
        for city in CITIES:
            forecast = self.calculate_city(city, ref_time)
            if forecast:
                results[city] = forecast
                top_brackets = sorted(forecast.bracket_probs.items(), key=lambda x: -x[1])[:3]
                bracket_str = ", ".join(f"{k}°F={v:.0%}" for k, v in top_brackets)
                logger.info(
                    f"  {city}: center={forecast.center}°F, std={forecast.std}, "
                    f"90%=[{forecast.interval_90[0]}, {forecast.interval_90[1]}] | {bracket_str}"
                )
        return results

    async def run(self) -> None:
        """Run probability engine on the same loop as BiasEngine."""
        logger.info("Starting Probability Engine")
        self.load_bias_cache()
        while True:
            try:
                self.calculate_all()
            except Exception as e:
                logger.error(f"Probability engine cycle failed: {e}")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
