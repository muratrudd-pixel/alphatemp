"""Bias Engine — real-time drift detection between observations and HRRR forecasts."""

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import duckdb
from loguru import logger
from scipy import stats

from core.constants import CITIES, STATION_COORDS
from core.db import get_connection
from core.heartbeat import record_heartbeat


def compute_drift(obs_temps: List[float], fcst_temps: List[float]) -> float:
    """Mean divergence between observed and forecast temperatures.
    Positive = observations warmer than forecast.
    """
    if not obs_temps or not fcst_temps or len(obs_temps) != len(fcst_temps):
        return 0.0
    diffs = [o - f for o, f in zip(obs_temps, fcst_temps)]
    return sum(diffs) / len(diffs)


def compute_slope_divergence(
    obs_temps: List[float], fcst_temps: List[float], hours: List[float]
) -> float:
    """Linear regression slope of the divergence time series.
    Positive slope = divergence is widening. Units: degrees F per hour.
    """
    if len(obs_temps) < 2:
        return 0.0
    diffs = [o - f for o, f in zip(obs_temps, fcst_temps)]
    slope, _, _, _, _ = stats.linregress(hours, diffs)
    return float(slope)


class BiasEngine:
    """Compares observed trajectories against HRRR forecast curves."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    def _get_forecast_curve(self, con, station_id: str, model_run) -> List[tuple]:
        rows = con.execute(
            "SELECT valid_at, temp_f FROM forecasts WHERE station_id = ? AND model_run = ? AND model_name = 'hrrr' ORDER BY valid_at",
            [station_id, model_run],
        ).fetchall()
        return rows

    def _get_observations_in_range(self, con, station_id: str, start, end) -> List[tuple]:
        rows = con.execute(
            "SELECT observed_at, temp_f FROM observations WHERE station_id = ? AND observed_at BETWEEN ? AND ? ORDER BY observed_at",
            [station_id, start, end],
        ).fetchall()
        return rows

    def _get_model_runs(self, con, station_id: str) -> List:
        rows = con.execute(
            "SELECT DISTINCT model_run FROM forecasts WHERE station_id = ? AND model_name = 'hrrr' ORDER BY model_run DESC LIMIT 3",
            [station_id],
        ).fetchall()
        return [r[0] for r in rows]

    def _interpolate_forecast_to_obs(self, forecast_curve: List[tuple], obs_times: List) -> List[Optional[float]]:
        """Linearly interpolate forecast curve to match observation timestamps."""
        if len(forecast_curve) < 2:
            return [None] * len(obs_times)

        fc_times = []
        for fc in forecast_curve:
            t = fc[0]
            fc_times.append(t.timestamp() if hasattr(t, 'timestamp') else float(t))
        fc_temps = [fc[1] for fc in forecast_curve]

        results = []
        for obs_t in obs_times:
            t = obs_t.timestamp() if hasattr(obs_t, 'timestamp') else float(obs_t)
            if t <= fc_times[0]:
                results.append(fc_temps[0])
            elif t >= fc_times[-1]:
                results.append(fc_temps[-1])
            else:
                for i in range(len(fc_times) - 1):
                    if fc_times[i] <= t <= fc_times[i + 1]:
                        frac = (t - fc_times[i]) / (fc_times[i + 1] - fc_times[i])
                        interp = fc_temps[i] + frac * (fc_temps[i + 1] - fc_temps[i])
                        results.append(interp)
                        break
                else:
                    results.append(None)
        return results

    def _compute_forecast_trend(self, con, station_id: str, model_runs: List) -> float:
        """How the forecasted daily high shifts across successive HRRR runs. Positive = successive runs forecasting higher."""
        if len(model_runs) < 2:
            return 0.0
        highs = []
        for run in model_runs:
            row = con.execute(
                "SELECT MAX(temp_f) FROM forecasts WHERE station_id = ? AND model_run = ? AND model_name = 'hrrr'",
                [station_id, run],
            ).fetchone()
            if row and row[0] is not None:
                highs.append(row[0])
        if len(highs) < 2:
            return 0.0
        highs.reverse()  # Chronological order
        deltas = [highs[i + 1] - highs[i] for i in range(len(highs) - 1)]
        return sum(deltas) / len(deltas)

    def _compute_confidence(self, obs_count: int, drift_values: List[float]) -> float:
        """Confidence score 0.0-1.0 based on observation count and divergence stability."""
        obs_factor = min(obs_count / 20.0, 1.0)
        if len(drift_values) < 2:
            stability_factor = 0.5
        else:
            mean_drift = sum(drift_values) / len(drift_values)
            variance = sum((d - mean_drift) ** 2 for d in drift_values) / len(drift_values)
            stability_factor = max(0.0, 1.0 - variance / 4.0)
        return round(obs_factor * 0.4 + stability_factor * 0.6, 3)

    def calculate_all(self, ref_time: Optional[datetime] = None) -> Dict[str, dict]:
        """Calculate drift reports for all cities."""
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)

        con = get_connection(self.db_path)
        reports = {}

        for city_name, city_cfg in CITIES.items():
            stid = city_cfg["settlement"]
            model_runs = self._get_model_runs(con, stid)
            if not model_runs:
                logger.debug(f"No forecast data for {city_name}/{stid}, skipping")
                continue

            latest_run = model_runs[0]
            forecast_curve = self._get_forecast_curve(con, stid, latest_run)
            if not forecast_curve:
                continue

            fc_start = forecast_curve[0][0]
            fc_end = forecast_curve[-1][0]
            observations = self._get_observations_in_range(con, stid, fc_start, fc_end)
            if not observations:
                logger.debug(f"No observations in forecast range for {city_name}")
                continue

            obs_times = [o[0] for o in observations]
            obs_temps = [o[1] for o in observations]

            fcst_interp = self._interpolate_forecast_to_obs(forecast_curve, obs_times)
            paired = [(o, f) for o, f in zip(obs_temps, fcst_interp) if f is not None]
            if not paired:
                continue

            obs_paired = [p[0] for p in paired]
            fcst_paired = [p[1] for p in paired]

            t0 = obs_times[0].timestamp() if hasattr(obs_times[0], 'timestamp') else float(obs_times[0])
            hours = []
            for ot in obs_times[:len(paired)]:
                t = ot.timestamp() if hasattr(ot, 'timestamp') else float(ot)
                hours.append((t - t0) / 3600.0)

            drift_score = compute_drift(obs_paired, fcst_paired)
            slope_div = compute_slope_divergence(obs_paired, fcst_paired, hours)
            forecast_trend = self._compute_forecast_trend(con, stid, model_runs)

            fcst_max_row = con.execute(
                "SELECT MAX(temp_f) FROM forecasts WHERE station_id = ? AND model_run = ? AND model_name = 'hrrr'",
                [stid, latest_run],
            ).fetchone()
            fcst_high = fcst_max_row[0] if fcst_max_row else 0.0

            # Hours remaining until end of forecast curve
            ref_ts = ref_time.replace(tzinfo=None).timestamp() if ref_time.tzinfo else ref_time.timestamp()
            fc_end_ts = fc_end.timestamp() if hasattr(fc_end, 'timestamp') else float(fc_end)
            hours_remaining = max(0, (fc_end_ts - ref_ts) / 3600.0)
            projected_high = round(fcst_high + drift_score + slope_div * hours_remaining, 1)

            drift_values = [o - f for o, f in paired]
            confidence = self._compute_confidence(len(observations), drift_values)

            reports[city_name] = {
                "model_run": latest_run,
                "drift_score": round(drift_score, 2),
                "slope_divergence": round(slope_div, 3),
                "forecast_trend": round(forecast_trend, 2),
                "confidence": confidence,
                "projected_high": projected_high,
            }

        con.close()
        return reports

    def calculate_and_store(self, ref_time: Optional[datetime] = None) -> int:
        """Calculate drift for all cities and store in drift_signals table."""
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)
        reports = self.calculate_all(ref_time)
        if not reports:
            return 0
        con = get_connection(self.db_path)
        stored = 0
        for city_name, report in reports.items():
            try:
                con.execute(
                    """INSERT INTO drift_signals
                       (city, calculated_at, model_run, drift_score, slope_divergence,
                        forecast_trend, confidence, projected_high)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [city_name, ref_time.replace(tzinfo=None), report["model_run"],
                     report["drift_score"], report["slope_divergence"],
                     report["forecast_trend"], report["confidence"],
                     report["projected_high"]],
                )
                stored += 1
            except duckdb.ConstraintException:
                pass
        con.close()
        logger.info(f"Stored {stored} drift signals")
        return stored

    async def run(self) -> None:
        """Run bias engine on a 60-second loop."""
        import asyncio
        from core.constants import POLL_INTERVAL_SECONDS
        logger.info("Starting Bias Engine — calculating drift every 60s")
        while True:
            try:
                cycle_start = time.monotonic()
                self.calculate_and_store()
                record_heartbeat(
                    "BiasEngine",
                    duration_ms=(time.monotonic() - cycle_start) * 1000,
                )
            except Exception as e:
                record_heartbeat("BiasEngine", duration_ms=0, status="error", error=str(e))
                logger.error(f"Bias engine cycle failed: {e}")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
