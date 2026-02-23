"""Historical bias model — computes per-station forecast error from backfilled data."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

from core.constants import STATION_COORDS
from core.db import get_connection


@dataclass
class StationBias:
    """Per-station historical bias statistics."""
    station_id: str
    mean_bias: float    # Mean(forecast_high - observed_high), positive = forecast runs hot
    std_error: float    # Std dev of daily peak errors
    sample_days: int    # Number of days with paired data


class BiasModel:
    """Computes per-station systematic forecast error from historical data.

    For each station, for each day with a 12z HRRR run:
      - peak_error = MAX(forecast) - MAX(observations) for that day's window
    Aggregates across all days to get mean_bias and std_error.
    Only includes days with sufficient observation coverage.
    """

    # Minimum observations in the forecast window to consider a day "complete".
    # With 1-min ASOS data across a 17-hour window, a full day has ~1020 obs.
    # 100 obs (~2 hours of coverage) filters out sparse/missing days.
    # KNYC (Central Park) only reports hourly METARs (~17 obs per window),
    # so it gets a lower threshold.
    MIN_OBS_COUNT = 100
    MIN_OBS_OVERRIDES = {"KNYC": 12}

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    def compute_station_bias(self, station_id: str) -> Optional[StationBias]:
        """Compute bias stats for a single station from all available paired data."""
        con = get_connection(self.db_path)

        # Get all distinct 12z model runs for this station
        runs = con.execute(
            """SELECT DISTINCT model_run FROM forecasts
               WHERE station_id = ? AND EXTRACT(HOUR FROM model_run) = 12
               ORDER BY model_run""",
            [station_id],
        ).fetchall()

        if not runs:
            con.close()
            return None

        peak_errors = []

        for (model_run,) in runs:
            # Forecast high for this run
            fcst_row = con.execute(
                "SELECT MAX(temp_f) FROM forecasts WHERE station_id = ? AND model_run = ?",
                [station_id, model_run],
            ).fetchone()
            fcst_high = fcst_row[0] if fcst_row and fcst_row[0] is not None else None
            if fcst_high is None:
                continue

            # Get the forecast time window
            window = con.execute(
                "SELECT MIN(valid_at), MAX(valid_at) FROM forecasts WHERE station_id = ? AND model_run = ?",
                [station_id, model_run],
            ).fetchone()
            if not window or window[0] is None:
                continue

            # Observed high in the same window — require complete day
            obs_row = con.execute(
                "SELECT MAX(temp_f), COUNT(*) FROM observations "
                "WHERE station_id = ? AND observed_at BETWEEN ? AND ?",
                [station_id, window[0], window[1]],
            ).fetchone()
            obs_high = obs_row[0] if obs_row and obs_row[0] is not None else None
            obs_count = obs_row[1] if obs_row else 0
            min_obs = self.MIN_OBS_OVERRIDES.get(station_id, self.MIN_OBS_COUNT)
            if obs_high is None or obs_count < min_obs:
                continue

            peak_errors.append(fcst_high - obs_high)

        con.close()

        if not peak_errors:
            return None

        n = len(peak_errors)
        mean_bias = sum(peak_errors) / n
        variance = sum((e - mean_bias) ** 2 for e in peak_errors) / (n - 1) if n > 1 else 0.0
        std_error = variance ** 0.5

        return StationBias(
            station_id=station_id,
            mean_bias=round(mean_bias, 2),
            std_error=round(std_error, 2),
            sample_days=n,
        )

    def compute_all(self) -> Dict[str, StationBias]:
        """Compute bias stats for all settlement stations."""
        results = {}
        for station_id in STATION_COORDS:
            bias = self.compute_station_bias(station_id)
            if bias:
                results[station_id] = bias
                logger.info(
                    f"  {station_id}: mean_bias={bias.mean_bias:+.2f}°F, "
                    f"std_error={bias.std_error:.2f}°F, sample_days={bias.sample_days}"
                )
            else:
                logger.warning(f"  {station_id}: no paired data available")
        return results

    def compute_and_store(self) -> int:
        """Compute bias for all stations and store in station_bias table."""
        biases = self.compute_all()
        if not biases:
            return 0

        con = get_connection(self.db_path)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        stored = 0

        for station_id, bias in biases.items():
            con.execute(
                """INSERT INTO station_bias
                   (station_id, calculated_at, mean_bias, std_error, sample_days)
                   VALUES (?, ?, ?, ?, ?)""",
                [station_id, now, bias.mean_bias, bias.std_error, bias.sample_days],
            )
            stored += 1

        con.close()
        logger.info(f"Stored bias stats for {stored} stations")
        return stored


if __name__ == "__main__":
    from core.db import init_db
    init_db()
    model = BiasModel()
    model.compute_and_store()
