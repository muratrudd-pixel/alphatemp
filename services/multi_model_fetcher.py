"""Live GFS + ECMWF forecast fetcher via Open-Meteo API.

Fills the same tables (forecasts, forecast_extended) that the backfill scripts
populate, so the feature builder gets all 23 features in live prediction.

Checks once per hour whether today's GFS/ECMWF data already exists in the DB.
If not, fetches from Open-Meteo forecast API and inserts.

Open-Meteo composites approximate the 00z run, and we store with model_run
at midnight UTC — matching the convention used by the backfill scripts and
expected by the feature builder (EXTRACT(HOUR FROM model_run) = 0).

Python 3.9 compatible.
"""

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import requests
from loguru import logger

from core.db import get_connection
from core.heartbeat import record_heartbeat

# Central Park
LAT = 40.7789
LON = -73.9692
STATION_ID = "KNYC"

# Open-Meteo forecast API (NOT the historical API)
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Check every 30 minutes whether we have today's data
POLL_INTERVAL_SECONDS = 1800

# Extended weather variables — must match backfill_openmeteo.py
EXTENDED_VARS = [
    "dewpoint_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "pressure_msl", "cloud_cover", "precipitation",
    "shortwave_radiation", "cape",
]

# Map Open-Meteo field names -> DB column names
_EXTENDED_COL_MAP = {
    "dewpoint_2m": "dewpoint_2m_f",
    "relative_humidity_2m": "humidity_2m",
    "wind_speed_10m": "wind_speed_10m",
    "wind_direction_10m": "wind_dir_10m",
    "wind_gusts_10m": "wind_gusts_10m",
    "pressure_msl": "pressure_msl",
    "cloud_cover": "cloud_cover",
    "precipitation": "precipitation",
    "shortwave_radiation": "shortwave_rad",
    "cape": "cape",
}


class MultiModelFetcher:
    """Fetches today's GFS and ECMWF forecasts from Open-Meteo."""

    def __init__(self, db_path="data/alphatemp.duckdb"):
        # type: (str) -> None
        self.db_path = db_path

    def _has_data_for_date(self, model, target_date):
        # type: (str, date) -> bool
        """Check if we already have forecast data for this model/date."""
        con = get_connection(self.db_path)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM forecasts "
                "WHERE model_name = ? AND model_run::DATE = ? "
                "AND EXTRACT(HOUR FROM model_run) = 0 AND station_id = ?",
                [model, target_date, STATION_ID],
            ).fetchone()
            return row[0] > 0 if row else False
        finally:
            con.close()

    def _has_extended_for_date(self, model, target_date):
        # type: (str, date) -> bool
        """Check if we already have extended data for this model/date."""
        con = get_connection(self.db_path)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM forecast_extended "
                "WHERE model_name = ? AND model_run::DATE = ? "
                "AND EXTRACT(HOUR FROM model_run) = 0 AND station_id = ?",
                [model, target_date, STATION_ID],
            ).fetchone()
            return row[0] > 0 if row else False
        finally:
            con.close()

    def _fetch_temp(self, model_id, target_date):
        # type: (str, date) -> dict
        """Fetch temperature forecast from Open-Meteo."""
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": model_id,
        }
        resp = requests.get(FORECAST_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fetch_extended(self, model_id, target_date):
        # type: (str, date) -> dict
        """Fetch extended variables from Open-Meteo."""
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(EXTENDED_VARS),
            "temperature_unit": "fahrenheit",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": model_id,
        }
        resp = requests.get(FORECAST_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _insert_temp(self, model, data):
        # type: (str, dict) -> int
        """Insert temperature forecast rows into DB."""
        hourly = data.get("hourly", {})
        times = hourly.get("temperature_2m") and hourly.get("time", [])
        temps_f = hourly.get("temperature_2m", [])
        if not times or not temps_f:
            return 0

        con = get_connection(self.db_path)
        inserted = 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            for ts_str, temp_f in zip(times, temps_f):
                if temp_f is None:
                    continue
                valid_at = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")
                model_run = datetime(valid_at.year, valid_at.month, valid_at.day, 0, 0)
                temp_c = round((temp_f - 32.0) * 5.0 / 9.0, 2)

                try:
                    con.execute(
                        "INSERT INTO forecasts "
                        "(station_id, model_run, valid_at, temp_f, temp_c, "
                        "ingested_at, model_name, fxx, is_spinup) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, FALSE)",
                        [STATION_ID, model_run, valid_at, temp_f, temp_c,
                         now, model, int((valid_at - model_run).total_seconds() // 3600)],
                    )
                    inserted += 1
                except duckdb.ConstraintException:
                    pass
        finally:
            con.close()
        return inserted

    def _insert_extended(self, model, data):
        # type: (str, dict) -> int
        """Insert extended variable rows into DB."""
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return 0

        var_arrays = {}  # type: dict
        for var_name in EXTENDED_VARS:
            var_arrays[var_name] = hourly.get(var_name, [])

        con = get_connection(self.db_path)
        inserted = 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            for i, ts_str in enumerate(times):
                valid_at = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")
                model_run = datetime(valid_at.year, valid_at.month, valid_at.day, 0, 0)

                vals = {}  # type: dict
                for var_name in EXTENDED_VARS:
                    arr = var_arrays[var_name]
                    val = arr[i] if i < len(arr) else None
                    vals[_EXTENDED_COL_MAP[var_name]] = val

                try:
                    con.execute(
                        "INSERT INTO forecast_extended "
                        "(station_id, model_run, valid_at, model_name, "
                        "dewpoint_2m_f, humidity_2m, wind_speed_10m, wind_dir_10m, "
                        "wind_gusts_10m, pressure_msl, cloud_cover, precipitation, "
                        "shortwave_rad, cape, ingested_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            STATION_ID, model_run, valid_at, model,
                            vals["dewpoint_2m_f"], vals["humidity_2m"],
                            vals["wind_speed_10m"], vals["wind_dir_10m"],
                            vals["wind_gusts_10m"], vals["pressure_msl"],
                            vals["cloud_cover"], vals["precipitation"],
                            vals["shortwave_rad"], vals["cape"],
                            now,
                        ],
                    )
                    inserted += 1
                except duckdb.ConstraintException:
                    pass
        finally:
            con.close()
        return inserted

    def fetch_model_for_date(self, model, model_id, target_date):
        # type: (str, str, date) -> int
        """Fetch both temp and extended data for one model/date. Returns rows inserted."""
        total = 0

        if not self._has_data_for_date(model, target_date):
            try:
                data = self._fetch_temp(model_id, target_date)
                count = self._insert_temp(model, data)
                total += count
                if count > 0:
                    logger.info("{} {} temp: {} rows inserted", model.upper(), target_date, count)
            except requests.RequestException as e:
                logger.warning("{} temp fetch failed for {}: {}", model.upper(), target_date, e)

        if not self._has_extended_for_date(model, target_date):
            try:
                data = self._fetch_extended(model_id, target_date)
                count = self._insert_extended(model, data)
                total += count
                if count > 0:
                    logger.info("{} {} extended: {} rows inserted", model.upper(), target_date, count)
            except requests.RequestException as e:
                logger.warning("{} extended fetch failed for {}: {}", model.upper(), target_date, e)

        return total

    def fetch_today(self):
        # type: () -> int
        """Fetch GFS and ECMWF for today (and tomorrow if needed)."""
        today = date.today()
        tomorrow = today + timedelta(days=1)
        total = 0

        # Fetch both models for today and tomorrow
        for model, model_id in [("gfs", "gfs_seamless"), ("ecmwf", "ecmwf_ifs")]:
            total += self.fetch_model_for_date(model, model_id, today)
            total += self.fetch_model_for_date(model, model_id, tomorrow)

        return total

    async def run(self):
        # type: () -> None
        """Run the multi-model fetcher loop."""
        logger.info("Starting GFS/ECMWF fetcher — polling every {}s", POLL_INTERVAL_SECONDS)
        loop = asyncio.get_event_loop()

        while True:
            try:
                cycle_start = time.monotonic()
                total = await loop.run_in_executor(None, self.fetch_today)
                duration = (time.monotonic() - cycle_start) * 1000
                record_heartbeat("MultiModelFetcher", duration_ms=duration)
                if total > 0:
                    logger.info("GFS/ECMWF fetch cycle: {} new rows", total)
            except Exception as e:
                record_heartbeat("MultiModelFetcher", duration_ms=0, status="error", error=str(e))
                logger.error("GFS/ECMWF fetch cycle failed: {}", e)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
