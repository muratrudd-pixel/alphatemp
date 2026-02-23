"""HRRR forecast fetcher via Herbie library."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS, FORECAST_POLL_INTERVAL_SECONDS
from core.db import get_connection


def get_recent_model_runs(
    ref_time: datetime, count: int = 3, delay_hours: int = 2
) -> List[datetime]:
    """Return the N most recent HRRR model run times, accounting for publication delay."""
    latest_hour = ref_time - timedelta(hours=delay_hours)
    latest_hour = latest_hour.replace(minute=0, second=0, microsecond=0)
    return [latest_hour - timedelta(hours=i) for i in range(count)]


class HRRRFetcher:
    """Fetches HRRR 2m temperature forecasts for settlement stations."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path
        self.stations = STATION_COORDS

    def _extract_nearest(self, msg, lat: float, lon: float) -> float:
        """Extract 2m temp at the nearest grid point to (lat, lon).

        pygrib messages provide latlons() as 2D arrays. HRRR uses negative
        longitudes (-134 to -60), matching standard convention.

        Longitude is weighted by cos(lat) to compensate for meridian
        convergence — at 40°N, 1° lon is ~22% shorter than 1° lat.
        """
        lat_grid, lon_grid = msg.latlons()
        cos_lat = np.cos(np.radians(lat))
        dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon) * cos_lat
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return float(msg.values[idx])

    def fetch_run(self, model_run: datetime, fxx_range: range = range(1, 19)) -> int:
        """Fetch a single HRRR run for all stations. Returns rows inserted."""
        inserted = 0
        con = get_connection(self.db_path)
        now = datetime.now(timezone.utc)

        for fxx in fxx_range:
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                )
                grib_path = H.download("TMP:2 m")
                grbs = pygrib.open(str(grib_path))
                msg = grbs.select(name="2 metre temperature")[0]
            except Exception as e:
                logger.debug(f"HRRR fxx={fxx} not available for {model_run}: {e}")
                continue

            valid_at = model_run + timedelta(hours=fxx)

            for stid, (lat, lon) in self.stations.items():
                try:
                    temp_k = self._extract_nearest(msg, lat, lon)
                    temp_c = round(temp_k - 273.15, 2)
                    temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                    con.execute(
                        """INSERT INTO forecasts
                           (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        [stid, model_run.replace(tzinfo=None), valid_at.replace(tzinfo=None),
                         temp_f, temp_c, now.replace(tzinfo=None)],
                    )
                    inserted += 1
                except duckdb.ConstraintException:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to extract point for {stid} fxx={fxx}: {e}")

        con.close()
        logger.info(f"HRRR {model_run.strftime('%Y-%m-%d %Hz')}: inserted {inserted} forecast points")
        return inserted

    async def fetch_latest(self) -> int:
        """Fetch the last 3 HRRR runs. Runs synchronous Herbie in executor."""
        runs = get_recent_model_runs(datetime.now(timezone.utc))
        total = 0
        loop = asyncio.get_event_loop()
        for run in runs:
            count = await loop.run_in_executor(None, self.fetch_run, run)
            total += count
        return total

    async def run(self) -> None:
        """Run the forecast fetcher loop indefinitely."""
        logger.info(f"Starting HRRR fetcher — polling every {FORECAST_POLL_INTERVAL_SECONDS}s")
        while True:
            try:
                await self.fetch_latest()
            except Exception as e:
                logger.error(f"HRRR fetch cycle failed: {e}")
            await asyncio.sleep(FORECAST_POLL_INTERVAL_SECONDS)
