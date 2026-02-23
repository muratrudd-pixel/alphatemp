"""HRRR forecast fetcher via Herbie library."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

import duckdb
import numpy as np
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

    def _extract_nearest(self, ds, lat: float, lon: float) -> float:
        """Extract 2m temp at the nearest grid point to (lat, lon).

        Herbie returns xarray datasets where latitude/longitude are 2D
        auxiliary coordinates on (y, x) dims. HRRR uses 0-360° longitude
        convention, so we convert negative longitudes before lookup.
        """
        lat_grid = ds["t2m"].coords["latitude"].values
        lon_grid = ds["t2m"].coords["longitude"].values
        lon_lookup = lon % 360  # Convert -87.75 → 272.25 to match HRRR grid
        dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon_lookup)
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return float(ds["t2m"].isel(y=idx[0], x=idx[1]).values)

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
                ds = H.xarray("TMP:2 m")
            except Exception as e:
                logger.debug(f"HRRR fxx={fxx} not available for {model_run}: {e}")
                continue

            valid_at = model_run + timedelta(hours=fxx)

            for stid, (lat, lon) in self.stations.items():
                try:
                    temp_k = self._extract_nearest(ds, lat, lon)
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
