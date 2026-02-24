"""HRRR forecast fetcher via Herbie library.

Polls for new HRRR runs every 2 minutes. Only fetches runs not already
in the database — checks the latest stored model_run and works forward.
HRRR publishes hourly; data typically lands on AWS ~45-90 min after run time.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS
from core.db import get_connection

# Poll every 2 minutes — frequent short checks to catch new runs ASAP
FETCH_INTERVAL_SECONDS = 120

# How many hours back to look on cold start (empty DB)
COLD_START_LOOKBACK_HOURS = 6


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

    def _get_latest_stored_run(self) -> Optional[datetime]:
        """Return the latest model_run already in the DB, or None."""
        con = get_connection(self.db_path)
        row = con.execute("SELECT MAX(model_run) FROM forecasts").fetchone()
        con.close()
        if row and row[0] is not None:
            mr = row[0]
            if mr.tzinfo is None:
                mr = mr.replace(tzinfo=timezone.utc)
            return mr
        return None

    def _get_missing_runs(self) -> List[datetime]:
        """Return model run times we should try to fetch, newest first.

        Checks DB for the latest stored run, then returns every hourly
        run between that and now. On cold start, looks back 6 hours.
        """
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        latest_stored = self._get_latest_stored_run()

        if latest_stored is None:
            # Cold start — grab the last N hours
            start_hour = current_hour - timedelta(hours=COLD_START_LOOKBACK_HOURS)
        else:
            # Start from the hour after the latest stored run
            start_hour = latest_stored + timedelta(hours=1)

        # Build list of candidate runs from start_hour up to current_hour
        runs = []
        candidate = current_hour
        while candidate >= start_hour:
            runs.append(candidate)
            candidate -= timedelta(hours=1)

        return runs  # newest first

    def fetch_run(self, model_run: datetime, fxx_range: range = range(1, 19)) -> int:
        """Fetch a single HRRR run for all stations. Returns rows inserted."""
        inserted = 0
        con = get_connection(self.db_path)
        now = datetime.now(timezone.utc)
        consecutive_misses = 0

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
                consecutive_misses = 0
            except Exception as e:
                consecutive_misses += 1
                if consecutive_misses >= 2:
                    logger.debug(f"HRRR {model_run.strftime('%Hz')} not published yet (fxx={fxx}), skipping run")
                    break
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
        if inserted > 0:
            logger.info(f"HRRR {model_run.strftime('%Y-%m-%d %Hz')}: inserted {inserted} forecast points")
        return inserted

    async def fetch_latest(self) -> int:
        """Fetch any HRRR runs not already in the DB. Newest first, stops on first miss."""
        runs = self._get_missing_runs()

        if not runs:
            return 0

        total = 0
        loop = asyncio.get_event_loop()
        for run in runs:
            count = await loop.run_in_executor(None, self.fetch_run, run)
            if count == 0 and run == runs[0]:
                # Newest run not published yet — normal, just wait
                logger.debug(f"HRRR {run.strftime('%Hz')} not available yet")
                break
            total += count
        return total

    async def run(self) -> None:
        """Run the forecast fetcher loop indefinitely."""
        logger.info(
            f"Starting HRRR fetcher — polling every {FETCH_INTERVAL_SECONDS}s, "
            f"no delay, DB-aware"
        )
        while True:
            try:
                await self.fetch_latest()
            except Exception as e:
                logger.error(f"HRRR fetch cycle failed: {e}")
            await asyncio.sleep(FETCH_INTERVAL_SECONDS)
