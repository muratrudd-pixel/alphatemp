"""HRRR forecast fetcher via Herbie library.

Polls for new HRRR runs every 2 minutes. Only fetches runs not already
in the database — checks the latest stored model_run and works forward.
HRRR publishes hourly; data typically lands on AWS ~45-90 min after run time.
"""

import asyncio
import signal as _signal
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS
from core.db import get_connection
from core.heartbeat import record_heartbeat

# Poll every 2 minutes — frequent short checks to catch new runs ASAP
FETCH_INTERVAL_SECONDS = 120

# Timeout for individual Herbie.download() calls — prevents hanging on
# missing GRIB files that block the entire async event loop.
HERBIE_TIMEOUT_SECONDS = 45


class _HerbieTimeout(Exception):
    """Raised when Herbie.download() exceeds timeout."""
    pass


def _alarm_handler(signum, frame):
    raise _HerbieTimeout("Herbie download timed out")

# How many hours back to look on cold start — 24h ensures today's 00z
# is always captured even if the system starts late in the day
COLD_START_LOOKBACK_HOURS = 24

# Minimum distinct fxx hours for a run to be considered "complete enough"
# to skip on retry. Feature builder needs fxx 5-18 for 00z, so 10 is a
# reasonable threshold that catches partially-fetched runs.
MIN_COMPLETE_FXX = 10


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
        row = con.execute("SELECT MAX(model_run) FROM forecasts WHERE model_name = 'hrrr'").fetchone()
        con.close()
        if row and row[0] is not None:
            mr = row[0]
            if mr.tzinfo is None:
                mr = mr.replace(tzinfo=timezone.utc)
            return mr
        return None

    def _get_incomplete_runs(self) -> List[datetime]:
        """Return recent runs with fewer than MIN_COMPLETE_FXX distinct fxx hours.

        These runs were partially fetched (e.g., fxx 1-2 available but 3-18
        not yet published) and need to be retried to fill in remaining hours.
        """
        con = get_connection(self.db_path)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=COLD_START_LOOKBACK_HOURS)
        rows = con.execute("""
            SELECT model_run, COUNT(DISTINCT fxx) as fxx_count
            FROM forecasts
            WHERE model_name = 'hrrr'
              AND model_run >= ?
            GROUP BY model_run
            HAVING COUNT(DISTINCT fxx) < ?
        """, [cutoff.replace(tzinfo=None), MIN_COMPLETE_FXX]).fetchall()
        con.close()

        result = []
        for row in rows:
            mr = row[0]
            if mr.tzinfo is None:
                mr = mr.replace(tzinfo=timezone.utc)
            result.append(mr)
        return result

    def _get_stored_fxx(self, model_run: datetime) -> set:
        """Return set of fxx values already stored for this run."""
        con = get_connection(self.db_path)
        rows = con.execute("""
            SELECT DISTINCT fxx FROM forecasts
            WHERE model_name = 'hrrr' AND model_run = ?
        """, [model_run.replace(tzinfo=None)]).fetchall()
        con.close()
        return {row[0] for row in rows}

    def _get_missing_runs(self) -> List[datetime]:
        """Return model run times we should try to fetch, oldest first.

        Checks DB for the latest stored run, then returns every hourly
        run between that and now. On cold start, looks back 24 hours.
        Oldest first so the consecutive-empty heuristic in fetch_latest
        doesn't prematurely skip available older runs when recent ones
        haven't been published yet.
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
        candidate = start_hour
        while candidate <= current_hour:
            runs.append(candidate)
            candidate += timedelta(hours=1)

        # Also retry runs that were partially fetched (e.g., only fxx 1-2
        # because higher hours weren't published yet on the first attempt)
        for incomplete_run in self._get_incomplete_runs():
            if incomplete_run not in runs:
                runs.append(incomplete_run)

        runs.sort()  # oldest first
        return runs

    def fetch_run(self, model_run: datetime, fxx_range: range = range(1, 19)) -> int:
        """Fetch a single HRRR run for all stations. Returns rows inserted."""
        inserted = 0
        con = get_connection(self.db_path)
        now = datetime.now(timezone.utc)
        consecutive_misses = 0
        stored_fxx = self._get_stored_fxx(model_run)

        for fxx in fxx_range:
            if fxx in stored_fxx:
                consecutive_misses = 0  # run exists, reset counter
                continue
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                )
                # Set alarm to prevent indefinite hang on missing GRIB
                old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
                _signal.alarm(HERBIE_TIMEOUT_SECONDS)
                try:
                    grib_path = H.download("TMP:2 m")
                finally:
                    _signal.alarm(0)  # cancel alarm
                    _signal.signal(_signal.SIGALRM, old_handler)
                grbs = pygrib.open(str(grib_path))
                msg = grbs.select(name="2 metre temperature")[0]
                consecutive_misses = 0
            except _HerbieTimeout:
                consecutive_misses += 1
                logger.warning(f"HRRR fxx={fxx} timed out after {HERBIE_TIMEOUT_SECONDS}s for {model_run}")
                if consecutive_misses >= 2:
                    break
                continue
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
                           (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name, fxx, is_spinup)
                           VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)""",
                        [stid, model_run.replace(tzinfo=None), valid_at.replace(tzinfo=None),
                         temp_f, temp_c, now.replace(tzinfo=None), fxx, fxx <= 3],
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
        consecutive_empty = 0
        for run in runs:
            count = await loop.run_in_executor(None, self.fetch_run, run)
            if count == 0:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    # 3 consecutive misses — remaining runs likely unavailable too
                    break
                continue
            consecutive_empty = 0
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
                cycle_start = time.monotonic()
                await self.fetch_latest()
                record_heartbeat(
                    "HRRRFetcher",
                    duration_ms=(time.monotonic() - cycle_start) * 1000,
                )
            except Exception as e:
                record_heartbeat("HRRRFetcher", duration_ms=0, status="error", error=str(e))
                logger.error(f"HRRR fetch cycle failed: {e}")
            await asyncio.sleep(FETCH_INTERVAL_SECONDS)
