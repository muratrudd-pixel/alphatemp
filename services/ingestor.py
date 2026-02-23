"""Synoptic API ingestor with METAR T-group parser."""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import duckdb
import httpx
import polars as pl
from loguru import logger

from core.constants import get_all_station_ids, POLL_INTERVAL_SECONDS
from core.db import get_connection
from core.retry import retry_async

T_GROUP_PATTERN = re.compile(r"\bT(\d)(\d{3})")
METAR_TIMESTAMP = re.compile(r"\d{6}Z")


def _is_metar_stub(metar_str: str) -> bool:
    """True if METAR is a bare stub with no weather data.

    Real METARs always contain a Zulu timestamp (e.g. 230505Z).
    Stubs like 'METAR KMIA AUTO' lack one and carry garbage temps.
    """
    if not metar_str:
        return True
    return METAR_TIMESTAMP.search(metar_str) is None


def parse_t_group(metar_remarks: str) -> Optional[float]:
    """Extract high-resolution Celsius temperature from METAR T-group.

    The T-group in METAR remarks encodes temperature to 0.1 C precision.
    Format: T[sign][temp_tenths][sign][dewpoint_tenths]
    Sign: 0 = positive, 1 = negative
    Example: T0228 -> +22.8 C, T1005 -> -0.5 C
    """
    match = T_GROUP_PATTERN.search(metar_remarks)
    if not match:
        return None
    sign = -1 if match.group(1) == "1" else 1
    temp = int(match.group(2)) / 10.0
    return sign * temp


SYNOPTIC_BASE_URL = "https://api.synopticdata.com/v2/stations/timeseries"


class SynopticIngestor:
    """Polls Synoptic API for 1-minute ASOS observations and stores in DuckDB."""

    def __init__(self, token: str, db_path: str = "data/alphatemp.duckdb"):
        self.token = token
        self.db_path = db_path
        self.stations = get_all_station_ids()

    @retry_async(max_retries=3, base_delay=2.0)
    async def _fetch_synoptic(self, params: dict) -> dict:
        """Fetch data from Synoptic API with automatic retry on failure."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(SYNOPTIC_BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()

    async def poll_once(self) -> int:
        """Execute a single poll cycle. Returns number of new rows inserted."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=90)
        params = {
            "stid": ",".join(self.stations),
            "start": start.strftime("%Y%m%d%H%M"),
            "end": now.strftime("%Y%m%d%H%M"),
            "vars": "air_temp,metar",
            "obtimezone": "UTC",
            "token": self.token,
        }

        try:
            data = await self._fetch_synoptic(params)
        except Exception as e:
            logger.warning(f"Synoptic API request failed after retries: {e}")
            return 0

        summary = data.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1 or "STATION" not in data:
            logger.warning(f"Synoptic API returned no data: {summary.get('RESPONSE_MESSAGE', 'unknown')}")
            return 0

        rows = []

        for station in data["STATION"]:
            stid = station["STID"]
            obs = station.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            temps = obs.get("air_temp_set_1", [])
            metars = obs.get("metar_set_1", [])

            for i, dt_str in enumerate(times):
                temp_f_val = temps[i] if i < len(temps) else None
                metar_str = metars[i] if i < len(metars) else ""

                # Convert Celsius API value to Fahrenheit
                temp_f = round(temp_f_val * 9.0 / 5.0 + 32.0, 1) if temp_f_val is not None else None

                # Parse T-group for high-res Celsius
                temp_c_tenth = parse_t_group(metar_str)

                # Stub METARs carry garbage temps — discard them
                if _is_metar_stub(metar_str):
                    temp_f = None
                    temp_c_tenth = None

                rows.append({
                    "station_id": stid,
                    "observed_at": dt_str,
                    "temp_f": temp_f,
                    "temp_c_tenth": temp_c_tenth,
                    "raw_metar": metar_str,
                    "ingested_at": now.isoformat(),
                })

        if not rows:
            logger.debug("No observations to insert")
            return 0

        df = pl.DataFrame(rows).with_columns(
            pl.col("observed_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
            pl.col("ingested_at").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%:z"),
        )

        con = get_connection(self.db_path)
        inserted = 0
        for row in df.iter_rows(named=True):
            try:
                con.execute(
                    """INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [row["station_id"], row["observed_at"], row["temp_f"],
                     row["temp_c_tenth"], row["raw_metar"], row["ingested_at"]],
                )
                inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate — skip silently
        con.close()

        logger.info(f"Inserted {inserted} new observations ({len(rows) - inserted} duplicates skipped)")
        return inserted

    async def _recover_gap(self) -> None:
        """Check for data gaps on startup and log accordingly."""
        con = get_connection(self.db_path)
        try:
            result = con.execute("SELECT MAX(observed_at) FROM observations").fetchone()
        finally:
            con.close()

        if result is None or result[0] is None:
            logger.info("No existing observations — starting fresh")
            return

        last_obs = result[0]
        if not isinstance(last_obs, datetime):
            last_obs = datetime.fromisoformat(str(last_obs))
        if last_obs.tzinfo is None:
            last_obs = last_obs.replace(tzinfo=timezone.utc)

        gap_minutes = (datetime.now(timezone.utc) - last_obs).total_seconds() / 60

        if gap_minutes > 90:
            logger.warning(
                f"Data gap of {gap_minutes:.0f} min detected (last obs: {last_obs.isoformat()}). "
                f"Gap exceeds 90-min lookback — some data is unrecoverable from the realtime API."
            )
        elif gap_minutes > 15:
            logger.info(
                f"Data gap of {gap_minutes:.0f} min detected (last obs: {last_obs.isoformat()}). "
                f"Extended 90-min lookback will recover it on next poll."
            )
        else:
            logger.info(f"Last observation {gap_minutes:.0f} min ago — no gap to recover")

    async def run(self) -> None:
        """Run the ingestor loop indefinitely."""
        await self._recover_gap()
        logger.info(f"Starting Synoptic ingestor — polling {len(self.stations)} stations every {POLL_INTERVAL_SECONDS}s")
        while True:
            await self.poll_once()
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
