"""AWC METAR ingestor — low-latency METAR + SPECI for settlement stations.

Uses the Aviation Weather Center (aviationweather.gov) API, which carries
both routine METARs and SPECI reports. IEM's ASOS feed was missing SPECIs
for non-airport stations like KNYC (Central Park).
"""

import asyncio
from datetime import datetime, timezone

import duckdb
import httpx
from loguru import logger

from core.constants import IEM_POLL_INTERVAL_SECONDS
from core.db import get_connection
from core.retry import retry_async
from services.ingestor import parse_t_group, parse_6h_max, parse_6h_min, _is_metar_stub

AWC_BASE_URL = "https://aviationweather.gov/api/data/metar"

# Settlement stations to poll
AWC_STATIONS = ["KNYC", "KPHL", "KMDW", "KMIA", "KLAX"]

USER_AGENT = "(alphatemp, contact@alphatemp.com)"


class IEMIngestor:
    """Polls AWC for METAR+SPECI on the 5 settlement stations.

    Named IEMIngestor for backwards compatibility with main.py imports,
    but now uses the AWC API which carries SPECIs that IEM lacks.
    """

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    @retry_async(max_retries=3, base_delay=2.0)
    async def _fetch_awc(self, client: httpx.AsyncClient, params: dict) -> list:
        """GET JSON from AWC METAR endpoint with retry."""
        resp = await client.get(
            AWC_BASE_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        return resp.json()

    async def poll_once(self) -> int:
        """Single poll cycle. Returns number of new rows inserted."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        params = {
            "ids": ",".join(AWC_STATIONS),
            "format": "json",
            "hours": "2",
            "taf": "false",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                data = await self._fetch_awc(client, params)
        except Exception as e:
            logger.warning(f"AWC METAR request failed after retries: {e}")
            return 0

        if not isinstance(data, list):
            logger.warning(f"AWC returned unexpected format: {type(data)}")
            return 0

        rows = self._parse_json(data, now)
        if not rows:
            logger.debug("AWC: no observations to insert")
            return 0

        con = get_connection(self.db_path)
        inserted = 0
        for row in rows:
            try:
                con.execute(
                    """INSERT INTO observations
                       (station_id, observed_at, temp_f, temp_c_tenth,
                        six_hr_max_c, six_hr_min_c, raw_metar, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [row["station_id"], row["observed_at"], row["temp_f"],
                     row["temp_c_tenth"], row["six_hr_max_c"], row["six_hr_min_c"],
                     row["raw_metar"], row["ingested_at"]],
                )
                inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate — first-to-arrive wins
        con.close()

        logger.info(f"AWC: inserted {inserted} new observations ({len(rows) - inserted} duplicates skipped)")
        return inserted

    def _parse_json(self, data: list, now: datetime) -> list[dict]:
        """Parse AWC JSON response into observation dicts."""
        rows = []

        for obs in data:
            station_id = obs.get("icaoId", "").strip()
            if station_id not in AWC_STATIONS:
                continue

            # Use obsTime (Unix epoch) — the actual observation time.
            # reportTime rounds hourly METARs to :00, which creates
            # phantom duplicates alongside Synoptic's real timestamps.
            obs_time = obs.get("obsTime")
            if obs_time is None:
                continue

            observed_at = datetime.utcfromtimestamp(obs_time)

            raw_ob = obs.get("rawOb", "").strip()

            temp_f = None
            temp_c_tenth = None
            six_hr_max_c = None
            six_hr_min_c = None

            if not _is_metar_stub(raw_ob):
                # T-group from raw METAR/SPECI string
                temp_c_tenth = parse_t_group(raw_ob)

                # 6-hour synoptic max/min
                six_hr_max_c = parse_6h_max(raw_ob)
                six_hr_min_c = parse_6h_min(raw_ob)

                # AWC provides temp in Celsius — convert to F
                temp_c = obs.get("temp")
                if temp_c is not None:
                    temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

            rows.append({
                "station_id": station_id,
                "observed_at": observed_at,
                "temp_f": temp_f,
                "temp_c_tenth": temp_c_tenth,
                "six_hr_max_c": six_hr_max_c,
                "six_hr_min_c": six_hr_min_c,
                "raw_metar": raw_ob,
                "ingested_at": now,
            })

        return rows

    async def run(self) -> None:
        """Run the AWC ingestor loop indefinitely."""
        logger.info(
            f"Starting AWC METAR+SPECI ingestor — polling {len(AWC_STATIONS)} settlement stations "
            f"every {IEM_POLL_INTERVAL_SECONDS}s"
        )
        while True:
            await self.poll_once()
            await asyncio.sleep(IEM_POLL_INTERVAL_SECONDS)
