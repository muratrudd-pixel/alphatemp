"""IEM ASOS ingestor — low-latency METAR + SPECI for settlement stations."""

import asyncio
import csv
import io
from datetime import datetime, timedelta, timezone

import duckdb
import httpx
from loguru import logger

from core.constants import IEM_POLL_INTERVAL_SECONDS
from core.db import get_connection
from core.retry import retry_async
from services.ingestor import parse_t_group, _is_metar_stub

IEM_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# IEM uses 3-letter FAA codes; map to 4-letter ICAO for our observations table
IEM_STATIONS = {
    "NYC": "KNYC",
    "PHL": "KPHL",
    "MDW": "KMDW",
    "MIA": "KMIA",
    "LAX": "KLAX",
}

USER_AGENT = "(alphatemp, contact@alphatemp.com)"


class IEMIngestor:
    """Polls IEM ASOS for METAR+SPECI on the 5 settlement stations."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    @retry_async(max_retries=3, base_delay=2.0)
    async def _fetch_iem(self, client: httpx.AsyncClient, params) -> str:
        """GET CSV from IEM ASOS endpoint with retry."""
        resp = await client.get(
            IEM_BASE_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        return resp.text

    async def poll_once(self) -> int:
        """Single poll cycle. Returns number of new rows inserted."""
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(minutes=90)
        # Naive UTC for DuckDB TIMESTAMP columns (same convention as Synoptic)
        now = now_utc.replace(tzinfo=None)

        # List of tuples so we can send report_type twice:
        #   report_type=3 (routine METAR) + report_type=4 (SPECI)
        params = [
            ("station", ",".join(IEM_STATIONS.keys())),
            ("data", "metar,tmpf"),
            ("year1", str(start.year)),
            ("month1", str(start.month)),
            ("day1", str(start.day)),
            ("hour1", str(start.hour)),
            ("minute1", str(start.minute)),
            ("year2", str(now_utc.year)),
            ("month2", str(now_utc.month)),
            ("day2", str(now_utc.day)),
            ("hour2", str(now_utc.hour)),
            ("minute2", str(now_utc.minute)),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("elev", "no"),
            ("missing", "M"),
            ("trace", "T"),
            ("direct", "no"),
            ("report_type", "3"),
            ("report_type", "4"),
        ]

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                csv_text = await self._fetch_iem(client, params)
        except Exception as e:
            logger.warning(f"IEM ASOS request failed after retries: {e}")
            return 0

        rows = self._parse_csv(csv_text, now)
        if not rows:
            logger.debug("IEM: no observations to insert")
            return 0

        con = get_connection(self.db_path)
        inserted = 0
        for row in rows:
            try:
                con.execute(
                    """INSERT INTO observations
                       (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [row["station_id"], row["observed_at"], row["temp_f"],
                     row["temp_c_tenth"], row["raw_metar"], row["ingested_at"]],
                )
                inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate — first-to-arrive wins
        con.close()

        logger.info(f"IEM: inserted {inserted} new observations ({len(rows) - inserted} duplicates skipped)")
        return inserted

    def _parse_csv(self, csv_text: str, now: datetime) -> list[dict]:
        """Parse IEM CSV response into observation dicts."""
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = []

        for record in reader:
            faa_code = record.get("station", "").strip()
            icao = IEM_STATIONS.get(faa_code)
            if icao is None:
                continue

            valid_str = record.get("valid", "").strip()
            if not valid_str:
                continue

            try:
                # Parse as naive UTC — DuckDB TIMESTAMP column has no tz,
                # same convention as Synoptic ingestor
                observed_at = datetime.strptime(valid_str, "%Y-%m-%d %H:%M")
            except ValueError:
                logger.debug(f"IEM: skipping unparseable timestamp: {valid_str}")
                continue

            metar_str = record.get("metar", "").strip()
            tmpf_str = record.get("tmpf", "").strip()

            # Parse temps
            temp_f = None
            temp_c_tenth = None

            if not _is_metar_stub(metar_str):
                # T-group from raw METAR
                temp_c_tenth = parse_t_group(metar_str)

                # tmpf from IEM's parsed field
                if tmpf_str and tmpf_str != "M":
                    try:
                        temp_f = round(float(tmpf_str), 1)
                    except ValueError:
                        pass

            rows.append({
                "station_id": icao,
                "observed_at": observed_at,
                "temp_f": temp_f,
                "temp_c_tenth": temp_c_tenth,
                "raw_metar": metar_str,
                "ingested_at": now,
            })

        return rows

    async def run(self) -> None:
        """Run the IEM ingestor loop indefinitely."""
        stations = list(IEM_STATIONS.values())
        logger.info(
            f"Starting IEM ASOS ingestor — polling {len(stations)} settlement stations "
            f"every {IEM_POLL_INTERVAL_SECONDS}s"
        )
        while True:
            await self.poll_once()
            await asyncio.sleep(IEM_POLL_INTERVAL_SECONDS)
