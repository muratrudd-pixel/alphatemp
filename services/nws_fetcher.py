"""NWS daily high/low fetcher via ACIS (Applied Climate Information System).

ACIS sources from the same CLImate Report (CLI) dataset that Kalshi uses for
settlement. This gives us the actual max/min thermometer reading — not the
running max of hourly ASOS snapshots, which reads 1-3°F lower.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import duckdb
import httpx
from loguru import logger

from core.constants import CITIES, NWS_DAILY_POLL_INTERVAL_SECONDS
from core.db import get_connection
from core.retry import retry_async

ACIS_URL = "https://data.rcc-acis.org/StnData"


class NWSFetcher:
    """Polls ACIS for NWS daily high/low temps and stores in DuckDB."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path
        self.stations = [cfg["settlement"] for cfg in CITIES.values()]

    @retry_async(max_retries=3, base_delay=2.0)
    async def _fetch_acis(self, client: httpx.AsyncClient, payload: dict) -> dict:
        """POST to ACIS StnData with automatic retry on failure."""
        resp = await client.post(ACIS_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _store_rows(self, rows: list[dict]) -> int:
        """Insert rows into nws_daily, skipping duplicates. Returns insert count."""
        if not rows:
            return 0
        con = get_connection(self.db_path)
        inserted = 0
        for row in rows:
            try:
                con.execute(
                    """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [row["station_id"], row["obs_date"], row["max_temp_f"],
                     row["min_temp_f"], row["source"], row["ingested_at"]],
                )
                inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate — skip silently
        con.close()
        return inserted

    async def poll_once(self) -> int:
        """Fetch yesterday + day-before NWS daily data for all settlement stations."""
        now = datetime.now(timezone.utc)
        today = now.date()
        sdate = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        edate = (today - timedelta(days=1)).strftime("%Y-%m-%d")

        all_rows = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for station_id in self.stations:
                payload = {
                    "sid": f"{station_id} 5",  # Type 5 = ICAO
                    "sdate": sdate,
                    "edate": edate,
                    "elems": "maxt,mint",
                }
                try:
                    data = await self._fetch_acis(client, payload)
                except Exception as e:
                    logger.warning(f"ACIS request failed for {station_id} after retries: {e}")
                    continue

                for row in data.get("data", []):
                    date_str, maxt, mint = row[0], row[1], row[2]
                    # "M" = missing value — skip
                    if maxt == "M" or mint == "M":
                        logger.debug(f"ACIS missing data for {station_id} on {date_str}")
                        continue
                    try:
                        max_f = float(maxt)
                        min_f = float(mint)
                    except (ValueError, TypeError):
                        logger.debug(f"ACIS non-numeric data for {station_id} on {date_str}: {maxt}/{mint}")
                        continue

                    all_rows.append({
                        "station_id": station_id,
                        "obs_date": date_str,
                        "max_temp_f": max_f,
                        "min_temp_f": min_f,
                        "source": "ACIS",
                        "ingested_at": now.replace(tzinfo=None),
                    })

        inserted = self._store_rows(all_rows)
        if inserted:
            logger.info(f"NWS daily: inserted {inserted} rows ({len(all_rows) - inserted} duplicates skipped)")
        else:
            logger.debug(f"NWS daily: no new rows ({len(all_rows)} duplicates skipped)")
        return inserted

    async def backfill(self, days_back: int = 365) -> int:
        """Backfill historical NWS daily data from ACIS."""
        now = datetime.now(timezone.utc)
        today = now.date()
        sdate = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        edate = (today - timedelta(days=1)).strftime("%Y-%m-%d")

        total_inserted = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            for station_id in self.stations:
                payload = {
                    "sid": f"{station_id} 5",
                    "sdate": sdate,
                    "edate": edate,
                    "elems": "maxt,mint",
                }
                try:
                    data = await self._fetch_acis(client, payload)
                except Exception as e:
                    logger.warning(f"ACIS backfill failed for {station_id}: {e}")
                    continue

                rows = []
                for row in data.get("data", []):
                    date_str, maxt, mint = row[0], row[1], row[2]
                    if maxt == "M" or mint == "M":
                        continue
                    try:
                        max_f = float(maxt)
                        min_f = float(mint)
                    except (ValueError, TypeError):
                        continue

                    rows.append({
                        "station_id": station_id,
                        "obs_date": date_str,
                        "max_temp_f": max_f,
                        "min_temp_f": min_f,
                        "source": "ACIS",
                        "ingested_at": now.replace(tzinfo=None),
                    })

                inserted = self._store_rows(rows)
                total_inserted += inserted
                logger.info(f"ACIS backfill {station_id}: {inserted}/{len(rows)} rows inserted")

        logger.info(f"NWS backfill complete: {total_inserted} total rows inserted")
        return total_inserted

    async def run(self) -> None:
        """Run the NWS daily fetcher loop indefinitely."""
        logger.info(
            f"Starting NWS daily fetcher — polling {len(self.stations)} stations "
            f"every {NWS_DAILY_POLL_INTERVAL_SECONDS}s"
        )
        while True:
            await self.poll_once()
            await asyncio.sleep(NWS_DAILY_POLL_INTERVAL_SECONDS)
