"""NWS daily high/low fetcher — direct CLI products + ACIS backfill.

Two channels:
  1. NWS CLI (fast) — polls api.weather.gov for same-day CLImate reports every 30 min.
     The afternoon CLI (~4:30-5 PM ET) carries today's preliminary high hours before ACIS.
  2. ACIS (reliable) — polls RCC-ACIS every 4 hours as backfill/catch-all.

CLI data overwrites ACIS when both exist for the same station+date, since CLI is
the authoritative NWS source that Kalshi settles against.
"""

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import duckdb
import httpx
from loguru import logger

from core.constants import (
    CITIES,
    NWS_CLI_POLL_INTERVAL_SECONDS,
    NWS_DAILY_POLL_INTERVAL_SECONDS,
)
from core.db import get_connection
from core.retry import retry_async

ACIS_URL = "https://data.rcc-acis.org/StnData"

# NWS CLI product endpoints
NWS_CLI_LIST_URL = "https://api.weather.gov/products/types/CLI"

# NWS office → settlement station → CLI text identifier
CLI_STATIONS = {
    "KOKX": {"station_id": "KNYC", "cli_code": "NYC"},
}

# Regex for temperature lines in CLI product text
TEMP_MAX_RE = re.compile(r"^\s+MAXIMUM\s+(\d+)", re.MULTILINE)
TEMP_MIN_RE = re.compile(r"^\s+MINIMUM\s+(\d+)", re.MULTILINE)

ET = ZoneInfo("America/New_York")


class NWSFetcher:
    """Polls ACIS for NWS daily high/low temps and stores in DuckDB."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path
        self.stations = [cfg["settlement"] for cfg in CITIES.values()]

    # ------------------------------------------------------------------
    # NWS CLI direct polling (fast channel)
    # ------------------------------------------------------------------

    @retry_async(max_retries=3, base_delay=2.0)
    async def _fetch_json(self, client: httpx.AsyncClient, url: str) -> dict:
        """GET a JSON endpoint from api.weather.gov with retry."""
        resp = await client.get(url, headers={"User-Agent": "(alphatemp, contact@alphatemp.com)"})
        resp.raise_for_status()
        return resp.json()

    def _parse_cli_product(self, product: dict) -> list[dict]:
        """Parse a single CLI product into nws_daily row dicts.

        Returns 0-2 rows (today section + yesterday section if present).
        """
        text = product.get("productText", "")
        issuance_str = product.get("issuanceTime", "")
        issuing_office = product.get("issuingOffice", "")

        station_cfg = CLI_STATIONS.get(issuing_office)
        if not station_cfg:
            return []

        # Verify the CLI header matches our expected station
        # CLI headers look like: "CLINYC", "CLIPHL", etc.
        expected_header = f"CLI{station_cfg['cli_code']}"
        if expected_header not in text:
            return []

        # Parse issuance time → ET date
        try:
            issuance_utc = datetime.fromisoformat(issuance_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            logger.warning(f"CLI: bad issuanceTime '{issuance_str}'")
            return []

        issuance_et = issuance_utc.astimezone(ET)
        now_utc = datetime.now(timezone.utc)

        rows = []

        # Split text into sections — look for TODAY and YESTERDAY
        # CLI format has sections like:
        #   WEATHER ITEM   OBSERVED   ...
        #   TODAY
        #     MAXIMUM         84   1:59 PM
        #   YESTERDAY
        #     MAXIMUM         79   2:15 PM
        sections = re.split(r"\n\s*(TODAY|YESTERDAY)\s*\n", text)

        for i, section_name in enumerate(sections):
            if section_name in ("TODAY", "YESTERDAY"):
                section_body = sections[i + 1] if i + 1 < len(sections) else ""

                if section_name == "TODAY":
                    obs_date = issuance_et.date()
                else:  # YESTERDAY
                    obs_date = issuance_et.date() - timedelta(days=1)

                max_match = TEMP_MAX_RE.search(section_body)
                if not max_match:
                    continue  # MM or missing — skip

                max_temp = float(max_match.group(1))

                min_match = TEMP_MIN_RE.search(section_body)
                min_temp = float(min_match.group(1)) if min_match else None

                rows.append({
                    "station_id": station_cfg["station_id"],
                    "obs_date": obs_date.isoformat(),
                    "max_temp_f": max_temp,
                    "min_temp_f": min_temp,
                    "source": "NWS_CLI",
                    "ingested_at": now_utc.replace(tzinfo=None),
                })

        return rows

    def _upsert_row(self, row: dict) -> bool:
        """Delete-then-insert upsert — CLI always overwrites ACIS."""
        con = get_connection(self.db_path)
        con.execute(
            "DELETE FROM nws_daily WHERE station_id = ? AND obs_date = ?",
            [row["station_id"], row["obs_date"]],
        )
        con.execute(
            """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [row["station_id"], row["obs_date"], row["max_temp_f"],
             row["min_temp_f"], row["source"], row["ingested_at"]],
        )
        con.close()
        return True

    async def poll_cli(self) -> int:
        """Fetch recent NWS CLI products and upsert settlement highs."""
        upserted = 0
        target_offices = set(CLI_STATIONS.keys())

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                listing = await self._fetch_json(client, NWS_CLI_LIST_URL)
            except Exception as e:
                logger.warning(f"CLI listing fetch failed: {e}")
                return 0

            products = listing.get("@graph", [])
            matched = [p for p in products if p.get("issuingOffice") in target_offices]

            if not matched:
                logger.debug("CLI poll: no products from target offices")
                return 0

            for product_meta in matched:
                product_url = product_meta.get("@id", "")
                if not product_url:
                    continue
                try:
                    product = await self._fetch_json(client, product_url)
                except Exception as e:
                    logger.warning(f"CLI product fetch failed ({product_url}): {e}")
                    continue

                rows = self._parse_cli_product(product)
                for row in rows:
                    try:
                        self._upsert_row(row)
                        upserted += 1
                    except Exception as e:
                        logger.warning(f"CLI upsert failed for {row['station_id']} {row['obs_date']}: {e}")

        if upserted:
            logger.info(f"NWS CLI: upserted {upserted} rows from {len(matched)} products")
        else:
            logger.debug(f"NWS CLI: 0 rows from {len(matched)} products")
        return upserted

    # ------------------------------------------------------------------
    # ACIS backfill (reliable channel)
    # ------------------------------------------------------------------

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
        """Run the NWS daily fetcher loop — CLI fast + ACIS backfill."""
        logger.info(
            f"Starting NWS daily fetcher — CLI every {NWS_CLI_POLL_INTERVAL_SECONDS}s, "
            f"ACIS every {NWS_DAILY_POLL_INTERVAL_SECONDS}s"
        )
        acis_interval = NWS_DAILY_POLL_INTERVAL_SECONDS
        cli_interval = NWS_CLI_POLL_INTERVAL_SECONDS
        last_acis = 0.0
        last_cli = 0.0
        while True:
            now = time.monotonic()
            if now - last_cli >= cli_interval:
                await self.poll_cli()
                last_cli = time.monotonic()
            if now - last_acis >= acis_interval:
                await self.poll_once()
                last_acis = time.monotonic()
            await asyncio.sleep(60)
