"""NWS daily high/low fetcher — CLI + DSM + ACIS channels.

Three channels with source hierarchy (CLI > DSM > ACIS):
  1. NWS CLI (authoritative) — polls api.weather.gov for CLImate reports every 30 min.
     The afternoon CLI (~4:30-5 PM ET) carries today's preliminary high. Final ~1:30 AM ET.
  2. DSM (early proxy) — polls api.weather.gov for Daily Summary Messages every 30 min.
     Arrives ~4-5 PM ET on settlement day, before CLI. Provides early max/min.
  3. ACIS (reliable) — polls RCC-ACIS every 4 hours as backfill/catch-all.

Source priority: CLI overwrites everything, DSM overwrites ACIS but not CLI.
"""

import asyncio
import re
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import duckdb
import httpx
from loguru import logger

from typing import Dict, List, Optional

from core.constants import (
    CITIES,
    NWS_CLI_POLL_INTERVAL_SECONDS,
    NWS_DAILY_POLL_INTERVAL_SECONDS,
)
from core.db import get_connection
from core.heartbeat import record_heartbeat
from core.retry import retry_async

ACIS_URL = "https://data.rcc-acis.org/StnData"

# NWS product endpoints
NWS_CLI_LIST_URL = "https://api.weather.gov/products/types/CLI"
NWS_DSM_LIST_URL = "https://api.weather.gov/products/types/DSM"

# NWS office → settlement station → CLI/DSM text identifier
CLI_STATIONS = {
    "KOKX": {"station_id": "KNYC", "cli_code": "NYC"},
}

# DSM station mapping: DSM uses ICAO code (KNYC) and 3-letter city code (NYC)
# in the product text. Map office → list of stations to extract.
DSM_STATIONS = {
    "KOKX": {"station_id": "KNYC", "dsm_code": "NYC"},
}

# Source priority — higher number wins. CLI is authoritative (Kalshi settlement).
# DSM is an early proxy that arrives before CLI. ACIS is the backfill catch-all.
SOURCE_PRIORITY: Dict[str, int] = {
    "NWS_CLI": 3,
    "DSM": 2,
    "ACIS": 1,
}

# Regex for temperature lines in CLI product text
TEMP_MAX_RE = re.compile(r"^\s+MAXIMUM\s+(\d+)", re.MULTILINE)
TEMP_MIN_RE = re.compile(r"^\s+MINIMUM\s+(\d+)", re.MULTILINE)

# DSM data line regex — extracts the encoded temperature data
# Format: KNYC DS [optional_time] [optional COR] DD/MM maxTempHHMM/ minTempHHMM// calMax/ calMin//...
DSM_LINE_RE = re.compile(
    r"^(K\w{3})\s+DS\s+"          # ICAO station code + "DS"
    r"(?:COR\s+)?"                 # optional correction marker
    r"(?:\d{4}\s+)?"               # optional observation time (e.g., 1600)
    r"(\d{2})/(\d{2})\s+"          # DD/MM
    r"(-?\d+)\d{4}/\s*"            # max temp (digits before last 4 HHMM) + time
    r"(-?\d+)\d{4}//"              # min temp + time, then //
    r"\s*(-?\d+)/\s*(-?\d+)//",    # calendar-day max/ min//
    re.MULTILINE,
)

ET = ZoneInfo("America/New_York")


def parse_dsm_product(product: dict) -> List[dict]:
    """Parse a DSM product into nws_daily row dicts.

    DSM (Daily Summary Message) is encoded WMO format issued by NWS offices.
    Format per station line:
        KNYC DS [COR] [HHMM] DD/MM maxTempHHMM/ minTempHHMM// calMax/ calMin//...

    The calendar-day max/min (after //) are the official daily values.
    We extract those rather than the observation-period values.

    Returns a list of row dicts (typically 1 per product for our target stations).
    """
    text = product.get("productText", "")
    issuance_str = product.get("issuanceTime", "")
    issuing_office = product.get("issuingOffice", "")

    station_cfg = DSM_STATIONS.get(issuing_office)
    if not station_cfg:
        return []

    # Parse issuance time for ingested_at timestamp
    try:
        issuance_utc = datetime.fromisoformat(issuance_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning(f"DSM: bad issuanceTime '{issuance_str}'")
        return []

    now_utc = datetime.now(timezone.utc)
    rows = []

    for match in DSM_LINE_RE.finditer(text):
        icao, dd, mm, _obs_max, _obs_min, cal_max_str, cal_min_str = match.groups()

        # Only process our target station
        if icao != station_cfg["station_id"]:
            continue

        # Build obs_date from DD/MM + year from issuance
        # DSM uses DD/MM without year — derive year from issuance time
        issuance_et = issuance_utc.astimezone(ET)
        try:
            day = int(dd)
            month = int(mm)
            year = issuance_et.year
            # Handle year boundary: if DSM date is Dec but issuance is Jan
            if month == 12 and issuance_et.month == 1:
                year -= 1
            obs_date = date_cls(year, month, day)
        except ValueError:
            logger.warning(f"DSM: bad date {dd}/{mm} from {issuing_office}")
            continue

        try:
            cal_max = float(cal_max_str)
            cal_min = float(cal_min_str)
        except (ValueError, TypeError):
            logger.warning(f"DSM: bad temps '{cal_max_str}'/'{cal_min_str}' for {icao}")
            continue

        rows.append({
            "station_id": icao,
            "obs_date": obs_date.isoformat(),
            "max_temp_f": cal_max,
            "min_temp_f": cal_min,
            "source": "DSM",
            "ingested_at": now_utc.replace(tzinfo=None),
        })

    return rows


class NWSFetcher:
    """Polls NWS CLI, DSM, and ACIS for daily high/low temps. Stores in DuckDB."""

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
                    "raw_text": text,
                })

        return rows

    def _upsert_row(self, row: dict) -> bool:
        """Upsert a row respecting source hierarchy: CLI > DSM > ACIS.

        A higher-priority source always overwrites a lower-priority source.
        A same-priority source updates the row (refreshes data).
        A lower-priority source is silently skipped.

        Returns True if the row was written, False if skipped.
        """
        con = get_connection(self.db_path)
        try:
            existing = con.execute(
                "SELECT source FROM nws_daily WHERE station_id = ? AND obs_date = ?",
                [row["station_id"], row["obs_date"]],
            ).fetchone()

            if existing:
                existing_source = existing[0]
                existing_priority = SOURCE_PRIORITY.get(existing_source, 0)
                new_priority = SOURCE_PRIORITY.get(row["source"], 0)

                if new_priority < existing_priority:
                    # Lower-priority source — don't overwrite
                    return False

            # Delete existing (if any) and insert new
            con.execute(
                "DELETE FROM nws_daily WHERE station_id = ? AND obs_date = ?",
                [row["station_id"], row["obs_date"]],
            )
            con.execute(
                """INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [row["station_id"], row["obs_date"], row["max_temp_f"],
                 row["min_temp_f"], row["source"], row["ingested_at"],
                 row.get("raw_text")],
            )
            return True
        finally:
            con.close()

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
    # DSM polling (early settlement proxy)
    # ------------------------------------------------------------------

    async def poll_dsm(self) -> int:
        """Fetch recent DSM products and upsert daily max/min temps.

        DSM arrives ~4-5 PM ET on the settlement day — hours before
        the final CLI report. Provides an early proxy for settlement.
        """
        upserted = 0
        target_offices = set(DSM_STATIONS.keys())

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                listing = await self._fetch_json(client, NWS_DSM_LIST_URL)
            except Exception as e:
                logger.warning(f"DSM listing fetch failed: {e}")
                return 0

            products = listing.get("@graph", [])
            matched = [p for p in products if p.get("issuingOffice") in target_offices]

            if not matched:
                logger.debug("DSM poll: no products from target offices")
                return 0

            for product_meta in matched:
                product_url = product_meta.get("@id", "")
                if not product_url:
                    continue
                try:
                    product = await self._fetch_json(client, product_url)
                except Exception as e:
                    logger.warning(f"DSM product fetch failed ({product_url}): {e}")
                    continue

                rows = parse_dsm_product(product)
                for row in rows:
                    try:
                        if self._upsert_row(row):
                            upserted += 1
                    except Exception as e:
                        logger.warning(f"DSM upsert failed for {row['station_id']} {row['obs_date']}: {e}")

        if upserted:
            logger.info(f"DSM: upserted {upserted} rows from {len(matched)} products")
        else:
            logger.debug(f"DSM: 0 rows from {len(matched)} products")
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
        """Run the NWS daily fetcher loop — CLI + DSM fast, ACIS backfill."""
        dsm_interval = NWS_CLI_POLL_INTERVAL_SECONDS  # Same cadence as CLI
        logger.info(
            f"Starting NWS daily fetcher — CLI/DSM every {NWS_CLI_POLL_INTERVAL_SECONDS}s, "
            f"ACIS every {NWS_DAILY_POLL_INTERVAL_SECONDS}s"
        )
        acis_interval = NWS_DAILY_POLL_INTERVAL_SECONDS
        cli_interval = NWS_CLI_POLL_INTERVAL_SECONDS
        last_acis = 0.0
        last_cli = 0.0
        last_dsm = 0.0
        while True:
            try:
                cycle_start = time.monotonic()
                now = time.monotonic()
                if now - last_cli >= cli_interval:
                    await self.poll_cli()
                    last_cli = time.monotonic()
                if now - last_dsm >= dsm_interval:
                    await self.poll_dsm()
                    last_dsm = time.monotonic()
                if now - last_acis >= acis_interval:
                    await self.poll_once()
                    last_acis = time.monotonic()
                record_heartbeat(
                    "NWSFetcher",
                    duration_ms=(time.monotonic() - cycle_start) * 1000,
                )
            except Exception as e:
                record_heartbeat("NWSFetcher", duration_ms=0, status="error", error=str(e))
                logger.error(f"NWS fetch cycle failed: {e}")
            await asyncio.sleep(60)
