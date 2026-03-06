#!/usr/bin/env python3
"""NWS CLI monitor — polls for Daily Climate Report and prints the high temp.

Standalone signal tool. Tells you when the CLI dropped and what the high was.
Does NOT write to the database.

Usage:
    python scripts/nws_cli_monitor.py --once       # check once, print result
    python scripts/nws_cli_monitor.py --poll        # poll every 30 min from 15:00-18:00 ET
"""

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NWS_CLI_URL = "https://api.weather.gov/products?type=CLI&office=OKX&limit=1"
USER_AGENT = "(alphatemp-cli-monitor, contact@alphatemp.com)"
ET = ZoneInfo("America/New_York")

# Same regex used by services/nws_fetcher.py
TEMP_MAX_RE = re.compile(r"^\s+MAXIMUM\s+(\d+)", re.MULTILINE)

POLL_INTERVAL_SECONDS = 30 * 60  # 30 minutes
POLL_START_HOUR = 15  # 3 PM ET
POLL_END_HOUR = 18    # 6 PM ET


# ---------------------------------------------------------------------------
# Fetch + Parse
# ---------------------------------------------------------------------------

def fetch_latest_cli() -> Optional[dict]:
    """Fetch the most recent CLI product from OKX.

    Returns the full product dict, or None on failure.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/ld+json"}

    try:
        resp = requests.get(NWS_CLI_URL, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch CLI listing: {}", e)
        return None

    data = resp.json()
    graphs = data.get("@graph", [])
    if not graphs:
        logger.info("No CLI products returned from OKX")
        return None

    # The listing gives us metadata; fetch the full product text
    product_url = graphs[0].get("@id", "")
    if not product_url:
        logger.warning("CLI listing entry has no @id URL")
        return None

    try:
        resp = requests.get(product_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch CLI product ({}): {}", product_url, e)
        return None

    return resp.json()


def parse_cli_high(product: dict) -> Optional[Tuple[str, int, str]]:
    """Extract the daily high from a CLI product.

    Mirrors the parsing logic in services/nws_fetcher.py:
    - Splits text on TODAY/YESTERDAY sections
    - Uses TEMP_MAX_RE to grab the MAXIMUM line
    - Derives obs_date from issuance time

    Returns (obs_date_str, high_temp_f, product_id) or None.
    """
    text = product.get("productText", "")
    issuance_str = product.get("issuanceTime", "")
    product_id = product.get("id", "unknown")

    if "CLINYC" not in text:
        logger.debug("Product {} is not CLINYC — skipping", product_id)
        return None

    # Parse issuance time
    try:
        issuance_utc = datetime.fromisoformat(issuance_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning("Bad issuanceTime '{}' in product {}", issuance_str, product_id)
        return None

    issuance_et = issuance_utc.astimezone(ET)

    # Split into sections and find TODAY first (primary), then YESTERDAY as fallback
    sections = re.split(r"\n\s*(TODAY|YESTERDAY)\s*\n", text)

    for i, section_name in enumerate(sections):
        if section_name == "TODAY":
            section_body = sections[i + 1] if i + 1 < len(sections) else ""
            obs_date = issuance_et.date()

            max_match = TEMP_MAX_RE.search(section_body)
            if max_match:
                high_temp = int(max_match.group(1))
                return (obs_date.isoformat(), high_temp, product_id)

    # If no TODAY section found, try YESTERDAY
    for i, section_name in enumerate(sections):
        if section_name == "YESTERDAY":
            section_body = sections[i + 1] if i + 1 < len(sections) else ""
            obs_date = issuance_et.date() - timedelta(days=1)

            max_match = TEMP_MAX_RE.search(section_body)
            if max_match:
                high_temp = int(max_match.group(1))
                return (obs_date.isoformat(), high_temp, product_id)

    logger.info("No MAXIMUM temperature found in product {}", product_id)
    return None


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_once() -> bool:
    """Check once for CLI and print result. Returns True if found."""
    logger.info("Checking NWS CLI from OKX...")
    product = fetch_latest_cli()
    if product is None:
        logger.warning("No CLI product available")
        return False

    result = parse_cli_high(product)
    if result is None:
        issuance = product.get("issuanceTime", "unknown")
        logger.info("CLI product found (issued {}) but no high temp extracted", issuance)
        return False

    obs_date, high_temp, product_id = result
    print(f"\n  CLI REPORT FOUND")
    print(f"  Date:       {obs_date}")
    print(f"  High Temp:  {high_temp} F")
    print(f"  Product ID: {product_id}")
    print(f"  Issuance:   {product.get('issuanceTime', 'unknown')}\n")
    return True


def run_poll() -> None:
    """Poll every 30 min during the 15:00-18:00 ET window."""
    logger.info("Starting CLI poll mode (15:00-18:00 ET, every 30 min)")

    while True:
        now_et = datetime.now(ET)

        if now_et.hour < POLL_START_HOUR:
            wait_until = now_et.replace(hour=POLL_START_HOUR, minute=0, second=0, microsecond=0)
            wait_seconds = (wait_until - now_et).total_seconds()
            logger.info(
                "Before poll window. Sleeping until {:02d}:00 ET ({:.0f} min)",
                POLL_START_HOUR, wait_seconds / 60,
            )
            time.sleep(wait_seconds)
            continue

        if now_et.hour >= POLL_END_HOUR:
            logger.info("Past poll window ({}:00 ET). Done for today.", POLL_END_HOUR)
            break

        found = run_once()
        if found:
            logger.info("CLI detected. Monitor complete.")
            break

        logger.info(
            "No CLI yet. Next check in {} min (window closes at {:02d}:00 ET)",
            POLL_INTERVAL_SECONDS // 60, POLL_END_HOUR,
        )
        time.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor NWS CLI products from OKX for daily high temperature."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Check once and exit")
    group.add_argument(
        "--poll", action="store_true",
        help="Poll every 30 min during 15:00-18:00 ET window",
    )
    args = parser.parse_args()

    # Configure loguru — remove default, add clean stderr output
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    if args.once:
        found = run_once()
        sys.exit(0 if found else 1)
    elif args.poll:
        run_poll()


if __name__ == "__main__":
    main()
