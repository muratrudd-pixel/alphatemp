"""Kalshi Market Fetcher — polls temperature bracket prices for all 5 cities."""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from core.constants import CITIES
from core.db import get_connection
from core.heartbeat import record_heartbeat
from services.exchange import KalshiClient, SERIES_MAP, get_temperature_markets

# Poll every 60 seconds — Kalshi books update frequently
MARKET_POLL_INTERVAL = 60


class MarketFetcher:
    """Polls Kalshi temperature markets and stores ticks in DuckDB."""

    def __init__(self):
        self._client: Optional[KalshiClient] = None
        self._first_fetch = True

    def _init_client(self) -> bool:
        """Initialize KalshiClient from environment variables."""
        api_key = os.getenv("KALSHI_API_KEY")
        api_secret = os.getenv("KALSHI_API_SECRET")

        if not api_key or not api_secret:
            logger.warning("KALSHI_API_KEY or KALSHI_API_SECRET not set — market fetcher disabled")
            return False

        # Handle newline escapes in the RSA private key (stored as single line in .env)
        api_secret = api_secret.replace("\\n", "\n")

        try:
            self._client = KalshiClient(api_key=api_key, private_key_pem=api_secret)
            status = self._client.get_exchange_status()
            logger.info(f"Kalshi auth OK — exchange status: {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kalshi client: {e}")
            return False

    def _fetch_and_store(self):
        """Fetch market data for all cities and store in DB."""
        if not self._client:
            return

        con = get_connection()
        now = datetime.now(timezone.utc)
        total_stored = 0

        for city in CITIES:
            series = SERIES_MAP.get(city)
            if not series:
                continue

            try:
                markets = get_temperature_markets(self._client, city)
                if not markets:
                    logger.debug(f"No open markets for {city} ({series})")
                    continue

                # Log first response so we can verify field names
                if self._first_fetch:
                    logger.info(f"Sample market response for {city}: {markets[0]}")
                    self._first_fetch = False

                for m in markets:
                    ticker = m.get("ticker", "")

                    # Kalshi API returns prices as dollar strings ("0.0400")
                    # with _dollars suffix on price fields
                    yes_bid = m.get("yes_bid_dollars")
                    yes_ask = m.get("yes_ask_dollars")
                    no_bid = m.get("no_bid_dollars")
                    no_ask = m.get("no_ask_dollars")
                    last_price = m.get("last_price_dollars")
                    volume = int(float(m.get("volume_fp", "0")))
                    floor_strike = m.get("floor_strike")
                    cap_strike = m.get("cap_strike")
                    open_interest = int(float(m.get("open_interest_fp", "0")))
                    liquidity = int(float(m.get("liquidity_dollars", "0") or "0"))
                    volume_24h = int(float(m.get("volume_24h_fp", "0")))

                    def to_decimal(v):
                        if v is None:
                            return None
                        # Dollar string ("0.04") → float (already 0-1 range)
                        return float(v)

                    con.execute(
                        """INSERT INTO market_ticks
                            (market_id, city, captured_at, yes_bid, yes_ask,
                             no_bid, no_ask, last_trade, volume,
                             floor_strike, cap_strike,
                             open_interest, liquidity, volume_24h)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            ticker, city, now,
                            to_decimal(yes_bid), to_decimal(yes_ask),
                            to_decimal(no_bid), to_decimal(no_ask),
                            to_decimal(last_price),
                            volume,
                            floor_strike, cap_strike,
                            open_interest, liquidity, volume_24h,
                        ],
                    )
                    total_stored += 1

                logger.info(f"  {city}: {len(markets)} brackets captured")

            except Exception as e:
                logger.error(f"Failed to fetch markets for {city}: {e}")

        con.close()
        if total_stored:
            logger.info(f"Market tick cycle complete: {total_stored} ticks across {len(CITIES)} cities")

    async def run(self) -> None:
        """Async polling loop for market data."""
        logger.info("Starting Market Fetcher")

        if not self._init_client():
            logger.warning("Market Fetcher disabled — no valid Kalshi credentials")
            # Don't kill the whole process; just idle
            while True:
                await asyncio.sleep(3600)

        while True:
            try:
                cycle_start = time.monotonic()
                self._fetch_and_store()
                record_heartbeat(
                    "MarketFetcher",
                    duration_ms=(time.monotonic() - cycle_start) * 1000,
                )
            except Exception as e:
                record_heartbeat("MarketFetcher", duration_ms=0, status="error", error=str(e))
                logger.error(f"Market fetch cycle failed: {e}")
            await asyncio.sleep(MARKET_POLL_INTERVAL)
