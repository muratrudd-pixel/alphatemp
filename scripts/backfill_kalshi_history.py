#!/usr/bin/env python3
"""Kalshi Historical Data Backfill — settlements and candlesticks.

Pulls all settled temperature markets and their price history before
Kalshi removes historical data from live endpoints (March 6, 2026).

Usage:
    python scripts/backfill_kalshi_history.py                # Both phases
    python scripts/backfill_kalshi_history.py --settlements   # Phase A only
    python scripts/backfill_kalshi_history.py --candlesticks  # Phase B only
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import duckdb
from dotenv import load_dotenv
from loguru import logger

# Add project root to path so imports work from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import DEFAULT_DB_PATH, init_db
from services.exchange import KalshiClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_SERIES = {
    "NYC": {"high": "KXHIGHNY"},
    # Add lows and other cities later:
    # "NYC": {"high": "KXHIGHNY", "low": "KXLOWTNYC"},
    # "CHI": {"high": "KXHIGHCHI"},
    # ...
}

# Rate limiting
REQUEST_DELAY = 0.5       # seconds between requests
BACKOFF_BASE = 2.0        # exponential backoff base for 429s
MAX_RETRIES = 5           # max retries per request

# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

# Kalshi event tickers encode the date as YYMMMDD, e.g.:
#   KXHIGHNY-26FEB25  -> year=26, month=FEB, day=25 -> 2026-02-25
#   HIGHNY-22JAN01    -> year=22, month=JAN, day=01 -> 2022-01-01
_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_event_date(ticker: str) -> Optional[datetime]:
    """Extract event date from a Kalshi market/event ticker.

    Format is YYMMMDD (year-month-day), e.g.:
        KXHIGHNY-26FEB25 -> 2026-02-25
        HIGHNY-22JAN01   -> 2022-01-01
    """
    m = _EVENT_DATE_RE.search(ticker)
    if not m:
        return None
    year = 2000 + int(m.group(1))
    month = _MONTH_MAP.get(m.group(2))
    day = int(m.group(3))
    if not month:
        return None
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Kalshi client init
# ---------------------------------------------------------------------------

def make_client() -> KalshiClient:
    """Create authenticated KalshiClient from .env credentials."""
    load_dotenv()
    api_key = os.getenv("KALSHI_API_KEY")
    api_secret = os.getenv("KALSHI_API_SECRET")

    if not api_key or not api_secret:
        logger.error("KALSHI_API_KEY or KALSHI_API_SECRET not set in environment")
        sys.exit(1)

    # Handle newline escapes in RSA key (stored as single line in .env)
    api_secret = api_secret.replace("\\n", "\n")

    client = KalshiClient(api_key=api_key, private_key_pem=api_secret)
    # Verify auth works
    status = client.get_exchange_status()
    logger.info(f"Kalshi auth OK — exchange: {status.get('exchange_active', '?')}")
    return client


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def api_call_with_retry(fn, *args, **kwargs):
    """Call fn with exponential backoff on rate limits (429)."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 429:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Rate limited (429), backing off {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded on rate limit")


# ---------------------------------------------------------------------------
# Phase A: Settlements
# ---------------------------------------------------------------------------

def fetch_all_settled_markets(client: KalshiClient, series_ticker: str) -> List[Dict]:
    """Fetch all settled markets for a series, handling pagination."""
    all_markets = []
    cursor = None

    while True:
        data = api_call_with_retry(
            client.get_settled_markets, series_ticker, cursor=cursor
        )
        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

        logger.debug(f"  Fetched {len(all_markets)} markets so far (cursor: {cursor[:20]}...)")
        time.sleep(REQUEST_DELAY)

    return all_markets


def ingest_settlements(
    con: duckdb.DuckDBPyConnection,
    markets: List[Dict],
    city: str,
    measure: str,
    series_ticker: str,
) -> Tuple[int, int]:
    """Insert settled markets into kalshi_settlements. Returns (inserted, skipped)."""
    now = datetime.now(timezone.utc)
    inserted = 0
    skipped = 0

    for m in markets:
        market_ticker = m.get("ticker", "")
        event_ticker = m.get("event_ticker", "")

        # Parse event date from event_ticker or market_ticker
        event_date = parse_event_date(event_ticker) or parse_event_date(market_ticker)

        # Settlement result: "yes" -> 1, "no" -> 0
        result = m.get("result", "")
        if result == "yes":
            settled_yes = 1
        elif result == "no":
            settled_yes = 0
        else:
            settled_yes = None

        floor_strike = m.get("floor_strike")
        cap_strike = m.get("cap_strike")
        volume = m.get("volume", 0)

        # close_time is ISO string from API
        close_time_str = m.get("close_time")
        close_time = None
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        try:
            con.execute(
                """INSERT INTO kalshi_settlements
                    (market_ticker, event_ticker, series_ticker, city, measure,
                     event_date, floor_strike, cap_strike, settled_yes,
                     volume, close_time, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    market_ticker, event_ticker, series_ticker, city, measure,
                    event_date, floor_strike, cap_strike, settled_yes,
                    volume, close_time, now,
                ],
            )
            inserted += 1
        except duckdb.ConstraintException:
            skipped += 1

    return inserted, skipped


def run_phase_a(client: KalshiClient, db_path: str) -> None:
    """Phase A: Pull all settled markets for every configured series."""
    logger.info("=" * 60)
    logger.info("PHASE A: Fetching settled markets")
    logger.info("=" * 60)

    con = duckdb.connect(db_path)

    for city, measures in ALL_SERIES.items():
        for measure, series_ticker in measures.items():
            logger.info(f"\n--- {city} {measure} ({series_ticker}) ---")

            markets = fetch_all_settled_markets(client, series_ticker)
            logger.info(f"  Fetched {len(markets)} settled markets from API")

            if not markets:
                logger.warning(f"  No settled markets found for {series_ticker}")
                continue

            # Log a sample for debugging
            sample = markets[0]
            logger.debug(f"  Sample market: ticker={sample.get('ticker')}, "
                        f"event_ticker={sample.get('event_ticker')}, "
                        f"result={sample.get('result')}, "
                        f"floor_strike={sample.get('floor_strike')}, "
                        f"cap_strike={sample.get('cap_strike')}")

            inserted, skipped = ingest_settlements(con, markets, city, measure, series_ticker)
            logger.info(f"  Inserted: {inserted}, Skipped (duplicates): {skipped}")

    # Summary
    result = con.execute("""
        SELECT series_ticker, city, measure, COUNT(*) as cnt,
               MIN(event_date) as min_date, MAX(event_date) as max_date
        FROM kalshi_settlements
        GROUP BY 1, 2, 3
        ORDER BY 1
    """).fetchall()

    logger.info("\n" + "=" * 60)
    logger.info("PHASE A SUMMARY")
    for row in result:
        logger.info(f"  {row[0]} ({row[1]} {row[2]}): {row[3]} markets, "
                    f"{row[4]} to {row[5]}")
    logger.info("=" * 60)

    con.close()


# ---------------------------------------------------------------------------
# Phase B: Candlesticks
# ---------------------------------------------------------------------------

def run_phase_b(client: KalshiClient, db_path: str, refetch: bool = False, source_db: str = None) -> None:
    """Phase B: Pull candlestick data for all settled markets.

    If refetch=True, re-fetches ALL markets with the wider time window
    (useful after expanding the window to capture pre-event trading).
    If source_db is provided, reads settlement list from source_db but writes
    candlesticks to db_path (temp DB pattern to avoid DuckDB long-write crashes).
    """
    logger.info("=" * 60)
    logger.info("PHASE B: Fetching candlestick data" + (" (REFETCH MODE)" if refetch else ""))
    logger.info("=" * 60)

    con = duckdb.connect(db_path)

    # Read settlement list from source DB if provided
    if source_db:
        source_con = duckdb.connect(source_db, read_only=True)
        logger.info(f"  Reading settlement list from {source_db}")
        if refetch:
            markets = source_con.execute("""
                SELECT s.market_ticker, s.series_ticker, s.close_time, s.event_date
                FROM kalshi_settlements s
                ORDER BY s.event_date
            """).fetchall()
        else:
            # Check what's already in the target DB
            have_candles = {t[0] for t in con.execute(
                "SELECT DISTINCT market_ticker FROM kalshi_candlesticks"
            ).fetchall()}
            all_markets = source_con.execute("""
                SELECT s.market_ticker, s.series_ticker, s.close_time, s.event_date
                FROM kalshi_settlements s
                ORDER BY s.event_date
            """).fetchall()
            markets = [m for m in all_markets if m[0] not in have_candles]
        source_con.close()
    elif refetch:
        markets = con.execute("""
            SELECT s.market_ticker, s.series_ticker, s.close_time, s.event_date
            FROM kalshi_settlements s
            ORDER BY s.event_date
        """).fetchall()
    else:
        markets = con.execute("""
            SELECT s.market_ticker, s.series_ticker, s.close_time, s.event_date
            FROM kalshi_settlements s
            WHERE s.market_ticker NOT IN (
                SELECT DISTINCT market_ticker FROM kalshi_candlesticks
            )
            ORDER BY s.event_date
        """).fetchall()

    total = len(markets)
    logger.info(f"  {total} markets need candlestick data")

    if total == 0:
        logger.info("  Nothing to do — all markets already have candlesticks")
        con.close()
        return

    succeeded = 0
    failed = 0

    for i, (market_ticker, series_ticker, close_time, event_date) in enumerate(markets):
        progress = f"[{i + 1}/{total}]"

        # In refetch mode, delete existing candles for this market first
        if refetch:
            con.execute("DELETE FROM kalshi_candlesticks WHERE market_ticker = ?", [market_ticker])

        # Determine time range: start 2 days before event_date to capture
        # full trading window (markets typically open ~1 day before event)
        if event_date:
            start_ts = int(datetime.combine(event_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            ).timestamp()) - 172800  # 2 days before event_date
        else:
            logger.warning(f"  {progress} {market_ticker}: no event_date, skipping")
            failed += 1
            continue

        if close_time:
            if hasattr(close_time, 'timestamp'):
                end_ts = int(close_time.timestamp())
            else:
                end_ts = start_ts + 172800 + 86400  # event_date + 24h
        else:
            end_ts = start_ts + 172800 + 86400  # event_date + 24h

        try:
            candles = api_call_with_retry(
                client.get_candlesticks,
                series_ticker, market_ticker, start_ts, end_ts,
            )

            if not candles:
                logger.debug(f"  {progress} {market_ticker}: 0 candles")
                succeeded += 1
                time.sleep(REQUEST_DELAY)
                continue

            inserted = 0
            for c in candles:
                end_period = c.get("end_period_ts")
                if end_period is None:
                    continue

                # Convert Unix seconds to datetime
                if isinstance(end_period, (int, float)):
                    end_period_dt = datetime.fromtimestamp(end_period, tz=timezone.utc)
                else:
                    end_period_dt = end_period

                try:
                    con.execute(
                        """INSERT INTO kalshi_candlesticks
                            (market_ticker, end_period_ts, period_minutes,
                             yes_bid_close, yes_ask_close,
                             price_open, price_high, price_low, price_close,
                             volume, open_interest)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            market_ticker, end_period_dt, 1,
                            c.get("yes_bid", {}).get("close") if isinstance(c.get("yes_bid"), dict) else c.get("yes_bid_close"),
                            c.get("yes_ask", {}).get("close") if isinstance(c.get("yes_ask"), dict) else c.get("yes_ask_close"),
                            c.get("price", {}).get("open") if isinstance(c.get("price"), dict) else c.get("price_open"),
                            c.get("price", {}).get("high") if isinstance(c.get("price"), dict) else c.get("price_high"),
                            c.get("price", {}).get("low") if isinstance(c.get("price"), dict) else c.get("price_low"),
                            c.get("price", {}).get("close") if isinstance(c.get("price"), dict) else c.get("price_close"),
                            c.get("volume", 0),
                            c.get("open_interest", 0),
                        ],
                    )
                    inserted += 1
                except duckdb.ConstraintException:
                    pass  # Duplicate — skip

            logger.info(f"  {progress} {market_ticker}: {inserted} candles inserted")
            succeeded += 1

        except Exception as e:
            logger.error(f"  {progress} {market_ticker}: FAILED — {e}")
            failed += 1

        time.sleep(REQUEST_DELAY)

        # Progress log every 100 markets (no CHECKPOINT — DuckDB 1.4.4 crashes on it)
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{total}")

    # Summary
    candle_stats = con.execute("""
        SELECT COUNT(DISTINCT market_ticker), COUNT(*)
        FROM kalshi_candlesticks
    """).fetchone()

    logger.info("\n" + "=" * 60)
    logger.info("PHASE B SUMMARY")
    logger.info(f"  Succeeded: {succeeded}, Failed: {failed}")
    logger.info(f"  Total: {candle_stats[0]} markets, {candle_stats[1]} candle rows")
    logger.info("=" * 60)

    con.close()


# ---------------------------------------------------------------------------
# Phase C: Trades
# ---------------------------------------------------------------------------

def run_phase_c(client: KalshiClient, db_path: str, source_db: str = None) -> None:
    """Phase C: Pull trade history for all settled markets.

    If source_db is provided, reads settlement list from source_db but writes
    trades to db_path (temp DB pattern to avoid DuckDB long-write crashes).
    """
    logger.info("=" * 60)
    logger.info("PHASE C: Fetching trade history")
    logger.info("=" * 60)

    con = duckdb.connect(db_path)

    # Read settlement list from source DB if provided, otherwise from same DB
    if source_db:
        source_con = duckdb.connect(source_db, read_only=True)
        all_tickers = source_con.execute("""
            SELECT market_ticker FROM kalshi_settlements ORDER BY event_date
        """).fetchall()
        source_con.close()
        logger.info(f"  Reading settlement list from {source_db}")
    else:
        all_tickers = con.execute("""
            SELECT market_ticker FROM kalshi_settlements ORDER BY event_date
        """).fetchall()
    all_tickers = [t[0] for t in all_tickers]

    # Find which ones already have trades (in the target DB)
    have_trades = set()
    existing = con.execute("""
        SELECT DISTINCT market_ticker FROM kalshi_trades
    """).fetchall()
    have_trades = {t[0] for t in existing}

    remaining = [t for t in all_tickers if t not in have_trades]
    total = len(remaining)

    logger.info(f"  {len(all_tickers)} total settled markets, {len(have_trades)} already have trades, {total} remaining")

    if total == 0:
        logger.info("  Nothing to do — all markets already have trades")
        con.close()
        return

    now = datetime.now(timezone.utc)
    succeeded = 0
    failed = 0
    total_trades = 0

    for i, market_ticker in enumerate(remaining):
        progress = f"[{i + 1}/{total}]"

        try:
            # Paginate through all trades for this market
            cursor = None
            market_trades = 0

            while True:
                data = api_call_with_retry(
                    client.get_trades, ticker=market_ticker, cursor=cursor, limit=1000
                )

                trades = data.get("trades", [])
                if not trades:
                    break

                for t in trades:
                    trade_id = t.get("trade_id")
                    if not trade_id:
                        continue

                    created_time_str = t.get("created_time", "")
                    try:
                        created_time = datetime.fromisoformat(
                            created_time_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        continue

                    try:
                        con.execute(
                            """INSERT INTO kalshi_trades
                                (trade_id, market_ticker, yes_price, no_price,
                                 count, taker_side, created_time, ingested_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            [
                                trade_id,
                                t.get("ticker", market_ticker),
                                t.get("yes_price"),
                                t.get("no_price"),
                                t.get("count"),
                                t.get("taker_side"),
                                created_time,
                                now,
                            ],
                        )
                        market_trades += 1
                    except duckdb.ConstraintException:
                        pass  # Duplicate

                cursor = data.get("cursor")
                if not cursor:
                    break
                time.sleep(REQUEST_DELAY)

            total_trades += market_trades
            if market_trades > 0:
                logger.info(f"  {progress} {market_ticker}: {market_trades} trades")
            else:
                logger.debug(f"  {progress} {market_ticker}: 0 trades")
            succeeded += 1

        except Exception as e:
            logger.error(f"  {progress} {market_ticker}: FAILED — {e}")
            failed += 1

        time.sleep(REQUEST_DELAY)

        # Progress log every 100 markets (no CHECKPOINT — DuckDB 1.4.4 crashes on it)
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{total} ({total_trades} trades so far)")

    # Summary
    trade_stats = con.execute("""
        SELECT COUNT(DISTINCT market_ticker), COUNT(*)
        FROM kalshi_trades
    """).fetchone()

    logger.info("\n" + "=" * 60)
    logger.info("PHASE C SUMMARY")
    logger.info(f"  Succeeded: {succeeded}, Failed: {failed}")
    logger.info(f"  Total: {trade_stats[0]} markets, {trade_stats[1]} trade rows")
    logger.info("=" * 60)

    con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi historical data backfill")
    parser.add_argument("--settlements", action="store_true", help="Phase A only: settlements")
    parser.add_argument("--candlesticks", action="store_true", help="Phase B only: candlesticks")
    parser.add_argument("--trades", action="store_true", help="Phase C only: trade history")
    parser.add_argument("--refetch", action="store_true", help="Re-fetch candlesticks with wider time window (deletes existing per-market)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    parser.add_argument("--source-db", default=None, help="Read settlements from this DB (for temp DB pattern)")
    args = parser.parse_args()

    # Default: run all phases
    run_all = not args.settlements and not args.candlesticks and not args.trades

    logger.info("Kalshi Historical Data Backfill")
    logger.info(f"Database: {args.db}")
    if args.source_db:
        logger.info(f"Source DB: {args.source_db}")

    # Ensure tables exist
    init_db(args.db)

    client = make_client()

    if args.settlements or run_all:
        run_phase_a(client, args.db)

    if args.candlesticks or run_all:
        run_phase_b(client, args.db, refetch=args.refetch, source_db=args.source_db)

    if args.trades or run_all:
        run_phase_c(client, args.db, source_db=args.source_db)

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
