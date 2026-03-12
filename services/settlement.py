"""Settlement service — polls for NWS CLI data and auto-settles open paper positions.

Runs as a standalone async loop with time-of-day-aware polling intervals.
Settlement math is consistent with paper_trader._settle_positions() and
strategy_backtester.compute_taker_fee().
"""
import asyncio
import math
import time
from datetime import datetime, timezone
from typing import List, Optional

import duckdb
from loguru import logger

from core.heartbeat import record_heartbeat


class SettlementService:
    def __init__(self, db_path, paper_trader=None):
        # type: (str, Optional[object]) -> None
        self.db_path = db_path
        self.paper_trader = paper_trader

    def _compute_fee(self, price_cents, contracts):
        # type: (int, int) -> float
        """Kalshi taker fee: max(ceil(0.07 * C * P * (1-P) * 100) / 100, C * $0.01).

        Returns fee in dollars. Matches paper_trader._compute_fee() exactly.
        """
        p = price_cents / 100.0
        raw = 0.07 * contracts * p * (1.0 - p)
        fee = max(math.ceil(round(raw * 100, 10)) / 100, contracts * 0.01)
        return round(fee, 2)

    def _get_poll_interval(self):
        # type: () -> int
        """Return poll interval in seconds based on current ET hour.

        - 4-6 PM ET: every 600s (10 min) — preliminary CLI expected
        - 1-2 AM ET: every 1800s (30 min) — final CLI expected
        - Otherwise: every 3600s (60 min)
        """
        try:
            import pytz
            et = datetime.now(pytz.timezone("US/Eastern"))
        except ImportError:
            # Fallback: UTC-5 approximation (close enough for polling intervals)
            from datetime import timedelta
            et = datetime.now(timezone.utc) - timedelta(hours=5)
        hour = et.hour

        if 16 <= hour < 18:
            return 600   # 10 min during preliminary CLI window
        elif 1 <= hour < 2:
            return 1800  # 30 min during final CLI window
        else:
            return 3600  # 60 min otherwise

    def _get_unsettled_dates(self):
        # type: () -> List[str]
        """Return distinct event_dates that have open positions."""
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT DISTINCT event_date
                FROM paper_positions
                WHERE status = 'open'
                  AND city = 'nyc'
                ORDER BY event_date
            """).fetchall()
            return [str(row[0]) for row in rows]
        finally:
            con.close()

    def settle_date(self, market_date):
        # type: (str) -> int
        """Settle all open positions for a given date.

        Returns count of positions settled.

        Settlement logic (matches paper_trader._settle_positions):
        - Bracket is [floor, cap) — floor inclusive, cap exclusive
        - YES wins if actual high is in the bracket
        - NO wins if actual high is outside the bracket
        - P&L in dollars (entry_price is in cents, divide by 100)
        - Fee was already computed at entry — no settlement fee on Kalshi
        """
        con = duckdb.connect(self.db_path)
        try:
            # 1. Check nws_daily for authoritative source (NWS_CLI or DSM)
            nws_row = con.execute("""
                SELECT max_temp_f, source
                FROM nws_daily
                WHERE station_id = 'KNYC'
                  AND obs_date = CAST(? AS DATE)
                  AND source IN ('NWS_CLI', 'DSM')
            """, [market_date]).fetchone()

            if nws_row is None:
                return 0

            actual_high = nws_row[0]

            # 2. Get all open positions for this date
            rows = con.execute("""
                SELECT id, direction, entry_price, contracts,
                       bracket_floor, bracket_cap, fees
                FROM paper_positions
                WHERE status = 'open'
                  AND city = 'nyc'
                  AND event_date = CAST(? AS DATE)
            """, [market_date]).fetchall()

            settled_count = 0
            for row in rows:
                pos_id, direction, entry_price, contracts, floor_val, cap_val, entry_fee = row

                # Bracket is [floor, cap) — floor inclusive, cap exclusive
                settled_yes = floor_val <= actual_high < cap_val

                if direction == "YES":
                    won = settled_yes
                else:  # NO
                    won = not settled_yes

                if won:
                    gross = (100 - entry_price) * contracts / 100.0
                else:
                    gross = -(entry_price * contracts / 100.0)

                net = round(gross - entry_fee, 2)

                con.execute("""
                    UPDATE paper_positions
                    SET status = 'closed',
                        exit_reason = 'settlement',
                        settled_yes = ?,
                        gross_pnl = ?,
                        net_pnl = ?,
                        fees = ?,
                        exit_time = ?,
                        exit_price = ?
                    WHERE id = ?
                """, [
                    settled_yes,
                    round(gross, 2),
                    net,
                    entry_fee,
                    datetime.now(timezone.utc),
                    100 if won else 0,
                    pos_id,
                ])
                settled_count += 1

            return settled_count
        finally:
            con.close()

    async def run(self):
        # type: () -> None
        """Main async loop — polls for settlement data.

        Adjusts poll interval by time of day:
        - 4-6 PM ET: every 10 min (preliminary CLI expected)
        - 1-2 AM ET: every 30 min (final CLI expected)
        - Otherwise: every 60 min
        """
        logger.info("SettlementService started")
        while True:
            try:
                cycle_start = time.monotonic()

                unsettled = self._get_unsettled_dates()
                total_settled = 0

                for market_date in unsettled:
                    count = self.settle_date(market_date)
                    if count > 0:
                        logger.info(
                            "Settled {} position(s) for {}".format(
                                count, market_date
                            )
                        )
                        total_settled += count

                duration_ms = (time.monotonic() - cycle_start) * 1000
                record_heartbeat(
                    "SettlementService",
                    duration_ms=duration_ms,
                )
                if total_settled:
                    logger.info(
                        "Settlement cycle complete: {} settled".format(
                            total_settled
                        )
                    )

            except Exception as e:
                logger.error("SettlementService error: {}".format(e))
                record_heartbeat(
                    "SettlementService",
                    duration_ms=0,
                    status="error",
                    error=str(e),
                )

            interval = self._get_poll_interval()
            await asyncio.sleep(interval)
