"""Automated paper trading service. Runs as async task in main.py.

Strategy logic is a PLACEHOLDER — entry/exit rules will be designed
in a dedicated brainstorming session. This shell provides:
- Position lifecycle (OPEN -> CLOSED)
- Fee computation (Kalshi taker fee model)
- Settlement resolution
- Heartbeat reporting
"""
import asyncio
import math
import time
from datetime import datetime, timezone
from typing import Optional

import duckdb
from loguru import logger

from core.heartbeat import record_heartbeat


PAPER_TRADE_INTERVAL = 60  # seconds


class PaperTrader:
    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    def _compute_fee(self, price_cents: int, contracts: int) -> float:
        """Kalshi taker fee: max(ceil(0.07 * C * P * (1-P) * 100) / 100, C * $0.01).

        Returns fee in dollars. Matches the formula in strategy_backtester.py
        but returns dollars instead of cents.
        """
        p = price_cents / 100.0
        raw = 0.07 * contracts * p * (1.0 - p)
        fee = max(math.ceil(round(raw * 100, 10)) / 100, contracts * 0.01)
        return round(fee, 2)

    def _next_id(self, con: duckdb.DuckDBPyConnection) -> int:
        """Generate the next paper_positions ID.

        The schema has no SEQUENCE or AUTOINCREMENT, so we compute
        max(id) + 1 manually. Safe for single-writer (one PaperTrader instance).
        """
        result = con.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM paper_positions"
        ).fetchone()
        return result[0]

    def _record_entry(
        self,
        city,           # type: str
        event_date,     # type: str
        bracket_floor,  # type: int
        bracket_cap,    # type: int
        direction,      # type: str
        model_prob,     # type: float
        market_price,   # type: float
        entry_price,    # type: float
        contracts=1,    # type: int
    ):
        # type: (...) -> None
        """Write a new open position to paper_positions."""
        edge = model_prob - (market_price / 100.0)
        fee = self._compute_fee(int(entry_price), contracts)
        con = duckdb.connect(self.db_path)
        try:
            next_id = self._next_id(con)
            con.execute("""
                INSERT INTO paper_positions
                (id, city, event_date, bracket_floor, bracket_cap, direction,
                 model_prob, market_price, edge, entry_price, entry_time,
                 fees, status, contracts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """, [
                next_id,
                city, event_date, bracket_floor, bracket_cap, direction,
                round(model_prob, 4), market_price, round(edge, 4),
                entry_price, datetime.now(timezone.utc),
                fee, contracts,
            ])
        finally:
            con.close()

    async def enter_position(
        self,
        city,           # type: str
        event_date,     # type: str
        bracket_floor,  # type: int
        bracket_cap,    # type: int
        direction,      # type: str
        model_prob,     # type: float
        market_price,   # type: float
        edge,           # type: float
    ):
        # type: (...) -> None
        """Public entry point called by StrategyEngine.

        Delegates to existing _record_entry() method. Paper trades at the
        ask price (market_price), always 1 contract.
        """
        self._record_entry(
            city=city,
            event_date=event_date,
            bracket_floor=bracket_floor,
            bracket_cap=bracket_cap,
            direction=direction,
            model_prob=model_prob,
            market_price=market_price,
            entry_price=market_price,  # paper trade at ask
            contracts=1,
        )

    async def exit_position(
        self,
        position_id,  # type: int
        exit_price,    # type: float
        reason,        # type: str
    ):
        # type: (...) -> None
        """Close a position early (edge reversal, kill switch, etc.).

        Computes P&L net of fees (entry + exit) and updates the DB.
        Both YES and NO use the same formula: gross = (exit - entry) * contracts / 100.
        You bought at entry_price, you sell at exit_price — direction is already
        baked into the price you paid.
        """
        con = duckdb.connect(self.db_path)
        try:
            row = con.execute(
                "SELECT entry_price, contracts, direction, fees "
                "FROM paper_positions WHERE id = ? AND status = 'open'",
                [position_id],
            ).fetchone()

            if row is None:
                raise ValueError(
                    "Position {} not found or already closed".format(position_id)
                )

            entry_price, contracts, direction, entry_fee = row

            # Gross P&L: bought at entry, selling at exit (in cents, convert to dollars)
            gross = (exit_price - entry_price) * contracts / 100.0

            # Exit fee on the sell side
            exit_fee = self._compute_fee(int(exit_price), contracts)
            total_fees = round(entry_fee + exit_fee, 2)

            net = round(gross - total_fees, 2)

            con.execute("""
                UPDATE paper_positions
                SET status = 'closed',
                    exit_price = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    gross_pnl = ?,
                    fees = ?,
                    net_pnl = ?
                WHERE id = ?
            """, [
                exit_price,
                datetime.now(timezone.utc),
                reason,
                round(gross, 2),
                total_fees,
                net,
                position_id,
            ])
        finally:
            con.close()

    async def _check_entries(self):
        """Evaluate brackets for tradeable edge. PLACEHOLDER STRATEGY.

        Current rule: no automatic entries. Replace with real strategy
        after dedicated brainstorm session.
        """
        # TODO: Real strategy logic — entry/exit rules, edge thresholds,
        # position sizing, NO-side logic, risk caps
        pass

    async def _check_exits(self):
        """Monitor open positions for early exit signals. PLACEHOLDER.

        Current rule: hold to settlement. Replace with real exit logic.
        """
        # TODO: Real exit logic (edge flip, stop-loss, etc.)
        pass

    async def _settle_positions(self):
        """Resolve open positions when NWS CLI settlement data arrives.

        Settlement logic:
        - Bracket is [floor, cap) — floor inclusive, cap exclusive
        - YES wins if actual high is in the bracket
        - NO wins if actual high is outside the bracket
        - P&L: winners get (100 - entry) per contract, losers lose entry per contract
        - All P&L in dollars (entry_price is in cents, divide by 100)
        """
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT p.id, p.direction, p.entry_price, p.contracts,
                       p.bracket_floor, p.bracket_cap, p.fees,
                       n.max_temp_f
                FROM paper_positions p
                JOIN nws_daily n ON n.obs_date = p.event_date
                    AND n.station_id = 'KNYC'
                    AND n.source = 'NWS_CLI'
                WHERE p.status = 'open'
                  AND p.city = 'nyc'
            """).fetchall()

            for row in rows:
                pos_id, direction, entry_price, contracts, floor_val, cap_val, entry_fee, actual_high = row
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
                    SET status = 'closed', exit_reason = 'settlement',
                        settled_yes = ?, gross_pnl = ?, net_pnl = ?,
                        exit_time = ?, exit_price = ?
                    WHERE id = ?
                """, [
                    settled_yes, round(gross, 2), net,
                    datetime.now(timezone.utc),
                    100 if won else 0,
                    pos_id,
                ])
        finally:
            con.close()

    async def _update_unrealized(self):
        """Update unrealized P&L for open positions using current market mid.

        Queries the latest market tick for each position's bracket and
        computes mark-to-market unrealized P&L.
        """
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT p.id, p.direction, p.entry_price, p.contracts,
                       p.bracket_floor, p.bracket_cap, p.city
                FROM paper_positions p
                WHERE p.status = 'open'
            """).fetchall()

            for row in rows:
                pos_id, direction, entry_price, contracts, floor_val, cap_val, city = row
                # Cast bracket ints to double for market_ticks join
                tick = con.execute("""
                    SELECT (yes_bid + yes_ask) / 2.0 AS mid
                    FROM market_ticks
                    WHERE city = ?
                      AND floor_strike = CAST(? AS DOUBLE)
                      AND cap_strike = CAST(? AS DOUBLE)
                    ORDER BY captured_at DESC LIMIT 1
                """, [city, floor_val, cap_val]).fetchone()

                if tick:
                    market_mid = tick[0]
                    if direction == "YES":
                        unrealized = (market_mid - entry_price) * contracts / 100.0
                    else:
                        unrealized = (entry_price - market_mid) * contracts / 100.0
                    con.execute(
                        "UPDATE paper_positions SET unrealized_pnl = ? WHERE id = ?",
                        [round(unrealized, 2), pos_id],
                    )
        finally:
            con.close()

    async def run(self):
        """Main loop — runs every 60s alongside other services."""
        logger.info("PaperTrader started (placeholder strategy)")
        while True:
            try:
                cycle_start = time.monotonic()
                await self._check_entries()
                await self._check_exits()
                # NOTE: settle + update_unrealized are not atomic (separate connections).
                # Acceptable for paper trading shell. Must share a connection/transaction
                # when real strategy is implemented.
                await self._settle_positions()
                await self._update_unrealized()
                record_heartbeat(
                    "PaperTrader",
                    duration_ms=(time.monotonic() - cycle_start) * 1000,
                )
            except Exception as e:
                logger.error(f"PaperTrader error: {e}")
                record_heartbeat("PaperTrader", duration_ms=0, status="error", error=str(e))
            await asyncio.sleep(PAPER_TRADE_INTERVAL)
