# services/circuit_breakers.py
"""Circuit breakers for paper trading — runtime-adjustable risk limits."""
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import duckdb
from loguru import logger


class CircuitBreakers:
    """Check all risk limits before allowing a trade."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_config(self) -> Dict[str, str]:
        """Load all paper_config key-value pairs."""
        con = duckdb.connect(self.db_path)
        rows = con.execute("SELECT key, value FROM paper_config").fetchall()
        con.close()
        return {k: v for k, v in rows}

    def check(
        self,
        bracket_floor: int,
        bracket_cap: int,
        edge_pct: float,
        market_date: str,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """Run all circuit breakers in order.

        Returns (allowed, reason). If blocked, reason identifies which breaker
        tripped. If all pass, returns (True, "all_clear").
        """
        if now is None:
            now = datetime.utcnow()

        cfg = self._get_config()

        # 1. Kill switch
        if cfg.get("kill_switch", "False").lower() == "true":
            logger.warning("Circuit breaker: kill_switch is ON")
            return (False, "kill_switch: trading disabled")

        # 2. Min edge
        min_edge = float(cfg.get("min_edge_pct", "5.0"))
        if edge_pct < min_edge:
            return (False, "edge: %.1f%% < min %.1f%%" % (edge_pct, min_edge))

        con = duckdb.connect(self.db_path)
        try:
            # 3. Max daily loss
            max_loss = float(cfg.get("max_daily_loss_cents", "-1000"))
            daily_pnl = con.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) FROM paper_positions "
                "WHERE status = 'closed' AND event_date = ?",
                [market_date],
            ).fetchone()[0]
            if daily_pnl <= max_loss:
                return (False, "daily_loss: %d cents (<= %d limit)" % (int(daily_pnl), int(max_loss)))

            # 4. Max open positions
            max_open = int(cfg.get("max_open_positions", "5"))
            open_count = con.execute(
                "SELECT COUNT(*) FROM paper_positions WHERE status = 'open'"
            ).fetchone()[0]
            if open_count >= max_open:
                return (False, "max_open: %d positions (>= %d limit)" % (open_count, max_open))

            # 5. Max per bracket
            max_bracket = int(cfg.get("max_per_bracket", "2"))
            bracket_contracts = con.execute(
                "SELECT COALESCE(SUM(contracts), 0) FROM paper_positions "
                "WHERE status = 'open' AND bracket_floor = ? AND bracket_cap = ?",
                [bracket_floor, bracket_cap],
            ).fetchone()[0]
            if bracket_contracts >= max_bracket:
                return (
                    False,
                    "max_per_bracket: %d contracts on [%d, %d) (>= %d limit)"
                    % (bracket_contracts, bracket_floor, bracket_cap, max_bracket),
                )

            # 6. Cooldown — time since last non-settlement exit on same bracket
            cooldown_min = int(cfg.get("cooldown_minutes", "30"))
            last_exit = con.execute(
                "SELECT MAX(exit_time) FROM paper_positions "
                "WHERE status = 'closed' "
                "AND exit_reason != 'settlement' "
                "AND bracket_floor = ? AND bracket_cap = ?",
                [bracket_floor, bracket_cap],
            ).fetchone()[0]
            if last_exit is not None:
                if isinstance(last_exit, str):
                    last_exit = datetime.fromisoformat(last_exit)
                elapsed = now - last_exit
                if elapsed < timedelta(minutes=cooldown_min):
                    remaining = cooldown_min - int(elapsed.total_seconds() / 60)
                    return (
                        False,
                        "cooldown: %d min remaining on [%d, %d)" % (remaining, bracket_floor, bracket_cap),
                    )

        finally:
            con.close()

        return (True, "all_clear")
