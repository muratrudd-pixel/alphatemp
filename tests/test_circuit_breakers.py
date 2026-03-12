# tests/test_circuit_breakers.py
"""Tests for CircuitBreakers — kill switch, edge, loss limits, position limits, cooldown."""
from datetime import datetime, timedelta

import duckdb
import pytest

from services.circuit_breakers import CircuitBreakers


def _create_test_db(tmp_path):
    """Create a test DB with paper_config defaults and paper_positions table."""
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE paper_config (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Seed defaults
    con.execute("""
        INSERT INTO paper_config (key, value) VALUES
        ('max_daily_loss_cents', '-1000'),
        ('max_open_positions', '5'),
        ('max_per_bracket', '2'),
        ('min_edge_pct', '5.0'),
        ('cooldown_minutes', '30'),
        ('kill_switch', 'False')
    """)

    con.execute("""
        CREATE TABLE paper_positions (
            id INTEGER,
            city VARCHAR,
            event_date DATE,
            bracket_floor INTEGER,
            bracket_cap INTEGER,
            direction VARCHAR,
            model_prob DOUBLE,
            market_price DOUBLE,
            edge DOUBLE,
            entry_price DOUBLE,
            entry_time TIMESTAMP,
            exit_price DOUBLE,
            exit_time TIMESTAMP,
            settled_yes BOOLEAN,
            gross_pnl DOUBLE,
            fees DOUBLE,
            net_pnl DOUBLE,
            status VARCHAR DEFAULT 'open',
            exit_reason VARCHAR,
            contracts INTEGER DEFAULT 1,
            unrealized_pnl DOUBLE DEFAULT 0.0
        )
    """)
    con.close()
    return db_path


@pytest.fixture
def test_db(tmp_path):
    return _create_test_db(tmp_path)


@pytest.fixture
def breakers(test_db):
    return CircuitBreakers(db_path=test_db)


class TestKillSwitch:
    def test_kill_switch_blocks_all(self, tmp_path):
        """Kill switch True -> blocked regardless of other params."""
        db_path = _create_test_db(tmp_path)
        con = duckdb.connect(db_path)
        con.execute("UPDATE paper_config SET value = 'True' WHERE key = 'kill_switch'")
        con.close()

        cb = CircuitBreakers(db_path=db_path)
        allowed, reason = cb.check(
            bracket_floor=46, bracket_cap=48, edge_pct=10.0,
            market_date="2026-03-10",
        )
        assert allowed is False
        assert "kill_switch" in reason


class TestMinEdge:
    def test_min_edge_blocks_low_edge(self, breakers):
        """3% edge with 5% threshold -> blocked."""
        allowed, reason = breakers.check(
            bracket_floor=46, bracket_cap=48, edge_pct=3.0,
            market_date="2026-03-10",
        )
        assert allowed is False
        assert "edge" in reason

    def test_sufficient_edge_passes(self, breakers):
        """8% edge with 5% threshold -> allowed (all other breakers clear)."""
        allowed, reason = breakers.check(
            bracket_floor=46, bracket_cap=48, edge_pct=8.0,
            market_date="2026-03-10",
        )
        assert allowed is True
        assert reason == "all_clear"


class TestMaxDailyLoss:
    def test_max_daily_loss_blocks(self, test_db):
        """After -$11 loss on the day -> blocked."""
        con = duckdb.connect(test_db)
        # Insert closed positions totaling -1100 cents net_pnl for the day
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             net_pnl, status, exit_reason, contracts)
            VALUES
            (1, 'nyc', '2026-03-10', 40, 42, 'YES',
             50, '2026-03-10 10:00:00', 0, '2026-03-10 12:00:00',
             -600, 'closed', 'settlement', 1),
            (2, 'nyc', '2026-03-10', 42, 44, 'YES',
             50, '2026-03-10 10:00:00', 0, '2026-03-10 13:00:00',
             -500, 'closed', 'settlement', 1)
        """)
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=44, bracket_cap=46, edge_pct=8.0,
            market_date="2026-03-10",
        )
        assert allowed is False
        assert "daily_loss" in reason


class TestMaxOpenPositions:
    def test_max_open_positions_blocks(self, test_db):
        """5 open positions -> blocked."""
        con = duckdb.connect(test_db)
        for i in range(5):
            con.execute("""
                INSERT INTO paper_positions
                (id, city, event_date, bracket_floor, bracket_cap, direction,
                 entry_price, entry_time, status, contracts)
                VALUES (?, 'nyc', '2026-03-10', ?, ?, 'YES',
                        50, '2026-03-10 10:00:00', 'open', 1)
            """, [i + 1, 40 + i * 2, 42 + i * 2])
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=50, bracket_cap=52, edge_pct=8.0,
            market_date="2026-03-10",
        )
        assert allowed is False
        assert "max_open" in reason


class TestMaxPerBracket:
    def test_max_per_bracket_blocks(self, test_db):
        """2 contracts on same bracket -> blocked."""
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, status, contracts)
            VALUES
            (1, 'nyc', '2026-03-10', 46, 48, 'YES',
             50, '2026-03-10 10:00:00', 'open', 2)
        """)
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=46, bracket_cap=48, edge_pct=8.0,
            market_date="2026-03-10",
        )
        assert allowed is False
        assert "max_per_bracket" in reason


class TestCooldown:
    def test_cooldown_blocks_recent_exit(self, test_db):
        """Exit 15 min ago, cooldown 30 min -> blocked."""
        now = datetime(2026, 3, 10, 14, 0, 0)
        exit_time = now - timedelta(minutes=15)

        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             net_pnl, status, exit_reason, contracts)
            VALUES
            (1, 'nyc', '2026-03-10', 46, 48, 'YES',
             50, '2026-03-10 10:00:00', 40, ?, -10, 'closed', 'stop_loss', 1)
        """, [exit_time])
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=46, bracket_cap=48, edge_pct=8.0,
            market_date="2026-03-10", now=now,
        )
        assert allowed is False
        assert "cooldown" in reason

    def test_cooldown_allows_after_expiry(self, test_db):
        """Exit 45 min ago, cooldown 30 min -> allowed."""
        now = datetime(2026, 3, 10, 14, 0, 0)
        exit_time = now - timedelta(minutes=45)

        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             net_pnl, status, exit_reason, contracts)
            VALUES
            (1, 'nyc', '2026-03-10', 46, 48, 'YES',
             50, '2026-03-10 10:00:00', 40, ?, -10, 'closed', 'stop_loss', 1)
        """, [exit_time])
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=46, bracket_cap=48, edge_pct=8.0,
            market_date="2026-03-10", now=now,
        )
        assert allowed is True
        assert reason == "all_clear"

    def test_cooldown_ignores_settlement_exits(self, test_db):
        """Settlement exit should not trigger cooldown."""
        now = datetime(2026, 3, 10, 14, 0, 0)
        exit_time = now - timedelta(minutes=5)

        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             entry_price, entry_time, exit_price, exit_time,
             net_pnl, status, exit_reason, contracts)
            VALUES
            (1, 'nyc', '2026-03-10', 46, 48, 'YES',
             50, '2026-03-10 10:00:00', 100, ?, 50, 'closed', 'settlement', 1)
        """, [exit_time])
        con.close()

        cb = CircuitBreakers(db_path=test_db)
        allowed, reason = cb.check(
            bracket_floor=46, bracket_cap=48, edge_pct=8.0,
            market_date="2026-03-10", now=now,
        )
        assert allowed is True
        assert reason == "all_clear"
