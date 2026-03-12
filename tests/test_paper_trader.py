# tests/test_paper_trader.py
"""Tests for PaperTrader service shell — fee math, position lifecycle, settlement."""
import asyncio

import duckdb
import pytest

from services.paper_trader import PaperTrader


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    # Minimal schema matching production paper_positions (no sequence — manual IDs)
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
    con.execute("""
        CREATE TABLE nws_daily (
            station_id VARCHAR,
            obs_date DATE,
            max_temp_f DOUBLE,
            min_temp_f DOUBLE,
            source VARCHAR,
            ingested_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE market_ticks (
            market_id VARCHAR,
            city VARCHAR,
            captured_at TIMESTAMP,
            yes_bid DOUBLE,
            yes_ask DOUBLE,
            no_bid DOUBLE,
            no_ask DOUBLE,
            last_trade DOUBLE,
            volume INTEGER,
            floor_strike DOUBLE,
            cap_strike DOUBLE
        )
    """)
    con.close()
    return db_path


@pytest.fixture
def trader(test_db):
    return PaperTrader(db_path=test_db)


class TestFeeComputation:
    def test_fee_at_50_cents(self, trader):
        """At 50c, fee = max(ceil(0.07 * 1 * 0.5 * 0.5 * 100) / 100, 0.01) = 0.02"""
        assert trader._compute_fee(50, 1) == 0.02

    def test_fee_minimum_floor(self, trader):
        """At extreme prices, fee should be at least $0.01 per contract."""
        assert trader._compute_fee(99, 1) == 0.01
        assert trader._compute_fee(1, 1) == 0.01

    def test_fee_scales_with_contracts(self, trader):
        """More contracts = higher fee."""
        fee_1 = trader._compute_fee(50, 1)
        fee_5 = trader._compute_fee(50, 5)
        assert fee_5 > fee_1

    def test_fee_symmetric(self, trader):
        """Fee at 30c should equal fee at 70c (P * (1-P) is symmetric)."""
        assert trader._compute_fee(30, 1) == trader._compute_fee(70, 1)

    def test_fee_zero_price_uses_minimum(self, trader):
        """Edge case: 0 cents means P=0, so P*(1-P)=0, falls to minimum."""
        assert trader._compute_fee(0, 1) == 0.01

    def test_fee_100_price_uses_minimum(self, trader):
        """Edge case: 100 cents means P=1, so P*(1-P)=0, falls to minimum."""
        assert trader._compute_fee(100, 1) == 0.01


class TestPositionLifecycle:
    def test_record_entry(self, trader, test_db):
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=2,
        )
        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT * FROM paper_positions").fetchone()
        con.close()
        assert row is not None
        # Verify key fields (column order matches schema)
        assert row[0] == 1       # id (auto-generated)
        assert row[1] == "NYC"   # city
        assert row[5] == "YES"   # direction
        assert row[17] == "open" # status
        assert row[19] == 2      # contracts

    def test_record_entry_increments_id(self, trader, test_db):
        """Second entry should get id=2."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57,
        )
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=48, bracket_cap=50,
            direction="NO", model_prob=0.38,
            market_price=42, entry_price=58,
        )
        con = duckdb.connect(test_db, read_only=True)
        ids = con.execute("SELECT id FROM paper_positions ORDER BY id").fetchall()
        con.close()
        assert ids == [(1,), (2,)]

    def test_edge_computation(self, trader, test_db):
        """Edge = model_prob - (market_price / 100)."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57,
        )
        con = duckdb.connect(test_db, read_only=True)
        edge = con.execute("SELECT edge FROM paper_positions").fetchone()[0]
        con.close()
        assert abs(edge - 0.04) < 0.001  # 0.62 - 0.58 = 0.04

    @pytest.mark.asyncio
    async def test_settle_position_yes_wins(self, trader, test_db):
        """YES position wins when actual high falls in bracket."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=1,
        )
        # Add settlement data — 47°F is in [46, 48)
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-02-28', 47, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute(
            "SELECT status, settled_yes, gross_pnl, net_pnl, exit_reason "
            "FROM paper_positions"
        ).fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is True   # settled_yes
        assert row[2] > 0       # gross_pnl positive (won)
        assert row[3] < row[2]  # net < gross (fees deducted)
        assert row[4] == "settlement"

    @pytest.mark.asyncio
    async def test_settle_position_yes_loses(self, trader, test_db):
        """YES position loses when actual high falls outside bracket."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=1,
        )
        # 50°F is outside [46, 48)
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-02-28', 50, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute(
            "SELECT status, settled_yes, gross_pnl, net_pnl "
            "FROM paper_positions"
        ).fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is False
        assert row[2] < 0  # gross_pnl negative (lost)

    @pytest.mark.asyncio
    async def test_settle_position_no_wins(self, trader, test_db):
        """NO position wins when actual high falls outside bracket."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="NO", model_prob=0.38,
            market_price=42, entry_price=58, contracts=1,
        )
        # 50°F is outside [46, 48) — NO wins
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-02-28', 50, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute(
            "SELECT status, settled_yes, gross_pnl, exit_reason "
            "FROM paper_positions"
        ).fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is False  # settled outside bracket
        assert row[2] > 0       # NO side wins
        assert row[3] == "settlement"

    @pytest.mark.asyncio
    async def test_settle_ignores_non_cli(self, trader, test_db):
        """Settlement only uses NWS_CLI source, not ACIS or other sources."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=1,
        )
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-02-28', 47, 'ACIS')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT status FROM paper_positions").fetchone()
        con.close()
        assert row[0] == "open"  # Should NOT have settled

    @pytest.mark.asyncio
    async def test_settle_bracket_boundary_exclusive_cap(self, trader, test_db):
        """Bracket is [floor, cap) — cap value itself is NOT in the bracket."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=1,
        )
        # 48°F is exactly at cap — should NOT be in bracket [46, 48)
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source)
            VALUES ('KNYC', '2026-02-28', 48, 'NWS_CLI')
        """)
        con.close()

        await trader._settle_positions()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute(
            "SELECT status, settled_yes FROM paper_positions"
        ).fetchone()
        con.close()
        assert row[0] == "closed"
        assert row[1] is False  # 48 is NOT in [46, 48)


class TestUnrealizedPnL:
    @pytest.mark.asyncio
    async def test_update_unrealized_yes(self, trader, test_db):
        """Unrealized P&L for YES: (market_mid - entry_price) * contracts / 100."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=1,
        )
        # Insert a market tick with mid = 62
        con = duckdb.connect(test_db)
        con.execute("""
            INSERT INTO market_ticks
            (market_id, city, captured_at, yes_bid, yes_ask, floor_strike, cap_strike)
            VALUES ('MKT1', 'NYC', '2026-02-28 12:00:00', 60, 64, 46, 48)
        """)
        con.close()

        await trader._update_unrealized()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT unrealized_pnl FROM paper_positions").fetchone()
        con.close()
        # mid = 62, entry = 57, unrealized = (62-57)*1/100 = 0.05
        assert abs(row[0] - 0.05) < 0.001

    @pytest.mark.asyncio
    async def test_update_unrealized_no_tick(self, trader, test_db):
        """No market tick available — unrealized should stay at default (0.0)."""
        trader._record_entry(
            city="NYC", event_date="2026-02-28",
            bracket_floor=46, bracket_cap=48,
            direction="YES", model_prob=0.62,
            market_price=58, entry_price=57, contracts=1,
        )

        await trader._update_unrealized()

        con = duckdb.connect(test_db, read_only=True)
        row = con.execute("SELECT unrealized_pnl FROM paper_positions").fetchone()
        con.close()
        assert row[0] == 0.0  # default, unchanged


class TestPlaceholderStrategy:
    @pytest.mark.asyncio
    async def test_check_entries_is_noop(self, trader):
        """Placeholder strategy should not create any positions."""
        await trader._check_entries()
        # No assertion on DB — just verifying it doesn't raise

    @pytest.mark.asyncio
    async def test_check_exits_is_noop(self, trader):
        """Placeholder exit logic should not modify anything."""
        await trader._check_exits()


class TestEntryExitMethods:
    @pytest.mark.asyncio
    async def test_enter_position_records_trade(self, trader, test_db):
        """enter_position() should insert an open position."""
        await trader.enter_position(
            city="NYC", event_date="2026-03-01",
            bracket_floor=50, bracket_cap=52,
            direction="YES", model_prob=0.65,
            market_price=55, edge=0.10,
        )
        con = duckdb.connect(test_db, read_only=True)
        try:
            row = con.execute(
                "SELECT id, city, direction, status, entry_price, contracts "
                "FROM paper_positions"
            ).fetchone()
        finally:
            con.close()
        assert row is not None
        assert row[0] == 1       # id
        assert row[1] == "NYC"   # city
        assert row[2] == "YES"   # direction
        assert row[3] == "open"  # status
        assert row[4] == 55      # entry_price = market_price (paper trade at ask)
        assert row[5] == 1       # contracts

    @pytest.mark.asyncio
    async def test_exit_position_closes_trade(self, trader, test_db):
        """exit_position() should mark position as closed with reason and P&L."""
        # Enter a position first
        await trader.enter_position(
            city="NYC", event_date="2026-03-01",
            bracket_floor=50, bracket_cap=52,
            direction="YES", model_prob=0.65,
            market_price=55, edge=0.10,
        )

        # Exit at a higher price (profit)
        await trader.exit_position(position_id=1, exit_price=70, reason="edge_reversal")

        con = duckdb.connect(test_db, read_only=True)
        try:
            row = con.execute(
                "SELECT status, exit_price, exit_reason, gross_pnl, net_pnl, exit_time "
                "FROM paper_positions WHERE id = 1"
            ).fetchone()
        finally:
            con.close()
        assert row[0] == "closed"
        assert row[1] == 70              # exit_price
        assert row[2] == "edge_reversal" # exit_reason
        assert row[5] is not None        # exit_time set

    @pytest.mark.asyncio
    async def test_exit_position_pnl_calculation(self, trader, test_db):
        """Exit P&L should be (exit - entry) * contracts / 100 - fees."""
        # Buy YES at 40c
        await trader.enter_position(
            city="NYC", event_date="2026-03-01",
            bracket_floor=50, bracket_cap=52,
            direction="YES", model_prob=0.65,
            market_price=40, edge=0.25,
        )

        # Sell at 60c — gross = (60 - 40) * 1 / 100 = 0.20
        await trader.exit_position(position_id=1, exit_price=60, reason="take_profit")

        con = duckdb.connect(test_db, read_only=True)
        try:
            row = con.execute(
                "SELECT gross_pnl, fees, net_pnl FROM paper_positions WHERE id = 1"
            ).fetchone()
        finally:
            con.close()
        gross_pnl, fees, net_pnl = row
        # Gross = (60 - 40) * 1 / 100 = 0.20
        assert abs(gross_pnl - 0.20) < 0.001
        # Fees = entry fee + exit fee
        entry_fee = trader._compute_fee(40, 1)
        exit_fee = trader._compute_fee(60, 1)
        expected_fees = round(entry_fee + exit_fee, 2)
        assert abs(fees - expected_fees) < 0.001
        # Net = gross - total fees
        expected_net = round(0.20 - expected_fees, 2)
        assert abs(net_pnl - expected_net) < 0.001

    @pytest.mark.asyncio
    async def test_exit_position_no_direction_loss(self, trader, test_db):
        """NO direction exit: bought NO at 70c, NO price drops to 50c = loss."""
        # Buy NO at 70c (YES price was 30c)
        await trader.enter_position(
            city="NYC", event_date="2026-03-01",
            bracket_floor=50, bracket_cap=52,
            direction="NO", model_prob=0.70,
            market_price=70, edge=0.05,
        )

        # Sell NO at 50c — gross = (50 - 70) * 1 / 100 = -0.20 (loss)
        await trader.exit_position(position_id=1, exit_price=50, reason="stop_loss")

        con = duckdb.connect(test_db, read_only=True)
        try:
            row = con.execute(
                "SELECT gross_pnl, net_pnl FROM paper_positions WHERE id = 1"
            ).fetchone()
        finally:
            con.close()
        assert row[0] < 0   # gross loss
        assert row[1] < row[0]  # net even worse after fees

    @pytest.mark.asyncio
    async def test_exit_nonexistent_position_raises(self, trader, test_db):
        """exit_position() on non-existent ID should raise ValueError."""
        with pytest.raises(ValueError, match="Position 999 not found"):
            await trader.exit_position(position_id=999, exit_price=50, reason="test")
