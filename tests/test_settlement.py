# tests/test_settlement.py
"""Tests for SettlementService — settlement resolution, P&L, and CLI gating."""
import asyncio
from datetime import datetime, timezone

import duckdb
import pytest

from services.settlement import SettlementService


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
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
    con.close()
    return db_path


@pytest.fixture
def service(test_db):
    return SettlementService(db_path=test_db)


def _insert_position(db_path, pos_id, event_date, bracket_floor, bracket_cap,
                      direction, entry_price, contracts=1, fees=0.02):
    """Helper to insert an open paper position."""
    con = duckdb.connect(db_path)
    try:
        con.execute("""
            INSERT INTO paper_positions
            (id, city, event_date, bracket_floor, bracket_cap, direction,
             model_prob, market_price, edge, entry_price, entry_time,
             fees, status, contracts)
            VALUES (?, 'NYC', ?, ?, ?, ?, 0.30, 40.0, 0.10, ?, ?, ?, 'open', ?)
        """, [
            pos_id, event_date, bracket_floor, bracket_cap, direction,
            entry_price, datetime.now(timezone.utc), fees, contracts,
        ])
    finally:
        con.close()


def _insert_nws(db_path, obs_date, max_temp_f, source="NWS_CLI"):
    """Helper to insert NWS daily observation."""
    con = duckdb.connect(db_path)
    try:
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
            VALUES ('KNYC', ?, ?, 30.0, ?, ?)
        """, [obs_date, max_temp_f, source, datetime.now(timezone.utc)])
    finally:
        con.close()


def _get_position(db_path, pos_id):
    """Helper to fetch a position by ID."""
    con = duckdb.connect(db_path)
    try:
        row = con.execute(
            "SELECT * FROM paper_positions WHERE id = ?", [pos_id]
        ).fetchone()
        cols = [d[0] for d in con.description]
        return dict(zip(cols, row)) if row else None
    finally:
        con.close()


class TestSettlementResolvesWinningYes:
    def test_yes_wins_when_actual_in_bracket(self, service, test_db):
        """YES position wins when actual high falls in [floor, cap)."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["status"] == "closed"
        assert pos["exit_reason"] == "settlement"
        assert pos["settled_yes"] is True
        # Winner: gross = (100 - 40) * 1 / 100.0 = 0.60
        assert pos["gross_pnl"] == 0.60
        assert pos["exit_price"] == 100

    def test_yes_wins_at_floor_boundary(self, service, test_db):
        """YES wins when actual high equals the floor exactly (inclusive)."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 74.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["settled_yes"] is True
        assert pos["gross_pnl"] == 0.60


class TestSettlementResolvesLosingYes:
    def test_yes_loses_when_actual_outside_bracket(self, service, test_db):
        """YES position loses when actual high outside bracket."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 73.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["status"] == "closed"
        assert pos["settled_yes"] is False
        # Loser: gross = -(40 * 1 / 100.0) = -0.40
        assert pos["gross_pnl"] == -0.40
        assert pos["exit_price"] == 0

    def test_yes_wins_at_cap_boundary(self, service, test_db):
        """YES wins when actual high equals the cap exactly (inclusive per Kalshi CFTC)."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 76.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["settled_yes"] is True
        assert pos["gross_pnl"] == 0.60


class TestSettlementResolvesWinningNo:
    def test_no_wins_when_actual_outside_bracket(self, service, test_db):
        """NO position wins when actual high outside bracket."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "NO", 60)
        _insert_nws(test_db, "2026-03-10", 73.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["status"] == "closed"
        assert pos["settled_yes"] is False
        # NO winner: gross = (100 - 60) * 1 / 100.0 = 0.40
        assert pos["gross_pnl"] == 0.40
        assert pos["exit_price"] == 100


class TestSettlementResolvesLosingNo:
    def test_no_loses_when_actual_in_bracket(self, service, test_db):
        """NO position loses when actual high in bracket."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "NO", 60)
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["status"] == "closed"
        assert pos["settled_yes"] is True
        # NO loser: gross = -(60 * 1 / 100.0) = -0.60
        assert pos["gross_pnl"] == -0.60
        assert pos["exit_price"] == 0


class TestNoSettlementWithoutCLI:
    def test_no_settlement_with_acis_source(self, service, test_db):
        """No settlement if nws_daily only has ACIS source (not CLI/DSM)."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 75.0, "ACIS")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 0
        assert pos["status"] == "open"

    def test_no_settlement_without_nws_data(self, service, test_db):
        """No settlement if no nws_daily row exists for the date."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 0
        assert pos["status"] == "open"


class TestSettlementExitPrices:
    def test_winner_gets_exit_price_100(self, service, test_db):
        """Winner gets exit_price=100."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert pos["exit_price"] == 100

    def test_loser_gets_exit_price_0(self, service, test_db):
        """Loser gets exit_price=0."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 73.0, "NWS_CLI")

        service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert pos["exit_price"] == 0


class TestSettlementFeeAndPnL:
    def test_net_pnl_is_gross_minus_entry_fee(self, service, test_db):
        """Net P&L = gross - entry fee. No settlement fee on Kalshi."""
        # Entry at 50c: fee = max(ceil(0.07 * 1 * 0.5 * 0.5 * 100)/100, 0.01) = 0.02
        fee = service._compute_fee(50, 1)
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 50, fees=fee)
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        # Winner: gross = (100-50)*1/100 = 0.50, fee = 0.02, net = 0.48
        assert pos["gross_pnl"] == 0.50
        assert pos["fees"] == fee
        assert pos["net_pnl"] == round(0.50 - fee, 2)

    def test_loser_net_pnl_includes_fee(self, service, test_db):
        """Loser net P&L: -(entry * qty / 100) - fee."""
        fee = service._compute_fee(50, 1)
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 50, fees=fee)
        _insert_nws(test_db, "2026-03-10", 73.0, "NWS_CLI")

        service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        # Loser: gross = -(50*1/100) = -0.50, net = -0.50 - 0.02 = -0.52
        assert pos["gross_pnl"] == -0.50
        assert pos["net_pnl"] == round(-0.50 - fee, 2)

    def test_multiple_contracts(self, service, test_db):
        """Fee and P&L scale with contract count."""
        fee = service._compute_fee(40, 3)
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40, contracts=3, fees=fee)
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        # Winner: gross = (100-40)*3/100 = 1.80
        assert pos["gross_pnl"] == 1.80
        assert pos["net_pnl"] == round(1.80 - fee, 2)


class TestSettlementMultiplePositions:
    def test_settles_multiple_positions_same_date(self, service, test_db):
        """Multiple positions on the same date are all settled."""
        fee = service._compute_fee(40, 1)
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40, fees=fee)
        _insert_position(test_db, 2, "2026-03-10", 76, 78, "YES", 30, fees=service._compute_fee(30, 1))
        _insert_position(test_db, 3, "2026-03-10", 74, 76, "NO", 60, fees=service._compute_fee(60, 1))
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        count = service.settle_date("2026-03-10")

        assert count == 3
        p1 = _get_position(test_db, 1)
        p2 = _get_position(test_db, 2)
        p3 = _get_position(test_db, 3)
        # p1: YES 74-76, actual=75 → winner
        assert p1["settled_yes"] is True
        assert p1["gross_pnl"] == 0.60
        # p2: YES 76-78, actual=75 → loser
        assert p2["settled_yes"] is False
        assert p2["gross_pnl"] == -0.30
        # p3: NO 74-76, actual=75 → loser (bracket hit, NO loses)
        assert p3["settled_yes"] is True
        assert p3["gross_pnl"] == -0.60

    def test_does_not_resettle_closed_positions(self, service, test_db):
        """Already-closed positions are not re-settled."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 75.0, "NWS_CLI")

        # Close it manually
        con = duckdb.connect(test_db)
        con.execute("UPDATE paper_positions SET status = 'closed' WHERE id = 1")
        con.close()

        count = service.settle_date("2026-03-10")
        assert count == 0


class TestGetUnsettledDates:
    def test_returns_dates_with_open_positions(self, service, test_db):
        """_get_unsettled_dates returns dates that have open positions."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_position(test_db, 2, "2026-03-11", 74, 76, "YES", 40)

        dates = service._get_unsettled_dates()
        assert len(dates) == 2

    def test_excludes_closed_positions(self, service, test_db):
        """_get_unsettled_dates excludes dates where all positions are closed."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        con = duckdb.connect(test_db)
        con.execute("UPDATE paper_positions SET status = 'closed' WHERE id = 1")
        con.close()

        dates = service._get_unsettled_dates()
        assert len(dates) == 0


@pytest.mark.asyncio
async def test_settlement_cap_inclusive(test_db):
    """Interior brackets: [floor, cap] both inclusive."""
    db_path = test_db
    con = duckdb.connect(db_path)
    con.execute("""
        INSERT INTO paper_positions (id, city, event_date, bracket_floor, bracket_cap,
            direction, model_prob, market_price, edge, entry_price, entry_time,
            fees, status, contracts)
        VALUES (999, 'NYC', '2026-01-15', 64, 66, 'YES', 0.5, 50, 5.0, 50,
            '2026-01-15 12:00:00', 0.04, 'open', 1)
    """)
    con.execute("""
        INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
        VALUES ('KNYC', '2026-01-15', 66, 40, 'NWS_CLI', '2026-01-15 22:00:00')
    """)
    con.close()

    svc = SettlementService(db_path)
    await svc._settle_date('2026-01-15')

    con = duckdb.connect(db_path, read_only=True)
    row = con.execute("SELECT settled_yes, net_pnl FROM paper_positions WHERE id = 999").fetchone()
    con.close()
    assert row[0] is True, "Temp at cap boundary (66) should settle YES"


@pytest.mark.asyncio
async def test_settlement_lower_tail(test_db):
    """Lower tail: temp < cap."""
    db_path = test_db
    con = duckdb.connect(db_path)
    con.execute("""
        INSERT INTO paper_positions (id, city, event_date, bracket_floor, bracket_cap,
            direction, model_prob, market_price, edge, entry_price, entry_time,
            fees, status, contracts)
        VALUES (998, 'NYC', '2026-01-15', NULL, 50, 'YES', 0.5, 50, 5.0, 50,
            '2026-01-15 12:00:00', 0.04, 'open', 1)
    """)
    con.execute("""
        INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
        VALUES ('KNYC', '2026-01-15', 48, 30, 'NWS_CLI', '2026-01-15 22:00:00')
    """)
    con.close()

    svc = SettlementService(db_path)
    await svc._settle_date('2026-01-15')

    con = duckdb.connect(db_path, read_only=True)
    row = con.execute("SELECT settled_yes FROM paper_positions WHERE id = 998").fetchone()
    con.close()
    assert row[0] is True, "Lower tail: 48 < 50 should settle YES"


class TestDSMSourceSettles:
    def test_dsm_source_triggers_settlement(self, service, test_db):
        """DSM is also an authoritative source — should trigger settlement."""
        _insert_position(test_db, 1, "2026-03-10", 74, 76, "YES", 40)
        _insert_nws(test_db, "2026-03-10", 75.0, "DSM")

        count = service.settle_date("2026-03-10")

        pos = _get_position(test_db, 1)
        assert count == 1
        assert pos["status"] == "closed"
        assert pos["settled_yes"] is True
