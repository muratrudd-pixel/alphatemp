"""Tests for strategy backtester data structures and fee math.

Validates the foundational dataclasses and fee calculation functions
defined in services/strategy_backtester.py. Every downstream P&L
calculation depends on these being correct.
"""

from datetime import date, datetime, timezone

import pytest

from services.strategy_backtester import (
    BacktestConfig,
    MarketSnapshot,
    ModelUpdate,
    Position,
    TradeRecord,
    compute_entry_cost,
    compute_exit_pnl,
    compute_settlement_pnl,
)


class TestFeeMath:
    def test_entry_cost_includes_trading_fee(self):
        cost = compute_entry_cost(price_cents=40, quantity=1)
        assert cost == pytest.approx(40.4)

    def test_entry_cost_multiple_contracts(self):
        cost = compute_entry_cost(price_cents=25, quantity=3)
        assert cost == pytest.approx(75.75)

    def test_settlement_pnl_winner(self):
        pnl = compute_settlement_pnl(entry_price_cents=40, quantity=1, settled_yes=True)
        assert pnl == pytest.approx(53.6)

    def test_settlement_pnl_loser(self):
        pnl = compute_settlement_pnl(entry_price_cents=40, quantity=1, settled_yes=False)
        assert pnl == pytest.approx(-40.4)

    def test_settlement_pnl_winner_multiple_contracts(self):
        pnl = compute_settlement_pnl(entry_price_cents=30, quantity=5, settled_yes=True)
        assert pnl == pytest.approx(313.5)

    def test_exit_pnl_profitable_sell(self):
        pnl = compute_exit_pnl(entry_price_cents=30, exit_price_cents=50, quantity=1)
        assert pnl == pytest.approx(19.2)

    def test_exit_pnl_loss_sell(self):
        pnl = compute_exit_pnl(entry_price_cents=40, exit_price_cents=25, quantity=1)
        assert pnl == pytest.approx(-15.65)

    def test_minimum_profitable_displacement(self):
        pnl_at_breakeven = compute_settlement_pnl(entry_price_cents=50, quantity=1, settled_yes=True)
        assert pnl_at_breakeven == pytest.approx(44.5)


class TestDataStructures:
    def test_backtest_config_defaults(self):
        cfg = BacktestConfig()
        assert cfg.starting_capital == 100.0
        assert cfg.start_date == date(2024, 11, 1)
        assert cfg.min_displacement == 0.12
        assert cfg.min_model_prob == 0.05
        assert cfg.bootstrap_iterations == 10000
        assert cfg.execution_latency_seconds == 60

    def test_position_capital_locked(self):
        pos = Position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price=35.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=2,
            capital_locked=70.7,
        )
        assert pos.capital_locked == pytest.approx(70.7)

    def test_market_snapshot_stale_flag(self):
        snap = MarketSnapshot(
            bracket=(70.0, 72.0),
            timestamp=datetime(2025, 6, 15, 3, 0, tzinfo=timezone.utc),
            yes_bid=0.20,
            yes_ask=0.25,
            volume=0,
            is_stale=True,
        )
        assert snap.is_stale is True
        assert snap.yes_ask - snap.yes_bid == pytest.approx(0.05)
