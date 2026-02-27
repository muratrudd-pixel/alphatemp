"""Tests for strategy backtester data structures and fee math.

Validates the foundational dataclasses and fee calculation functions
defined in services/strategy_backtester.py. Every downstream P&L
calculation depends on these being correct.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from services.strategy_backtester import (
    BacktestConfig,
    EventPortfolio,
    MarketSnapshot,
    ModelUpdate,
    Position,
    SanityFilter,
    TradeRecord,
    _ET,
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


class TestSanityFilter:
    """Tests for SanityFilter.check() and is_post_peak_check()."""

    def setup_method(self):
        self.cfg = BacktestConfig()
        self.sf = SanityFilter(self.cfg)
        # Defaults for passing all checks
        self.base_kwargs = dict(
            model_prob=0.25,
            market_ask_cents=30,
            spread_cents=5,
            model_std=2.0,
            bracket=(72.0, 74.0),
            is_post_peak=False,
            recently_exited=set(),
        )

    def test_reject_low_probability_bracket(self):
        ok, reason = self.sf.check(**{**self.base_kwargs, "model_prob": 0.03})
        assert ok is False
        assert "model_prob" in reason

    def test_accept_sufficient_probability(self):
        ok, reason = self.sf.check(**{**self.base_kwargs, "model_prob": 0.15})
        assert ok is True
        assert reason == ""

    def test_reject_post_peak(self):
        ok, reason = self.sf.check(**{**self.base_kwargs, "is_post_peak": True})
        assert ok is False
        assert "post_peak" in reason

    def test_reject_wide_spread(self):
        ok, reason = self.sf.check(**{**self.base_kwargs, "spread_cents": 15})
        assert ok is False
        assert "spread" in reason

    def test_reject_high_model_uncertainty(self):
        ok, reason = self.sf.check(**{**self.base_kwargs, "model_std": 4.0})
        assert ok is False
        assert "uncertainty" in reason or "std" in reason

    def test_reject_recently_exited_bracket(self):
        bracket = (72.0, 74.0)
        ok, reason = self.sf.check(
            **{**self.base_kwargs, "bracket": bracket, "recently_exited": {bracket}}
        )
        assert ok is False
        assert "churn" in reason or "reentry" in reason

    def test_post_peak_detection_declining_temps(self):
        """5 obs, peak at 1 PM ET, declining at 3 PM ET -> True."""
        # Build observations in UTC (ET + 5h)
        base = datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc)  # 10 AM ET
        obs = [
            (base, 78.0),                                          # 10 AM ET
            (base + timedelta(hours=2), 82.0),                     # 12 PM ET
            (base + timedelta(hours=3), 84.0),                     # 1 PM ET (peak)
            (base + timedelta(hours=4), 83.0),                     # 2 PM ET
            (base + timedelta(hours=5), 81.5),                     # 3 PM ET
        ]
        current = base + timedelta(hours=5)  # 3 PM ET = 20:00 UTC
        assert SanityFilter.is_post_peak_check(obs, current) is True

    def test_not_post_peak_still_rising(self):
        """3 obs, still rising at 1:30 PM ET -> False."""
        base = datetime(2025, 7, 15, 16, 0, tzinfo=timezone.utc)  # 11 AM ET
        obs = [
            (base, 78.0),
            (base + timedelta(hours=1), 80.0),
            (base + timedelta(hours=2), 82.0),  # still rising
        ]
        current = base + timedelta(hours=2, minutes=30)  # 1:30 PM ET
        assert SanityFilter.is_post_peak_check(obs, current) is False

    def test_not_post_peak_too_early(self):
        """Declining but before 2 PM ET -> False."""
        base = datetime(2025, 7, 15, 15, 0, tzinfo=timezone.utc)  # 10 AM ET
        obs = [
            (base, 80.0),
            (base + timedelta(hours=1), 82.0),
            (base + timedelta(hours=2), 81.0),  # declining
            (base + timedelta(hours=3), 79.0),  # declining
        ]
        current = base + timedelta(hours=3)  # 1 PM ET = 18:00 UTC
        assert SanityFilter.is_post_peak_check(obs, current) is False


class TestEventPortfolio:
    """Tests for EventPortfolio.combined_ev() and optimal_subset()."""

    def setup_method(self):
        self.portfolio = EventPortfolio(event_date=date(2025, 7, 15))

    def test_single_bracket_ev(self):
        ev = self.portfolio.combined_ev(entry_prices=[40], model_probs=[0.50])
        assert ev == pytest.approx(6.6, abs=0.1)

    def test_single_bracket_negative_ev(self):
        ev = self.portfolio.combined_ev(entry_prices=[45], model_probs=[0.48])
        assert ev < 0

    def test_two_bracket_portfolio_ev(self):
        ev = self.portfolio.combined_ev(
            entry_prices=[22, 17], model_probs=[0.35, 0.30]
        )
        assert ev == pytest.approx(20.39, abs=0.1)

    def test_adding_bracket_can_decrease_ev(self):
        """Adding an expensive, low-prob bracket hurts the portfolio."""
        ev_one = self.portfolio.combined_ev(
            entry_prices=[25], model_probs=[0.40]
        )
        ev_two = self.portfolio.combined_ev(
            entry_prices=[25, 60], model_probs=[0.40, 0.10]
        )
        assert ev_two < ev_one

    def test_optimal_subset_picks_best_single(self):
        """One good bracket, one bad -> picks only the good one."""
        candidates = [
            ((72.0, 74.0), 30, 0.45),  # good
            ((80.0, 82.0), 55, 0.10),  # bad (negative EV)
        ]
        selected = self.portfolio.optimal_subset(candidates)
        assert len(selected) == 1
        assert selected[0][0] == (72.0, 74.0)

    def test_optimal_subset_picks_both_when_beneficial(self):
        """Two cheap adjacent brackets -> picks both."""
        candidates = [
            ((72.0, 74.0), 20, 0.35),
            ((74.0, 76.0), 18, 0.30),
        ]
        selected = self.portfolio.optimal_subset(candidates)
        assert len(selected) == 2

    def test_empty_candidates_returns_empty(self):
        selected = self.portfolio.optimal_subset([])
        assert selected == []
