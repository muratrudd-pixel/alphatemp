"""Tests for strategy backtester data structures and fee math.

Validates the foundational dataclasses and fee calculation functions
defined in services/strategy_backtester.py. Every downstream P&L
calculation depends on these being correct.
"""

import os
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from core.db import init_db
from services.strategy_backtester import (
    BacktestConfig,
    EdgeAnalyzer,
    EventPortfolio,
    MarketDataLoader,
    MarketSnapshot,
    ModelUpdate,
    PnLSimulator,
    Position,
    PositionManager,
    SanityFilter,
    TradeRecord,
    TriggerDetector,
    _ET,
    compute_entry_cost,
    compute_exit_pnl,
    compute_settlement_pnl,
)


TEST_DB = "tests/test_strategy_backtest.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


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


class TestPositionManager:
    """Tests for PositionManager capital tracking and settlement."""

    def setup_method(self):
        self.t0 = datetime(2025, 7, 15, 14, 0, tzinfo=timezone.utc)
        self.event_date = date(2025, 7, 15)

    def test_initial_state(self):
        pm = PositionManager(bankroll=100.0)
        assert pm.capital_available == pytest.approx(100.0)
        assert pm.capital_locked == pytest.approx(0.0)

    def test_open_position_locks_capital(self):
        pm = PositionManager(bankroll=100.0)
        pos = pm.open_position(
            self.event_date, (72.0, 74.0), 30, self.t0, 1
        )
        assert pos is not None
        # 30 + 30*0.01 = 30.3 cents = $0.303
        assert pm.capital_locked == pytest.approx(0.303)

    def test_reject_when_insufficient_capital(self):
        pm = PositionManager(bankroll=0.20)  # only $0.20
        pos = pm.open_position(
            self.event_date, (72.0, 74.0), 30, self.t0, 1
        )
        # 30.3 cents = $0.303 > $0.20
        assert pos is None

    def test_close_position_frees_capital(self):
        pm = PositionManager(bankroll=100.0)
        pm.open_position(self.event_date, (72.0, 74.0), 30, self.t0, 1)
        pnl = pm.close_position(
            self.event_date, (72.0, 74.0), 45, self.t0 + timedelta(hours=1)
        )
        # exit_pnl(30, 45, 1) = 45 - 0.45 - 30 - 0.3 = 14.25
        assert pnl == pytest.approx(14.25)
        assert len(pm.positions) == 0

    def test_settle_day_winner(self):
        pm = PositionManager(bankroll=100.0)
        pm.open_position(self.event_date, (72.0, 74.0), 30, self.t0, 1)
        records = pm.settle_day(self.event_date, (72.0, 74.0))
        assert len(records) == 1
        # settlement_pnl(30, 1, True) = 70*0.9 - 30*0.01 = 63 - 0.3 = 62.7
        assert records[0].pnl == pytest.approx(62.7)
        assert records[0].settlement_result == 1

    def test_settle_day_loser(self):
        pm = PositionManager(bankroll=100.0)
        pm.open_position(self.event_date, (72.0, 74.0), 25, self.t0, 1)
        records = pm.settle_day(self.event_date, (76.0, 78.0))  # different bracket wins
        assert len(records) == 1
        # settlement_pnl(25, 1, False) = -(25 + 0.25) = -25.25
        assert records[0].pnl == pytest.approx(-25.25)
        assert records[0].settlement_result == 0

    def test_settle_day_mixed(self):
        pm = PositionManager(bankroll=100.0)
        pm.open_position(self.event_date, (72.0, 74.0), 30, self.t0, 1)
        pm.open_position(self.event_date, (74.0, 76.0), 25, self.t0, 1)
        records = pm.settle_day(self.event_date, (72.0, 74.0))
        assert len(records) == 2
        # One winner, one loser
        results = {r.bracket: r for r in records}
        assert results[(72.0, 74.0)].settlement_result == 1
        assert results[(74.0, 76.0)].settlement_result == 0

    def test_positions_for_date(self):
        pm = PositionManager(bankroll=100.0)
        d1 = date(2025, 7, 15)
        d2 = date(2025, 7, 16)
        pm.open_position(d1, (72.0, 74.0), 30, self.t0, 1)
        pm.open_position(d2, (74.0, 76.0), 25, self.t0, 1)
        assert len(pm.positions_for(d1)) == 1
        assert len(pm.positions_for(d2)) == 1
        assert pm.positions_for(d1)[0].bracket == (72.0, 74.0)

    def test_bankroll_updates_after_settlement(self):
        pm = PositionManager(bankroll=100.0)
        pm.open_position(self.event_date, (72.0, 74.0), 30, self.t0, 1)
        initial_bankroll = pm.bankroll
        records = pm.settle_day(self.event_date, (72.0, 74.0))
        pnl_dollars = records[0].pnl / 100.0
        assert pm.bankroll == pytest.approx(initial_bankroll + pnl_dollars)


class TestMarketDataLoader:
    def _seed_kalshi_data(self, db_path):
        con = duckdb.connect(db_path)
        con.execute("""
            INSERT INTO kalshi_settlements VALUES
            ('KXHIGHNY-25JUN15-T72-B74', 'KXHIGHNY-25JUN15', 'KXHIGHNY',
             'NYC', 'high', '2025-06-15', 72.0, 74.0, 1, 500,
             '2025-06-15 23:00:00', CURRENT_TIMESTAMP),
            ('KXHIGHNY-25JUN15-T70-B72', 'KXHIGHNY-25JUN15', 'KXHIGHNY',
             'NYC', 'high', '2025-06-15', 70.0, 72.0, 0, 300,
             '2025-06-15 23:00:00', CURRENT_TIMESTAMP)
        """)
        con.execute("""
            INSERT INTO kalshi_candlesticks VALUES
            ('KXHIGHNY-25JUN15-T72-B74', '2025-06-15 14:00:00', 1,
             0.30, 0.35, 0.32, 0.34, 0.31, 0.33, 10, 50),
            ('KXHIGHNY-25JUN15-T72-B74', '2025-06-15 14:01:00', 1,
             0.31, 0.36, 0.33, 0.35, 0.32, 0.34, 5, 50),
            ('KXHIGHNY-25JUN15-T72-B74', '2025-06-15 14:03:00', 1,
             0.32, 0.37, 0.34, 0.36, 0.33, 0.35, 8, 55)
        """)
        con.close()

    @pytest.fixture
    def market_db(self, test_db):
        self._seed_kalshi_data(test_db)
        return test_db

    def test_load_brackets_for_date(self, market_db):
        loader = MarketDataLoader(market_db)
        brackets = loader.get_brackets(date(2025, 6, 15))
        assert len(brackets) == 2
        settled = [b for b in brackets if b["settled_yes"] == 1]
        assert len(settled) == 1
        assert settled[0]["bracket"] == (72.0, 74.0)

    def test_load_snapshots_for_bracket(self, market_db):
        loader = MarketDataLoader(market_db)
        snaps = loader.get_snapshots(
            market_ticker='KXHIGHNY-25JUN15-T72-B74',
            bracket=(72.0, 74.0),
            start_ts=datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc),
            end_ts=datetime(2025, 6, 15, 14, 4, tzinfo=timezone.utc),
        )
        assert len(snaps) == 4  # 3 real + 1 forward-filled at 14:02
        stale = [s for s in snaps if s.is_stale]
        assert len(stale) == 1

    def test_snapshot_prices_are_probabilities(self, market_db):
        loader = MarketDataLoader(market_db)
        snaps = loader.get_snapshots(
            market_ticker='KXHIGHNY-25JUN15-T72-B74',
            bracket=(72.0, 74.0),
            start_ts=datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc),
            end_ts=datetime(2025, 6, 15, 14, 1, tzinfo=timezone.utc),
        )
        assert 0 <= snaps[0].yes_bid <= 1.0
        assert 0 <= snaps[0].yes_ask <= 1.0


class TestTriggerDetection:
    def _seed_obs_data(self, db_path):
        con = duckdb.connect(db_path)
        for hour in range(12, 22):
            con.execute("""
                INSERT INTO observations (station_id, observed_at, temp_f,
                    ingest_source, ingested_at)
                VALUES ('KNYC', ?, ?, 'synoptic', CURRENT_TIMESTAMP)
            """, [datetime(2025, 6, 15, hour, 53), 70.0 + hour - 12])
        con.execute("""
            INSERT INTO observations (station_id, observed_at, temp_f,
                ingest_source, ingested_at)
            VALUES ('KNYC', '2025-06-15 14:23:00', 76.5, 'awc', CURRENT_TIMESTAMP)
        """)
        con.close()

    @pytest.fixture
    def obs_db(self, test_db):
        self._seed_obs_data(test_db)
        return test_db

    def test_get_obs_triggers(self, obs_db):
        detector = TriggerDetector(obs_db, station_id='KNYC')
        triggers = detector.get_triggers(
            event_date=date(2025, 6, 15),
            market_open_utc=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            market_close_utc=datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc),
        )
        obs_triggers = [t for t in triggers if t[2] == 'observation']
        assert len(obs_triggers) == 11

    def test_triggers_are_sorted(self, obs_db):
        detector = TriggerDetector(obs_db, station_id='KNYC')
        triggers = detector.get_triggers(
            event_date=date(2025, 6, 15),
            market_open_utc=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            market_close_utc=datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc),
        )
        model_times = [t[0] for t in triggers]
        assert model_times == sorted(model_times)

    def test_execution_time_offset(self, obs_db):
        detector = TriggerDetector(obs_db, station_id='KNYC', execution_latency_seconds=60)
        triggers = detector.get_triggers(
            event_date=date(2025, 6, 15),
            market_open_utc=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            market_close_utc=datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc),
        )
        for model_time, exec_time, _ in triggers:
            delta = (exec_time - model_time).total_seconds()
            assert delta == 60

    def test_speci_included_at_correct_time(self, obs_db):
        detector = TriggerDetector(obs_db, station_id='KNYC')
        triggers = detector.get_triggers(
            event_date=date(2025, 6, 15),
            market_open_utc=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            market_close_utc=datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc),
        )
        speci_time = datetime(2025, 6, 15, 14, 23, tzinfo=timezone.utc)
        model_times = [t[0] for t in triggers]
        assert speci_time in model_times

    def test_forecast_run_triggers_included(self, obs_db):
        detector = TriggerDetector(obs_db, station_id='KNYC')
        triggers = detector.get_triggers(
            event_date=date(2025, 6, 15),
            market_open_utc=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            market_close_utc=datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc),
        )
        fcst_triggers = [t for t in triggers if t[2] == 'forecast_run']
        assert len(fcst_triggers) >= 3


class TestEdgeAnalyzer:
    def test_market_brier_from_midpoints(self):
        analyzer = EdgeAnalyzer()
        market_probs = [0.05, 0.20, 0.40, 0.25, 0.10]
        settled = [0, 0, 1, 0, 0]
        brier = analyzer.compute_brier(market_probs, settled)
        assert brier == pytest.approx(0.475)

    def test_perfect_brier_is_zero(self):
        analyzer = EdgeAnalyzer()
        assert analyzer.compute_brier([0, 0, 1, 0, 0], [0, 0, 1, 0, 0]) == pytest.approx(0.0)

    def test_displacement_positive_means_model_higher(self):
        analyzer = EdgeAnalyzer()
        assert analyzer.compute_displacement(0.35, 0.22) == pytest.approx(0.13)

    def test_record_and_summarize(self):
        analyzer = EdgeAnalyzer()
        analyzer.record(
            event_date=date(2025, 6, 15),
            trigger_time=datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc),
            model_probs={(72.0, 74.0): 0.40, (70.0, 72.0): 0.25, (74.0, 76.0): 0.20},
            market_mids={(72.0, 74.0): 0.35, (70.0, 72.0): 0.20, (74.0, 76.0): 0.15},
            settled_bracket=(72.0, 74.0),
            et_hour=13,
        )
        summary = analyzer.summary_by_hour()
        assert 13 in summary
        assert "model_brier" in summary[13]
        assert "market_brier" in summary[13]
        assert summary[13]["edge"] > 0


class TestPnLSimulator:
    def _make_trades(self):
        trades = []
        for i in range(10):
            trades.append(TradeRecord(
                event_date=date(2025, 3, 1) + timedelta(days=i),
                bracket=(72.0, 74.0), direction='BUY_YES',
                entry_price=30.0, entry_time=datetime(2025, 3, 1, 14, 0, tzinfo=timezone.utc),
                exit_price=None, exit_time=None, exit_type='settlement',
                settlement_result=1, pnl=62.7, fees_paid=7.3,
                displacement_at_entry=0.15, model_prob_at_entry=0.40,
                capital_locked_hours=24.0,
            ))
        for i in range(5):
            trades.append(TradeRecord(
                event_date=date(2025, 3, 11) + timedelta(days=i),
                bracket=(74.0, 76.0), direction='BUY_YES',
                entry_price=25.0, entry_time=datetime(2025, 3, 11, 14, 0, tzinfo=timezone.utc),
                exit_price=None, exit_time=None, exit_type='settlement',
                settlement_result=0, pnl=-25.25, fees_paid=0.25,
                displacement_at_entry=0.12, model_prob_at_entry=0.20,
                capital_locked_hours=24.0,
            ))
        return trades

    def test_aggregate_metrics(self):
        trades = self._make_trades()
        sim = PnLSimulator()
        metrics = sim.aggregate(trades, starting_capital=100.0)
        assert metrics["total_trades"] == 15
        assert metrics["win_rate"] == pytest.approx(10 / 15)
        assert metrics["total_pnl"] > 0
        assert metrics["profit_factor"] > 1.0

    def test_bootstrap_confidence_interval(self):
        daily_pnls = [(t.event_date, t.pnl) for t in self._make_trades()]
        sim = PnLSimulator()
        ci = sim.bootstrap(daily_pnls, n_iterations=1000, seed=42)
        assert "pnl_ci_95" in ci
        assert ci["pnl_ci_95"][0] < ci["pnl_ci_95"][1]
        assert 0 <= ci["prob_profitable"] <= 1.0

    def test_regime_split_by_season(self):
        trades = self._make_trades()
        sim = PnLSimulator()
        def season_fn(trade):
            m = trade.event_date.month
            if m in (12, 1, 2):
                return "winter"
            elif m in (3, 4, 5):
                return "spring"
            elif m in (6, 7, 8):
                return "summer"
            return "fall"
        splits = sim.regime_split(trades, season_fn)
        assert "spring" in splits
        assert splits["spring"]["total_trades"] == 15

    def test_sharpe_ratio(self):
        trades = self._make_trades()
        sim = PnLSimulator()
        metrics = sim.aggregate(trades, starting_capital=100.0)
        assert metrics["sharpe_ratio"] > 0
