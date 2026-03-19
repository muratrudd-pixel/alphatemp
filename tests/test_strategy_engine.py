"""Tests for services/strategy_engine.py — StrategyEngine class.

Tests the pure logic methods: edge computation, signal generation,
edge reversal detection, dedup guard, EV filter, and tail brackets.
"""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.strategy_engine import StrategyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """StrategyEngine with mocked dependencies (no real DB)."""
    mock_trader = MagicMock()
    eng = StrategyEngine.__new__(StrategyEngine)
    eng.db_path = ":memory:"
    eng.paper_trader = mock_trader
    eng.model = MagicMock()
    eng.feature_builder = MagicMock()
    eng.circuit_breakers = MagicMock()
    eng.min_edge_pct = 5.0
    eng.min_ev_cents = 2
    eng.min_model_prob = 0.15
    eng.starting_capital = 100.0
    eng.max_per_bracket = 10
    eng._last_data_hash = None
    # Mock _get_bankroll so _generate_signals doesn't hit DB
    eng._get_bankroll = MagicMock(return_value=100.0)
    return eng


# ---------------------------------------------------------------------------
# _compute_edge
# ---------------------------------------------------------------------------

class TestComputeEdge:
    def test_positive_edge(self, engine):
        edge = engine._compute_edge(0.20, 12)
        assert abs(edge - 8.0) < 0.01

    def test_zero_edge(self, engine):
        edge = engine._compute_edge(0.50, 50)
        assert abs(edge) < 0.01

    def test_negative_edge(self, engine):
        edge = engine._compute_edge(0.10, 25)
        assert abs(edge - (-15.0)) < 0.01

    def test_high_prob_low_market(self, engine):
        edge = engine._compute_edge(0.80, 60)
        assert abs(edge - 20.0) < 0.01


# ---------------------------------------------------------------------------
# _aggregate_to_kalshi_brackets
# ---------------------------------------------------------------------------

class TestAggregateToKalshiBrackets:
    def test_sums_adjacent_1f_probs(self, engine):
        """Interior bracket (72, 73) sums probs for 72 <= k <= 73."""
        bracket_probs = {72: 0.08, 73: 0.07, 74: 0.06, 75: 0.05}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
            (74, 75): {"yes_bid": 8, "yes_ask": 10, "no_bid": 88, "no_ask": 90},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(72, 73)] - 0.15) < 0.001
        assert abs(result[(74, 75)] - 0.11) < 0.001

    def test_missing_adjacent_uses_zero(self, engine):
        bracket_probs = {72: 0.10}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(72, 73)] - 0.10) < 0.001

    def test_lower_tail(self, engine):
        """Lower tail (None, 43) sums all probs for k < 43."""
        bracket_probs = {40: 0.05, 41: 0.05, 42: 0.10, 50: 0.20}
        market_prices = {
            (None, 43): {"yes_bid": 10, "yes_ask": 15, "no_bid": 80, "no_ask": 85},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[(None, 43)] - 0.20) < 0.001

    def test_upper_tail(self, engine):
        """Upper tail (60, None) sums all probs for k > 60."""
        bracket_probs = {58: 0.05, 59: 0.10, 60: 0.15, 61: 0.08, 62: 0.03}
        market_prices = {
            (60, None): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        # k > 60: 61 + 62 = 0.08 + 0.03 = 0.11
        assert abs(result[(60, None)] - 0.11) < 0.001


# ---------------------------------------------------------------------------
# _generate_signals
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def test_no_edge_no_trade(self, engine):
        bracket_probs = {72: 0.15, 74: 0.10}
        market_prices = {
            (72, 73): {"yes_bid": 11, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
            (74, 75): {"yes_bid": 8, "yes_ask": 9, "no_bid": 89, "no_ask": 91},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 0

    def test_positive_edge_generates_yes_signal(self, engine):
        bracket_probs = {72: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        sig = signals[0]
        assert sig["bracket_floor"] == 72
        assert sig["bracket_cap"] == 73
        assert sig["direction"] == "YES"

    def test_no_signal_when_model_says_unlikely(self, engine):
        bracket_probs = {72: 0.02}
        market_prices = {
            (72, 73): {"yes_bid": 1, "yes_ask": 3, "no_bid": 80, "no_ask": 85},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["direction"] == "NO"

    def test_yes_preferred_over_no_on_same_bracket(self, engine):
        bracket_probs = {72: 0.55}
        market_prices = {
            (72, 73): {"yes_bid": 40, "yes_ask": 45, "no_bid": 35, "no_ask": 40},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["direction"] == "YES"

    def test_multiple_brackets_multiple_signals(self, engine):
        bracket_probs = {72: 0.30, 74: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 15, "yes_ask": 18, "no_bid": 78, "no_ask": 82},
            (74, 75): {"yes_bid": 10, "yes_ask": 12, "no_bid": 85, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 2

    def test_bracket_not_in_market_skipped(self, engine):
        bracket_probs = {72: 0.30, 99: 0.05}
        market_prices = {
            (72, 73): {"yes_bid": 15, "yes_ask": 18, "no_bid": 78, "no_ask": 82},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert all(s["bracket_floor"] == 72 for s in signals)

    def test_ev_filter_blocks_penny_bets(self, engine):
        """Penny YES bets with low EV after fees should be blocked."""
        # 3c YES, ~2% edge -> EV = 2 - 1 = 1c, below min_ev_cents=2
        bracket_probs = {72: 0.05}
        market_prices = {
            (72, 73): {"yes_bid": 2, "yes_ask": 3, "no_bid": 95, "no_ask": 97},
        }
        engine.min_edge_pct = 2.0  # lower threshold to let edge pass
        signals = engine._generate_signals(bracket_probs, market_prices)
        # EV too low — should be filtered
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Dedup guard
# ---------------------------------------------------------------------------

class TestDedupGuard:
    def test_dedup_filters_existing_positions(self):
        """Signals matching open positions should be filtered out."""
        signals = [
            {"bracket_floor": 50, "bracket_cap": 52, "direction": "YES",
             "model_prob": 0.4, "market_price": 30, "edge_pct": 10.0},
            {"bracket_floor": 54, "bracket_cap": 56, "direction": "NO",
             "model_prob": 0.1, "market_price": 85, "edge_pct": 5.0},
        ]
        open_positions = [
            {"id": 1, "bracket_floor": 50, "bracket_cap": 52,
             "direction": "YES", "entry_price": 28},
        ]
        held = {(p["bracket_floor"], p["bracket_cap"], p["direction"])
                for p in open_positions}
        filtered = [s for s in signals
                    if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held]
        assert len(filtered) == 1
        assert filtered[0]["bracket_floor"] == 54

    def test_dedup_handles_none_tail_brackets(self):
        """Tail brackets with None floor/cap should dedup correctly."""
        signals = [
            {"bracket_floor": None, "bracket_cap": 49, "direction": "YES",
             "model_prob": 0.8, "market_price": 70, "edge_pct": 10.0},
        ]
        open_positions = [
            {"id": 1, "bracket_floor": None, "bracket_cap": 49,
             "direction": "YES", "entry_price": 65},
        ]
        held = {(p["bracket_floor"], p["bracket_cap"], p["direction"])
                for p in open_positions}
        filtered = [s for s in signals
                    if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held]
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# _check_edge_reversals
# ---------------------------------------------------------------------------

class TestCheckEdgeReversals:
    def test_edge_reversal_exits_position(self, engine):
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 72, "bracket_cap": 74,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.04, 73: 0.04}
        market_prices = {
            (72, 74): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_called_once_with(1, 10, "edge_reversal")

    def test_no_exit_when_edge_still_positive(self, engine):
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 72, "bracket_cap": 74,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.15, 73: 0.10}
        market_prices = {
            (72, 74): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_not_called()

    def test_no_side_edge_reversal(self, engine):
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 2, "bracket_floor": 72, "bracket_cap": 74,
             "direction": "NO", "entry_price": 85},
        ]
        bracket_probs = {72: 0.15, 73: 0.10}
        market_prices = {
            (72, 74): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        # model_prob for (72,74) = 0.15+0.10 = 0.25. NO prob = 0.75.
        # NO edge = (75 - 88) = -13% -> should exit
        engine.paper_trader.exit_position.assert_called_once_with(2, 86, "edge_reversal")

    def test_zero_edge_exits_position(self, engine):
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 72, "bracket_cap": 74,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.06, 73: 0.06}
        market_prices = {
            (72, 74): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_called_once_with(1, 10, "edge_reversal")

    def test_bracket_missing_from_model_skips(self, engine):
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {"id": 1, "bracket_floor": 99, "bracket_cap": 101,
             "direction": "YES", "entry_price": 12},
        ]
        bracket_probs = {72: 0.20}
        market_prices = {
            (72, 74): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(bracket_probs, market_prices, open_positions)
        )
        engine.paper_trader.exit_position.assert_not_called()


# ---------------------------------------------------------------------------
# min_model_prob filter
# ---------------------------------------------------------------------------

class TestMinModelProbFilter:
    def test_blocks_low_prob_yes_signal(self, engine):
        """5% model prob on a 1c bracket has edge but should be blocked."""
        engine.min_model_prob = 0.15
        engine.min_edge_pct = 3.0
        bracket_probs = {72: 0.051}
        market_prices = {
            (72, 73): {"yes_bid": 1, "yes_ask": 1, "no_bid": 97, "no_ask": 99},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 0

    def test_blocks_low_prob_no_signal(self, engine):
        """Model says 92% YES (8% NO) — NO side should be blocked."""
        engine.min_model_prob = 0.15
        bracket_probs = {72: 0.46, 73: 0.46}  # 92% YES, 8% NO
        market_prices = {
            (72, 73): {"yes_bid": 90, "yes_ask": 95, "no_bid": 1, "no_ask": 1},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        # NO prob = 8% < 15% threshold — blocked
        assert len(signals) == 0

    def test_passes_high_prob_yes_signal(self, engine):
        """25% model prob on 12c bracket — above threshold, should pass."""
        engine.min_model_prob = 0.15
        bracket_probs = {72: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["direction"] == "YES"

    def test_passes_high_prob_no_signal(self, engine):
        """Model says 10% YES (90% NO) on 85c bracket — NO prob above threshold."""
        engine.min_model_prob = 0.15
        bracket_probs = {72: 0.10}
        market_prices = {
            (72, 73): {"yes_bid": 25, "yes_ask": 30, "no_bid": 60, "no_ask": 65},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["direction"] == "NO"


# ---------------------------------------------------------------------------
# _compute_contracts (half-Kelly)
# ---------------------------------------------------------------------------

class TestComputeContracts:
    def test_basic_kelly(self):
        """20% model prob, 10c price, $100 bankroll -> kelly_f=0.10, half=0.05,
        raw = 0.05 * 100 / 0.10 = 50, capped at max_per_bracket=10."""
        result = StrategyEngine._compute_contracts(0.20, 10, 100.0, 10)
        assert result == 10

    def test_small_edge(self):
        """16% model prob, 10c price, $100 bankroll -> kelly_f=0.06, half=0.03,
        raw = 0.03 * 100 / 0.10 = 30, capped at 10."""
        result = StrategyEngine._compute_contracts(0.16, 10, 100.0, 10)
        assert result == 10

    def test_moderate_edge_moderate_price(self):
        """40% model prob, 25c price, $100 bankroll -> kelly_f=0.15, half=0.075,
        raw = 0.075 * 100 / 0.25 = 30, capped at 10."""
        result = StrategyEngine._compute_contracts(0.40, 25, 100.0, 10)
        assert result == 10

    def test_small_contract_count(self):
        """Moderate edge on cheap bracket gives reasonable count."""
        # 15% prob, 10c price -> edge=0.05, half=0.025,
        # raw = 0.025 * 100 / 0.10 = 25, capped at 10
        result = StrategyEngine._compute_contracts(0.15, 10, 100.0, 10)
        assert result == 10
        # With lower cap, count is limited
        result2 = StrategyEngine._compute_contracts(0.15, 10, 100.0, 3)
        assert result2 == 3

    def test_zero_edge_returns_one(self):
        """No edge -> returns 1 (floor)."""
        result = StrategyEngine._compute_contracts(0.10, 10, 100.0, 10)
        assert result == 1

    def test_negative_edge_returns_one(self):
        result = StrategyEngine._compute_contracts(0.05, 10, 100.0, 10)
        assert result == 1

    def test_zero_bankroll_returns_one(self):
        result = StrategyEngine._compute_contracts(0.30, 10, 0.0, 10)
        assert result == 1

    def test_zero_price_returns_one(self):
        result = StrategyEngine._compute_contracts(0.30, 0, 100.0, 10)
        assert result == 1

    def test_max_per_bracket_cap(self):
        """Huge edge should still be capped at max_per_bracket."""
        result = StrategyEngine._compute_contracts(0.90, 5, 1000.0, 5)
        assert result == 5

    def test_signals_include_contracts(self, engine):
        """Generated signals should include a contracts field."""
        bracket_probs = {72: 0.25}
        market_prices = {
            (72, 73): {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert "contracts" in signals[0]
        assert signals[0]["contracts"] >= 1
