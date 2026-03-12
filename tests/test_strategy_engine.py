"""Tests for services/strategy_engine.py — StrategyEngine class.

Tests the pure logic methods: edge computation, signal generation, and
edge reversal detection. Does NOT test the async run loop or DB queries.
"""

import asyncio
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
    eng._last_data_hash = None
    return eng


# ---------------------------------------------------------------------------
# _compute_edge
# ---------------------------------------------------------------------------

class TestComputeEdge:
    def test_positive_edge(self, engine):
        """Edge = (model_prob - market_cents/100) * 100.
        model 20%, market 12c -> +8% edge.
        """
        edge = engine._compute_edge(0.20, 12)
        assert abs(edge - 8.0) < 0.01

    def test_zero_edge(self, engine):
        """Model matches market -> 0% edge."""
        edge = engine._compute_edge(0.50, 50)
        assert abs(edge) < 0.01

    def test_negative_edge(self, engine):
        """Model below market -> negative edge."""
        edge = engine._compute_edge(0.10, 25)
        assert edge < 0.0
        assert abs(edge - (-15.0)) < 0.01

    def test_high_prob_low_market(self, engine):
        """Model 80%, market 60c -> +20% edge."""
        edge = engine._compute_edge(0.80, 60)
        assert abs(edge - 20.0) < 0.01


# ---------------------------------------------------------------------------
# _aggregate_to_kalshi_brackets
# ---------------------------------------------------------------------------

class TestAggregateToKalshiBrackets:
    def test_sums_adjacent_1f_probs(self, engine):
        """Two 1°F probs should sum to one 2°F Kalshi bracket prob."""
        bracket_probs = {72: 0.08, 73: 0.07, 74: 0.06, 75: 0.05}
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
            74: {"yes_bid": 8, "yes_ask": 10, "no_bid": 88, "no_ask": 90},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        # [72,74) = probs[72] + probs[73] = 0.08 + 0.07 = 0.15
        assert abs(result[72] - 0.15) < 0.001
        # [74,76) = probs[74] + probs[75] = 0.06 + 0.05 = 0.11
        assert abs(result[74] - 0.11) < 0.001

    def test_missing_adjacent_uses_zero(self, engine):
        """If model only has one of the two 1°F components, other defaults to 0."""
        bracket_probs = {72: 0.10}  # no key 73
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        result = engine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        assert abs(result[72] - 0.10) < 0.001


# ---------------------------------------------------------------------------
# _generate_signals
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def test_no_edge_no_trade(self, engine):
        """Below threshold -> no signal."""
        # Model says 15% for bracket 72, market ask is 12c -> 3% edge (< 5% min)
        bracket_probs = {72: 0.15, 74: 0.10}
        market_prices = {
            72: {"yes_bid": 11, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
            74: {"yes_bid": 8, "yes_ask": 9, "no_bid": 89, "no_ask": 91},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 0

    def test_positive_edge_generates_yes_signal(self, engine):
        """Model > market ask -> YES signal when edge >= min_edge_pct."""
        # Model says 25% for bracket 72, market yes_ask is 12c -> 13% edge
        bracket_probs = {72: 0.25}
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        sig = signals[0]
        assert sig["bracket_floor"] == 72
        assert sig["direction"] == "YES"
        assert abs(sig["model_prob"] - 0.25) < 0.001
        assert sig["market_price"] == 12
        assert abs(sig["edge_pct"] - 13.0) < 0.01

    def test_no_signal_when_model_says_unlikely(self, engine):
        """Model says bracket unlikely -> NO signal if (1-model) > no_ask/100."""
        # Model says 2% for bracket 72. NO prob = 98%.
        # no_ask = 85c -> market implied NO = 85%. Edge = (98-85) = 13%
        bracket_probs = {72: 0.02}
        market_prices = {
            72: {"yes_bid": 1, "yes_ask": 3, "no_bid": 80, "no_ask": 85},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        sig = signals[0]
        assert sig["direction"] == "NO"
        assert abs(sig["model_prob"] - 0.02) < 0.001
        assert sig["market_price"] == 85
        assert abs(sig["edge_pct"] - 13.0) < 0.01

    def test_yes_preferred_over_no_on_same_bracket(self, engine):
        """If both YES and NO have edge, prefer YES."""
        # Model says 55% for bracket 72.
        # yes_ask = 45c -> YES edge = (55 - 45) = 10%
        # no_ask = 40c  -> NO edge = (45 - 40) = 5% (exactly at threshold)
        # YES should win
        bracket_probs = {72: 0.55}
        market_prices = {
            72: {"yes_bid": 40, "yes_ask": 45, "no_bid": 35, "no_ask": 40},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        assert len(signals) == 1
        assert signals[0]["direction"] == "YES"

    def test_multiple_brackets_multiple_signals(self, engine):
        """Multiple brackets can each generate signals."""
        bracket_probs = {72: 0.30, 74: 0.25}
        market_prices = {
            72: {"yes_bid": 15, "yes_ask": 18, "no_bid": 78, "no_ask": 82},
            74: {"yes_bid": 10, "yes_ask": 12, "no_bid": 85, "no_ask": 88},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        # Bracket 72: YES edge = (30-18) = 12% -> YES
        # Bracket 74: YES edge = (25-12) = 13% -> YES
        assert len(signals) == 2
        floors = {s["bracket_floor"] for s in signals}
        assert floors == {72, 74}

    def test_bracket_not_in_market_skipped(self, engine):
        """Brackets with no market data are silently skipped."""
        bracket_probs = {72: 0.30, 99: 0.05}
        market_prices = {
            72: {"yes_bid": 15, "yes_ask": 18, "no_bid": 78, "no_ask": 82},
        }
        signals = engine._generate_signals(bracket_probs, market_prices)
        # Only bracket 72 has market data
        assert all(s["bracket_floor"] == 72 for s in signals)


# ---------------------------------------------------------------------------
# _check_edge_reversals
# ---------------------------------------------------------------------------

class TestCheckEdgeReversals:
    def test_edge_reversal_exits_position(self, engine):
        """When edge flips negative, call paper_trader.exit_position."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {
                "id": 1,
                "bracket_floor": 72,
                "bracket_cap": 74,
                "direction": "YES",
                "entry_price": 12,
            }
        ]
        # Now model says only 8% (was ~20% when entered)
        bracket_probs = {72: 0.08}
        # Market yes_bid = 10c -> exit price
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(
                bracket_probs, market_prices, open_positions
            )
        )
        # Edge = (8 - 12) * 100 = -4% -> reversed, should exit
        engine.paper_trader.exit_position.assert_called_once_with(
            1, 10, "edge_reversal"
        )

    def test_no_exit_when_edge_still_positive(self, engine):
        """When edge is still positive, don't exit."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {
                "id": 1,
                "bracket_floor": 72,
                "bracket_cap": 74,
                "direction": "YES",
                "entry_price": 12,
            }
        ]
        bracket_probs = {72: 0.25}
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(
                bracket_probs, market_prices, open_positions
            )
        )
        engine.paper_trader.exit_position.assert_not_called()

    def test_no_side_edge_reversal(self, engine):
        """NO position: edge reversal when (1 - model_prob) < no_ask/100."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {
                "id": 2,
                "bracket_floor": 72,
                "bracket_cap": 74,
                "direction": "NO",
                "entry_price": 85,
            }
        ]
        # Model now says 25% -> NO prob = 75%. no_ask = 88c -> 75 < 88 -> reversed
        bracket_probs = {72: 0.25}
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(
                bracket_probs, market_prices, open_positions
            )
        )
        # NO edge = ((1 - 0.25) - 88/100) * 100 = (0.75 - 0.88)*100 = -13% -> exit
        engine.paper_trader.exit_position.assert_called_once_with(
            2, 86, "edge_reversal"
        )

    def test_bracket_missing_from_model_skips(self, engine):
        """Position on a bracket the model didn't predict -> skip, don't exit."""
        engine.paper_trader.exit_position = AsyncMock()
        open_positions = [
            {
                "id": 1,
                "bracket_floor": 99,
                "bracket_cap": 101,
                "direction": "YES",
                "entry_price": 12,
            }
        ]
        bracket_probs = {72: 0.20}
        market_prices = {
            72: {"yes_bid": 10, "yes_ask": 12, "no_bid": 86, "no_ask": 88},
        }
        asyncio.get_event_loop().run_until_complete(
            engine._check_edge_reversals(
                bracket_probs, market_prices, open_positions
            )
        )
        engine.paper_trader.exit_position.assert_not_called()
