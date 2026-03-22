# tests/test_strategy_comparison.py
"""Tests for strategy comparison runner."""
import math
import pytest


# ── Fee calculation ──────────────────────────────────────────────────────

class TestComputeFee:
    def test_basic_fee(self):
        from scripts.strategy_comparison import compute_fee
        assert compute_fee(1, 50) == 0.02

    def test_minimum_fee(self):
        from scripts.strategy_comparison import compute_fee
        assert compute_fee(1, 5) == 0.01

    def test_multi_contract(self):
        from scripts.strategy_comparison import compute_fee
        assert compute_fee(3, 50) == 0.06


# ── Kelly sizing ─────────────────────────────────────────────────────────

class TestKellySize:
    def test_basic_sizing(self):
        from scripts.strategy_comparison import kelly_size
        # 0.6 - 0.5 = 0.09999... (float), raw = 9.999... -> int truncates to 9
        assert kelly_size(0.6, 50, 100.0, 10) == 9

    def test_clamp_to_max(self):
        from scripts.strategy_comparison import kelly_size
        assert kelly_size(0.8, 20, 200.0, 5) == 5

    def test_floor_to_one(self):
        from scripts.strategy_comparison import kelly_size
        assert kelly_size(0.51, 50, 10.0, 10) == 1

    def test_zero_bankroll(self):
        from scripts.strategy_comparison import kelly_size
        assert kelly_size(0.6, 50, 0.0, 10) == 1

    def test_negative_edge(self):
        from scripts.strategy_comparison import kelly_size
        assert kelly_size(0.4, 50, 100.0, 10) == 1


# ── P&L calculation ──────────────────────────────────────────────────────

class TestPnl:
    def test_yes_win(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=True, direction='YES', entry_price_cents=30, contracts=2, entry_fee=0.04)
        assert net == pytest.approx(1.36, abs=0.01)

    def test_yes_loss(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=False, direction='YES', entry_price_cents=30, contracts=2, entry_fee=0.04)
        assert net == pytest.approx(-0.64, abs=0.01)

    def test_no_win(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=False, direction='NO', entry_price_cents=40, contracts=1, entry_fee=0.02)
        assert net == pytest.approx(0.58, abs=0.01)

    def test_no_loss(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=True, direction='NO', entry_price_cents=40, contracts=1, entry_fee=0.02)
        assert net == pytest.approx(-0.42, abs=0.01)


# ── Settlement ───────────────────────────────────────────────────────────

class TestSettlement:
    def test_interior_both_inclusive(self):
        from scripts.strategy_comparison import resolve_settlement
        assert resolve_settlement(64, 66, 66) is True
        assert resolve_settlement(64, 66, 64) is True
        assert resolve_settlement(64, 66, 65) is True
        assert resolve_settlement(64, 66, 63) is False
        assert resolve_settlement(64, 66, 67) is False

    def test_lower_tail(self):
        from scripts.strategy_comparison import resolve_settlement
        assert resolve_settlement(None, 50, 49) is True
        assert resolve_settlement(None, 50, 50) is False
        assert resolve_settlement(None, 50, 51) is False

    def test_upper_tail(self):
        from scripts.strategy_comparison import resolve_settlement
        assert resolve_settlement(80, None, 81) is True
        assert resolve_settlement(80, None, 80) is False
        assert resolve_settlement(80, None, 79) is False


# ── Model median ─────────────────────────────────────────────────────────

class TestModelMedian:
    def test_basic_median(self):
        from scripts.strategy_comparison import compute_model_median
        probs = {60: 0.1, 61: 0.15, 62: 0.25, 63: 0.25, 64: 0.15, 65: 0.1}
        assert compute_model_median(probs) == 62.0

    def test_median_crosses_at_boundary(self):
        from scripts.strategy_comparison import compute_model_median
        probs = {70: 0.4, 71: 0.2, 72: 0.4}
        assert compute_model_median(probs) == 71.0


# ── Strategy Filters ─────────────────────────────────────────────────────

def _make_signal(floor, cap, direction, edge_pct, model_prob=0.5, market_price=40, contracts=1):
    return {
        "bracket_floor": floor, "bracket_cap": cap, "direction": direction,
        "model_prob": model_prob, "market_price": market_price,
        "edge_pct": edge_pct, "contracts": contracts,
    }


class TestBaselineFilter:
    def test_passes_everything(self):
        from scripts.strategy_comparison import BaselineFilter
        signals = [_make_signal(60, 62, "YES", 8.0), _make_signal(70, 72, "YES", 6.0)]
        assert len(BaselineFilter().filter(signals, {}, 65.0)) == 2


class TestAdjacencyFilter:
    def test_within_distance(self):
        from scripts.strategy_comparison import AdjacencyFilter
        signals = [
            _make_signal(64, 66, "YES", 8.0),
            _make_signal(60, 62, "YES", 6.0),
            _make_signal(70, 72, "YES", 7.0),
        ]
        result = AdjacencyFilter(max_distance=4).filter(signals, {}, 65.0)
        assert len(result) == 2

    def test_tail_bracket(self):
        from scripts.strategy_comparison import AdjacencyFilter
        signals = [_make_signal(None, 50, "YES", 8.0)]
        result = AdjacencyFilter(max_distance=4).filter(signals, {}, 65.0)
        assert len(result) == 0


class TestTopKFilter:
    def test_top_2(self):
        from scripts.strategy_comparison import TopKFilter
        signals = [
            _make_signal(60, 62, "YES", 5.0),
            _make_signal(64, 66, "YES", 10.0),
            _make_signal(68, 70, "YES", 7.0),
        ]
        result = TopKFilter(k=2).filter(signals, {}, 65.0)
        assert len(result) == 2
        edges = [s["edge_pct"] for s in result]
        assert 10.0 in edges and 7.0 in edges

    def test_fewer_than_k(self):
        from scripts.strategy_comparison import TopKFilter
        assert len(TopKFilter(k=3).filter([_make_signal(60, 62, "YES", 8.0)], {}, 65.0)) == 1


class TestPortfolioEVFilter:
    def test_mutually_exclusive_ev(self):
        from scripts.strategy_comparison import PortfolioEVFilter
        signals = [
            _make_signal(64, 66, "YES", 10.0, model_prob=0.5, market_price=40, contracts=1),
            _make_signal(60, 62, "YES", 6.0, model_prob=0.3, market_price=24, contracts=1),
        ]
        probs = {i: 0.05 for i in range(55, 75)}
        result = PortfolioEVFilter().filter(signals, probs, 65.0)
        assert len(result) >= 1


class TestHybridFilter:
    def test_adjacency_then_top_k(self):
        from scripts.strategy_comparison import HybridFilter
        signals = [
            _make_signal(64, 66, "YES", 10.0),
            _make_signal(63, 65, "YES", 8.0),
            _make_signal(60, 62, "YES", 12.0),
            _make_signal(70, 72, "YES", 15.0),
        ]
        result = HybridFilter(max_distance=4, k=2).filter(signals, {}, 65.0)
        assert len(result) == 2
        edges = sorted([s["edge_pct"] for s in result], reverse=True)
        assert edges[0] == 12.0
        assert edges[1] == 10.0


# ── Market Data ──────────────────────────────────────────────────────────

class TestParseTickerBracket:
    def test_interior_bracket(self):
        from scripts.strategy_comparison import parse_ticker_bracket
        floor, cap = parse_ticker_bracket("KXHIGHNY-26JAN15-B61.5")
        assert floor == 61
        assert cap == 62

    def test_lower_tail(self):
        from scripts.strategy_comparison import parse_ticker_bracket
        floor, cap = parse_ticker_bracket("KXHIGHNY-26JAN15-T41")
        assert floor is None
        assert cap == 41

    def test_t_ticker_returns_none_strike(self):
        from scripts.strategy_comparison import parse_ticker_bracket
        floor, cap = parse_ticker_bracket("KXHIGHNY-26JAN15-T80")
        assert floor is None
        assert cap == 80


class TestDeriveNoPrices:
    def test_no_from_yes(self):
        from scripts.strategy_comparison import derive_no_prices
        no_ask, no_bid = derive_no_prices(yes_bid=40, yes_ask=45)
        assert no_ask == 60
        assert no_bid == 55


class TestGenerateSignals:
    def test_yes_signal(self):
        from scripts.strategy_comparison import generate_signals
        probs = {60: 0.05, 61: 0.1, 62: 0.2, 63: 0.3, 64: 0.2, 65: 0.1, 66: 0.05}
        prices = {(62, 64): {"yes_ask": 30, "yes_bid": 28, "no_ask": 72, "no_bid": 70}}
        config = {"min_edge_pct": 5.0, "min_model_prob": 0.15, "min_ev_cents": 2, "max_per_bracket": 10}
        signals = generate_signals(probs, prices, config, bankroll=100.0)
        assert len(signals) >= 1
        assert signals[0]["direction"] == "YES"
        assert signals[0]["edge_pct"] == pytest.approx(40.0, abs=0.1)

    def test_no_signal(self):
        from scripts.strategy_comparison import generate_signals
        probs = {60: 0.05, 61: 0.05, 62: 0.05, 63: 0.05, 64: 0.3, 65: 0.3, 66: 0.2}
        prices = {(60, 62): {"yes_ask": 70, "yes_bid": 68, "no_ask": 32, "no_bid": 30}}
        config = {"min_edge_pct": 5.0, "min_model_prob": 0.15, "min_ev_cents": 2, "max_per_bracket": 10}
        signals = generate_signals(probs, prices, config, bankroll=100.0)
        assert len(signals) == 1
        assert signals[0]["direction"] == "NO"
        assert signals[0]["edge_pct"] == pytest.approx(53.0, abs=0.1)
