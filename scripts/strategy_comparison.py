#!/usr/bin/env python3
"""Strategy Comparison Runner — backtest 12 signal filtering strategies.

Mirrors the live StrategyEngine exactly, using the same FeatureBuilder and QRModel.
Compares strategy filters to find the most profitable approach.

Usage:
    python scripts/strategy_comparison.py --quick          # 100 sample days
    python scripts/strategy_comparison.py --full           # all ~1659 days
    python scripts/strategy_comparison.py --quick --seed 42
"""
import argparse
import math
import os
import re
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ── Pure utility functions ───────────────────────────────────────────────


def compute_fee(contracts, price_cents):
    # type: (int, int) -> float
    """Taker fee in dollars. Mirrors PaperTrader._compute_fee()."""
    p = price_cents / 100.0
    raw = 0.07 * contracts * p * (1.0 - p)
    fee = max(math.ceil(round(raw * 100, 10)) / 100.0, contracts * 0.01)
    return round(fee, 2)


def kelly_size(model_prob, price_cents, bankroll, max_per_bracket):
    # type: (float, int, float, int) -> int
    """Half-Kelly position sizing. Mirrors StrategyEngine._compute_contracts()."""
    if price_cents <= 0 or bankroll <= 0:
        return 1
    edge_decimal = model_prob - price_cents / 100.0
    if edge_decimal <= 0:
        return 1
    half_kelly = edge_decimal / 2.0
    price_dollars = price_cents / 100.0
    raw = half_kelly * bankroll / price_dollars
    return max(1, min(int(raw), max_per_bracket))


def compute_pnl(settled_yes, direction, entry_price_cents, contracts, entry_fee):
    # type: (bool, str, int, int, float) -> float
    """Net P&L for a position held to settlement. No exit fee.

    Direction should be uppercase "YES" or "NO" (matching live engine convention).
    """
    d = direction.upper()
    won = (settled_yes and d == 'YES') or (not settled_yes and d == 'NO')
    if won:
        gross = (100 - entry_price_cents) * contracts / 100.0
    else:
        gross = -(entry_price_cents * contracts / 100.0)
    return round(gross - entry_fee, 2)


def resolve_settlement(floor_strike, cap_strike, actual_high):
    # type: (Optional[int], Optional[int], int) -> bool
    """Determine if a bracket settles YES given actual high temp.

    Both bounds inclusive for interior brackets (per Kalshi CFTC filing).
    NWS CLI always reports integer degrees, so boundary ambiguity cannot occur.
    """
    if floor_strike is None:
        return actual_high < cap_strike           # lower tail (strict)
    elif cap_strike is None:
        return actual_high > floor_strike          # upper tail (strict)
    else:
        return floor_strike <= actual_high <= cap_strike  # interior (both inclusive)


def compute_model_median(bracket_probs):
    # type: (Dict[int, float]) -> float
    """Compute median temperature from 1-degree bracket probabilities."""
    sorted_brackets = sorted(bracket_probs.items())
    cumulative = 0.0
    for temp, prob in sorted_brackets:
        cumulative += prob
        if cumulative >= 0.5:
            return float(temp)
    return float(sorted_brackets[-1][0]) if sorted_brackets else 0.0


# ── Strategy Filters ─────────────────────────────────────────────────────


def bracket_distance(floor, cap, median):
    # type: (Optional[int], Optional[int], float) -> float
    """Distance from a bracket to the model median temperature."""
    if floor is None:
        return max(0.0, median - cap)       # lower tail
    if cap is None:
        return max(0.0, floor - median)      # upper tail
    if floor <= median <= cap:
        return 0.0
    return min(abs(median - floor), abs(median - cap))


class BaselineFilter:
    """No filtering — pass all signals (current live behavior)."""
    name = "baseline"

    def filter(self, signals, bracket_probs, model_median):
        return list(signals)


class AdjacencyFilter:
    """Only trade brackets within max_distance degrees of model median."""

    def __init__(self, max_distance):
        self.max_distance = max_distance
        self.name = "adjacency_{}".format(max_distance)

    def filter(self, signals, bracket_probs, model_median):
        return [s for s in signals
                if bracket_distance(s["bracket_floor"], s["bracket_cap"], model_median)
                <= self.max_distance]


class TopKFilter:
    """Keep only the K signals with highest edge_pct."""

    def __init__(self, k):
        self.k = k
        self.name = "top_k_{}".format(k)

    def filter(self, signals, bracket_probs, model_median):
        return sorted(signals, key=lambda s: s["edge_pct"], reverse=True)[:self.k]


class PortfolioEVFilter:
    """Mutually-exclusive EV optimizer for temperature brackets.

    Brackets are mutually exclusive (only one can settle YES).
    Greedy: sort by EV, add signal if portfolio EV improves.
    """
    name = "portfolio_ev"

    def filter(self, signals, bracket_probs, model_median):
        if not signals:
            return []

        def signal_ev(s):
            p = s["model_prob"]
            price = s["market_price"] / 100.0
            fee = compute_fee(s["contracts"], s["market_price"])
            win_pnl = (1.0 - price) * s["contracts"] - fee
            lose_pnl = -price * s["contracts"] - fee
            return p * win_pnl + (1.0 - p) * lose_pnl

        ranked = sorted(signals, key=signal_ev, reverse=True)
        portfolio = [ranked[0]]
        best_ev = signal_ev(ranked[0])

        for sig in ranked[1:]:
            candidate = portfolio + [sig]
            total_ev = 0.0
            all_lose_pnl = 0.0
            for s in candidate:
                p = s["model_prob"]
                price = s["market_price"] / 100.0
                fee = compute_fee(s["contracts"], s["market_price"])
                win_pnl = (1.0 - price) * s["contracts"] - fee
                lose_pnl = -price * s["contracts"] - fee
                total_ev += p * (win_pnl - lose_pnl)
                all_lose_pnl += lose_pnl
            total_ev += all_lose_pnl

            if total_ev > best_ev:
                portfolio = candidate
                best_ev = total_ev

        return portfolio


class HybridFilter:
    """Adjacency first, then Top-K on the survivors."""

    def __init__(self, max_distance, k):
        self.max_distance = max_distance
        self.k = k
        self.name = "hybrid_{}_{}".format(max_distance, k)

    def filter(self, signals, bracket_probs, model_median):
        adj = AdjacencyFilter(self.max_distance)
        topk = TopKFilter(self.k)
        return topk.filter(adj.filter(signals, bracket_probs, model_median),
                           bracket_probs, model_median)


ALL_STRATEGIES = [
    BaselineFilter(),
    AdjacencyFilter(2),
    AdjacencyFilter(4),
    AdjacencyFilter(6),
    AdjacencyFilter(8),
    TopKFilter(1),
    TopKFilter(2),
    TopKFilter(3),
    TopKFilter(5),
    PortfolioEVFilter(),
    HybridFilter(4, 2),
    HybridFilter(6, 3),
]


# ── Market Data ──────────────────────────────────────────────────────────


def parse_ticker_bracket(ticker):
    # type: (str) -> Tuple[Optional[int], Optional[int]]
    """Extract floor/cap from a Kalshi ticker string.

    Interior: KXHIGHNY-26JAN15-B61.5 -> floor=61, cap=62
    T-tickers: KXHIGHNY-26JAN15-T41 -> (None, 41) — caller resolves upper vs lower
    """
    parts = ticker.split("-")
    bracket_part = parts[-1]

    if bracket_part.startswith("B"):
        midpoint = float(bracket_part[1:])
        floor_val = int(midpoint - 0.5)
        cap_val = int(midpoint + 0.5)
        return floor_val, cap_val
    elif bracket_part.startswith("T"):
        strike = int(bracket_part[1:])
        return None, strike
    return None, None


def derive_no_prices(yes_bid, yes_ask):
    # type: (int, int) -> Tuple[int, int]
    """Derive NO prices from YES prices (binary contract identity)."""
    no_ask = 100 - yes_bid
    no_bid = 100 - yes_ask
    return no_ask, no_bid


def aggregate_to_kalshi_brackets(bracket_probs, market_keys):
    # type: (Dict[int, float], List[Tuple[Optional[int], Optional[int]]]) -> Dict[Tuple, float]
    """Aggregate 1-deg model probs to Kalshi bracket format.

    Mirrors StrategyEngine._aggregate_to_kalshi_brackets().
    """
    aggregated = {}
    for key in market_keys:
        floor, cap = key
        if floor is None and cap is not None:
            total = sum(p for k, p in bracket_probs.items() if k < cap)
        elif cap is None and floor is not None:
            total = sum(p for k, p in bracket_probs.items() if k > floor)
        elif floor is not None and cap is not None:
            total = sum(p for k, p in bracket_probs.items() if floor <= k <= cap)
        else:
            continue
        aggregated[key] = total
    return aggregated


def generate_signals(bracket_probs, market_prices, config, bankroll):
    # type: (Dict[int, float], Dict, Dict, float) -> List[Dict]
    """Generate trade signals. Mirrors StrategyEngine._generate_signals().

    Args:
        bracket_probs: {int: float} — 1-deg model probabilities
        market_prices: {(floor, cap): {yes_ask, yes_bid, no_ask, no_bid}} — cents
        config: {min_edge_pct, min_model_prob, min_ev_cents, max_per_bracket}
        bankroll: current bankroll in dollars
    """
    kalshi_probs = aggregate_to_kalshi_brackets(bracket_probs, list(market_prices.keys()))
    signals = []

    for key, model_prob in kalshi_probs.items():
        floor, cap = key
        prices = market_prices[key]
        yes_ask = prices["yes_ask"]
        no_ask = prices["no_ask"]

        yes_edge = (model_prob - yes_ask / 100.0) * 100.0
        no_prob = 1.0 - model_prob
        no_edge = (no_prob - no_ask / 100.0) * 100.0

        if yes_edge >= config["min_edge_pct"]:
            if model_prob < config["min_model_prob"]:
                continue
            price_decimal = yes_ask / 100.0
            fee_cents = max(math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)), 1)
            if yes_edge - fee_cents < config["min_ev_cents"]:
                continue
            contracts = kelly_size(model_prob, yes_ask, bankroll, config["max_per_bracket"])
            signals.append({
                "bracket_floor": floor, "bracket_cap": cap, "direction": "YES",
                "model_prob": model_prob, "market_price": yes_ask,
                "edge_pct": yes_edge, "contracts": contracts,
            })
        elif no_edge >= config["min_edge_pct"]:
            if no_prob < config["min_model_prob"]:
                continue
            price_decimal = no_ask / 100.0
            fee_cents = max(math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)), 1)
            if no_edge - fee_cents < config["min_ev_cents"]:
                continue
            contracts = kelly_size(no_prob, no_ask, bankroll, config["max_per_bracket"])
            signals.append({
                "bracket_floor": floor, "bracket_cap": cap, "direction": "NO",
                "model_prob": model_prob, "market_price": no_ask,
                "edge_pct": no_edge, "contracts": contracts,
            })

    return signals


def main():
    parser = argparse.ArgumentParser(description="Strategy comparison backtest runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="100 season-stratified sample days")
    mode.add_argument("--full", action="store_true", help="All ~1659 days")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy names (default: all 12)")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of sample days for --quick mode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--db", type=str, default="data/alphatemp.duckdb",
                        help="Path to DuckDB database")
    args = parser.parse_args()
    print("Strategy comparison runner — not yet implemented (Tasks 4-5)")


if __name__ == "__main__":
    main()
