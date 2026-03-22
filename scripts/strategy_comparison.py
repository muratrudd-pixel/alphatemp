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
import json
import math
import os
import random
import re
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
from tqdm import tqdm

# Add project root to path for service imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.feature_builder import FeatureBuilder
from services.model import QRModel


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


# ── DB Query Functions ───────────────────────────────────────────────────


def get_candlestick_prices(con, event_date, update_hour):
    # type: (duckdb.DuckDBPyConnection, date, int) -> Dict[Tuple, Dict]
    """Get latest candlestick prices for each bracket at the given hour.

    Returns {(floor, cap): {yes_ask, yes_bid, no_ask, no_bid}} in cents.
    Converts ET update_hour to UTC for candlestick timestamp comparison.
    """
    # ET to UTC: EST = UTC-5 (conservative, matches NWS CLI convention)
    utc_hour = update_hour + 5
    if utc_hour >= 24:
        lookup_date = event_date + timedelta(days=1)
        utc_hour -= 24
    else:
        lookup_date = event_date
    cutoff = datetime(lookup_date.year, lookup_date.month, lookup_date.day, utc_hour, 59, 59)

    # Build event ticker pattern: KXHIGHNY-YYMMMDD
    month_abbr = event_date.strftime("%b").upper()
    yr = event_date.strftime("%y")
    day = event_date.strftime("%d")
    event_ticker = "KXHIGHNY-{}{}{}".format(yr, month_abbr, day)

    rows = con.execute("""
        WITH ranked AS (
            SELECT market_ticker,
                   yes_bid_close, yes_ask_close,
                   ROW_NUMBER() OVER (PARTITION BY market_ticker ORDER BY end_period_ts DESC) as rn
            FROM kalshi_candlesticks
            WHERE market_ticker LIKE ? || '-%'
              AND end_period_ts <= ?
        )
        SELECT market_ticker, yes_bid_close, yes_ask_close
        FROM ranked WHERE rn = 1
    """, [event_ticker, cutoff]).fetchall()

    prices = {}
    for ticker, yes_bid, yes_ask in rows:
        if yes_bid is None or yes_ask is None:
            continue
        floor, cap = parse_ticker_bracket(ticker)
        if floor is None and cap is None:
            continue
        # Candlestick prices are already in cents (0-100 range)
        yb = int(round(yes_bid))
        ya = int(round(yes_ask))
        if ya <= 0 or ya >= 100:
            continue
        no_ask, no_bid = derive_no_prices(yb, ya)
        prices[(floor, cap)] = {
            "yes_ask": ya, "yes_bid": yb,
            "no_ask": no_ask, "no_bid": no_bid,
        }

    # Resolve T-tickers: lowest strike T = lower tail, highest = upper tail
    t_keys = [(f, c) for f, c in prices if f is None]
    if len(t_keys) == 2:
        strikes = sorted([c for _, c in t_keys])
        low_data = prices.pop((None, strikes[0]))
        high_data = prices.pop((None, strikes[1]))
        prices[(None, strikes[0])] = low_data
        prices[(strikes[1], None)] = high_data
    elif len(t_keys) == 1:
        interior_floors = [f for f, c in prices if f is not None and c is not None]
        t_strike = t_keys[0][1]
        if interior_floors and t_strike > max(interior_floors):
            data = prices.pop((None, t_strike))
            prices[(t_strike, None)] = data

    return prices


def get_settlement_data(con, event_date):
    # type: (duckdb.DuckDBPyConnection, date) -> Optional[Dict]
    """Get settlement outcomes from kalshi_settlements.

    Returns {(floor, cap): settled_yes} or NWS fallback dict, or None.
    """
    rows = con.execute("""
        SELECT market_ticker, floor_strike, cap_strike, settled_yes
        FROM kalshi_settlements
        WHERE event_date = ?
    """, [event_date]).fetchall()

    if rows:
        result = {}
        for ticker, floor_s, cap_s, settled in rows:
            floor = int(floor_s) if floor_s is not None else None
            cap = int(cap_s) if cap_s is not None else None
            result[(floor, cap)] = bool(settled)
        return result

    # Fallback: NWS daily actual high + bracket logic
    nws_row = con.execute("""
        SELECT max_temp_f FROM nws_daily
        WHERE station_id = 'KNYC' AND obs_date = ? AND max_temp_f IS NOT NULL
        ORDER BY CASE source WHEN 'NWS_CLI' THEN 0 ELSE 1 END
        LIMIT 1
    """, [event_date]).fetchone()
    if nws_row is None:
        return None
    return {"_nws_fallback": True, "_actual_high": int(nws_row[0])}


def get_running_max(con, event_date, update_hour):
    # type: (duckdb.DuckDBPyConnection, date, int) -> Optional[float]
    """Max observed temp up to update_hour ET on event_date."""
    row = con.execute("""
        SELECT MAX(temp_f) FROM observations
        WHERE station_id = 'KNYC' AND observed_at::DATE = ?
          AND EXTRACT(HOUR FROM observed_at AT TIME ZONE 'UTC' AT TIME ZONE 'EST')::INTEGER <= ?
          AND temp_f IS NOT NULL
    """, [event_date, update_hour]).fetchone()
    return row[0] if row and row[0] is not None else None


# ── Metrics & Output ─────────────────────────────────────────────────────


def compute_metrics(trades, starting_capital):
    # type: (List[Dict], float) -> Dict
    """Compute summary metrics for a list of trades."""
    if not trades:
        return {
            "total_pnl": 0.0, "roi_pct": 0.0, "win_rate": 0.0,
            "avg_edge": 0.0, "trade_count": 0, "max_drawdown": 0.0,
            "sharpe": 0.0, "profit_factor": 0.0,
        }

    total_pnl = sum(t["net_pnl"] for t in trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    gross_win = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))

    # Cumulative P&L for drawdown
    cum_pnl = []
    running = 0.0
    for t in trades:
        running += t["net_pnl"]
        cum_pnl.append(running)
    peak = cum_pnl[0]
    max_dd = 0.0
    for val in cum_pnl:
        if val > peak:
            peak = val
        dd = val - peak
        if dd < max_dd:
            max_dd = dd

    # Daily P&L for Sharpe
    daily_pnl = {}  # type: Dict[str, float]
    for t in trades:
        d = str(t["event_date"])
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t["net_pnl"]
    daily_vals = list(daily_pnl.values())
    if len(daily_vals) > 1:
        mean_d = sum(daily_vals) / len(daily_vals)
        std_d = (sum((v - mean_d) ** 2 for v in daily_vals) / (len(daily_vals) - 1)) ** 0.5
        sharpe = (mean_d / std_d * (252 ** 0.5)) if std_d > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / starting_capital * 100, 2),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_edge": round(sum(t["edge_pct"] for t in trades) / len(trades), 2),
        "trade_count": len(trades),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float('inf'),
    }


def sample_dates(con, sample_size, seed):
    # type: (duckdb.DuckDBPyConnection, int, Optional[int]) -> List[date]
    """Season-stratified random sample of backtest dates."""
    rows = con.execute("""
        SELECT DISTINCT event_date FROM kalshi_settlements
        WHERE event_date >= '2021-08-06' AND event_date <= '2026-02-23'
        ORDER BY event_date
    """).fetchall()
    all_dates = [r[0] for r in rows]

    seasons = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
    for d in all_dates:
        m = d.month
        if m in (12, 1, 2):
            seasons["DJF"].append(d)
        elif m in (3, 4, 5):
            seasons["MAM"].append(d)
        elif m in (6, 7, 8):
            seasons["JJA"].append(d)
        else:
            seasons["SON"].append(d)

    rng = random.Random(seed)
    per_season = sample_size // 4
    sampled = []
    for name in ["DJF", "MAM", "JJA", "SON"]:
        pool = seasons[name]
        n = min(per_season, len(pool))
        sampled.extend(rng.sample(pool, n))

    return sorted(sampled)


def all_backtest_dates(con):
    # type: (duckdb.DuckDBPyConnection) -> List[date]
    """All dates with settlement data."""
    rows = con.execute("""
        SELECT DISTINCT event_date FROM kalshi_settlements
        WHERE event_date >= '2021-08-06' AND event_date <= '2026-02-23'
        ORDER BY event_date
    """).fetchall()
    return [r[0] for r in rows]


def print_results_table(results):
    # type: (List[Tuple[str, Dict]]) -> None
    """Print ranked strategy comparison table."""
    results.sort(key=lambda x: x[1]["total_pnl"], reverse=True)

    header = "{:<20s} {:>8s} {:>7s} {:>7s} {:>8s} {:>8s} {:>7s} {:>6s}".format(
        "Strategy", "P&L($)", "ROI(%)", "Trades", "WinRate", "MaxDD", "Sharpe", "PF")
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for name, m in results:
        disq = " [DQ]" if m["max_drawdown"] < -20.0 else ""
        print("{:<20s} {:>+8.2f} {:>6.1f}% {:>7d} {:>7.1f}% {:>+8.2f} {:>7.2f} {:>6.2f}{}".format(
            name, m["total_pnl"], m["roi_pct"], m["trade_count"],
            m["win_rate"], m["max_drawdown"], m["sharpe"], m["profit_factor"], disq))
    print()


def run_backtest(args):
    # type: (argparse.Namespace) -> None
    """Main walk-forward backtest loop."""
    db_path = args.db
    fb = FeatureBuilder(db_path)
    model = QRModel()
    config = {
        "min_edge_pct": 5.0, "min_model_prob": 0.15, "min_ev_cents": 2,
        "max_per_bracket": 10, "starting_capital": 100.0,
    }

    con = duckdb.connect(db_path, read_only=True)
    try:
        if args.quick:
            dates = sample_dates(con, args.sample_size, args.seed)
            mode_label = "{} sample days".format(len(dates))
        else:
            dates = all_backtest_dates(con)
            mode_label = "{} days (full)".format(len(dates))
    finally:
        con.close()

    # Filter strategies if specified
    if args.strategies:
        wanted = set(args.strategies.split(","))
        strategies = [s for s in ALL_STRATEGIES if s.name in wanted]
    else:
        strategies = list(ALL_STRATEGIES)

    print("Strategy Comparison — {} — {} strategies".format(mode_label, len(strategies)))

    # Per-strategy state
    ledgers = {s.name: [] for s in strategies}
    bankrolls = {s.name: config["starting_capital"] for s in strategies}

    for event_date in tqdm(dates, desc="Backtesting", unit="day"):
        # Fetch settlement first, then close so FeatureBuilder can open its own connection
        con = duckdb.connect(db_path, read_only=True)
        try:
            settlement = get_settlement_data(con, event_date)
        finally:
            con.close()
        if settlement is None:
            continue

        held = {s.name: set() for s in strategies}
        day_entries = {s.name: [] for s in strategies}

        cached_run_hour = None
        model_fitted = False
        fit_date_key = None

        for update_hour in range(24):
            # 1. Train model (cache when run_hour unchanged)
            # FeatureBuilder manages its own DuckDB connection internally
            result = fb.get_training_data(event_date, update_hour)
            if result is None:
                continue
            X, y, train_dates, run_hour = result

            if run_hour != cached_run_hour:
                fit_date_key = "{}-rh{}".format(event_date, run_hour)
                fit_result = model.fit(X, y, run_hour=run_hour, date_key=fit_date_key)
                if fit_result is None:
                    continue
                cached_run_hour = run_hour
                model_fitted = True

            if not model_fitted:
                continue

            # 2. Build live features
            feat_result = fb.build_features(event_date, update_hour)
            if feat_result is None:
                continue
            features, fcst_high, rh = feat_result

            # 3. Predict + get market prices (read-only queries)
            con = duckdb.connect(db_path, read_only=True)
            try:
                running_max = get_running_max(con, event_date, update_hour)
                probs = model.predict_bracket_probs(features, fcst_high, rh, fit_date_key,
                                                     running_max=running_max)
                if not probs:
                    continue

                # 4. Get market prices
                prices = get_candlestick_prices(con, event_date, update_hour)
            finally:
                con.close()
            if not prices:
                continue

            model_median = compute_model_median(probs)

            # 5. For each strategy: generate signals, filter, size, dedup, record
            for strategy in strategies:
                sname = strategy.name
                raw_signals = generate_signals(probs, prices, config, bankrolls[sname])
                filtered = strategy.filter(raw_signals, probs, model_median)

                for sig in filtered:
                    key = (sig["bracket_floor"], sig["bracket_cap"], sig["direction"])
                    if key in held[sname]:
                        continue
                    held[sname].add(key)

                    entry_fee = compute_fee(sig["contracts"], sig["market_price"])

                    bracket_key = (sig["bracket_floor"], sig["bracket_cap"])
                    if settlement.get("_nws_fallback"):
                        actual = settlement["_actual_high"]
                        settled_yes = resolve_settlement(
                            sig["bracket_floor"], sig["bracket_cap"], actual)
                    elif bracket_key in settlement:
                        settled_yes = settlement[bracket_key]
                    else:
                        continue

                    net = compute_pnl(settled_yes, sig["direction"],
                                      sig["market_price"], sig["contracts"], entry_fee)

                    day_entries[sname].append({
                        "event_date": str(event_date),
                        "bracket_floor": sig["bracket_floor"],
                        "bracket_cap": sig["bracket_cap"],
                        "direction": sig["direction"],
                        "model_prob": sig["model_prob"],
                        "market_price": sig["market_price"],
                        "edge_pct": sig["edge_pct"],
                        "contracts": sig["contracts"],
                        "entry_fee": entry_fee,
                        "settled_yes": settled_yes,
                        "net_pnl": net,
                    })

        # End of day: update bankrolls
        for strategy in strategies:
            sname = strategy.name
            ledgers[sname].extend(day_entries[sname])
            day_pnl = sum(e["net_pnl"] for e in day_entries[sname])
            bankrolls[sname] += day_pnl

    # Output results
    results = []
    for strategy in strategies:
        m = compute_metrics(ledgers[strategy.name], config["starting_capital"])
        results.append((strategy.name, m))

    print_results_table(results)

    # Save detailed results
    os.makedirs("data/backtest_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if args.quick else "full"
    out_path = "data/backtest_results/{}_{}.json".format(mode, ts)
    output = {
        "mode": mode, "dates": len(dates), "seed": args.seed,
        "strategies": {s.name: {"metrics": m, "trades": ledgers[s.name]}
                       for s, (_, m) in zip(strategies, results)},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Detailed results saved to {}".format(out_path))


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
    run_backtest(args)


if __name__ == "__main__":
    main()
