# Strategy Comparison Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtest runner that mirrors the live trading engine exactly, then compare 12 signal filtering strategies to find the best betting approach.

**Architecture:** Single standalone script (`scripts/strategy_comparison.py`) that imports and calls the live engine's `FeatureBuilder` and `QRModel` directly. Strategy filters, Kelly sizing, P&L ledger, and market data lookup are owned by the script. All positions hold to settlement (no intra-day exit modeling).

**Tech Stack:** Python 3.9, DuckDB (read-only), numpy, scipy (via QRModel), tqdm, argparse

**Spec:** `docs/superpowers/specs/2026-03-22-strategy-backtest-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/strategy_comparison.py` | CLI, walk-forward loop, signal generation, strategy filters, Kelly sizing, P&L, settlement, output |
| `tests/test_strategy_comparison.py` | All unit + integration tests |
| `services/paper_trader.py` | Bug fix: settlement boundary (line 225) |
| `services/settlement.py` | Bug fix: settlement boundary (line 125) + tail handling |

---

## Task 0: Fix Settlement Boundary Bugs

**Files:**
- Modify: `services/paper_trader.py:225`
- Modify: `services/settlement.py:124-125`
- Modify: `tests/test_paper_trader.py`
- Modify: `tests/test_settlement.py`

- [ ] **Step 1: Write failing test for paper_trader cap-inclusive settlement**

Add to `tests/test_paper_trader.py`:

```python
@pytest.mark.asyncio
async def test_settlement_cap_inclusive(test_db):
    """Kalshi CFTC filing: interior brackets are [floor, cap] — both inclusive."""
    db_path, _ = test_db
    con = duckdb.connect(db_path)
    # Position: YES on [64, 66], entry at 50c, 1 contract
    con.execute("""
        INSERT INTO paper_positions (id, city, event_date, bracket_floor, bracket_cap,
            direction, model_prob, market_price, edge, entry_price, entry_time,
            fees, status, contracts)
        VALUES (999, 'NYC', '2026-01-15', 64, 66, 'YES', 0.5, 50, 5.0, 50,
            '2026-01-15 12:00:00', 0.04, 'open', 1)
    """)
    # Settlement at exactly 66 — should settle YES (cap inclusive)
    con.execute("""
        INSERT INTO nws_daily (station_id, obs_date, max_temp_f, min_temp_f, source, ingested_at)
        VALUES ('KNYC', '2026-01-15', 66, 40, 'NWS_CLI', '2026-01-15 22:00:00')
    """)
    con.close()

    pt = PaperTrader(db_path)
    await pt._settle_positions()

    con = duckdb.connect(db_path, read_only=True)
    row = con.execute("SELECT settled_yes, net_pnl FROM paper_positions WHERE id = 999").fetchone()
    con.close()
    assert row[0] is True, "Temp at cap boundary (66) should settle YES"
    assert row[1] > 0, "Should be a winning trade"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_paper_trader.py::test_settlement_cap_inclusive -v`
Expected: FAIL — current code uses `< cap_val`, so 66 == 66 returns False.

- [ ] **Step 3: Fix paper_trader.py settlement boundary**

In `services/paper_trader.py` line 225, change:
```python
# OLD
settled_yes = floor_val <= actual_high < cap_val  # interior
# NEW
settled_yes = floor_val <= actual_high <= cap_val  # interior (both inclusive per Kalshi CFTC)
```

Also update the docstring at line 198:
```python
# OLD
- Bracket is [floor, cap) — floor inclusive, cap exclusive
# NEW
- Bracket is [floor, cap] — both inclusive (per Kalshi CFTC filing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_paper_trader.py::test_settlement_cap_inclusive -v`
Expected: PASS

- [ ] **Step 5: Write failing test for settlement.py boundary + tail handling**

Add to `tests/test_settlement.py`:

```python
@pytest.mark.asyncio
async def test_settlement_cap_inclusive(test_db):
    """Interior brackets: [floor, cap] both inclusive."""
    db_path, _ = test_db
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
    db_path, _ = test_db
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_settlement.py::test_settlement_cap_inclusive tests/test_settlement.py::test_settlement_lower_tail -v`
Expected: FAIL — cap-inclusive test fails, tail test crashes with TypeError.

- [ ] **Step 7: Fix settlement.py boundary + tail handling**

In `services/settlement.py` line 124-125, replace:
```python
# OLD
# Bracket is [floor, cap) — floor inclusive, cap exclusive
settled_yes = floor_val <= actual_high < cap_val
```
```python
# NEW
if floor_val is None:
    settled_yes = actual_high < cap_val          # lower tail
elif cap_val is None:
    settled_yes = actual_high > floor_val         # upper tail
else:
    settled_yes = floor_val <= actual_high <= cap_val  # interior (both inclusive)
```

- [ ] **Step 8: Run all settlement + paper_trader tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_paper_trader.py tests/test_settlement.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/paper_trader.py services/settlement.py tests/test_paper_trader.py tests/test_settlement.py
git commit -m "fix: settlement boundary — both inclusive per Kalshi CFTC filing

Interior brackets now use [floor, cap] (both inclusive) instead of
[floor, cap) (cap exclusive). Also fixes settlement.py tail bracket
handling that would crash with TypeError on NULL floor/cap."
```

---

## Task 1: Script Scaffold + Pure Functions

**Files:**
- Create: `scripts/strategy_comparison.py`
- Create: `tests/test_strategy_comparison.py`

- [ ] **Step 1: Write tests for pure utility functions**

Create `tests/test_strategy_comparison.py`:

```python
# tests/test_strategy_comparison.py
"""Tests for strategy comparison runner."""
import math
import pytest


# ── Fee calculation ──────────────────────────────────────────────────────

class TestComputeFee:
    def test_basic_fee(self):
        from scripts.strategy_comparison import compute_fee
        # 1 contract at 50c → 0.07 * 1 * 0.5 * 0.5 = 0.0175 → ceil(1.75) = 2 cents = $0.02
        assert compute_fee(1, 50) == 0.02

    def test_minimum_fee(self):
        from scripts.strategy_comparison import compute_fee
        # 1 contract at 5c → raw very small, minimum is $0.01/contract
        assert compute_fee(1, 5) == 0.01

    def test_multi_contract(self):
        from scripts.strategy_comparison import compute_fee
        # 3 contracts at 50c → 0.07 * 3 * 0.5 * 0.5 = 0.0525 → ceil(5.25) = 6 cents = $0.06
        # min = 3 * $0.01 = $0.03
        assert compute_fee(3, 50) == 0.06


# ── Kelly sizing ─────────────────────────────────────────────────────────

class TestKellySize:
    def test_basic_sizing(self):
        from scripts.strategy_comparison import kelly_size
        # model_prob=0.6, price=50c, bankroll=$100, max=10
        # edge = 0.1, half_kelly = 0.05, raw = 0.05 * 100 / 0.50 = 10
        assert kelly_size(0.6, 50, 100.0, 10) == 10

    def test_clamp_to_max(self):
        from scripts.strategy_comparison import kelly_size
        # Large edge → raw > max_per_bracket → clamp
        assert kelly_size(0.8, 20, 200.0, 5) == 5

    def test_floor_to_one(self):
        from scripts.strategy_comparison import kelly_size
        # Tiny edge → raw < 1 → floor to 1
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
        # gross = (100-30)*2/100 = 1.40, net = 1.40 - 0.04 = 1.36
        assert net == pytest.approx(1.36, abs=0.01)

    def test_yes_loss(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=False, direction='YES', entry_price_cents=30, contracts=2, entry_fee=0.04)
        # gross = -(30*2/100) = -0.60, net = -0.60 - 0.04 = -0.64
        assert net == pytest.approx(-0.64, abs=0.01)

    def test_no_win(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=False, direction='NO', entry_price_cents=40, contracts=1, entry_fee=0.02)
        # gross = (100-40)*1/100 = 0.60, net = 0.60 - 0.02 = 0.58
        assert net == pytest.approx(0.58, abs=0.01)

    def test_no_loss(self):
        from scripts.strategy_comparison import compute_pnl
        net = compute_pnl(settled_yes=True, direction='NO', entry_price_cents=40, contracts=1, entry_fee=0.02)
        # gross = -(40*1/100) = -0.40, net = -0.40 - 0.02 = -0.42
        assert net == pytest.approx(-0.42, abs=0.01)


# ── Settlement ───────────────────────────────────────────────────────────

class TestSettlement:
    def test_interior_both_inclusive(self):
        from scripts.strategy_comparison import resolve_settlement
        assert resolve_settlement(64, 66, 66) is True   # cap inclusive
        assert resolve_settlement(64, 66, 64) is True   # floor inclusive
        assert resolve_settlement(64, 66, 65) is True   # middle
        assert resolve_settlement(64, 66, 63) is False   # below
        assert resolve_settlement(64, 66, 67) is False   # above

    def test_lower_tail(self):
        from scripts.strategy_comparison import resolve_settlement
        assert resolve_settlement(None, 50, 49) is True
        assert resolve_settlement(None, 50, 50) is False  # strict less-than
        assert resolve_settlement(None, 50, 51) is False

    def test_upper_tail(self):
        from scripts.strategy_comparison import resolve_settlement
        assert resolve_settlement(80, None, 81) is True
        assert resolve_settlement(80, None, 80) is False  # strict greater-than
        assert resolve_settlement(80, None, 79) is False


# ── Model median ─────────────────────────────────────────────────────────

class TestModelMedian:
    def test_basic_median(self):
        from scripts.strategy_comparison import compute_model_median
        probs = {60: 0.1, 61: 0.15, 62: 0.25, 63: 0.25, 64: 0.15, 65: 0.1}
        median = compute_model_median(probs)
        # cumulative: 60→0.1, 61→0.25, 62→0.50 → median is 62
        assert median == 62.0

    def test_median_crosses_at_boundary(self):
        from scripts.strategy_comparison import compute_model_median
        probs = {70: 0.4, 71: 0.2, 72: 0.4}
        # cumulative: 70→0.4, 71→0.6 → crosses 0.5 at 71
        assert compute_model_median(probs) == 71.0
```

- [ ] **Step 2: Create script with pure functions**

Create `scripts/strategy_comparison.py`:

```python
#!/usr/bin/env python3
"""Strategy Comparison Runner — backtest 12 signal filtering strategies.

Mirrors the live StrategyEngine exactly, using the same FeatureBuilder and QRModel.
Compares strategy filters to find the best betting approach.

Usage:
    python scripts/strategy_comparison.py --quick          # 100 sample days
    python scripts/strategy_comparison.py --full           # all ~1659 days
    python scripts/strategy_comparison.py --quick --seed 42
"""
import argparse
import math
import sys
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
    print("Strategy comparison runner — not yet implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_strategy_comparison.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add scripts/strategy_comparison.py tests/test_strategy_comparison.py
git commit -m "feat: scaffold strategy comparison runner with pure utility functions

Adds compute_fee, kelly_size, compute_pnl, resolve_settlement,
compute_model_median — all tested. CLI skeleton in place."
```

---

## Task 2: Strategy Filters

**Files:**
- Modify: `scripts/strategy_comparison.py`
- Modify: `tests/test_strategy_comparison.py`

- [ ] **Step 1: Write tests for strategy filters**

Add to `tests/test_strategy_comparison.py`:

```python
# ── Strategy Filters ─────────────────────────────────────────────────────

def _make_signal(floor, cap, direction, edge_pct, model_prob=0.5, market_price=40, contracts=1):
    """Helper to build signal dicts for filter tests."""
    return {
        "bracket_floor": floor, "bracket_cap": cap, "direction": direction,
        "model_prob": model_prob, "market_price": market_price,
        "edge_pct": edge_pct, "contracts": contracts,
    }


class TestBaselineFilter:
    def test_passes_everything(self):
        from scripts.strategy_comparison import BaselineFilter
        signals = [_make_signal(60, 62, "yes", 8.0), _make_signal(70, 72, "yes", 6.0)]
        result = BaselineFilter().filter(signals, {}, 65.0)
        assert len(result) == 2


class TestAdjacencyFilter:
    def test_within_distance(self):
        from scripts.strategy_comparison import AdjacencyFilter
        signals = [
            _make_signal(64, 66, "yes", 8.0),   # distance 0 from median 65
            _make_signal(60, 62, "yes", 6.0),    # distance 3 from median 65
            _make_signal(70, 72, "yes", 7.0),    # distance 5 from median 65
        ]
        result = AdjacencyFilter(max_distance=4).filter(signals, {}, 65.0)
        assert len(result) == 2  # 64-66 and 60-62, not 70-72

    def test_tail_bracket(self):
        from scripts.strategy_comparison import AdjacencyFilter
        signals = [_make_signal(None, 50, "yes", 8.0)]  # lower tail
        result = AdjacencyFilter(max_distance=4).filter(signals, {}, 65.0)
        assert len(result) == 0  # distance = 65-50 = 15, too far


class TestTopKFilter:
    def test_top_2(self):
        from scripts.strategy_comparison import TopKFilter
        signals = [
            _make_signal(60, 62, "yes", 5.0),
            _make_signal(64, 66, "yes", 10.0),
            _make_signal(68, 70, "yes", 7.0),
        ]
        result = TopKFilter(k=2).filter(signals, {}, 65.0)
        assert len(result) == 2
        edges = [s["edge_pct"] for s in result]
        assert 10.0 in edges and 7.0 in edges

    def test_fewer_than_k(self):
        from scripts.strategy_comparison import TopKFilter
        signals = [_make_signal(60, 62, "yes", 8.0)]
        result = TopKFilter(k=3).filter(signals, {}, 65.0)
        assert len(result) == 1  # only 1 signal, return it


class TestPortfolioEVFilter:
    def test_mutually_exclusive_ev(self):
        from scripts.strategy_comparison import PortfolioEVFilter
        # Two signals — adding the second should only help if combined EV improves
        signals = [
            _make_signal(64, 66, "yes", 10.0, model_prob=0.5, market_price=40, contracts=1),
            _make_signal(60, 62, "yes", 6.0, model_prob=0.3, market_price=24, contracts=1),
        ]
        probs = {i: 0.05 for i in range(55, 75)}  # uniform-ish
        result = PortfolioEVFilter().filter(signals, probs, 65.0)
        assert len(result) >= 1  # at least the best signal


class TestHybridFilter:
    def test_adjacency_then_top_k(self):
        from scripts.strategy_comparison import HybridFilter
        signals = [
            _make_signal(64, 66, "yes", 10.0),  # near median 65, high edge
            _make_signal(63, 65, "yes", 8.0),   # near median, lower edge
            _make_signal(60, 62, "yes", 12.0),   # far from median, highest edge
            _make_signal(70, 72, "yes", 15.0),   # far from median
        ]
        # Adjacency(4) keeps 64-66 and 63-65 (60-62 is 3 away, also passes; 70-72 is 5 away, fails)
        # Top-K(2) then takes the 2 highest edges from remaining
        result = HybridFilter(max_distance=4, k=2).filter(signals, {}, 65.0)
        assert len(result) == 2
        edges = sorted([s["edge_pct"] for s in result], reverse=True)
        assert edges[0] == 12.0  # 60-62 passes adjacency(4) and has highest edge
        assert edges[1] == 10.0  # 64-66
```

- [ ] **Step 2: Implement all strategy filters**

Add to `scripts/strategy_comparison.py`:

```python
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
            # Portfolio EV with mutual exclusivity:
            # sum of (prob_i * win_pnl_i) + (1 - sum_probs) * sum(lose_pnl)
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
```

- [ ] **Step 3: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_strategy_comparison.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add scripts/strategy_comparison.py tests/test_strategy_comparison.py
git commit -m "feat: add 12 strategy filters for comparison runner

Baseline, Adjacency(2/4/6/8), TopK(1/2/3/5), PortfolioEV (mutually
exclusive), Hybrid(4-2, 6-3). All tested."
```

---

## Task 3: Market Data + Signal Generation

**Files:**
- Modify: `scripts/strategy_comparison.py`
- Modify: `tests/test_strategy_comparison.py`

- [ ] **Step 1: Write tests for candlestick ticker parsing and NO-side price derivation**

Add to `tests/test_strategy_comparison.py`:

```python
class TestParseTickerBracket:
    def test_interior_bracket(self):
        from scripts.strategy_comparison import parse_ticker_bracket
        # B61.5 → floor=61, cap=62 (ticker is midpoint of 2°F bracket)
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
        # T-tickers are ambiguous (could be upper or lower tail).
        # parse_ticker_bracket returns (None, strike) for all T-tickers.
        # The caller (get_candlestick_prices) resolves upper vs lower
        # by comparing strikes across all brackets for the event.
        assert floor is None
        assert cap == 80


class TestDeriveNoPrices:
    def test_no_from_yes(self):
        from scripts.strategy_comparison import derive_no_prices
        no_ask, no_bid = derive_no_prices(yes_bid=40, yes_ask=45)
        assert no_ask == 60   # 100 - yes_bid
        assert no_bid == 55   # 100 - yes_ask


class TestGenerateSignals:
    def test_yes_signal(self):
        from scripts.strategy_comparison import generate_signals
        probs = {60: 0.05, 61: 0.1, 62: 0.2, 63: 0.3, 64: 0.2, 65: 0.1, 66: 0.05}
        prices = {
            (62, 64): {"yes_ask": 30, "yes_bid": 28, "no_ask": 72, "no_bid": 70},
        }
        config = {"min_edge_pct": 5.0, "min_model_prob": 0.15, "min_ev_cents": 2,
                  "max_per_bracket": 10}
        signals = generate_signals(probs, prices, config, bankroll=100.0)
        # model_prob for [62,64] = 0.2 + 0.3 + 0.2 = 0.7
        # yes_edge = (0.7 - 0.30) * 100 = 40%
        assert len(signals) >= 1
        assert signals[0]["direction"] == "YES"
        assert signals[0]["edge_pct"] == pytest.approx(40.0, abs=0.1)

    def test_no_signal(self):
        from scripts.strategy_comparison import generate_signals
        # Model says bracket is unlikely (prob 0.1), market asks 70c for YES
        # → NO edge: no_prob=0.9, no_ask=100-68=32 → no_edge=(0.9-0.32)*100=58%
        probs = {60: 0.05, 61: 0.05, 62: 0.05, 63: 0.05, 64: 0.3, 65: 0.3, 66: 0.2}
        prices = {
            (60, 62): {"yes_ask": 70, "yes_bid": 68, "no_ask": 32, "no_bid": 30},
        }
        config = {"min_edge_pct": 5.0, "min_model_prob": 0.15, "min_ev_cents": 2,
                  "max_per_bracket": 10}
        signals = generate_signals(probs, prices, config, bankroll=100.0)
        # model_prob for [60,62] = 0.05+0.05+0.05 = 0.15
        # yes_edge = (0.15-0.70)*100 = -55% → fails
        # no_prob = 0.85, no_edge = (0.85-0.32)*100 = 53%
        assert len(signals) == 1
        assert signals[0]["direction"] == "NO"
        assert signals[0]["edge_pct"] == pytest.approx(53.0, abs=0.1)
```

- [ ] **Step 2: Implement ticker parsing, price derivation, and signal generation**

Add to `scripts/strategy_comparison.py`:

```python
import re

# ── Market Data ──────────────────────────────────────────────────────────


def parse_ticker_bracket(ticker):
    # type: (str) -> Tuple[Optional[int], Optional[int]]
    """Extract floor/cap from a Kalshi ticker string.

    Interior: KXHIGHNY-26JAN15-B61.5 → floor=61, cap=62
    Lower tail: KXHIGHNY-26JAN15-T41 → floor=None, cap=41
    Upper tail: determined by being the highest-strike T-ticker for an event.
    """
    parts = ticker.split("-")
    bracket_part = parts[-1]  # e.g. "B61.5" or "T41"

    if bracket_part.startswith("B"):
        midpoint = float(bracket_part[1:])
        floor_val = int(midpoint - 0.5)
        cap_val = int(midpoint + 0.5)
        return floor_val, cap_val
    elif bracket_part.startswith("T"):
        strike = int(bracket_part[1:])
        # T-tickers: the lower tail has the lowest strike, upper tail has highest.
        # We return (None, strike) and let the caller resolve upper vs lower
        # by comparing against other brackets for the same event.
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
    """Aggregate 1°F model probs to Kalshi bracket format.

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
        bracket_probs: {int: float} — 1°F model probabilities
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
```

- [ ] **Step 3: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_strategy_comparison.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add scripts/strategy_comparison.py tests/test_strategy_comparison.py
git commit -m "feat: add market data parsing and signal generation

Ticker parsing, NO-price derivation, bracket aggregation, and signal
generation — all mirroring live StrategyEngine exactly."
```

---

## Task 4: Walk-Forward Loop + Candlestick Queries

**Files:**
- Modify: `scripts/strategy_comparison.py`
- Modify: `tests/test_strategy_comparison.py`

This is the core loop. It ties everything together.

- [ ] **Step 1: Write integration test against real DB**

Add to `tests/test_strategy_comparison.py`:

```python
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "alphatemp.duckdb")


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Requires alphatemp.duckdb")
class TestCandlestickLookup:
    def test_get_candlestick_prices(self):
        from scripts.strategy_comparison import get_candlestick_prices
        import duckdb
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            prices = get_candlestick_prices(con, date(2025, 7, 15), 12)
            assert len(prices) > 0, "Should find brackets for a mid-2025 date"
            # Each entry should have yes_ask, yes_bid, no_ask, no_bid
            for key, p in prices.items():
                assert "yes_ask" in p and "no_ask" in p
                assert 0 < p["yes_ask"] < 100
        finally:
            con.close()


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Requires alphatemp.duckdb")
class TestGetSettlement:
    def test_settlement_from_kalshi(self):
        from scripts.strategy_comparison import get_settlement_data
        import duckdb
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            result = get_settlement_data(con, date(2025, 7, 15))
            assert result is not None, "Should have settlement data for mid-2025"
            # result = {(floor, cap): settled_yes}
            assert len(result) > 0
        finally:
            con.close()


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="Requires alphatemp.duckdb")
class TestRunningMax:
    def test_running_max_filtered_by_hour(self):
        from scripts.strategy_comparison import get_running_max
        import duckdb
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            # At hour 0, running max should be None or very low (no daytime obs yet)
            rm_0 = get_running_max(con, date(2025, 7, 15), 0)
            # At hour 18, should have daytime peak
            rm_18 = get_running_max(con, date(2025, 7, 15), 18)
            if rm_0 is not None and rm_18 is not None:
                assert rm_18 >= rm_0, "Afternoon max should be >= midnight max"
        finally:
            con.close()
```

- [ ] **Step 2: Implement data query functions**

Add to `scripts/strategy_comparison.py`:

```python
import duckdb
from datetime import date, datetime, timedelta


def get_candlestick_prices(con, event_date, update_hour):
    # type: (duckdb.DuckDBPyConnection, date, int) -> Dict[Tuple, Dict]
    """Get latest candlestick prices for each bracket at the given hour.

    Returns {(floor, cap): {yes_ask, yes_bid, no_ask, no_bid}} in cents.
    Converts ET update_hour to UTC for candlestick timestamp comparison.
    """
    # ET to UTC: EST = UTC-5, EDT = UTC-4. Use EST (conservative, matches NWS CLI convention)
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
        no_ask, no_bid = derive_no_prices(yb, ya)
        prices[(floor, cap)] = {
            "yes_ask": ya, "yes_bid": yb,
            "no_ask": no_ask, "no_bid": no_bid,
        }

    # Resolve T-tickers: lowest strike T = lower tail, highest = upper tail
    t_keys = [(f, c) for f, c in prices if f is None]
    if len(t_keys) == 2:
        strikes = sorted([c for _, c in t_keys])
        # Lower tail: (None, low_strike), Upper tail: (high_strike, None)
        low_data = prices.pop((None, strikes[0]))
        high_data = prices.pop((None, strikes[1]))
        prices[(None, strikes[0])] = low_data
        prices[(strikes[1], None)] = high_data
    elif len(t_keys) == 1:
        # Single T-ticker — need interior brackets to determine if upper or lower
        interior_floors = [f for f, c in prices if f is not None and c is not None]
        t_strike = t_keys[0][1]
        if interior_floors and t_strike > max(interior_floors):
            data = prices.pop((None, t_strike))
            prices[(t_strike, None)] = data

    return prices


def get_settlement_data(con, event_date):
    # type: (duckdb.DuckDBPyConnection, date) -> Optional[Dict[Tuple, bool]]
    """Get settlement outcomes from kalshi_settlements.

    Returns {(floor, cap): settled_yes} or None if no settlement data.
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
```

- [ ] **Step 3: Run tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_strategy_comparison.py -v`
Expected: ALL PASS (DB-dependent tests skip if DB not found)

- [ ] **Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add scripts/strategy_comparison.py tests/test_strategy_comparison.py
git commit -m "feat: add candlestick price lookups, settlement data, running_max

Queries kalshi_candlesticks for historical market prices, resolves
T-ticker ambiguity (upper vs lower tail), derives NO-side prices."
```

---

## Task 5: Walk-Forward Loop + Output

**Files:**
- Modify: `scripts/strategy_comparison.py`
- Modify: `tests/test_strategy_comparison.py`

- [ ] **Step 1: Write test for metrics calculation**

Add to `tests/test_strategy_comparison.py`:

```python
class TestMetrics:
    def test_compute_metrics(self):
        from scripts.strategy_comparison import compute_metrics
        trades = [
            {"net_pnl": 0.50, "event_date": "2025-01-01", "edge_pct": 8.0},
            {"net_pnl": -0.30, "event_date": "2025-01-02", "edge_pct": 6.0},
            {"net_pnl": 0.40, "event_date": "2025-01-03", "edge_pct": 10.0},
            {"net_pnl": 0.20, "event_date": "2025-01-04", "edge_pct": 7.0},
        ]
        m = compute_metrics(trades, starting_capital=100.0)
        assert m["total_pnl"] == pytest.approx(0.80, abs=0.01)
        assert m["roi_pct"] == pytest.approx(0.80, abs=0.01)
        assert m["win_rate"] == pytest.approx(75.0, abs=0.1)
        assert m["trade_count"] == 4
        assert m["avg_edge"] == pytest.approx(7.75, abs=0.01)
        assert m["max_drawdown"] <= 0  # drawdown is negative
        assert m["profit_factor"] > 1.0

    def test_empty_trades(self):
        from scripts.strategy_comparison import compute_metrics
        m = compute_metrics([], starting_capital=100.0)
        assert m["total_pnl"] == 0.0
        assert m["trade_count"] == 0
```

- [ ] **Step 2: Implement metrics, sampling, and the main walk-forward loop**

Add to `scripts/strategy_comparison.py`:

```python
import json
import os
import random
from datetime import date

import numpy as np
from tqdm import tqdm

# Import live engine components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.feature_builder import FeatureBuilder
from services.model import QRModel


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

    # Stratify by season: DJF=Dec/Jan/Feb, MAM, JJA, SON
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
    # Sort by total P&L descending
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
    ledgers = {s.name: [] for s in strategies}       # type: Dict[str, List[Dict]]
    bankrolls = {s.name: config["starting_capital"] for s in strategies}

    for event_date in tqdm(dates, desc="Backtesting", unit="day"):
        con = duckdb.connect(db_path, read_only=True)
        try:
            # Get settlement for this date
            settlement = get_settlement_data(con, event_date)
            if settlement is None:
                continue

            # Per-strategy held positions for this day
            held = {s.name: set() for s in strategies}  # type: Dict[str, set]
            day_entries = {s.name: [] for s in strategies}

            cached_run_hour = None
            model_fitted = False
            fit_date_key = None  # stable key for QRModel cache lookup

            for update_hour in range(24):
                # 1. Train model (cache when run_hour unchanged)
                result = fb.get_training_data(event_date, update_hour)
                if result is None:
                    continue
                X, y, train_dates, run_hour = result

                if run_hour != cached_run_hour:
                    # CRITICAL: use stable date_key tied to run_hour, not update_hour.
                    # QRModel caches coefficients by (run_hour, date_key).
                    # predict_bracket_probs must use the SAME date_key as fit().
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

                # 3. Predict (use same date_key as fit for cache hit)
                running_max = get_running_max(con, event_date, update_hour)
                probs = model.predict_bracket_probs(features, fcst_high, rh, fit_date_key,
                                                     running_max=running_max)
                if not probs:
                    continue

                # 4. Get market prices
                prices = get_candlestick_prices(con, event_date, update_hour)
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
                            continue  # dedup
                        held[sname].add(key)

                        entry_fee = compute_fee(sig["contracts"], sig["market_price"])
                        # Settle this position
                        bracket_key = (sig["bracket_floor"], sig["bracket_cap"])
                        if settlement.get("_nws_fallback"):
                            # NWS fallback — apply bracket logic
                            actual = settlement["_actual_high"]
                            settled_yes = resolve_settlement(
                                sig["bracket_floor"], sig["bracket_cap"], actual)
                        elif bracket_key in settlement:
                            settled_yes = settlement[bracket_key]
                        else:
                            # Bracket not in settlement data — skip
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

            # End of day: update bankrolls, extend ledgers
            for strategy in strategies:
                sname = strategy.name
                ledgers[sname].extend(day_entries[sname])
                day_pnl = sum(e["net_pnl"] for e in day_entries[sname])
                bankrolls[sname] += day_pnl
        finally:
            con.close()

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
```

Update `main()`:

```python
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
```

- [ ] **Step 3: Run all tests**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/test_strategy_comparison.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add scripts/strategy_comparison.py tests/test_strategy_comparison.py
git commit -m "feat: complete walk-forward loop, metrics, and output

Full backtest runner with model caching, season-stratified sampling,
per-strategy bankroll tracking, ranked results table, JSON export."
```

---

## Task 6: Smoke Test + Quick Run

**Files:** None new — validation only.

- [ ] **Step 1: Run all project tests to verify no regressions**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 -m pytest tests/ -q --timeout=60`
Expected: All existing tests pass, no regressions.

- [ ] **Step 2: Smoke test with 5 sample days**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 scripts/strategy_comparison.py --quick --sample-size 5 --seed 42`
Expected: Runs to completion, prints results table, saves JSON. Verify output is reasonable (no NaN, no obviously wrong numbers).

- [ ] **Step 3: Run the real quick-mode comparison (100 days)**

Run: `cd ~/Projects/alphatemp/alphatemp && python3 scripts/strategy_comparison.py --quick --seed 42`
Expected: Completes in ~15-20 minutes. Prints ranked table of all 12 strategies.

- [ ] **Step 4: Review results with Russell**

Present the ranked results table. Discuss which strategy to implement in the live engine. No commit needed — this is a decision point.

---

## Execution Notes

- **Task 0** (bug fixes) is independent and can be done first or in parallel.
- **Tasks 1-5** are sequential — each builds on the previous.
- **Task 6** is the validation/decision point.
- The script uses `sys.path.insert` to import from `services/` — standard pattern for scripts in this project.
- All tests use `pytest.approx` for floating point comparisons.
- DB-dependent integration tests use `@pytest.mark.skipif` so they pass in CI without the DB.
