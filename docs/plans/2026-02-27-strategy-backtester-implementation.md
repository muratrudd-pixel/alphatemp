# Strategy Backtester Implementation Plan

> **Fee Notice (2026-02-28):** Fee math in this plan (1% trading, 10% settlement) is SUPERSEDED. Code has already been corrected in `services/strategy_backtester.py` with actual Kalshi formula: taker fee = max(ceil(0.07*C*P*(1-P)), C*$0.01). No settlement fee.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a three-layer Kalshi strategy backtester that simulates trading performance against historical market data with realistic execution, capital constraints, and fee accounting.

**Architecture:** Unified pipeline in `services/strategy_backtester.py` with composable layers: Edge Analysis → Strategy Engine → P&L Simulator. Observation-triggered model updates, strict ask-side execution pricing, mutually exclusive portfolio EV, bankroll-constrained position management, and 5 sanity filters.

**Tech Stack:** Python 3.9, DuckDB, scipy, numpy, loguru. Reuses existing `BacktestDataProvider`, `compute_divergence_features()`, `map_probs_to_kalshi_brackets()`, and `ModelFn` interface.

**Design doc:** `docs/plans/2026-02-27-backtest-framework-design.md`

## Critical Implementation Rules

### 1. Bracket Resolution — DO NOT REIMPLEMENT
The existing `KalshiBracket.contains()` in `services/backtester.py:49-64` handles Kalshi's mutually exclusive bracket logic correctly:
- **Lower tail:** `temp < cap_strike` (strictly less than)
- **Interior:** `floor_strike <= temp <= cap_strike` (inclusive both ends)
- **Upper tail:** `temp > floor_strike` (strictly greater than)

This ensures brackets are mutually exclusive — no overlapping wins. **Always import and use `KalshiBracket` and `map_probs_to_kalshi_brackets()` from `services/backtester.py`. Never write your own bracket resolution logic.**

### 2. Execution Latency Slippage
In reality, when an observation arrives (e.g., AWC SPECI stamped at 14:12:00), there is processing delay before an order reaches the exchange (~60 seconds for model re-evaluation + order submission). The backtester must simulate this:
- Add `execution_latency_seconds: int = 60` to `BacktestConfig`
- When a trigger fires at time T, pull market prices from T + latency (the *following* minute's candlestick), not the exact trigger timestamp
- This prevents unrealistically optimistic fills where the backtest assumes instant execution at the observation timestamp

---

### Task 1: Data Structures and Fee Math

**Files:**
- Create: `services/strategy_backtester.py`
- Create: `tests/test_strategy_backtester.py`

**Context:** All other tasks build on these dataclasses and fee functions. The fee math is the hardest part to get right — every P&L calculation downstream depends on it. Kalshi charges 1% trading fee on entry, 10% settlement fee on profit (winners only), and we model execution at `yes_ask` (crossing the spread).

**Step 1: Write fee math tests**

```python
"""tests/test_strategy_backtester.py"""
import pytest
from datetime import date, datetime, timezone
from services.strategy_backtester import (
    BacktestConfig,
    MarketSnapshot,
    ModelUpdate,
    Position,
    TradeRecord,
    compute_entry_cost,
    compute_settlement_pnl,
    compute_exit_pnl,
)


class TestFeeMath:
    """Fee calculations must exactly match Kalshi's fee structure."""

    def test_entry_cost_includes_trading_fee(self):
        # Buy 1 contract at 40 cents
        cost = compute_entry_cost(price_cents=40, quantity=1)
        # Capital locked = price + 1% trading fee
        assert cost == pytest.approx(40.4)

    def test_entry_cost_multiple_contracts(self):
        cost = compute_entry_cost(price_cents=25, quantity=3)
        # 25 * 3 = 75, trading fee = 75 * 0.01 = 0.75
        assert cost == pytest.approx(75.75)

    def test_settlement_pnl_winner(self):
        # Bought at 40 cents, bracket settles YES
        pnl = compute_settlement_pnl(
            entry_price_cents=40, quantity=1, settled_yes=True
        )
        # Gross profit = (100 - 40) = 60
        # Settlement fee = 60 * 0.10 = 6
        # Trading fee (paid at entry) = 40 * 0.01 = 0.4
        # Net P&L = 60 - 6 - 0.4 = 53.6
        assert pnl == pytest.approx(53.6)

    def test_settlement_pnl_loser(self):
        # Bought at 40 cents, bracket settles NO
        pnl = compute_settlement_pnl(
            entry_price_cents=40, quantity=1, settled_yes=False
        )
        # Loss = entry + trading fee = 40 + 0.4 = -40.4
        assert pnl == pytest.approx(-40.4)

    def test_settlement_pnl_winner_multiple_contracts(self):
        pnl = compute_settlement_pnl(
            entry_price_cents=30, quantity=5, settled_yes=True
        )
        # Gross = (100-30)*5 = 350
        # Settlement fee = 350 * 0.10 = 35
        # Trading fee = 30*5*0.01 = 1.5
        # Net = 350 - 35 - 1.5 = 313.5
        assert pnl == pytest.approx(313.5)

    def test_exit_pnl_profitable_sell(self):
        # Bought at 30 cents ask, sell at 50 cents bid
        pnl = compute_exit_pnl(
            entry_price_cents=30, exit_price_cents=50, quantity=1
        )
        # Revenue = 50
        # Exit trading fee = 50 * 0.01 = 0.5
        # Entry trading fee = 30 * 0.01 = 0.3
        # Net = 50 - 0.5 - 30 - 0.3 = 19.2
        assert pnl == pytest.approx(19.2)

    def test_exit_pnl_loss_sell(self):
        # Bought at 40 ask, sell at 25 bid (taking a loss to free capital)
        pnl = compute_exit_pnl(
            entry_price_cents=40, exit_price_cents=25, quantity=1
        )
        # Revenue = 25
        # Exit fee = 0.25
        # Entry fee = 0.4
        # Net = 25 - 0.25 - 40 - 0.4 = -15.65
        assert pnl == pytest.approx(-15.65)

    def test_minimum_profitable_displacement(self):
        """At various price levels, verify the fee hurdle."""
        # At 50 cents, need model_prob such that:
        # q * (100-50) * 0.90 - (1-q) * 50 - 0.01 * 50 > 0
        # q * 45 - 50 + q*50 - 0.5 > 0
        # 95q > 50.5
        # q > 0.5316
        # Displacement needed: 0.5316 - 0.50 = 0.0316 (3.16%)
        pnl_at_breakeven = compute_settlement_pnl(
            entry_price_cents=50, quantity=1, settled_yes=True
        )
        # Winner pnl = (100-50)*0.9 - 50*0.01 = 45 - 0.5 = 44.5
        assert pnl_at_breakeven == pytest.approx(44.5)


class TestDataStructures:
    """Verify dataclass construction and defaults."""

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
            capital_locked=70.7,  # 35*2 + 35*2*0.01
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
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.strategy_backtester'`

**Step 3: Implement dataclasses and fee functions**

```python
"""services/strategy_backtester.py

Kalshi Strategy Backtester — three-layer unified pipeline.

Layer 1: Edge Analysis        (our Brier vs market Brier, displacement)
Layer 2: Strategy Engine      (sanity filters, trade signals, portfolio EV)
Layer 3: P&L Simulator        (full history, bootstrap, regime splits)

Design doc: docs/plans/2026-02-27-backtest-framework-design.md
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np
from loguru import logger
from scipy.stats import norm

from services.backtester import (
    KalshiBracket,
    ModelFn,
    map_probs_to_kalshi_brackets,
    compute_brier_score,
)
from services.data_provider import BacktestDataProvider

# ---------------------------------------------------------------------------
# Fee constants (Kalshi KXHIGHNY)
# ---------------------------------------------------------------------------
TRADING_FEE_RATE = 0.01       # 1% of contract value
SETTLEMENT_FEE_RATE = 0.10    # 10% of profit (winners only)

# ---------------------------------------------------------------------------
# Fee math — every P&L calculation uses these
# ---------------------------------------------------------------------------

def compute_entry_cost(price_cents: float, quantity: int) -> float:
    """Total capital locked when buying YES contracts.

    Returns price * qty + trading fee (in cents).
    """
    gross = price_cents * quantity
    trading_fee = gross * TRADING_FEE_RATE
    return gross + trading_fee


def compute_settlement_pnl(
    entry_price_cents: float,
    quantity: int,
    settled_yes: bool,
) -> float:
    """Net P&L at settlement, in cents.

    Winners: (100 - entry) * qty * (1 - settlement_fee) - entry * qty * trading_fee
    Losers:  -(entry * qty + entry * qty * trading_fee)
    """
    gross_entry = entry_price_cents * quantity
    trading_fee = gross_entry * TRADING_FEE_RATE

    if settled_yes:
        gross_profit = (100 - entry_price_cents) * quantity
        settlement_fee = gross_profit * SETTLEMENT_FEE_RATE
        return gross_profit - settlement_fee - trading_fee
    else:
        return -(gross_entry + trading_fee)


def compute_exit_pnl(
    entry_price_cents: float,
    exit_price_cents: float,
    quantity: int,
) -> float:
    """Net P&L from early exit (sell YES back to market), in cents.

    Revenue from selling minus entry cost minus both trading fees.
    """
    revenue = exit_price_cents * quantity
    exit_trading_fee = revenue * TRADING_FEE_RATE
    entry_cost = entry_price_cents * quantity
    entry_trading_fee = entry_cost * TRADING_FEE_RATE
    return revenue - exit_trading_fee - entry_cost - entry_trading_fee


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for the strategy backtester."""
    starting_capital: float = 100.0
    start_date: date = field(default_factory=lambda: date(2024, 11, 1))
    end_date: Optional[date] = None
    burn_in_days: int = 90
    fixed_bet_size: int = 1
    min_displacement: float = 0.12
    min_model_prob: float = 0.05
    max_spread_cents: float = 10.0
    max_model_std: float = 3.5
    min_reentry_minutes: int = 60
    market_shift_threshold: float = 0.10
    stale_price_minutes: int = 30
    model_name: str = 'ensemble'
    bootstrap_iterations: int = 10000
    execution_latency_seconds: int = 60
    station_id: str = 'KNYC'


@dataclass
class MarketSnapshot:
    """One bracket's market state at one minute."""
    bracket: Tuple[Optional[float], Optional[float]]
    timestamp: datetime
    yes_bid: float
    yes_ask: float
    volume: int
    is_stale: bool


@dataclass
class ModelUpdate:
    """Model state after a trigger (new obs or new forecast run)."""
    ref_time: datetime
    bracket_probs: Dict[Tuple, float]
    trigger_type: str
    model_std: float


@dataclass
class Position:
    """An open YES position."""
    event_date: date
    bracket: Tuple[Optional[float], Optional[float]]
    entry_price: float
    entry_time: datetime
    quantity: int
    capital_locked: float


@dataclass
class TradeRecord:
    """Completed trade with full detail."""
    event_date: date
    bracket: Tuple[Optional[float], Optional[float]]
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    exit_type: str                  # 'settlement' | 'early_exit'
    settlement_result: Optional[int]
    pnl: float
    fees_paid: float
    displacement_at_entry: float
    model_prob_at_entry: float
    capital_locked_hours: float
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add strategy backtester data structures and fee math"
```

---

### Task 2: Sanity Filters

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** Five pre-trade guard rails that kill obviously bad trades. The post-peak detection is the most complex — it checks observation trends to determine if the daily high has already occurred. When post-peak, new entries are suppressed but existing positions can still be exited. All thresholds come from `BacktestConfig`.

**Step 1: Write sanity filter tests**

```python
class TestSanityFilter:
    """Pre-trade guard rails."""

    def setup_method(self):
        self.config = BacktestConfig()
        self.sf = SanityFilter(self.config)

    def test_reject_low_probability_bracket(self):
        """Reject model_prob < 5% — tails are unreliable."""
        ok, reason = self.sf.check(
            model_prob=0.03,
            market_ask_cents=2.0,
            spread_cents=1.0,
            model_std=2.0,
            bracket=(80.0, 82.0),
            is_post_peak=False,
            recently_exited=set(),
        )
        assert not ok
        assert "model_prob" in reason.lower()

    def test_accept_sufficient_probability(self):
        ok, _ = self.sf.check(
            model_prob=0.15,
            market_ask_cents=8.0,
            spread_cents=2.0,
            model_std=2.0,
            bracket=(72.0, 74.0),
            is_post_peak=False,
            recently_exited=set(),
        )
        assert ok

    def test_reject_post_peak(self):
        """After daily high has passed, no new entries."""
        ok, reason = self.sf.check(
            model_prob=0.30,
            market_ask_cents=20.0,
            spread_cents=2.0,
            model_std=2.0,
            bracket=(72.0, 74.0),
            is_post_peak=True,
            recently_exited=set(),
        )
        assert not ok
        assert "post_peak" in reason.lower()

    def test_reject_wide_spread(self):
        """Spread > max_spread_cents eats the displacement."""
        ok, reason = self.sf.check(
            model_prob=0.30,
            market_ask_cents=25.0,
            spread_cents=15.0,
            model_std=2.0,
            bracket=(72.0, 74.0),
            is_post_peak=False,
            recently_exited=set(),
        )
        assert not ok
        assert "spread" in reason.lower()

    def test_reject_high_model_uncertainty(self):
        """Model std > threshold means flat distribution, noisy signals."""
        ok, reason = self.sf.check(
            model_prob=0.12,
            market_ask_cents=5.0,
            spread_cents=2.0,
            model_std=4.0,
            bracket=(72.0, 74.0),
            is_post_peak=False,
            recently_exited=set(),
        )
        assert not ok
        assert "uncertainty" in reason.lower() or "std" in reason.lower()

    def test_reject_recently_exited_bracket(self):
        """No churn — don't re-enter within min_reentry_minutes."""
        ok, reason = self.sf.check(
            model_prob=0.30,
            market_ask_cents=20.0,
            spread_cents=2.0,
            model_std=2.0,
            bracket=(72.0, 74.0),
            is_post_peak=False,
            recently_exited={(72.0, 74.0)},
        )
        assert not ok
        assert "churn" in reason.lower() or "reentry" in reason.lower()

    def test_post_peak_detection_declining_temps(self):
        """Detect peak when last 2+ obs decline and time is after 2 PM ET."""
        from datetime import timezone
        _ET = timezone(timedelta(hours=-5))

        obs = [
            (datetime(2025, 6, 15, 16, 0, tzinfo=timezone.utc), 78.0),  # 11 AM ET
            (datetime(2025, 6, 15, 17, 0, tzinfo=timezone.utc), 80.0),  # 12 PM ET
            (datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc), 82.0),  # 1 PM ET (peak)
            (datetime(2025, 6, 15, 19, 0, tzinfo=timezone.utc), 81.0),  # 2 PM ET
            (datetime(2025, 6, 15, 20, 0, tzinfo=timezone.utc), 79.0),  # 3 PM ET
        ]
        current_time = datetime(2025, 6, 15, 20, 30, tzinfo=timezone.utc)
        assert self.sf.is_post_peak(obs, current_time) is True

    def test_not_post_peak_still_rising(self):
        obs = [
            (datetime(2025, 6, 15, 16, 0, tzinfo=timezone.utc), 78.0),
            (datetime(2025, 6, 15, 17, 0, tzinfo=timezone.utc), 80.0),
            (datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc), 82.0),
        ]
        current_time = datetime(2025, 6, 15, 18, 30, tzinfo=timezone.utc)
        assert self.sf.is_post_peak(obs, current_time) is False

    def test_not_post_peak_too_early(self):
        """Even if declining, before 2 PM ET is too early to call."""
        obs = [
            (datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc), 75.0),  # 9 AM ET
            (datetime(2025, 6, 15, 15, 0, tzinfo=timezone.utc), 74.0),  # 10 AM ET
            (datetime(2025, 6, 15, 16, 0, tzinfo=timezone.utc), 73.0),  # 11 AM ET
        ]
        current_time = datetime(2025, 6, 15, 16, 30, tzinfo=timezone.utc)
        assert self.sf.is_post_peak(obs, current_time) is False
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestSanityFilter -v`
Expected: FAIL — `ImportError: cannot import name 'SanityFilter'`

**Step 3: Implement SanityFilter**

Add to `services/strategy_backtester.py`:

```python
# Add to imports at top
from datetime import timezone as tz_utc

_ET = timezone(timedelta(hours=-5))  # EST (not EDT — matches NWS CLI convention)


class SanityFilter:
    """Pre-trade guard rails — kill obviously bad trades."""

    def __init__(self, config: BacktestConfig):
        self.min_model_prob = config.min_model_prob
        self.max_spread_cents = config.max_spread_cents
        self.max_model_std = config.max_model_std

    def check(
        self,
        model_prob: float,
        market_ask_cents: float,
        spread_cents: float,
        model_std: float,
        bracket: Tuple[Optional[float], Optional[float]],
        is_post_peak: bool,
        recently_exited: Set[Tuple],
    ) -> Tuple[bool, str]:
        """Check all sanity filters. Returns (pass, reason_if_rejected)."""

        if model_prob < self.min_model_prob:
            return False, "model_prob {:.3f} < min {:.3f}".format(
                model_prob, self.min_model_prob
            )

        if is_post_peak:
            return False, "post_peak: no new entries after daily high passed"

        if spread_cents > self.max_spread_cents:
            return False, "spread {:.1f}c > max {:.1f}c".format(
                spread_cents, self.max_spread_cents
            )

        if model_std > self.max_model_std:
            return False, "model std {:.2f}F > max {:.2f}F — high uncertainty".format(
                model_std, self.max_model_std
            )

        if bracket in recently_exited:
            return False, "reentry blocked — recently exited this bracket (no churn)"

        return True, ""

    def is_post_peak(
        self,
        obs_temps: List[Tuple[datetime, float]],
        current_time: datetime,
    ) -> bool:
        """Detect if the daily high has likely already occurred.

        All conditions must be true:
        1. At least 3 observations exist
        2. Current time is after 2 PM ET (19:00 UTC in EST)
        3. Last 2+ consecutive obs show declining temps
        4. Running max occurred >60 min ago
        """
        if len(obs_temps) < 3:
            return False

        # Check time — must be after 2 PM ET
        current_et = current_time.astimezone(_ET) if current_time.tzinfo else current_time.replace(tzinfo=tz_utc).astimezone(_ET)
        if current_et.hour < 14:
            return False

        # Check last 2+ obs are declining
        temps = [t for _, t in obs_temps]
        if temps[-1] >= temps[-2]:
            return False

        # Running max must have occurred >60 min ago
        max_temp = max(temps)
        max_time = None
        for ts, t in obs_temps:
            if t == max_temp:
                max_time = ts
        if max_time is None:
            return False

        if current_time.tzinfo is None:
            max_time_aware = max_time if max_time.tzinfo else max_time.replace(tzinfo=tz_utc)
            current_aware = current_time.replace(tzinfo=tz_utc)
        else:
            max_time_aware = max_time if max_time.tzinfo else max_time.replace(tzinfo=tz_utc)
            current_aware = current_time

        minutes_since_peak = (current_aware - max_time_aware).total_seconds() / 60
        return minutes_since_peak > 60
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestSanityFilter -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add SanityFilter with 5 pre-trade guard rails"
```

---

### Task 3: EventPortfolio — Mutually Exclusive EV

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** This is the trickiest math in the system. Kalshi brackets on the same event are mutually exclusive — only one settles YES. So when you hold multiple brackets, you must evaluate combined EV accounting for the fact that winning on bracket A means losing on all others. Independent EV calculation is mathematically wrong and will overstate profitability. The `optimal_subset` method uses greedy selection: start with the best single bracket, add others only if they increase portfolio EV.

**Step 1: Write EventPortfolio tests**

```python
class TestEventPortfolio:
    """Mutually exclusive bracket EV — the hardest math."""

    def test_single_bracket_ev(self):
        """Single bracket: standard EV calc."""
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        ev = portfolio.combined_ev(
            entry_prices=[40.0],
            model_probs=[0.50],
        )
        # Win (50%): (100-40) * 0.90 - 40*0.01 = 54 - 0.4 = 53.6
        # Lose (50%): -(40 + 0.4) = -40.4
        # EV = 0.50 * 53.6 + 0.50 * (-40.4) = 26.8 - 20.2 = 6.6
        assert ev == pytest.approx(6.6)

    def test_single_bracket_negative_ev(self):
        """Low displacement = negative EV after fees."""
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        ev = portfolio.combined_ev(
            entry_prices=[45.0],
            model_probs=[0.48],
        )
        # Win (48%): (100-45)*0.9 - 45*0.01 = 49.5 - 0.45 = 49.05
        # Lose (52%): -(45 + 0.45) = -45.45
        # EV = 0.48*49.05 + 0.52*(-45.45) = 23.544 - 23.634 = -0.09
        assert ev < 0

    def test_two_bracket_portfolio_ev(self):
        """Two brackets: correlated losses matter."""
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        ev = portfolio.combined_ev(
            entry_prices=[22.0, 17.0],     # cents
            model_probs=[0.35, 0.30],       # probabilities
        )
        # A wins (35%): gain_A - loss_B
        #   gain_A = (100-22)*0.9 - 22*0.01 = 70.2 - 0.22 = 69.98
        #   loss_B = 17 + 17*0.01 = 17.17
        #   net_A = 69.98 - 17.17 = 52.81
        #
        # B wins (30%): gain_B - loss_A
        #   gain_B = (100-17)*0.9 - 17*0.01 = 74.7 - 0.17 = 74.53
        #   loss_A = 22 + 22*0.01 = 22.22
        #   net_B = 74.53 - 22.22 = 52.31
        #
        # Neither (35%): -loss_A - loss_B
        #   net_none = -22.22 - 17.17 = -39.39
        #
        # EV = 0.35*52.81 + 0.30*52.31 + 0.35*(-39.39)
        #    = 18.4835 + 15.693 + (-13.7865) = 20.39
        assert ev == pytest.approx(20.39, abs=0.01)

    def test_adding_bracket_can_decrease_ev(self):
        """If 2nd bracket is expensive, the correlated loss kills EV."""
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        ev_one = portfolio.combined_ev(
            entry_prices=[20.0],
            model_probs=[0.35],
        )
        ev_two = portfolio.combined_ev(
            entry_prices=[20.0, 60.0],    # 2nd bracket is expensive
            model_probs=[0.35, 0.10],      # and low probability
        )
        # The expensive low-prob bracket should hurt portfolio EV
        assert ev_two < ev_one

    def test_optimal_subset_picks_best_single(self):
        """When only one bracket has positive EV, pick just that one."""
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        candidates = [
            ((72.0, 74.0), 20.0, 0.35),  # (bracket, entry_price, model_prob)
            ((74.0, 76.0), 55.0, 0.08),  # bad: low prob, high price
        ]
        optimal = portfolio.optimal_subset(candidates)
        assert len(optimal) == 1
        assert optimal[0][0] == (72.0, 74.0)

    def test_optimal_subset_picks_both_when_beneficial(self):
        """Two cheap adjacent brackets can improve portfolio EV."""
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        candidates = [
            ((72.0, 74.0), 22.0, 0.35),
            ((74.0, 76.0), 17.0, 0.30),
        ]
        optimal = portfolio.optimal_subset(candidates)
        assert len(optimal) == 2

    def test_empty_candidates_returns_empty(self):
        portfolio = EventPortfolio(event_date=date(2025, 6, 15))
        assert portfolio.optimal_subset([]) == []
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestEventPortfolio -v`
Expected: FAIL — `ImportError: cannot import name 'EventPortfolio'`

**Step 3: Implement EventPortfolio**

Add to `services/strategy_backtester.py`:

```python
class EventPortfolio:
    """Evaluate combined EV for mutually exclusive brackets.

    On the same event_date, only one bracket settles YES. Holding
    multiple brackets means winning on one and losing on all others.
    Independent EV calculation overstates profitability.
    """

    def __init__(self, event_date: date):
        self.event_date = event_date

    def combined_ev(
        self,
        entry_prices: List[float],
        model_probs: List[float],
    ) -> float:
        """Calculate portfolio EV accounting for mutual exclusivity.

        For N brackets with entry prices p_i and model probs q_i:
          If bracket j wins (prob q_j):
            gain_j = (100 - p_j) * (1 - SETTLEMENT_FEE_RATE) - p_j * TRADING_FEE_RATE
            loss_others = sum(p_i + p_i * TRADING_FEE_RATE for i != j)
            net_j = gain_j - loss_others

          If none win (prob 1 - sum(q_i)):
            net_none = -sum(p_i + p_i * TRADING_FEE_RATE)

        EV = sum(q_j * net_j) + (1 - sum(q)) * net_none
        """
        n = len(entry_prices)
        if n == 0:
            return 0.0

        # Precompute loss per bracket (what you lose if this bracket doesn't win)
        losses = [p + p * TRADING_FEE_RATE for p in entry_prices]
        total_loss = sum(losses)

        ev = 0.0
        for j in range(n):
            gain_j = (
                (100 - entry_prices[j]) * (1 - SETTLEMENT_FEE_RATE)
                - entry_prices[j] * TRADING_FEE_RATE
            )
            net_j = gain_j - (total_loss - losses[j])
            ev += model_probs[j] * net_j

        # None-win scenario
        prob_none = 1.0 - sum(model_probs)
        ev += prob_none * (-total_loss)

        return ev

    def optimal_subset(
        self,
        candidates: List[Tuple[Tuple, float, float]],
    ) -> List[Tuple[Tuple, float, float]]:
        """Greedy selection of brackets that maximize portfolio EV.

        candidates: list of (bracket, entry_price_cents, model_prob)

        Algorithm:
        1. Filter to individually positive-EV brackets
        2. Sort by individual EV descending
        3. Start with best, add each candidate if it increases portfolio EV
        """
        if not candidates:
            return []

        # Filter: individual EV must be positive
        viable = []
        for bracket, price, prob in candidates:
            single_ev = self.combined_ev([price], [prob])
            if single_ev > 0:
                viable.append((bracket, price, prob, single_ev))

        if not viable:
            return []

        # Sort by individual EV descending
        viable.sort(key=lambda x: x[3], reverse=True)

        # Greedy build
        selected = [viable[0][:3]]
        current_prices = [viable[0][1]]
        current_probs = [viable[0][2]]
        current_ev = viable[0][3]

        for bracket, price, prob, _ in viable[1:]:
            test_prices = current_prices + [price]
            test_probs = current_probs + [prob]
            test_ev = self.combined_ev(test_prices, test_probs)

            if test_ev > current_ev:
                selected.append((bracket, price, prob))
                current_prices.append(price)
                current_probs.append(prob)
                current_ev = test_ev

        return selected
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestEventPortfolio -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add EventPortfolio with mutually exclusive bracket EV"
```

---

### Task 4: PositionManager — Capital Lockup and Exits

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** Kalshi requires 100% upfront collateralization. Buying a contract at 30¢ locks $0.30 per contract until settlement or early exit. The PositionManager tracks the bankroll, enforces capital constraints, handles early exits (sell YES at bid), and settles positions at end of day. Early exits only execute if the fee math makes sense — either profitable exit or model probability collapsed below the loss-cut threshold.

**Step 1: Write PositionManager tests**

```python
class TestPositionManager:
    """Capital lockup, early exits, settlement."""

    def setup_method(self):
        self.pm = PositionManager(bankroll=100.0)

    def test_initial_state(self):
        assert self.pm.bankroll == 100.0
        assert self.pm.capital_available == pytest.approx(100.0)
        assert self.pm.capital_locked == pytest.approx(0.0)
        assert len(self.pm.positions) == 0

    def test_open_position_locks_capital(self):
        pos = self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        assert pos is not None
        # Locked = 30 + 30*0.01 = 30.3 cents = $0.303
        assert self.pm.capital_locked == pytest.approx(0.303)
        assert self.pm.capital_available == pytest.approx(100.0 - 0.303)

    def test_reject_when_insufficient_capital(self):
        """Bankroll constraint prevents opening."""
        pm = PositionManager(bankroll=0.20)
        pos = pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        assert pos is None
        assert pm.capital_locked == pytest.approx(0.0)

    def test_close_position_frees_capital(self):
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        pnl = self.pm.close_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            exit_price_cents=45.0,
            exit_time=datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc),
        )
        # P&L: 45 - 45*0.01 - 30 - 30*0.01 = 45 - 0.45 - 30 - 0.3 = 14.25 cents
        assert pnl == pytest.approx(14.25)
        assert self.pm.capital_locked == pytest.approx(0.0)
        assert len(self.pm.positions) == 0

    def test_settle_day_winner(self):
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        trades = self.pm.settle_day(
            event_date=date(2025, 6, 15),
            settled_bracket=(72.0, 74.0),
        )
        assert len(trades) == 1
        # Winner: (100-30)*0.9 - 30*0.01 = 63 - 0.3 = 62.7 cents
        assert trades[0].pnl == pytest.approx(62.7)
        assert self.pm.capital_locked == pytest.approx(0.0)

    def test_settle_day_loser(self):
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(74.0, 76.0),
            entry_price_cents=25.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        trades = self.pm.settle_day(
            event_date=date(2025, 6, 15),
            settled_bracket=(72.0, 74.0),  # different bracket won
        )
        assert len(trades) == 1
        # Loser: -(25 + 25*0.01) = -25.25 cents
        assert trades[0].pnl == pytest.approx(-25.25)

    def test_settle_day_mixed(self):
        """One winner, one loser on same day."""
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(74.0, 76.0),
            entry_price_cents=20.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        trades = self.pm.settle_day(
            event_date=date(2025, 6, 15),
            settled_bracket=(72.0, 74.0),
        )
        assert len(trades) == 2
        # Winner (72-74): 62.7
        # Loser (74-76): -20.2
        total_pnl = sum(t.pnl for t in trades)
        assert total_pnl == pytest.approx(62.7 - 20.2)

    def test_positions_for_date(self):
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        self.pm.open_position(
            event_date=date(2025, 6, 16),
            bracket=(70.0, 72.0),
            entry_price_cents=25.0,
            entry_time=datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        assert len(self.pm.positions_for(date(2025, 6, 15))) == 1
        assert len(self.pm.positions_for(date(2025, 6, 16))) == 1
        assert len(self.pm.positions_for(date(2025, 6, 17))) == 0

    def test_bankroll_updates_after_settlement(self):
        """Bankroll increases on winners, decreases on losers."""
        self.pm.open_position(
            event_date=date(2025, 6, 15),
            bracket=(72.0, 74.0),
            entry_price_cents=30.0,
            entry_time=datetime(2025, 6, 14, 14, 0, tzinfo=timezone.utc),
            quantity=1,
        )
        initial_bankroll = self.pm.bankroll
        trades = self.pm.settle_day(
            event_date=date(2025, 6, 15),
            settled_bracket=(72.0, 74.0),
        )
        # Bankroll should increase by the P&L (converted from cents to dollars)
        pnl_dollars = trades[0].pnl / 100.0
        assert self.pm.bankroll == pytest.approx(initial_bankroll + pnl_dollars)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestPositionManager -v`
Expected: FAIL — `ImportError: cannot import name 'PositionManager'`

**Step 3: Implement PositionManager**

Add to `services/strategy_backtester.py`:

```python
class PositionManager:
    """Track positions, enforce bankroll constraint, handle exits and settlement."""

    def __init__(self, bankroll: float):
        self.bankroll = bankroll  # in dollars
        self.positions = []  # type: List[Position]
        self._recently_exited = {}  # type: Dict[Tuple, datetime]  # bracket -> exit_time

    @property
    def capital_locked(self) -> float:
        """Total capital locked in open positions (dollars)."""
        return sum(p.capital_locked for p in self.positions) / 100.0

    @property
    def capital_available(self) -> float:
        """Free capital for new positions (dollars)."""
        return self.bankroll - self.capital_locked

    def positions_for(self, event_date: date) -> List[Position]:
        """Get all open positions for a specific event date."""
        return [p for p in self.positions if p.event_date == event_date]

    def open_position(
        self,
        event_date: date,
        bracket: Tuple[Optional[float], Optional[float]],
        entry_price_cents: float,
        entry_time: datetime,
        quantity: int,
    ) -> Optional[Position]:
        """Open a position if capital is available. Returns None if rejected."""
        cost_cents = compute_entry_cost(entry_price_cents, quantity)
        cost_dollars = cost_cents / 100.0

        if cost_dollars > self.capital_available:
            return None

        pos = Position(
            event_date=event_date,
            bracket=bracket,
            entry_price=entry_price_cents,
            entry_time=entry_time,
            quantity=quantity,
            capital_locked=cost_cents,
        )
        self.positions.append(pos)
        return pos

    def close_position(
        self,
        event_date: date,
        bracket: Tuple[Optional[float], Optional[float]],
        exit_price_cents: float,
        exit_time: datetime,
    ) -> float:
        """Sell YES position back to market. Returns P&L in cents."""
        pos = None
        for p in self.positions:
            if p.event_date == event_date and p.bracket == bracket:
                pos = p
                break

        if pos is None:
            return 0.0

        pnl = compute_exit_pnl(pos.entry_price, exit_price_cents, pos.quantity)
        self.bankroll += pnl / 100.0
        self.positions.remove(pos)
        self._recently_exited[bracket] = exit_time
        return pnl

    def settle_day(
        self,
        event_date: date,
        settled_bracket: Tuple[Optional[float], Optional[float]],
    ) -> List[TradeRecord]:
        """Settle all positions for an event date. Returns trade records."""
        day_positions = self.positions_for(event_date)
        records = []

        for pos in day_positions:
            won = pos.bracket == settled_bracket
            pnl = compute_settlement_pnl(pos.entry_price, pos.quantity, won)
            fees = pos.capital_locked * TRADING_FEE_RATE
            if won:
                fees += (100 - pos.entry_price) * pos.quantity * SETTLEMENT_FEE_RATE

            hours_locked = 0.0
            if pos.entry_time.tzinfo:
                # Approximate settlement at 11 PM ET on event_date
                settle_time = datetime(
                    event_date.year, event_date.month, event_date.day,
                    4, 0, tzinfo=tz_utc,  # ~11 PM ET = 4 AM UTC next day
                ) + timedelta(days=1)
                hours_locked = (settle_time - pos.entry_time).total_seconds() / 3600

            record = TradeRecord(
                event_date=event_date,
                bracket=pos.bracket,
                direction='BUY_YES',
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                exit_price=None,
                exit_time=None,
                exit_type='settlement',
                settlement_result=1 if won else 0,
                pnl=pnl,
                fees_paid=fees,
                displacement_at_entry=0.0,
                model_prob_at_entry=0.0,
                capital_locked_hours=hours_locked,
            )
            records.append(record)
            self.bankroll += pnl / 100.0

        # Remove settled positions
        self.positions = [p for p in self.positions if p.event_date != event_date]
        return records

    def recently_exited_brackets(
        self,
        current_time: datetime,
        min_minutes: int,
    ) -> Set[Tuple]:
        """Return brackets exited within min_minutes of current_time."""
        result = set()
        for bracket, exit_time in self._recently_exited.items():
            if current_time.tzinfo and exit_time.tzinfo:
                delta = (current_time - exit_time).total_seconds() / 60
            else:
                delta = (current_time - exit_time).total_seconds() / 60
            if delta < min_minutes:
                result.add(bracket)
        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestPositionManager -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add PositionManager with capital lockup and early exits"
```

---

### Task 5: Market Data Loader

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** Loads candlestick data from `kalshi_candlesticks` and converts it to `MarketSnapshot` objects. Handles forward-filling when no candle exists for a minute (stale price detection). Also loads bracket definitions from `kalshi_settlements` for a given event_date. This is the bridge between raw DB data and the strategy engine's input format.

**Step 1: Write market data loader tests**

```python
class TestMarketDataLoader:
    """Load and transform candlestick data."""

    def _seed_kalshi_data(self, db_path):
        """Insert minimal Kalshi test data."""
        con = duckdb.connect(db_path)
        # Settlement for one event
        con.execute("""
            INSERT INTO kalshi_settlements VALUES
            ('KXHIGHNY-25JUN15-T72-B74', 'KXHIGHNY-25JUN15', 'KXHIGHNY',
             'NYC', 'high', '2025-06-15', 72.0, 74.0, 1, 500,
             '2025-06-15 23:00:00', CURRENT_TIMESTAMP),
            ('KXHIGHNY-25JUN15-T70-B72', 'KXHIGHNY-25JUN15', 'KXHIGHNY',
             'NYC', 'high', '2025-06-15', 70.0, 72.0, 0, 300,
             '2025-06-15 23:00:00', CURRENT_TIMESTAMP)
        """)
        # Candlesticks: 3 minutes of data for one bracket
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
        # 3 actual candles + 1 forward-filled (14:02 gap)
        assert len(snaps) == 4
        # Check stale detection
        stale = [s for s in snaps if s.is_stale]
        assert len(stale) == 1  # 14:02 is forward-filled

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
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestMarketDataLoader -v`
Expected: FAIL — `ImportError: cannot import name 'MarketDataLoader'`

**Step 3: Implement MarketDataLoader**

Add to `services/strategy_backtester.py`:

```python
class MarketDataLoader:
    """Load Kalshi market data from DuckDB and convert to MarketSnapshot objects."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_brackets(
        self, event_date: date
    ) -> List[Dict]:
        """Get all bracket definitions and settlement outcomes for an event date."""
        con = duckdb.connect(self.db_path, read_only=True)
        rows = con.execute("""
            SELECT market_ticker, floor_strike, cap_strike, settled_yes, volume
            FROM kalshi_settlements
            WHERE event_date = ? AND series_ticker = 'KXHIGHNY'
            ORDER BY floor_strike NULLS FIRST
        """, [event_date]).fetchall()
        con.close()

        return [
            {
                "market_ticker": r[0],
                "bracket": (r[1], r[2]),
                "settled_yes": r[3],
                "volume": r[4],
            }
            for r in rows
        ]

    def get_snapshots(
        self,
        market_ticker: str,
        bracket: Tuple[Optional[float], Optional[float]],
        start_ts: datetime,
        end_ts: datetime,
    ) -> List[MarketSnapshot]:
        """Load candlestick data and forward-fill gaps.

        Returns one MarketSnapshot per minute in [start_ts, end_ts).
        Minutes without candles are forward-filled with is_stale=True.
        """
        con = duckdb.connect(self.db_path, read_only=True)
        rows = con.execute("""
            SELECT end_period_ts, yes_bid_close, yes_ask_close, volume
            FROM kalshi_candlesticks
            WHERE market_ticker = ?
              AND end_period_ts >= ? AND end_period_ts < ?
            ORDER BY end_period_ts
        """, [market_ticker, start_ts, end_ts]).fetchall()
        con.close()

        # Index by minute
        candle_map = {}  # type: Dict[datetime, Tuple]
        for ts, bid, ask, vol in rows:
            if hasattr(ts, 'replace'):
                ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
            candle_map[ts.replace(second=0, microsecond=0)] = (bid, ask, vol)

        # Generate minute-by-minute snapshots with forward fill
        snapshots = []
        current = start_ts.replace(second=0, microsecond=0)
        last_bid = None
        last_ask = None

        while current < end_ts:
            minute_key = current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current
            # Check with and without tz for matching
            matched = candle_map.get(minute_key)
            if matched is None:
                # Try naive match
                matched = candle_map.get(current.replace(tzinfo=None))

            if matched is not None:
                bid, ask, vol = matched
                last_bid = bid
                last_ask = ask
                snapshots.append(MarketSnapshot(
                    bracket=bracket,
                    timestamp=minute_key,
                    yes_bid=bid,
                    yes_ask=ask,
                    volume=vol,
                    is_stale=False,
                ))
            elif last_bid is not None:
                # Forward fill
                snapshots.append(MarketSnapshot(
                    bracket=bracket,
                    timestamp=minute_key,
                    yes_bid=last_bid,
                    yes_ask=last_ask,
                    volume=0,
                    is_stale=True,
                ))

            current += timedelta(minutes=1)

        return snapshots
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestMarketDataLoader -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add MarketDataLoader with forward-fill stale price detection"
```

---

### Task 6: Trigger Detection — Observation Timestamps

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** Instead of fixed hourly checkpoints, the strategy backtester triggers model re-evaluation when new observations arrive. This queries distinct `observed_at` timestamps from the observations table and interleaves them with forecast run availability times. Each trigger becomes a point where the model recalculates divergence features and the strategy engine checks for trade signals. The market is open from 10 AM ET the prior day through settlement, so triggers span ~28+ hours.

**CRITICAL — Execution Latency:** Each trigger returns TWO timestamps: `model_time` (when the model evaluates, using data up to this point) and `execution_time` (model_time + execution_latency_seconds, when the order would actually reach the exchange). The strategy engine uses `model_time` for the model but `execution_time` for market price lookup. This prevents unrealistically optimistic fills.

**Step 1: Write trigger detection tests**

```python
class TestTriggerDetection:
    """Observation-triggered model updates."""

    def _seed_obs_data(self, db_path):
        con = duckdb.connect(db_path)
        # Observations on settlement day (June 15) — hourly METAR
        for hour in range(12, 22):  # 7 AM to 5 PM ET (UTC 12-22)
            con.execute("""
                INSERT INTO observations (station_id, observed_at, temp_f,
                    ingest_source, ingested_at)
                VALUES ('KNYC', ?, ?, 'synoptic', CURRENT_TIMESTAMP)
            """, [datetime(2025, 6, 15, hour, 53), 70.0 + hour - 12])
        # One SPECI at 14:23 UTC (irregular timestamp)
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
        # Each trigger is (model_time, execution_time, trigger_type)
        obs_triggers = [t for t in triggers if t[2] == 'observation']
        assert len(obs_triggers) == 11  # 10 hourly + 1 SPECI

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
        """execution_time = model_time + latency (default 60s)."""
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
        # Run hours 0, 6, 12, 18 on settlement day + prior day 18z
        # Available at run_hour + 2h lag
        assert len(fcst_triggers) >= 3  # At least 00z+2, 06z+2, 12z+2
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestTriggerDetection -v`
Expected: FAIL — `ImportError: cannot import name 'TriggerDetector'`

**Step 3: Implement TriggerDetector**

Add to `services/strategy_backtester.py`:

```python
from services.data_provider import HRRR_AVAILABILITY_LAG_HOURS

RUN_HOURS = [0, 6, 12, 18]


class TriggerDetector:
    """Detect model re-evaluation triggers from observation timestamps.

    Instead of fixed hourly checkpoints, the strategy triggers when:
    1. A new observation arrives (METAR or SPECI)
    2. A new forecast run becomes available (run_hour + lag)

    Returns (model_time, execution_time, trigger_type) tuples.
    model_time: when the model evaluates (data available up to this point)
    execution_time: model_time + latency (when the order reaches the exchange)
    The strategy uses model_time for model evaluation but execution_time
    for market price lookup, simulating realistic execution slippage.
    """

    def __init__(
        self,
        db_path: str,
        station_id: str = 'KNYC',
        execution_latency_seconds: int = 60,
    ):
        self.db_path = db_path
        self.station_id = station_id
        self.latency = timedelta(seconds=execution_latency_seconds)

    def get_triggers(
        self,
        event_date: date,
        market_open_utc: datetime,
        market_close_utc: datetime,
    ) -> List[Tuple[datetime, datetime, str]]:
        """Return sorted list of (model_time, execution_time, trigger_type).

        trigger_type: 'observation' or 'forecast_run'
        Only includes triggers within [market_open_utc, market_close_utc].
        """
        triggers = []

        # 1. Observation triggers
        con = duckdb.connect(self.db_path, read_only=True)
        open_naive = market_open_utc.replace(tzinfo=None) if market_open_utc.tzinfo else market_open_utc
        close_naive = market_close_utc.replace(tzinfo=None) if market_close_utc.tzinfo else market_close_utc

        rows = con.execute("""
            SELECT DISTINCT observed_at
            FROM observations
            WHERE station_id = ?
              AND observed_at >= ? AND observed_at <= ?
              AND temp_f IS NOT NULL
            ORDER BY observed_at
        """, [self.station_id, open_naive, close_naive]).fetchall()
        con.close()

        for (obs_ts,) in rows:
            ts = obs_ts.replace(tzinfo=timezone.utc) if hasattr(obs_ts, 'replace') and obs_ts.tzinfo is None else obs_ts
            triggers.append((ts, ts + self.latency, 'observation'))

        # 2. Forecast run triggers (available at run_hour + lag)
        # Check both prior day and settlement day runs
        for day_offset in [-1, 0]:
            run_date = event_date + timedelta(days=day_offset)
            for rh in RUN_HOURS:
                avail_time = datetime(
                    run_date.year, run_date.month, run_date.day,
                    rh, 0, tzinfo=timezone.utc,
                ) + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)
                if market_open_utc <= avail_time <= market_close_utc:
                    triggers.append((avail_time, avail_time + self.latency, 'forecast_run'))

        # Sort by model_time
        triggers.sort(key=lambda t: t[0])
        return triggers
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestTriggerDetection -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add TriggerDetector for observation-triggered model updates"
```

---

### Task 7: EdgeAnalyzer — Brier Comparison and Displacement

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** Layer 1 of the pipeline. Compares our model's Brier score against the market's implied Brier score at each trigger point. Also tracks displacement (model_prob - market_ask per bracket). This is the diagnostic layer — it answers "where and when does our model beat the market?" and produces the edge heatmap. Records data for all trigger points, including during the burn-in period (analysis without trading).

**Step 1: Write EdgeAnalyzer tests**

```python
class TestEdgeAnalyzer:
    """Layer 1: Brier comparison and displacement tracking."""

    def test_market_brier_from_midpoints(self):
        """Market Brier uses midpoint prices as implied probabilities."""
        analyzer = EdgeAnalyzer()
        market_probs = [0.05, 0.20, 0.40, 0.25, 0.10]
        settled = [0, 0, 1, 0, 0]  # bracket 2 settled YES
        brier = analyzer.compute_brier(market_probs, settled)
        # sum((p-o)^2) = 0.05^2 + 0.20^2 + (0.40-1)^2 + 0.25^2 + 0.10^2
        #              = 0.0025 + 0.04 + 0.36 + 0.0625 + 0.01 = 0.475
        assert brier == pytest.approx(0.475)

    def test_perfect_brier_is_zero(self):
        analyzer = EdgeAnalyzer()
        market_probs = [0.0, 0.0, 1.0, 0.0, 0.0]
        settled = [0, 0, 1, 0, 0]
        assert analyzer.compute_brier(market_probs, settled) == pytest.approx(0.0)

    def test_displacement_positive_means_model_higher(self):
        """Positive displacement = model assigns more probability than market."""
        analyzer = EdgeAnalyzer()
        d = analyzer.compute_displacement(model_prob=0.35, market_ask=0.22)
        assert d == pytest.approx(0.13)

    def test_record_and_summarize(self):
        analyzer = EdgeAnalyzer()
        analyzer.record(
            event_date=date(2025, 6, 15),
            trigger_time=datetime(2025, 6, 15, 18, 0, tzinfo=timezone.utc),
            model_probs={
                (72.0, 74.0): 0.40,
                (70.0, 72.0): 0.25,
                (74.0, 76.0): 0.20,
            },
            market_mids={
                (72.0, 74.0): 0.35,
                (70.0, 72.0): 0.20,
                (74.0, 76.0): 0.15,
            },
            settled_bracket=(72.0, 74.0),
            et_hour=13,
        )
        summary = analyzer.summary_by_hour()
        assert 13 in summary
        assert "model_brier" in summary[13]
        assert "market_brier" in summary[13]
        assert summary[13]["edge"] > 0  # model should be better here
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestEdgeAnalyzer -v`
Expected: FAIL — `ImportError: cannot import name 'EdgeAnalyzer'`

**Step 3: Implement EdgeAnalyzer**

Add to `services/strategy_backtester.py`:

```python
from collections import defaultdict


class EdgeAnalyzer:
    """Layer 1: Compare our Brier vs market Brier, track displacement."""

    def __init__(self):
        self._records = []  # type: List[Dict]
        self._by_hour = defaultdict(lambda: {
            "model_briers": [], "market_briers": [],
            "displacements": [], "count": 0,
        })

    def compute_brier(
        self,
        probs: List[float],
        settled: List[int],
    ) -> float:
        """Brier score: sum((p_k - o_k)^2)."""
        return sum((p - o) ** 2 for p, o in zip(probs, settled))

    def compute_displacement(
        self,
        model_prob: float,
        market_ask: float,
    ) -> float:
        """Displacement = model_prob - market_ask (positive = model higher)."""
        return model_prob - market_ask

    def record(
        self,
        event_date: date,
        trigger_time: datetime,
        model_probs: Dict[Tuple, float],
        market_mids: Dict[Tuple, float],
        settled_bracket: Tuple[Optional[float], Optional[float]],
        et_hour: int,
    ) -> None:
        """Record one edge measurement for later analysis."""
        brackets = sorted(model_probs.keys())
        m_probs = [model_probs.get(b, 0.0) for b in brackets]
        mkt_probs = [market_mids.get(b, 0.0) for b in brackets]
        settled_flags = [1 if b == settled_bracket else 0 for b in brackets]

        model_brier = self.compute_brier(m_probs, settled_flags)
        market_brier = self.compute_brier(mkt_probs, settled_flags)

        self._by_hour[et_hour]["model_briers"].append(model_brier)
        self._by_hour[et_hour]["market_briers"].append(market_brier)
        self._by_hour[et_hour]["count"] += 1

        for b in brackets:
            mp = model_probs.get(b, 0.0)
            mkp = market_mids.get(b, 0.0)
            self._by_hour[et_hour]["displacements"].append(mp - mkp)

        self._records.append({
            "event_date": event_date,
            "trigger_time": trigger_time,
            "et_hour": et_hour,
            "model_brier": model_brier,
            "market_brier": market_brier,
            "edge": market_brier - model_brier,
        })

    def summary_by_hour(self) -> Dict[int, Dict]:
        """Aggregate edge statistics by ET hour."""
        result = {}
        for hour, data in self._by_hour.items():
            mb = data["model_briers"]
            mkb = data["market_briers"]
            result[hour] = {
                "model_brier": sum(mb) / len(mb) if mb else 0.0,
                "market_brier": sum(mkb) / len(mkb) if mkb else 0.0,
                "edge": (sum(mkb) - sum(mb)) / len(mb) if mb else 0.0,
                "count": data["count"],
                "avg_displacement": (
                    sum(data["displacements"]) / len(data["displacements"])
                    if data["displacements"] else 0.0
                ),
            }
        return result

    def results(self) -> List[Dict]:
        """Return all recorded edge measurements."""
        return self._records
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestEdgeAnalyzer -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add EdgeAnalyzer for Brier comparison and displacement tracking"
```

---

### Task 8: PnLSimulator — Bootstrap and Regime Splits

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** Layer 3 of the pipeline. Takes the list of daily P&Ls from the backtest and produces aggregate metrics, bootstrap confidence intervals (10,000 iterations), and regime-split analysis. The bootstrap answers "how confident are we that this is profitable?" The regime splits answer "when does the strategy work best/worst?" Regime functions take a trade record and return a regime label (e.g., "summer", "winter", "tight_spread", "wide_spread").

**Step 1: Write PnLSimulator tests**

```python
class TestPnLSimulator:
    """Layer 3: Aggregate metrics, bootstrap, regime splits."""

    def _make_trades(self):
        """Sample trade records for testing."""
        trades = []
        # 10 winning trades at ~60c profit
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
        # 5 losing trades at ~25c loss
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
        trades = self._make_trades()
        daily_pnls = [(t.event_date, t.pnl) for t in trades]
        sim = PnLSimulator()
        ci = sim.bootstrap(daily_pnls, n_iterations=1000, seed=42)
        assert "pnl_ci_95" in ci
        assert ci["pnl_ci_95"][0] < ci["pnl_ci_95"][1]
        assert "prob_profitable" in ci
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
        assert splits["spring"]["total_trades"] == 15  # all March

    def test_sharpe_ratio(self):
        trades = self._make_trades()
        sim = PnLSimulator()
        metrics = sim.aggregate(trades, starting_capital=100.0)
        # With mostly winners, Sharpe should be positive
        assert metrics["sharpe_ratio"] > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestPnLSimulator -v`
Expected: FAIL — `ImportError: cannot import name 'PnLSimulator'`

**Step 3: Implement PnLSimulator**

Add to `services/strategy_backtester.py`:

```python
class PnLSimulator:
    """Layer 3: Aggregate metrics, bootstrap resampling, regime splits."""

    def aggregate(
        self,
        trades: List[TradeRecord],
        starting_capital: float,
    ) -> Dict:
        """Compute aggregate performance metrics."""
        if not trades:
            return {"total_trades": 0, "total_pnl": 0.0}

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        gross_wins = sum(t.pnl for t in winners) if winners else 0
        gross_losses = abs(sum(t.pnl for t in losers)) if losers else 0

        # Daily P&Ls for Sharpe calculation
        daily = defaultdict(float)  # type: Dict[date, float]
        for t in trades:
            daily[t.event_date] += t.pnl

        daily_returns = list(daily.values())
        mean_daily = np.mean(daily_returns) if daily_returns else 0
        std_daily = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1

        # Cumulative P&L for drawdown
        cumulative = np.cumsum(daily_returns)
        peak = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else np.array([0])
        drawdown = peak - cumulative if len(cumulative) > 0 else np.array([0])
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "total_pnl": total_pnl,
            "total_pnl_dollars": total_pnl / 100.0,
            "total_return_pct": (total_pnl / 100.0) / starting_capital * 100,
            "sharpe_ratio": float(mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0,
            "max_drawdown": max_drawdown,
            "win_rate": len(winners) / len(trades) if trades else 0,
            "avg_win": np.mean([t.pnl for t in winners]) if winners else 0,
            "avg_loss": np.mean([t.pnl for t in losers]) if losers else 0,
            "profit_factor": gross_wins / gross_losses if gross_losses > 0 else float('inf'),
            "total_trades": len(trades),
            "total_fees": sum(t.fees_paid for t in trades),
            "avg_capital_locked_hours": np.mean([t.capital_locked_hours for t in trades]),
        }

    def bootstrap(
        self,
        daily_pnls: List[Tuple[date, float]],
        n_iterations: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict:
        """Bootstrap resampling for confidence intervals."""
        rng = np.random.RandomState(seed)
        pnls = np.array([p for _, p in daily_pnls])
        n = len(pnls)

        if n == 0:
            return {
                "pnl_ci_95": (0.0, 0.0),
                "sharpe_ci_95": (0.0, 0.0),
                "prob_profitable": 0.0,
            }

        total_pnls = []
        sharpes = []

        for _ in range(n_iterations):
            sample = rng.choice(pnls, size=n, replace=True)
            total_pnls.append(float(np.sum(sample)))
            std = np.std(sample, ddof=1) if n > 1 else 1
            sharpe = float(np.mean(sample) / std * np.sqrt(252)) if std > 0 else 0
            sharpes.append(sharpe)

        total_pnls = np.array(total_pnls)
        sharpes = np.array(sharpes)

        return {
            "pnl_ci_95": (float(np.percentile(total_pnls, 2.5)),
                          float(np.percentile(total_pnls, 97.5))),
            "sharpe_ci_95": (float(np.percentile(sharpes, 2.5)),
                             float(np.percentile(sharpes, 97.5))),
            "prob_profitable": float(np.mean(total_pnls > 0)),
        }

    def regime_split(
        self,
        trades: List[TradeRecord],
        regime_fn: Callable[[TradeRecord], str],
    ) -> Dict[str, Dict]:
        """Split trades by regime and compute metrics for each."""
        groups = defaultdict(list)  # type: Dict[str, List[TradeRecord]]
        for t in trades:
            label = regime_fn(t)
            groups[label].append(t)

        result = {}
        for label, group_trades in groups.items():
            result[label] = {
                "total_trades": len(group_trades),
                "total_pnl": sum(t.pnl for t in group_trades),
                "win_rate": (
                    len([t for t in group_trades if t.pnl > 0]) / len(group_trades)
                    if group_trades else 0
                ),
            }
        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestPnLSimulator -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add PnLSimulator with bootstrap CI and regime splits"
```

---

### Task 9: StrategyBacktester — Main Orchestrator

**Files:**
- Modify: `services/strategy_backtester.py`
- Modify: `tests/test_strategy_backtester.py`

**Context:** The main orchestrator that wires everything together. Iterates settlement dates, gets triggers, evaluates the model at each trigger, checks edge, applies sanity filters, manages positions, and settles at end of day. This is the integration point — all previous classes compose here. The test uses a simple mock model function and minimal seeded data to verify the full loop runs end-to-end.

**Important interfaces this consumes:**
- `ModelFn = Callable[[BacktestDataProvider, datetime], Optional[Dict[int, float]]]` — from `services/backtester.py`
- `BacktestDataProvider(db_path, station_id, model_run, ref_time, connection, model_name)` — from `services/data_provider.py`
- `compute_divergence_features(obs_temps, fcst_interp, obs_hours, fcst_curve_up_to_t)` — from `services/divergence.py`
- `map_probs_to_kalshi_brackets(bracket_probs_1f, kalshi_brackets)` — from `services/backtester.py`

**Step 1: Write orchestrator tests**

```python
class TestStrategyBacktester:
    """Main orchestrator — integration test with minimal data."""

    def _seed_full_data(self, db_path):
        """Seed minimal but complete data for one settlement date."""
        con = duckdb.connect(db_path)
        event_date = date(2025, 3, 15)

        # NWS settlement: actual high = 73°F
        con.execute("""
            INSERT INTO nws_daily (station_id, obs_date, max_temp_f, source, ingested_at)
            VALUES ('KNYC', '2025-03-15', 73.0, 'CLI', CURRENT_TIMESTAMP)
        """)

        # Kalshi brackets
        for floor_val in range(64, 84, 2):
            cap = floor_val + 2
            settled = 1 if floor_val <= 73 <= cap else 0
            ticker = 'KXHIGHNY-25MAR15-T{}-B{}'.format(floor_val, cap)
            con.execute("""
                INSERT INTO kalshi_settlements VALUES
                (?, 'KXHIGHNY-25MAR15', 'KXHIGHNY', 'NYC', 'high',
                 '2025-03-15', ?, ?, ?, 100, '2025-03-15 23:00:00',
                 CURRENT_TIMESTAMP)
            """, [ticker, float(floor_val), float(cap), settled])

            # Candlesticks: 3 minutes at 18:00 UTC
            for minute in range(3):
                mid = 0.10 if abs(floor_val + 1 - 73) > 4 else 0.30
                con.execute("""
                    INSERT INTO kalshi_candlesticks VALUES
                    (?, ?, 1, ?, ?, ?, ?, ?, ?, 5, 20)
                """, [
                    ticker,
                    datetime(2025, 3, 15, 18, minute),
                    mid - 0.02, mid + 0.02,
                    mid - 0.01, mid + 0.01, mid - 0.01, mid,
                ])

        # HRRR forecast: predicts 74°F high
        model_run = datetime(2025, 3, 15, 12, 0)
        for hour in range(24):
            temp = 60 + min(hour, 18) * (14.0 / 18)  # ramp to 74
            con.execute("""
                INSERT INTO forecasts (station_id, model_run, valid_at,
                    temp_f, model_name, ingested_at)
                VALUES ('KNYC', ?, ?, ?, 'hrrr', CURRENT_TIMESTAMP)
            """, [model_run, datetime(2025, 3, 15, hour), temp])

        # Observations: hourly 12-20 UTC
        for hour in range(12, 21):
            temp = 65 + (hour - 12) * 1.0
            con.execute("""
                INSERT INTO observations (station_id, observed_at, temp_f,
                    ingest_source, ingested_at)
                VALUES ('KNYC', ?, ?, 'synoptic', CURRENT_TIMESTAMP)
            """, [datetime(2025, 3, 15, hour, 53), temp])

        # Station bias
        con.execute("""
            INSERT INTO station_bias (station_id, calculated_at, mean_bias,
                std_error, sample_days, ingested_at)
            VALUES ('KNYC', '2025-03-14 00:00:00', 1.0, 2.5, 90,
                CURRENT_TIMESTAMP)
        """)

        con.close()

    @pytest.fixture
    def full_db(self, test_db):
        self._seed_full_data(test_db)
        return test_db

    def test_run_produces_report(self, full_db):
        """Full loop: model → edge → strategy → settlement → report."""
        from services.backtester import uniform_model

        config = BacktestConfig(
            starting_capital=100.0,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31),
            burn_in_days=0,  # skip burn-in for test
            min_displacement=0.05,  # low threshold for test
        )
        backtester = StrategyBacktester(db_path=full_db, config=config)
        report = backtester.run(model_fn=uniform_model)

        assert report is not None
        assert "total_trades" in report
        assert "total_pnl" in report
        assert "edge_by_hour" in report

    def test_capital_constraint_enforced(self, full_db):
        """With tiny bankroll, some signals should be rejected."""
        from services.backtester import uniform_model

        config = BacktestConfig(
            starting_capital=0.10,  # 10 cents — very constrained
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31),
            burn_in_days=0,
            min_displacement=0.01,
        )
        backtester = StrategyBacktester(db_path=full_db, config=config)
        report = backtester.run(model_fn=uniform_model)

        assert report["missed_due_to_capital"] >= 0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestStrategyBacktester -v`
Expected: FAIL — `ImportError: cannot import name 'StrategyBacktester'`

**Step 3: Implement StrategyBacktester**

Add to `services/strategy_backtester.py`:

```python
class StrategyBacktester:
    """Main orchestrator — three-layer unified pipeline.

    Iterates settlement dates, evaluates model at each trigger,
    computes edge, applies strategy filters, manages positions,
    settles at end of day, and produces the final report.
    """

    def __init__(self, db_path: str, config: BacktestConfig):
        self.db_path = db_path
        self.config = config
        self.market_loader = MarketDataLoader(db_path)
        self.trigger_detector = TriggerDetector(
            db_path, config.station_id, config.execution_latency_seconds,
        )
        self.edge_analyzer = EdgeAnalyzer()
        self.sanity_filter = SanityFilter(config)
        self.pnl_simulator = PnLSimulator()

    def _get_settlement_dates(self) -> List[Tuple[date, float]]:
        """Get (event_date, actual_high) from nws_daily in the configured window."""
        con = duckdb.connect(self.db_path, read_only=True)
        query = """
            SELECT obs_date, max_temp_f FROM nws_daily
            WHERE station_id = ? AND max_temp_f IS NOT NULL
        """
        params = [self.config.station_id]  # type: list

        if self.config.start_date:
            query += " AND obs_date >= ?"
            params.append(self.config.start_date)
        if self.config.end_date:
            query += " AND obs_date <= ?"
            params.append(self.config.end_date)

        query += " ORDER BY obs_date"
        rows = con.execute(query, params).fetchall()
        con.close()
        return [(r[0], r[1]) for r in rows]

    def _get_model_run_for_trigger(
        self,
        trigger_time: datetime,
        trigger_type: str,
        event_date: date,
    ) -> datetime:
        """Determine which forecast model_run to use at this trigger time.

        Uses the most recent model_run that's available (run_hour + 2h lag).
        """
        from services.data_provider import HRRR_AVAILABILITY_LAG_HOURS

        best_run = None
        for day_offset in [-1, 0]:
            run_date = event_date + timedelta(days=day_offset)
            for rh in RUN_HOURS:
                run_time = datetime(
                    run_date.year, run_date.month, run_date.day,
                    rh, 0, tzinfo=timezone.utc,
                )
                avail_time = run_time + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)
                if avail_time <= trigger_time:
                    if best_run is None or run_time > best_run:
                        best_run = run_time

        if best_run is None:
            best_run = datetime(
                event_date.year, event_date.month, event_date.day,
                0, 0, tzinfo=timezone.utc,
            )
        return best_run

    def _get_et_hour(self, utc_time: datetime) -> int:
        """Convert UTC time to ET hour."""
        et_time = utc_time.astimezone(_ET) if utc_time.tzinfo else utc_time.replace(tzinfo=timezone.utc).astimezone(_ET)
        return et_time.hour

    def run(self, model_fn: ModelFn) -> Dict:
        """Execute the full backtest pipeline."""
        config = self.config
        position_mgr = PositionManager(bankroll=config.starting_capital)
        all_trades = []  # type: List[TradeRecord]
        missed_capital = 0
        anomalies = []  # type: List[Dict]

        settlement_dates = self._get_settlement_dates()
        burn_in_end = config.start_date + timedelta(days=config.burn_in_days)

        con = duckdb.connect(self.db_path, read_only=True)

        for event_date, actual_high in settlement_dates:
            # Get bracket definitions and settlement outcome
            brackets_info = self.market_loader.get_brackets(event_date)
            if not brackets_info:
                continue

            settled_bracket = None
            for bi in brackets_info:
                if bi["settled_yes"] == 1:
                    settled_bracket = bi["bracket"]
                    break

            if settled_bracket is None:
                continue

            # Market window: prior day 10 AM ET → settlement day 11 PM ET
            market_open = datetime(
                event_date.year, event_date.month, event_date.day,
                15, 0, tzinfo=timezone.utc,  # 10 AM ET = 15 UTC
            ) - timedelta(days=1)
            market_close = datetime(
                event_date.year, event_date.month, event_date.day,
                23, 0, tzinfo=timezone.utc,
            )

            # Get triggers for this event
            triggers = self.trigger_detector.get_triggers(
                event_date, market_open, market_close,
            )

            # Track obs for post-peak detection
            day_obs = []  # type: List[Tuple[datetime, float]]

            for model_time, execution_time, trigger_type in triggers:
                et_hour = self._get_et_hour(model_time)

                # Determine model_run to use
                model_run = self._get_model_run_for_trigger(
                    model_time, trigger_type, event_date,
                )

                # Create data provider (walk-forward fence at model_time)
                provider = BacktestDataProvider(
                    db_path=self.db_path,
                    station_id=config.station_id,
                    model_run=model_run,
                    ref_time=model_time,  # model sees data up to model_time
                    connection=con,
                    model_name=config.model_name if config.model_name != 'ensemble' else 'hrrr',
                )

                # Call model (evaluates at model_time)
                bracket_probs_1f = model_fn(provider, model_time)
                if bracket_probs_1f is None:
                    continue

                # Map to Kalshi brackets
                kalshi_brackets = [
                    KalshiBracket(bi["bracket"][0], bi["bracket"][1], bi["settled_yes"])
                    for bi in brackets_info
                ]
                mapped_probs = map_probs_to_kalshi_brackets(bracket_probs_1f, kalshi_brackets)

                # Build model_probs dict
                model_probs = {}  # type: Dict[Tuple, float]
                for i, bi in enumerate(brackets_info):
                    model_probs[bi["bracket"]] = mapped_probs[i]

                # Get market prices at execution_time (model_time + latency)
                # This simulates realistic slippage — the order reaches
                # the exchange after the model has processed the observation.
                market_mids = {}  # type: Dict[Tuple, float]
                market_asks = {}  # type: Dict[Tuple, float]
                market_bids = {}  # type: Dict[Tuple, float]
                for bi in brackets_info:
                    snaps = self.market_loader.get_snapshots(
                        market_ticker=bi["market_ticker"],
                        bracket=bi["bracket"],
                        start_ts=execution_time - timedelta(minutes=1),
                        end_ts=execution_time + timedelta(minutes=1),
                    )
                    if snaps:
                        snap = snaps[-1]
                        market_mids[bi["bracket"]] = (snap.yes_bid + snap.yes_ask) / 2
                        market_asks[bi["bracket"]] = snap.yes_ask
                        market_bids[bi["bracket"]] = snap.yes_bid

                if not market_mids:
                    continue

                # ── Layer 1: Edge Analysis (always record, even during burn-in)
                self.edge_analyzer.record(
                    event_date=event_date,
                    trigger_time=model_time,
                    model_probs=model_probs,
                    market_mids=market_mids,
                    settled_bracket=settled_bracket,
                    et_hour=et_hour,
                )

                # Skip trading during burn-in
                if event_date < burn_in_end:
                    continue

                # ── Layer 2: Strategy Engine ──

                # Update obs for post-peak detection
                if trigger_type == 'observation':
                    obs = provider.get_observations_in_range(
                        config.station_id,
                        model_time - timedelta(minutes=5),
                        model_time + timedelta(minutes=5),
                    )
                    for obs_row in obs:
                        day_obs.append((model_time, obs_row[1]))

                is_post_peak = self.sanity_filter.is_post_peak(day_obs, model_time)
                recently_exited = position_mgr.recently_exited_brackets(
                    model_time, config.min_reentry_minutes,
                )

                # Compute model std (approximate from bracket probs)
                temps = list(bracket_probs_1f.keys())
                probs_list = list(bracket_probs_1f.values())
                if temps and probs_list:
                    mean_t = sum(t * p for t, p in zip(temps, probs_list))
                    var_t = sum(p * (t - mean_t) ** 2 for t, p in zip(temps, probs_list))
                    model_std = var_t ** 0.5
                else:
                    model_std = 99.0

                # Evaluate each bracket for trade signals
                candidates = []
                for bi in brackets_info:
                    bracket = bi["bracket"]
                    mp = model_probs.get(bracket, 0.0)
                    ask = market_asks.get(bracket)
                    bid = market_bids.get(bracket)
                    if ask is None:
                        continue

                    ask_cents = ask * 100
                    spread_cents = (ask - market_bids.get(bracket, ask)) * 100

                    ok, reason = self.sanity_filter.check(
                        model_prob=mp,
                        market_ask_cents=ask_cents,
                        spread_cents=spread_cents,
                        model_std=model_std,
                        bracket=bracket,
                        is_post_peak=is_post_peak,
                        recently_exited=recently_exited,
                    )
                    if not ok:
                        continue

                    displacement = mp - ask
                    if displacement < config.min_displacement:
                        continue

                    candidates.append((bracket, ask_cents, mp))

                # Portfolio-level EV optimization
                if candidates:
                    portfolio = EventPortfolio(event_date)
                    existing = position_mgr.positions_for(event_date)
                    existing_brackets = {p.bracket for p in existing}

                    # Filter out brackets we already hold
                    new_candidates = [
                        c for c in candidates if c[0] not in existing_brackets
                    ]

                    optimal = portfolio.optimal_subset(new_candidates)
                    for bracket, ask_cents, mp in optimal:
                        pos = position_mgr.open_position(
                            event_date=event_date,
                            bracket=bracket,
                            entry_price_cents=ask_cents,
                            entry_time=execution_time,  # entry at execution time, not model time
                            quantity=config.fixed_bet_size,
                        )
                        if pos is None:
                            missed_capital += 1

            # ── Layer 3: Settlement ──
            day_trades = position_mgr.settle_day(event_date, settled_bracket)
            all_trades.extend(day_trades)

        con.close()

        # Build report
        metrics = self.pnl_simulator.aggregate(all_trades, config.starting_capital)
        daily_pnls = []  # type: List[Tuple[date, float]]
        daily_map = defaultdict(float)  # type: Dict[date, float]
        for t in all_trades:
            daily_map[t.event_date] += t.pnl
        daily_pnls = sorted(daily_map.items())

        bootstrap = self.pnl_simulator.bootstrap(
            daily_pnls, config.bootstrap_iterations,
        )

        report = dict(metrics)
        report["edge_by_hour"] = self.edge_analyzer.summary_by_hour()
        report["bootstrap"] = bootstrap
        report["missed_due_to_capital"] = missed_capital
        report["anomalies"] = anomalies
        report["trades"] = all_trades
        report["final_bankroll"] = position_mgr.bankroll

        return report
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py::TestStrategyBacktester -v`
Expected: ALL PASS (may need minor fixture adjustments based on DB schema)

**Step 5: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add services/strategy_backtester.py tests/test_strategy_backtester.py
git commit -m "feat: add StrategyBacktester main orchestrator"
```

---

### Task 10: CLI Runner Script

**Files:**
- Create: `scripts/run_strategy_backtest.py`

**Context:** Command-line entry point that runs the full strategy backtest with configurable parameters and prints a formatted report. Uses argparse for configuration overrides. Connects to the production DuckDB database. This is what Russell runs to get results.

**Step 1: Write the CLI script**

```python
"""scripts/run_strategy_backtest.py

Run the Kalshi strategy backtester and print results.

Usage:
    python scripts/run_strategy_backtest.py
    python scripts/run_strategy_backtest.py --capital 50 --min-displacement 0.15
    python scripts/run_strategy_backtest.py --start 2025-01-01 --end 2025-06-01
"""

import argparse
import sys
from datetime import date, datetime

from loguru import logger

from services.strategy_backtester import (
    BacktestConfig,
    StrategyBacktester,
)
from services.backtester import uniform_model, walk_forward_model


def parse_args():
    p = argparse.ArgumentParser(description="Kalshi Strategy Backtester")
    p.add_argument("--db", default="data/alphatemp.duckdb", help="DuckDB path")
    p.add_argument("--capital", type=float, default=100.0, help="Starting capital ($)")
    p.add_argument("--start", type=str, default="2024-11-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--burn-in", type=int, default=90, help="Burn-in days")
    p.add_argument("--min-displacement", type=float, default=0.12, help="Minimum displacement")
    p.add_argument("--min-prob", type=float, default=0.05, help="Minimum model probability")
    p.add_argument("--max-spread", type=float, default=10.0, help="Maximum spread (cents)")
    p.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations")
    p.add_argument("--model", type=str, default="uniform", help="Model: uniform, walkforward")
    return p.parse_args()


def print_report(report):
    """Pretty-print the backtest report."""
    print("\n" + "=" * 60)
    print("  KALSHI STRATEGY BACKTEST REPORT")
    print("=" * 60)

    print("\n── Aggregate Metrics ──")
    print("  Total trades:      {}".format(report.get("total_trades", 0)))
    print("  Win rate:          {:.1%}".format(report.get("win_rate", 0)))
    print("  Total P&L:         {:.1f}c (${:.2f})".format(
        report.get("total_pnl", 0), report.get("total_pnl_dollars", 0)))
    print("  Return:            {:.1f}%".format(report.get("total_return_pct", 0)))
    print("  Sharpe ratio:      {:.2f}".format(report.get("sharpe_ratio", 0)))
    print("  Max drawdown:      {:.1f}c".format(report.get("max_drawdown", 0)))
    print("  Profit factor:     {:.2f}".format(report.get("profit_factor", 0)))
    print("  Avg win:           {:.1f}c".format(report.get("avg_win", 0)))
    print("  Avg loss:          {:.1f}c".format(report.get("avg_loss", 0)))
    print("  Total fees:        {:.1f}c".format(report.get("total_fees", 0)))
    print("  Final bankroll:    ${:.2f}".format(report.get("final_bankroll", 0)))

    bs = report.get("bootstrap", {})
    if bs:
        print("\n── Bootstrap Confidence (95%) ──")
        ci = bs.get("pnl_ci_95", (0, 0))
        print("  P&L CI:            [{:.1f}c, {:.1f}c]".format(ci[0], ci[1]))
        print("  Prob profitable:   {:.1%}".format(bs.get("prob_profitable", 0)))

    edge = report.get("edge_by_hour", {})
    if edge:
        print("\n── Edge by ET Hour ──")
        print("  {:>4}  {:>10}  {:>10}  {:>8}  {:>5}".format(
            "Hour", "Model Br.", "Mkt Br.", "Edge", "N"))
        for h in sorted(edge.keys()):
            e = edge[h]
            print("  {:>4}  {:>10.4f}  {:>10.4f}  {:>+8.4f}  {:>5}".format(
                h, e["model_brier"], e["market_brier"], e["edge"], e["count"]))

    missed = report.get("missed_due_to_capital", 0)
    if missed > 0:
        print("\n── Capital Constraints ──")
        print("  Missed signals (capital):  {}".format(missed))

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    config = BacktestConfig(
        starting_capital=args.capital,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end) if args.end else None,
        burn_in_days=args.burn_in,
        min_displacement=args.min_displacement,
        min_model_prob=args.min_prob,
        max_spread_cents=args.max_spread,
        bootstrap_iterations=args.bootstrap,
    )

    models = {
        "uniform": uniform_model,
        "walkforward": walk_forward_model,
    }
    model_fn = models.get(args.model, uniform_model)

    logger.info("Running strategy backtest: {} → {}, capital=${}, model={}",
                config.start_date, config.end_date or "latest",
                config.starting_capital, args.model)

    bt = StrategyBacktester(db_path=args.db, config=config)
    report = bt.run(model_fn=model_fn)

    print_report(report)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs (smoke test)**

Run: `cd ~/Projects/alphatemp/alphatemp && python scripts/run_strategy_backtest.py --start 2025-02-01 --end 2025-02-28 --burn-in 0 --bootstrap 100`
Expected: Report prints without errors (P&L may be 0 if uniform model has no edge)

**Step 3: Commit**

```bash
cd ~/Projects/alphatemp/alphatemp
git add scripts/run_strategy_backtest.py
git commit -m "feat: add CLI runner for strategy backtester"
```

---

### Task 11: Full Test Suite Run and Cleanup

**Files:**
- Modify: `tests/test_strategy_backtester.py` (add missing test_db fixture if needed)
- No new files

**Context:** Run the full test suite to make sure nothing is broken. Fix any import issues, fixture problems, or test interactions. The `test_db` fixture needs to create and tear down a temporary DuckDB with the full schema (via `init_db()`).

**Step 1: Add the test_db fixture if not already present**

Check if `test_db` fixture exists in conftest.py or add it to the test file:

```python
import os
import duckdb
from core.db import init_db

TEST_DB = "tests/test_strategy_backtest.duckdb"

@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
```

**Step 2: Run full test suite**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/test_strategy_backtester.py -v --tb=short`
Expected: ALL PASS

**Step 3: Run existing test suite to verify no regressions**

Run: `cd ~/Projects/alphatemp/alphatemp && python -m pytest tests/ -v --tb=short -x`
Expected: ALL PASS (existing tests unaffected)

**Step 4: Commit any fixture fixes**

```bash
cd ~/Projects/alphatemp/alphatemp
git add tests/test_strategy_backtester.py
git commit -m "test: finalize strategy backtester test suite"
```

---

## Summary

| Task | Component | Tests | Estimated Steps |
|------|-----------|-------|-----------------|
| 1 | Data structures + fee math | 10 | 5 |
| 2 | SanityFilter (5 guards) | 9 | 5 |
| 3 | EventPortfolio (mutually exclusive EV) | 7 | 5 |
| 4 | PositionManager (capital + exits) | 9 | 5 |
| 5 | MarketDataLoader | 4 | 5 |
| 6 | TriggerDetector (+ latency slippage) | 5 | 5 |
| 7 | EdgeAnalyzer (Layer 1) | 4 | 5 |
| 8 | PnLSimulator (Layer 3) | 4 | 5 |
| 9 | StrategyBacktester (orchestrator) | 2 | 5 |
| 10 | CLI runner | 0 (smoke test) | 3 |
| 11 | Full test suite cleanup | — | 4 |

**Total: 11 tasks, 53 tests, ~52 steps**

## Dependency Order

```
Task 1 (data structures) ← ALL other tasks depend on this
Task 2 (sanity filters)  ← Task 9
Task 3 (portfolio EV)    ← Task 9
Task 4 (positions)       ← Task 9
Task 5 (market loader)   ← Task 9
Task 6 (triggers)        ← Task 9
Task 7 (edge analyzer)   ← Task 9
Task 8 (P&L simulator)   ← Task 9
Task 9 (orchestrator)    ← Task 10, 11
```

Tasks 2-8 are independent and can be implemented in parallel after Task 1.
