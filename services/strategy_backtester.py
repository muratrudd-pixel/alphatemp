"""AlphaTemp Kalshi strategy backtester — data structures and fee math.

Foundation layer for the strategy backtester. Defines:
- Fee constants and calculation functions (entry cost, settlement P&L, exit P&L)
- Core dataclasses (BacktestConfig, MarketSnapshot, ModelUpdate, Position, TradeRecord)

All P&L calculations are in cents unless stated otherwise.
Kalshi fee structure: 1% trading fee on entry, 10% settlement fee on profit (winners only).
Execution modeled at yes_ask (crossing the spread).

Design doc: docs/plans/2026-02-27-backtest-framework-design.md
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

from services.backtester import (
    KalshiBracket,
    ModelFn,
    map_probs_to_kalshi_brackets,
    compute_brier_score,
)
from services.data_provider import BacktestDataProvider

# ── Fee Constants ────────────────────────────────────────────────────────────

TRADING_FEE_RATE = 0.01       # 1% of contract value
SETTLEMENT_FEE_RATE = 0.10    # 10% of profit (winners only)


# ── Fee Functions ────────────────────────────────────────────────────────────

def compute_entry_cost(price_cents: float, quantity: int) -> float:
    """Total capital locked when buying YES contracts.

    Returns price * qty + trading fee (in cents).
    """
    gross = price_cents * quantity
    trading_fee = gross * TRADING_FEE_RATE
    return gross + trading_fee


def compute_settlement_pnl(
    entry_price_cents: float, quantity: int, settled_yes: bool
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
    entry_price_cents: float, exit_price_cents: float, quantity: int
) -> float:
    """Net P&L from early exit (sell YES back to market), in cents."""
    revenue = exit_price_cents * quantity
    exit_trading_fee = revenue * TRADING_FEE_RATE
    entry_cost = entry_price_cents * quantity
    entry_trading_fee = entry_cost * TRADING_FEE_RATE
    return revenue - exit_trading_fee - entry_cost - entry_trading_fee


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
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
    bracket: Tuple[Optional[float], Optional[float]]
    timestamp: datetime
    yes_bid: float
    yes_ask: float
    volume: int
    is_stale: bool


@dataclass
class ModelUpdate:
    ref_time: datetime
    bracket_probs: Dict[Tuple, float]
    trigger_type: str
    model_std: float


@dataclass
class Position:
    event_date: date
    bracket: Tuple[Optional[float], Optional[float]]
    entry_price: float
    entry_time: datetime
    quantity: int
    capital_locked: float


@dataclass
class TradeRecord:
    event_date: date
    bracket: Tuple[Optional[float], Optional[float]]
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    exit_type: str
    settlement_result: Optional[int]
    pnl: float
    fees_paid: float
    displacement_at_entry: float
    model_prob_at_entry: float
    capital_locked_hours: float


# ── Timezone Constant ──────────────────────────────────────────────────────

_ET = timezone(timedelta(hours=-5))


# ── Sanity Filter ──────────────────────────────────────────────────────────

class SanityFilter:
    """Pre-trade guard rails — rejects entries that fail any of 5 checks."""

    def __init__(self, config):
        # type: (BacktestConfig) -> None
        self.min_model_prob = config.min_model_prob
        self.max_spread_cents = config.max_spread_cents
        self.max_model_std = config.max_model_std

    def check(
        self,
        model_prob,       # type: float
        market_ask_cents,  # type: float
        spread_cents,     # type: float
        model_std,        # type: float
        bracket,          # type: Tuple
        is_post_peak,     # type: bool
        recently_exited,  # type: Set[Tuple]
    ):
        # type: (...) -> Tuple[bool, str]
        """Run 5 filters in order. Return (True, '') or (False, reason)."""
        if model_prob < self.min_model_prob:
            return (False, "model_prob %.3f below min %.3f" % (model_prob, self.min_model_prob))
        if is_post_peak:
            return (False, "post_peak: daily high likely passed")
        if spread_cents > self.max_spread_cents:
            return (False, "spread %.1f exceeds max %.1f" % (spread_cents, self.max_spread_cents))
        if model_std > self.max_model_std:
            return (False, "uncertainty (std=%.2f) exceeds max %.2f" % (model_std, self.max_model_std))
        if bracket in recently_exited:
            return (False, "churn/reentry: bracket recently exited")
        return (True, "")

    @staticmethod
    def is_post_peak_check(
        obs_temps,     # type: List[Tuple[datetime, float]]
        current_time,  # type: datetime
    ):
        # type: (...) -> bool
        """Detect if the daily high has already passed.

        ALL conditions must be true:
        1. At least 3 observations
        2. Current time after 2 PM ET (14:00 ET)
        3. Last 2+ observations are declining
        4. Running max occurred >60 min ago
        """
        if len(obs_temps) < 3:
            return False

        # Convert current_time to ET for the 2 PM check
        current_et = current_time.astimezone(_ET)
        if current_et.hour < 14:
            return False

        # Check last 2+ obs declining
        if len(obs_temps) >= 2:
            if obs_temps[-1][1] >= obs_temps[-2][1]:
                return False

        if len(obs_temps) >= 3:
            if obs_temps[-2][1] >= obs_temps[-3][1]:
                return False

        # Running max occurred >60 min ago
        max_temp = max(obs_temps, key=lambda x: x[1])
        max_time = max_temp[0]
        if (current_time - max_time).total_seconds() <= 3600:
            return False

        return True
