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
from datetime import date, datetime
from typing import Dict, Optional, Tuple

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
