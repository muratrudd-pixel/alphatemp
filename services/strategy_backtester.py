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


# ── Event Portfolio ────────────────────────────────────────────────────────

class EventPortfolio:
    """Mutually exclusive bracket portfolio — EV and optimal subset selection."""

    def __init__(self, event_date):
        # type: (date) -> None
        self.event_date = event_date

    def combined_ev(self, entry_prices, model_probs):
        # type: (List[float], List[float]) -> float
        """Mutually exclusive EV calculation.

        For N brackets held simultaneously:
        - If bracket j wins (prob q_j): gain on j, lose on all others.
        - If none win (prob 1 - sum(q)): lose all entries.

        Loss per bracket = price + price * TRADING_FEE_RATE
        Gain per bracket = (100-price) * (1-SETTLEMENT_FEE_RATE) - price * TRADING_FEE_RATE
        """
        n = len(entry_prices)
        if n == 0:
            return 0.0

        # Pre-compute per-bracket loss and gain
        losses = []  # type: List[float]
        gains = []   # type: List[float]
        for p in entry_prices:
            losses.append(p + p * TRADING_FEE_RATE)
            gains.append((100 - p) * (1 - SETTLEMENT_FEE_RATE) - p * TRADING_FEE_RATE)

        total_loss = sum(losses)
        total_prob = sum(model_probs)

        ev = 0.0
        for j in range(n):
            # If bracket j wins: gain on j, lose on all others
            net_if_j_wins = gains[j] - (total_loss - losses[j])
            ev += model_probs[j] * net_if_j_wins

        # If none wins
        ev += (1.0 - total_prob) * (-total_loss)

        return ev

    def optimal_subset(self, candidates):
        # type: (List[Tuple[Tuple, float, float]]) -> List[Tuple[Tuple, float, float]]
        """Greedy selection of brackets maximizing portfolio EV.

        candidates = [(bracket, entry_price_cents, model_prob), ...]

        Algorithm:
        1. Filter to positive individual EV
        2. Sort by individual EV descending
        3. Greedily add if portfolio EV increases
        """
        if not candidates:
            return []

        # Step 1: filter to positive individual EV
        positive = []  # type: List[Tuple[Tuple, float, float]]
        for bracket, price, prob in candidates:
            single_ev = self.combined_ev([price], [prob])
            if single_ev > 0:
                positive.append((bracket, price, prob))

        # Step 2: sort by individual EV descending
        positive.sort(key=lambda c: self.combined_ev([c[1]], [c[2]]), reverse=True)

        # Step 3: greedy addition
        selected = []  # type: List[Tuple[Tuple, float, float]]
        current_ev = 0.0
        for candidate in positive:
            trial_prices = [c[1] for c in selected] + [candidate[1]]
            trial_probs = [c[2] for c in selected] + [candidate[2]]
            trial_ev = self.combined_ev(trial_prices, trial_probs)
            if trial_ev > current_ev:
                selected.append(candidate)
                current_ev = trial_ev

        return selected


# ── Position Manager ──────────────────────────────────────────────────────

class PositionManager:
    """Tracks open positions, capital lockup, and early exits.

    Capital (bankroll) is tracked in DOLLARS.
    Positions and P&L use CENTS. Conversion: cents / 100.
    """

    def __init__(self, bankroll):
        # type: (float) -> None
        self.bankroll = bankroll  # dollars
        self.positions = []  # type: List[Position]
        self._recently_exited = {}  # type: Dict[Tuple, datetime]

    @property
    def capital_locked(self):
        # type: () -> float
        """Total capital locked across all open positions, in dollars."""
        return sum(p.capital_locked for p in self.positions) / 100.0

    @property
    def capital_available(self):
        # type: () -> float
        """Bankroll minus locked capital, in dollars."""
        return self.bankroll - self.capital_locked

    def positions_for(self, event_date):
        # type: (date) -> List[Position]
        """Return all open positions for a given event date."""
        return [p for p in self.positions if p.event_date == event_date]

    def open_position(self, event_date, bracket, entry_price_cents, entry_time, quantity):
        # type: (date, Tuple, float, datetime, int) -> Optional[Position]
        """Open a new position if sufficient capital is available.

        Returns the Position or None if insufficient capital.
        """
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

    def close_position(self, event_date, bracket, exit_price_cents, exit_time):
        # type: (date, Tuple, float, datetime) -> float
        """Close a position early (sell back to market).

        Returns P&L in cents. Updates bankroll. Tracks bracket in recently_exited.
        """
        pos = None  # type: Optional[Position]
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

    def settle_day(self, event_date, settled_bracket):
        # type: (date, Tuple) -> List[TradeRecord]
        """Settle all positions for a given event date.

        Returns a list of TradeRecords. Updates bankroll for each settlement.
        """
        day_positions = self.positions_for(event_date)
        records = []  # type: List[TradeRecord]

        for pos in day_positions:
            won = pos.bracket == settled_bracket
            pnl = compute_settlement_pnl(pos.entry_price, pos.quantity, won)
            self.bankroll += pnl / 100.0

            # Compute fees for the record
            entry_trading_fee = pos.entry_price * pos.quantity * TRADING_FEE_RATE
            if won:
                gross_profit = (100 - pos.entry_price) * pos.quantity
                settlement_fee = gross_profit * SETTLEMENT_FEE_RATE
                fees = entry_trading_fee + settlement_fee
            else:
                fees = entry_trading_fee

            # Capital locked hours
            hours_locked = 0.0  # will be filled by caller if needed

            record = TradeRecord(
                event_date=pos.event_date,
                bracket=pos.bracket,
                direction="buy_yes",
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                exit_price=100.0 if won else 0.0,
                exit_time=None,
                exit_type="settlement",
                settlement_result=1 if won else 0,
                pnl=pnl,
                fees_paid=fees,
                displacement_at_entry=0.0,
                model_prob_at_entry=0.0,
                capital_locked_hours=hours_locked,
            )
            records.append(record)

        # Remove settled positions
        self.positions = [p for p in self.positions if p.event_date != event_date]

        return records

    def recently_exited_brackets(self, current_time, min_minutes):
        # type: (datetime, int) -> Set[Tuple]
        """Return brackets exited within the last min_minutes."""
        cutoff = current_time - timedelta(minutes=min_minutes)
        return {
            bracket for bracket, exit_time in self._recently_exited.items()
            if exit_time >= cutoff
        }
