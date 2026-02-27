"""AlphaTemp Kalshi strategy backtester — data structures and fee math.

Foundation layer for the strategy backtester. Defines:
- Fee constants and calculation functions (entry cost, settlement P&L, exit P&L)
- Core dataclasses (BacktestConfig, MarketSnapshot, ModelUpdate, Position, TradeRecord)

All P&L calculations are in cents unless stated otherwise.
Kalshi fee structure: 1% trading fee on entry, 10% settlement fee on profit (winners only).
Execution modeled at yes_ask (crossing the spread).

Design doc: docs/plans/2026-02-27-backtest-framework-design.md
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np

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


# ── Market Data Loader ────────────────────────────────────────────────────

class MarketDataLoader:
    """Load Kalshi candlestick data and convert to MarketSnapshot objects."""

    def __init__(self, db_path):
        # type: (str) -> None
        self.db_path = db_path

    def get_brackets(self, event_date):
        # type: (date) -> List[Dict]
        """Get bracket definitions from kalshi_settlements for KXHIGHNY."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = con.execute("""
                SELECT market_ticker, floor_strike, cap_strike, settled_yes, volume
                FROM kalshi_settlements
                WHERE event_date = ? AND series_ticker = 'KXHIGHNY'
            """, [event_date]).fetchall()
        finally:
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

    def get_snapshots(self, market_ticker, bracket, start_ts, end_ts):
        # type: (str, Tuple, datetime, datetime) -> List[MarketSnapshot]
        """Load candlestick data and forward-fill gaps.

        One MarketSnapshot per minute in [start_ts, end_ts).
        Minutes without candles forward-fill with is_stale=True.
        """
        # Strip tz for DuckDB TIMESTAMP columns (stored as naive UTC)
        start_naive = start_ts.replace(tzinfo=None) if start_ts.tzinfo else start_ts
        end_naive = end_ts.replace(tzinfo=None) if end_ts.tzinfo else end_ts

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = con.execute("""
                SELECT end_period_ts, yes_bid_close, yes_ask_close, volume
                FROM kalshi_candlesticks
                WHERE market_ticker = ?
                  AND end_period_ts >= ?
                  AND end_period_ts < ?
                ORDER BY end_period_ts
            """, [market_ticker, start_naive, end_naive]).fetchall()
        finally:
            con.close()

        # Index candle data by minute
        candle_map = {}  # type: Dict[datetime, tuple]
        for row in rows:
            ts = row[0]
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            candle_map[ts] = (row[1], row[2], row[3])

        # Generate one snapshot per minute in [start_ts, end_ts)
        snapshots = []  # type: List[MarketSnapshot]
        current = start_ts
        last_bid = 0.0
        last_ask = 0.0
        last_vol = 0

        while current < end_ts:
            if current in candle_map:
                bid, ask, vol = candle_map[current]
                last_bid = bid
                last_ask = ask
                last_vol = vol
                snapshots.append(MarketSnapshot(
                    bracket=bracket,
                    timestamp=current,
                    yes_bid=bid,
                    yes_ask=ask,
                    volume=vol,
                    is_stale=False,
                ))
            else:
                snapshots.append(MarketSnapshot(
                    bracket=bracket,
                    timestamp=current,
                    yes_bid=last_bid,
                    yes_ask=last_ask,
                    volume=last_vol,
                    is_stale=True,
                ))
            current += timedelta(minutes=1)

        return snapshots


# ── Trigger Detector ──────────────────────────────────────────────────────

RUN_HOURS = [0, 6, 12, 18]


class TriggerDetector:
    """Returns (model_time, execution_time, trigger_type) tuples.

    execution_time = model_time + latency (simulates real execution delay).
    """

    def __init__(self, db_path, station_id='KNYC', execution_latency_seconds=60):
        # type: (str, str, int) -> None
        self.db_path = db_path
        self.station_id = station_id
        self.latency = timedelta(seconds=execution_latency_seconds)

    def get_triggers(self, event_date, market_open_utc, market_close_utc):
        # type: (date, datetime, datetime) -> List[Tuple[datetime, datetime, str]]
        """Return sorted (model_time, execution_time, trigger_type).

        1. Query distinct observed_at from observations table in [open, close]
        2. Add forecast run availability times (run_hour + HRRR_AVAILABILITY_LAG_HOURS)
        3. Each gets execution_time = model_time + self.latency
        4. Sort by model_time
        """
        from services.data_provider import HRRR_AVAILABILITY_LAG_HOURS

        # Strip tz for DuckDB queries
        open_naive = market_open_utc.replace(tzinfo=None) if market_open_utc.tzinfo else market_open_utc
        close_naive = market_close_utc.replace(tzinfo=None) if market_close_utc.tzinfo else market_close_utc

        triggers = []  # type: List[Tuple[datetime, datetime, str]]

        # 1. Observation triggers
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            rows = con.execute("""
                SELECT DISTINCT observed_at
                FROM observations
                WHERE station_id = ?
                  AND observed_at >= ?
                  AND observed_at <= ?
                ORDER BY observed_at
            """, [self.station_id, open_naive, close_naive]).fetchall()
        finally:
            con.close()

        for (obs_at,) in rows:
            if isinstance(obs_at, datetime) and obs_at.tzinfo is None:
                obs_at = obs_at.replace(tzinfo=timezone.utc)
            triggers.append((obs_at, obs_at + self.latency, 'observation'))

        # 2. Forecast run triggers
        # Generate run times that fall within [open, close] after adding lag
        # Check the day of event_date and the day before
        for day_offset in range(-1, 2):
            check_date = event_date + timedelta(days=day_offset)
            for run_hour in RUN_HOURS:
                run_time = datetime(
                    check_date.year, check_date.month, check_date.day,
                    run_hour, 0, 0, tzinfo=timezone.utc,
                )
                avail_time = run_time + timedelta(hours=HRRR_AVAILABILITY_LAG_HOURS)
                if market_open_utc <= avail_time <= market_close_utc:
                    triggers.append((avail_time, avail_time + self.latency, 'forecast_run'))

        # Sort by model_time
        triggers.sort(key=lambda t: t[0])
        return triggers


# ── Edge Analyzer ─────────────────────────────────────────────────────────

class EdgeAnalyzer:
    """Layer 1: Brier comparison + displacement tracking."""

    def __init__(self):
        # type: () -> None
        self._records = []  # type: List[Dict]
        self._by_hour = defaultdict(
            lambda: {"model_briers": [], "market_briers": [], "displacements": [], "count": 0}
        )

    def compute_brier(self, probs, settled):
        # type: (List[float], List[int]) -> float
        """Brier score: sum of (predicted - actual)^2."""
        return sum((p - o) ** 2 for p, o in zip(probs, settled))

    def compute_displacement(self, model_prob, market_ask):
        # type: (float, float) -> float
        """Displacement = model_prob - market_ask (positive = model higher)."""
        return model_prob - market_ask

    def record(self, event_date, trigger_time, model_probs, market_mids, settled_bracket, et_hour):
        # type: (date, datetime, Dict[Tuple, float], Dict[Tuple, float], Tuple, int) -> None
        """Record edge stats for one trigger point.

        model_probs and market_mids: {bracket: probability}
        settled_bracket: the bracket that actually settled YES.
        """
        # Sort brackets for consistent ordering
        brackets = sorted(set(model_probs.keys()) | set(market_mids.keys()))

        model_list = []  # type: List[float]
        market_list = []  # type: List[float]
        settled_list = []  # type: List[int]
        displacements = []  # type: List[float]

        for bracket in brackets:
            mp = model_probs.get(bracket, 0.0)
            mk = market_mids.get(bracket, 0.0)
            s = 1 if bracket == settled_bracket else 0
            model_list.append(mp)
            market_list.append(mk)
            settled_list.append(s)
            displacements.append(self.compute_displacement(mp, mk))

        model_brier = self.compute_brier(model_list, settled_list)
        market_brier = self.compute_brier(market_list, settled_list)

        record = {
            "event_date": event_date,
            "trigger_time": trigger_time,
            "model_brier": model_brier,
            "market_brier": market_brier,
            "edge": market_brier - model_brier,
            "et_hour": et_hour,
            "avg_displacement": sum(displacements) / len(displacements) if displacements else 0.0,
        }
        self._records.append(record)

        # Per-hour aggregation
        hour_data = self._by_hour[et_hour]
        hour_data["model_briers"].append(model_brier)
        hour_data["market_briers"].append(market_brier)
        hour_data["displacements"].extend(displacements)
        hour_data["count"] += 1

    def summary_by_hour(self):
        # type: () -> Dict[int, Dict]
        """Aggregate edge stats per ET hour."""
        result = {}  # type: Dict[int, Dict]
        for hour, data in self._by_hour.items():
            n = data["count"]
            model_brier = sum(data["model_briers"]) / n if n else 0.0
            market_brier = sum(data["market_briers"]) / n if n else 0.0
            avg_disp = sum(data["displacements"]) / len(data["displacements"]) if data["displacements"] else 0.0
            result[hour] = {
                "model_brier": model_brier,
                "market_brier": market_brier,
                "edge": market_brier - model_brier,
                "count": n,
                "avg_displacement": avg_disp,
            }
        return result

    def results(self):
        # type: () -> List[Dict]
        return self._records


# ── P&L Simulator ─────────────────────────────────────────────────────────

class PnLSimulator:
    """Layer 3: aggregate metrics, bootstrap CI, and regime splits."""

    def aggregate(self, trades, starting_capital):
        # type: (List[TradeRecord], float) -> Dict
        """Compute aggregate P&L metrics from a list of TradeRecords.

        All P&L values are in cents. Returns:
        total_pnl, total_pnl_dollars, total_return_pct, sharpe_ratio,
        max_drawdown, win_rate, avg_win, avg_loss, profit_factor,
        total_trades, total_fees, avg_capital_locked_hours
        """
        if not trades:
            return {
                "total_pnl": 0.0, "total_pnl_dollars": 0.0, "total_return_pct": 0.0,
                "sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
                "total_trades": 0, "total_fees": 0.0, "avg_capital_locked_hours": 0.0,
            }

        total_pnl = sum(t.pnl for t in trades)
        total_fees = sum(t.fees_paid for t in trades)
        total_trades = len(trades)

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]

        win_rate = len(wins) / total_trades if total_trades else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        # Daily P&L for Sharpe ratio
        daily_pnl = defaultdict(float)  # type: Dict
        for t in trades:
            daily_pnl[t.event_date] += t.pnl
        daily_returns = list(daily_pnl.values())

        mean_daily = np.mean(daily_returns) if daily_returns else 0.0
        std_daily = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = float(mean_daily / std_daily) if std_daily > 0 else 0.0

        # Max drawdown (cumulative P&L)
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in sorted(daily_pnl.items()):
            cumulative += pnl[1]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_drawdown:
                max_drawdown = dd

        avg_locked_hours = (
            sum(t.capital_locked_hours for t in trades) / total_trades
            if total_trades else 0.0
        )

        total_pnl_dollars = total_pnl / 100.0
        total_return_pct = (total_pnl_dollars / starting_capital * 100.0) if starting_capital else 0.0

        return {
            "total_pnl": total_pnl,
            "total_pnl_dollars": total_pnl_dollars,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "total_fees": total_fees,
            "avg_capital_locked_hours": avg_locked_hours,
        }

    def bootstrap(self, daily_pnls, n_iterations=10000, seed=None):
        # type: (List[Tuple[date, float]], int, Optional[int]) -> Dict
        """Bootstrap confidence intervals via resampling.

        Returns pnl_ci_95, sharpe_ci_95, prob_profitable.
        """
        if not daily_pnls:
            return {"pnl_ci_95": (0.0, 0.0), "sharpe_ci_95": (0.0, 0.0), "prob_profitable": 0.0}

        rng = np.random.RandomState(seed)
        pnl_values = np.array([p[1] for p in daily_pnls])
        n_days = len(pnl_values)

        boot_pnls = []  # type: List[float]
        boot_sharpes = []  # type: List[float]
        profitable_count = 0

        for _ in range(n_iterations):
            sample = rng.choice(pnl_values, size=n_days, replace=True)
            total = float(np.sum(sample))
            boot_pnls.append(total)
            if total > 0:
                profitable_count += 1
            mean_s = float(np.mean(sample))
            std_s = float(np.std(sample, ddof=1)) if n_days > 1 else 0.0
            sharpe = mean_s / std_s if std_s > 0 else 0.0
            boot_sharpes.append(sharpe)

        pnl_ci = (float(np.percentile(boot_pnls, 2.5)), float(np.percentile(boot_pnls, 97.5)))
        sharpe_ci = (float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5)))
        prob_profitable = profitable_count / n_iterations

        return {
            "pnl_ci_95": pnl_ci,
            "sharpe_ci_95": sharpe_ci,
            "prob_profitable": prob_profitable,
        }

    def regime_split(self, trades, regime_fn):
        # type: (List[TradeRecord], ...) -> Dict[str, Dict]
        """Group trades by regime_fn(trade) -> label.

        Returns {label: {total_trades, total_pnl, win_rate}}.
        """
        groups = defaultdict(list)  # type: Dict[str, List[TradeRecord]]
        for t in trades:
            label = regime_fn(t)
            groups[label].append(t)

        result = {}  # type: Dict[str, Dict]
        for label, group_trades in groups.items():
            wins = [t for t in group_trades if t.pnl > 0]
            result[label] = {
                "total_trades": len(group_trades),
                "total_pnl": sum(t.pnl for t in group_trades),
                "win_rate": len(wins) / len(group_trades) if group_trades else 0.0,
            }
        return result


# ── Strategy Backtester (Main Orchestrator) ──────────────────────────────

class StrategyBacktester:
    """Main orchestrator — three-layer unified pipeline.

    Iterates settlement dates, evaluates model at each trigger,
    computes edge, applies strategy filters, manages positions,
    settles at end of day, and produces the final report.
    """

    def __init__(self, db_path, config):
        # type: (str, BacktestConfig) -> None
        self.db_path = db_path
        self.config = config
        self.market_loader = MarketDataLoader(db_path)
        self.trigger_detector = TriggerDetector(
            db_path, config.station_id, config.execution_latency_seconds,
        )
        self.edge_analyzer = EdgeAnalyzer()
        self.sanity_filter = SanityFilter(config)
        self.pnl_simulator = PnLSimulator()

    def _get_settlement_dates(self):
        # type: () -> List[Tuple[date, float]]
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

    def _get_model_run_for_trigger(self, trigger_time, trigger_type, event_date):
        # type: (datetime, str, date) -> datetime
        """Determine which forecast model_run to use at this trigger time."""
        from services.data_provider import HRRR_AVAILABILITY_LAG_HOURS

        best_run = None  # type: Optional[datetime]
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

    def _get_et_hour(self, utc_time):
        # type: (datetime) -> int
        """Convert UTC time to ET hour."""
        et_time = utc_time.astimezone(_ET) if utc_time.tzinfo else utc_time.replace(tzinfo=timezone.utc).astimezone(_ET)
        return et_time.hour

    def run(self, model_fn):
        # type: (ModelFn) -> Dict
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

            settled_bracket = None  # type: Optional[Tuple]
            for bi in brackets_info:
                if bi["settled_yes"] == 1:
                    settled_bracket = bi["bracket"]
                    break

            if settled_bracket is None:
                continue

            # Market window: prior day 10 AM ET -> settlement day 11 PM ET
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

                # -- Layer 1: Edge Analysis (always record, even during burn-in)
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

                # -- Layer 2: Strategy Engine --

                # Update obs for post-peak detection
                if trigger_type == 'observation':
                    obs = provider.get_observations_in_range(
                        config.station_id,
                        model_time - timedelta(minutes=5),
                        model_time + timedelta(minutes=5),
                    )
                    for obs_row in obs:
                        day_obs.append((model_time, obs_row[1]))

                is_post_peak = SanityFilter.is_post_peak_check(day_obs, model_time)
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
                candidates = []  # type: List[Tuple[Tuple, float, float]]
                for bi in brackets_info:
                    bracket = bi["bracket"]
                    mp = model_probs.get(bracket, 0.0)
                    ask = market_asks.get(bracket)
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
                            entry_time=execution_time,
                            quantity=config.fixed_bet_size,
                        )
                        if pos is None:
                            missed_capital += 1

            # -- Layer 3: Settlement --
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
