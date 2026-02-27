# AlphaTemp Kalshi Strategy Backtester — Final Design

## Project Context

AlphaTemp predicts daily high temperature settlement brackets for Kalshi KXHIGHNY markets (NYC, Central Park / KNYC). We have a 3-model ensemble (HRRR + GFS + ECMWF) with walk-forward adaptive weights that produces 1°F bracket probabilities, mapped to Kalshi's 2°F brackets. Current best Brier score: 0.7705 at 18 ET.

### Goals
1. **Model optimization** — compare our Brier vs Kalshi's implied Brier by hour + displacement analysis
2. **Strategy development** — entry thresholds, selective situations, bet sizing, fee-adjusted EV
3. **P&L simulation** — full history backtest, bootstrap resampling, regime splits

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 services/strategy_backtester.py              │
│                                                             │
│  Layer 0: Data Foundation                                   │
│  ├── Market price reconstruction (1-min candlesticks)       │
│  ├── Model probability reconstruction (obs-triggered)       │
│  ├── Walk-forward discipline (model + strategy layers)      │
│  └── Anomaly detection (market shifts, rounding, staleness) │
│                                                             │
│  Layer 1: Edge Analysis                                     │
│  ├── Our Brier vs Market Brier by hour                      │
│  ├── Displacement distribution per bracket                  │
│  └── Edge heatmap by (hour, bracket position, season)       │
│                                                             │
│  Layer 2: Strategy Engine                                   │
│  ├── Sanity filters (5 pre-trade guard rails)               │
│  ├── Observation-triggered trade signals                    │
│  ├── Mutually exclusive portfolio EV (EventPortfolio)       │
│  ├── Position manager (capital lockup, early exits)         │
│  └── Walk-forward strategy parameter tuning                 │
│                                                             │
│  Layer 3: P&L Simulator                                     │
│  ├── Full history backtest                                  │
│  ├── Bootstrap resampling (confidence intervals)            │
│  ├── Regime splits (season, volatility, spread width)       │
│  └── Bankroll tracking with capital utilization             │
└─────────────────────────────────────────────────────────────┘
```

Each layer consumes the output of the one above it. Can run Layer 1 standalone for model optimization, or all three for full simulation.

---

## Layer 0: Data Foundation

### Time Window
- **Primary backtest period**: November 2024 → present (~16 months of liquid data)
- **Liquidity inflection**: Nov 2024 — volume jumped 18x (668K → 11.9M/month), spreads collapsed from 31¢ to 8¢
- **Current spreads**: Sub-3¢ (Feb 2026: 1.82¢ avg) — genuinely tradeable
- Pre-Nov 2024 data available for robustness checks but NOT used for calibration

### Market Price Reconstruction
- For each `(event_date, bracket, minute_ts)`, pull prices from `kalshi_candlesticks`
- **Entry price**: `yes_ask` (crossing the spread — conservative, no synthetic YES from NO book)
- **Exit price**: `yes_bid` (selling back to market)
- Covers full market window: **prior-day 10 AM ET through settlement** (~28+ hours)
- When no candle exists for a minute, forward-fill from last available price with `is_stale = True`
- 24-hour trading confirmed: candles exist at every UTC hour

### Model Probability Reconstruction — Observation-Triggered
Instead of fixed hourly checkpoints, the model re-evaluates when new data arrives:

```
Trigger types:
  1. New forecast run (00z/06z/12z/18z) → full model re-evaluation
  2. New observation timestamp          → divergence update only

For each settlement_date:
  - Query all distinct observed_at timestamps for KNYC (+ neighbors)
  - Each new obs = a model trigger (~24-30 per day for METAR, + SPECIs)
  - Between triggers, model probabilities hold constant
  - compute_divergence_features() already supports arbitrary ref_times
```

The base forecast (HRRR/GFS/ECMWF) only changes at run_hours. Between run_hours, only the divergence correction updates as new obs arrive. `BacktestDataProvider` enforces `observed_at <= ref_time` (walk-forward preserved).

### Walk-Forward Discipline (Critical — No Data Leakage)

Two independent walk-forward layers:

1. **Model walk-forward** (already built): Expanding-window bias/regression. Each day's prediction uses only data before that day. Per-run-hour bias estimation. Minimum 90 days of paired data.

2. **Strategy walk-forward** (new): Displacement thresholds, edge filters, and tuned parameters are learned from an expanding window of *prior strategy outcomes*. Strategy never sees future performance when setting current parameters.

- **Burn-in period**: ~90 days (Nov 2024 – Jan 2025) — seeds both walk-forward windows. No P&L reported.
- **Evaluation period**: Feb 2025 onward — P&L results measured here.

### Anomaly Detection
- **Market-shift detector**: Flag minutes where market_mid moves >10% within a 30-min window while model_prob is stable. Log as "market knows something we don't" for manual review.
- **Rounding edge tracker**: For boundary temps (temp == floor_strike or temp == cap_strike), track market pricing accuracy. Identify systematic rounding errors.
- **Stale price windows**: Flag stretches where same price persists >30 min with no trades.

---

## Layer 1: Edge Analysis

### Our Brier vs Market Brier

For each `(event_date, ET_hour)`:
- **Model Brier**: `sum((model_prob_k - settled_k)^2)` across all brackets
- **Market Brier**: `sum((market_mid_k - settled_k)^2)` across all brackets
- **Edge**: `market_brier - model_brier` (positive = we're better)

Output: edge heatmap by `(ET_hour, month, bracket_position)` showing where and when the model outperforms the market.

### Displacement Analysis

For each `(event_date, bracket, trigger_time)`:
- **Displacement**: `model_prob - market_ask` (since we'd pay ask to enter)
- Track: displacement magnitude, sign, duration, and whether the trade would have been profitable
- Distribution of displacements by hour, season, bracket position (center vs tail)

### Key Questions Layer 1 Answers
- At which hours does our model beat the market?
- Is the edge larger overnight (thin crowd) or during the day?
- Which bracket positions (center, near-tail, far-tail) have the most edge?
- Does edge vary by season (summer vs winter)?
- How often does the market reprice aggressively while our model is stable (gap risk)?

---

## Layer 2: Strategy Engine

### Sanity Filters (Pre-Trade Guard Rails)

Five filters that kill obviously bad trades before they reach the EV calculator:

```python
class SanityFilter:
    # 1. Tail Bracket Minimum
    # Reject bets where model_prob < 5%.
    # Gaussian model is LEAST accurate in the tails.
    # Calibration error dominates any displacement signal.
    MIN_MODEL_PROB = 0.05

    # 2. Post-Peak Cutoff
    # After the daily high has likely occurred:
    #   - No new temperature data will change the outcome
    #   - Remaining uncertainty = rounding, NWS measurement
    #   - Our model has ZERO edge on rounding debates
    # Detection: last 2+ obs declining AND current time > 2 PM ET
    #            AND running max occurred >60 min ago
    # Effect: suppress new entries, allow exits only.

    # 3. Spread Too Wide
    # If ask - bid > threshold, displacement is eaten by execution.
    MAX_SPREAD_CENTS = 10

    # 4. Model Uncertainty Too High
    # If model std > threshold, probability distribution is flat
    # and displacement signals are noise.
    MAX_MODEL_STD = 3.5  # °F

    # 5. No Churn
    # Don't re-enter a bracket within N minutes of exiting.
    # Prevents fee-burning oscillation.
    MIN_REENTRY_MINUTES = 60
```

All thresholds are configurable via `BacktestConfig` and subject to walk-forward optimization.

### Trade Signal Generation

At each model trigger (new obs or new forecast run):

1. Compute updated bracket probabilities
2. For each bracket, compute displacement vs current market ask
3. Apply sanity filters — reject any bracket that fails
4. Compute fee-adjusted EV for passing brackets
5. Check existing positions — should any be exited?
6. Build `EventPortfolio` and compute optimal bracket subset

### Mutually Exclusive Portfolio EV

**Critical**: brackets on the same event_date are mutually exclusive — only one settles YES. Independent EV calculation is WRONG.

```python
class EventPortfolio:
    """Correct EV accounting for mutually exclusive outcomes."""

    def combined_ev(self, positions, model_probs):
        """
        For N held brackets with entry prices p_i and model probs q_i:

        If bracket j wins:
          gain_j = (100 - p_j) × (1 - settlement_fee) - trading_fee
          loss = sum(p_i + trading_fee for i != j)
          net_j = gain_j - loss

        If none win (prob = 1 - sum(q_i)):
          net = -sum(p_i + trading_fee)

        Portfolio EV = sum_j(q_j × net_j) + (1-sum(q_j)) × net_none
        """

    def optimal_subset(self, candidates, model_probs):
        """
        Greedy algorithm:
        1. Start with highest individual EV bracket
        2. For each remaining candidate, add it if portfolio EV increases
        3. Stop when no candidate improves portfolio EV

        Adding a 2nd bracket INCREASES loss in losing scenarios.
        The 2nd bracket must add enough win-probability to compensate.
        """
```

### Position Manager (Capital Lockup + Early Exits)

```python
class PositionManager:
    """Enforces bankroll constraint and manages position lifecycle."""

    bankroll: float          # total capital (e.g., $100)
    positions: List[Position]

    # ── Capital Tracking ──
    # Kalshi requires 100% upfront collateralization.
    # A bet at 22¢ locks $0.22 per contract until settlement (~28 hours).
    # capital_available = bankroll - sum(pos.capital_locked)

    def can_open(self, price_cents, qty) -> bool:
        """Reject if insufficient free capital."""

    def open_position(self, signal, qty) -> Optional[Position]:
        """Lock capital, record entry price (yes_ask) and time."""

    # ── Early Exit Logic ──
    # If model updates and our held bracket's probability collapses,
    # sell YES contracts back to free capital.
    #
    # Exit price: yes_bid (we're selling, so we hit the bid)
    # Only execute if the exit clears the fee math:
    #   exit_pnl = (exit_bid - entry_ask) × qty - trading_fees
    #   Must be > 0 (don't exit at a loss unless model says probability
    #   dropped below the loss-cut threshold)

    def should_exit(self, position, model_update, market) -> bool:
        """
        Exit conditions:
        1. Model prob for this bracket dropped below entry-implied breakeven
        2. Post-peak detected (no new data coming, exit to free capital)
        3. Profitable exit available (yes_bid > entry_ask + fees)
        """

    def close_position(self, position, exit_bid) -> float:
        """Sell at yes_bid. Returns realized P&L. Frees locked capital."""

    def settle_day(self, event_date, settled_bracket) -> float:
        """
        End-of-day settlement:
        - Winners: profit = (100 - entry) × (1 - 0.10) - trading_fee
        - Losers: loss = entry + trading_fee
        Returns net P&L. Frees all locked capital for this event.
        """
```

---

## Layer 3: P&L Simulator

### Full History Backtest
- Run strategy across all dates in evaluation period (Feb 2025+)
- Track daily P&L, cumulative P&L, capital utilization
- Record every trade: entry time, exit time/settlement, bracket, prices, P&L

### Bootstrap Resampling
- Resample daily P&L with replacement, 10,000 iterations
- Compute confidence intervals on: total return, Sharpe ratio, max drawdown, win rate
- Answer: "How confident are we that this strategy is profitable?"

### Regime Splits
- **Season**: summer (Jun-Aug) vs winter (Dec-Feb) vs shoulder (Mar-May, Sep-Nov)
- **Volatility**: high temp variance days vs low variance days
- **Spread**: tight spread days (<3¢) vs wide spread days (>5¢)
- **Model confidence**: high-certainty days (std < 2°F) vs uncertain days (std > 3°F)
- **Day of week**: weekday vs weekend (different Kalshi liquidity patterns?)

### Bankroll Tracking
- Starting capital: $100 (configurable)
- Track capital utilization rate (locked / total) over time
- Flag days where capital constraint prevented a trade (missed opportunity cost)
- Track peak drawdown and recovery time

### Output: BacktestReport

```python
@dataclass
class BacktestReport:
    # ── Aggregate Metrics ──
    total_pnl: float              # net of ALL fees
    total_return_pct: float       # pnl / starting_capital
    sharpe_ratio: float           # annualized
    max_drawdown: float           # peak-to-trough
    max_drawdown_duration: int    # days
    win_rate: float               # winning trades / total trades
    avg_win: float                # avg P&L on winners
    avg_loss: float               # avg P&L on losers
    profit_factor: float          # gross_wins / gross_losses
    total_trades: int
    avg_trades_per_day: float
    avg_capital_utilization: float

    # ── Bootstrap Confidence ──
    pnl_ci_95: Tuple[float, float]         # 95% CI on total P&L
    sharpe_ci_95: Tuple[float, float]       # 95% CI on Sharpe
    prob_profitable: float                  # % of bootstrap runs with positive P&L

    # ── Edge Analysis (Layer 1) ──
    brier_comparison_by_hour: Dict[int, Tuple[float, float]]  # hour → (our_brier, market_brier)
    displacement_distribution: Dict          # stats on displacement magnitudes
    edge_heatmap: Dict                       # (hour, season) → avg_edge

    # ── Regime Analysis ──
    regime_splits: Dict[str, BacktestMetrics]  # regime_name → metrics

    # ── Anomaly Log ──
    market_shift_events: List[Dict]    # times market moved, model didn't
    rounding_edge_cases: List[Dict]    # boundary temp events
    missed_due_to_capital: List[Dict]  # signals rejected for capital constraint

    # ── Trade Log ──
    trades: List[TradeRecord]          # every individual trade with full detail
```

---

## Class Architecture

```python
# ─── Configuration ─────────────────────────────────────────

@dataclass
class BacktestConfig:
    starting_capital: float = 100.0
    start_date: date = date(2024, 11, 1)
    end_date: Optional[date] = None        # None = latest available
    burn_in_days: int = 90
    fixed_bet_size: int = 1                 # contracts per signal
    min_displacement: float = 0.12          # > fee hurdle
    min_model_prob: float = 0.05
    max_spread_cents: float = 10.0
    max_model_std: float = 3.5
    min_reentry_minutes: int = 60
    market_shift_threshold: float = 0.10    # 10% move = anomaly
    stale_price_minutes: int = 30
    model_name: str = 'ensemble'            # or 'hrrr', 'gfs', 'ecmwf'
    bootstrap_iterations: int = 10000

# ─── Data Structures ──────────────────────────────────────

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
    trigger_type: str                       # 'forecast_run' | 'observation'
    model_std: float                        # for sanity filter

@dataclass
class Position:
    event_date: date
    bracket: Tuple[Optional[float], Optional[float]]
    entry_price: float                      # yes_ask at entry (cents)
    entry_time: datetime
    quantity: int
    capital_locked: float                   # entry_price × quantity

@dataclass
class TradeRecord:
    event_date: date
    bracket: Tuple[Optional[float], Optional[float]]
    direction: str                          # 'BUY_YES' (NO deferred)
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]             # yes_bid if early exit, None if settled
    exit_time: Optional[datetime]
    settlement_result: Optional[int]        # 1=YES, 0=NO
    pnl: float                              # net of fees
    fees_paid: float
    displacement_at_entry: float
    model_prob_at_entry: float
    capital_locked_hours: float             # duration of capital lockup

# ─── Core Classes ──────────────────────────────────────────

class SanityFilter:
    """Pre-trade guard rails."""
    def check(self, signal, context) -> Tuple[bool, str]
    def is_post_peak(self, obs_temps, current_time) -> bool

class EventPortfolio:
    """Mutually exclusive bracket EV calculation."""
    def combined_ev(self, positions, model_probs) -> float
    def optimal_subset(self, candidates, model_probs) -> List

class PositionManager:
    """Capital lockup, early exits, settlement."""
    def can_open(self, price, qty) -> bool
    def open_position(self, signal, qty) -> Optional[Position]
    def should_exit(self, position, model_update, market) -> bool
    def close_position(self, position, exit_bid) -> float
    def settle_day(self, event_date, settled_bracket) -> float

class EdgeAnalyzer:
    """Layer 1: Brier comparison + displacement analysis."""
    def compute_edge(self, model_update, market_state) -> Dict
    def brier_vs_market(self, date_range) -> Dict

class StrategyEngine:
    """Layer 2: Signal generation + portfolio optimization."""
    def generate_signals(self, model_update, market, filters) -> List
    def evaluate_portfolio(self, signals, existing_positions) -> List

class PnLSimulator:
    """Layer 3: Full simulation + bootstrap + regime splits."""
    def run_full_history(self, config) -> BacktestReport
    def bootstrap(self, daily_pnls, n_iterations) -> ConfidenceIntervals
    def regime_split(self, trades, regime_fn) -> Dict[str, BacktestMetrics]

class StrategyBacktester:
    """Main orchestrator — composes all layers."""
    def run(self, model_fn, config) -> BacktestReport
```

---

## Main Simulation Loop

```python
class StrategyBacktester:
    def run(self, model_fn: ModelFn, config: BacktestConfig) -> BacktestReport:
        position_mgr = PositionManager(bankroll=config.starting_capital)
        edge_analyzer = EdgeAnalyzer()
        strategy = StrategyEngine(config)
        filters = SanityFilter(config)
        trades = []
        daily_pnls = []

        for event_date in liquid_era_dates(config):
            brackets = get_kalshi_brackets(event_date)
            settled_bracket = get_settlement(event_date)

            # Get all trigger timestamps: obs arrivals + forecast runs
            triggers = get_trigger_timestamps(event_date)

            for trigger_time in triggers:
                # ── Layer 1: Edge Analysis ──
                model_update = evaluate_model(model_fn, trigger_time)
                market_state = get_market_at(brackets, trigger_time)
                edge_analyzer.record(model_update, market_state)

                # Skip during burn-in (still record edge for analysis)
                if event_date < config.start_date + burn_in_days:
                    continue

                # ── Layer 2: Strategy Engine ──

                # Check existing positions — should we exit any?
                for pos in position_mgr.positions_for(event_date):
                    if position_mgr.should_exit(pos, model_update, market_state):
                        pnl = position_mgr.close_position(pos, market_bid)
                        trades.append(make_trade_record(pos, 'early_exit', pnl))

                # Evaluate new entries
                candidates = strategy.generate_signals(
                    model_update, market_state, filters
                )
                portfolio = EventPortfolio(
                    event_date,
                    existing=position_mgr.positions_for(event_date),
                    candidates=candidates
                )
                optimal = portfolio.optimal_subset(candidates, model_update)

                for signal in optimal:
                    pos = position_mgr.open_position(signal, config.fixed_bet_size)
                    if pos is None:  # capital constraint
                        record_missed_opportunity(signal)

            # ── Layer 3: Settlement ──
            day_pnl = position_mgr.settle_day(event_date, settled_bracket)
            daily_pnls.append((event_date, day_pnl))

        # ── Post-Simulation Analysis ──
        report = PnLSimulator.build_report(
            trades=trades,
            daily_pnls=daily_pnls,
            edge_data=edge_analyzer.results(),
            config=config,
        )
        report.bootstrap_ci = PnLSimulator.bootstrap(daily_pnls, config.bootstrap_iterations)
        report.regime_splits = PnLSimulator.regime_split(trades, REGIME_FUNCTIONS)

        return report
```

---

## Fee Math Reference

All P&L calculations use these exact formulas:

```python
# ── Entry (Buy YES) ──
# Cost: yes_ask × quantity
# Trading fee: cost × 0.01
# Total capital locked: cost + trading_fee

# ── Early Exit (Sell YES) ──
# Revenue: yes_bid × quantity
# Trading fee: revenue × 0.01
# P&L: revenue - trading_fee - entry_cost - entry_trading_fee

# ── Settlement (YES wins) ──
# Gross profit: (100 - entry_price) × quantity
# Settlement fee: gross_profit × 0.10
# Trading fee: already paid at entry
# Net P&L: gross_profit - settlement_fee - entry_trading_fee

# ── Settlement (YES loses) ──
# Loss: entry_price × quantity + entry_trading_fee
# No settlement fee (nothing to settle)

# ── Minimum displacement for profitability ──
# For a bracket with model_prob q and entry_price p (cents):
#   EV = q × (100-p) × 0.90 - (1-q) × p - 0.01 × p
#   EV > 0 requires displacement > ~11% (varies by price level)
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entry price | `yes_ask` (cross the spread) | Conservative. Real execution sometimes finds cheaper fills via NO book — backtest P&L is a lower bound. |
| Exit price | `yes_bid` (hit the bid) | Conservative. Selling back into the market. |
| NO book synthetic | Deferred to live engine | Candlestick data lacks `no_bid`. Will implement in live execution where we have full order book. |
| Trade direction | Buy YES only | Validate core strategy first. NO trades added later. |
| Bet sizing | Fixed → Kelly | Prove edge with fixed sizing. Graduate to fractional Kelly once edge confirmed. |
| Model triggers | Observation-triggered (sub-hourly) | Fixed hourly checkpoints lose latency to market makers. Re-evaluate on each new obs timestamp. |
| Backtest window | Nov 2024+ (liquid era) | Pre-Nov market is fundamentally different (18x volume gap). |
| Overfitting protection | Walk-forward on both model AND strategy | 16 months of data — can't afford static train/test split. |
| Multi-bracket | Portfolio-level EV (mutually exclusive) | Independent bracket EV is mathematically wrong. Combined EV accounts for correlated losses. |
| Capital | 100% collateral, bankroll-constrained | Matches Kalshi's actual margin rules. Prevents simulating infinite capital. |
| Early exits | Supported | Frees locked capital. Only exits that clear fee hurdle or where model prob collapsed. |
| Post-peak | Suppress new entries after peak detected | Model has no edge on rounding debates. Switch to position management mode. |
| Tail bets | Reject model_prob < 5% | Gaussian model least accurate in tails. Calibration error dominates. |

---

## File Structure

```
services/
  strategy_backtester.py    # Main module — all classes above
  backtester.py             # Existing model backtester (unchanged)
  data_provider.py          # Existing DataProvider (unchanged)
  divergence.py             # Existing divergence features (unchanged)

scripts/
  run_strategy_backtest.py  # CLI entry point — configurable, runs report

docs/plans/
  2026-02-27-backtest-framework-design.md  # This document
```

Existing code is NOT modified. The strategy backtester imports from existing modules:
- `BacktestDataProvider` for walk-forward data access
- `compute_divergence_features()` for sub-hourly model updates
- `map_probs_to_kalshi_brackets()` for bracket mapping
- Model functions (ensemble, phase2b, etc.) via the `ModelFn` interface
