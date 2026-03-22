# Strategy Backtest Comparison Runner

**Date:** 2026-03-22
**Status:** Draft
**Goal:** Build a backtest runner that exactly mirrors the live trading engine, then compare 12 strategy configurations to find the best betting approach.

## Problem Statement

The current live engine (`StrategyEngine`) enters every bracket that independently clears `min_edge_pct`, which produces incoherent positions (e.g., buying YES on non-adjacent brackets like <=62 AND [66-67)). We need to backtest alternative signal filtering strategies to find the most profitable approach.

The existing `strategy_backtester.py` has critical divergences from the live engine (fixed 1-contract sizing vs half-Kelly, different portfolio selection logic, wrong settlement boundaries) and cannot be trusted for this comparison. This spec describes a new, clean runner built from scratch that reuses the live engine's actual components.

## Architecture

### Module

`scripts/strategy_comparison.py` — a standalone script, not a service.

### Component Reuse

The runner imports and calls the same classes the live engine uses:

| Component | Source | What it provides |
|-----------|--------|-----------------|
| `FeatureBuilder` | `services/feature_builder.py` | `get_training_data()`, `build_features()` |
| `QRModel` | `services/model.py` | `fit()`, `predict_bracket_probs()` |
| Fee formula | `services/paper_trader.py` | `_compute_fee()` logic (reimplemented as standalone function) |

### What the runner owns

- Walk-forward simulation loop
- Signal filtering (strategy configurations)
- Half-Kelly position sizing
- Bankroll tracking (live, per-strategy)
- Candlestick price lookups
- Settlement resolution
- P&L ledger and reporting
- Quick mode (sample) vs full mode

## Walk-Forward Simulation Loop

For each event date in the backtest window:

```
for each event_date in backtest_dates:
    bankroll = starting_capital + realized_pnl_so_far
    held_positions = []

    for update_hour in 0..23:
        # 1. Train model (same FeatureBuilder, 180-day window)
        result = feature_builder.get_training_data(event_date, update_hour)
        if result is None: continue
        X, y, dates, run_hour = result

        # 2. Fit QR model
        model.fit(X, y, run_hour=run_hour, date_key=date_key)

        # 3. Build live features for this hour
        feat_result = feature_builder.build_features(event_date, update_hour)
        if feat_result is None: continue
        features, fcst_high, rh = feat_result

        # 4. Predict bracket probabilities
        running_max = get_running_max(event_date, update_hour)
        probs = model.predict_bracket_probs(features, fcst_high, rh, date_key, running_max)

        # 5. Get market prices from candlesticks
        prices = get_candlestick_prices(event_date, update_hour)
        if not prices: continue

        # 6. Generate raw signals (same edge/fee math as live)
        raw_signals = generate_signals(probs, prices, config)

        # 7. Apply strategy filter <-- THIS VARIES PER STRATEGY
        filtered = strategy.filter(raw_signals, probs, model_median)

        # 8. Size with half-Kelly (same as live)
        sized = kelly_size(filtered, bankroll, config)

        # 9. Dedup against positions already held today
        new_entries = dedup(sized, held_positions)

        # 10. Record entries
        held_positions.extend(new_entries)

    # End of day: settle all positions
    settlement_temp = get_settlement_temp(event_date)
    realize_pnl(held_positions, settlement_temp)
```

### Timezone Convention

The `update_hour` loop (0..23) is in **Eastern Time (ET)**, matching the live engine (`strategy_engine.py` line 84: `update_hour = now_et.hour`). Candlestick queries must convert ET to UTC for timestamp comparison. `FeatureBuilder.build_features()` uses ET internally.

### Computing `model_median`

`QRModel.predict_bracket_probs()` returns `Dict[int, float]` (1°F bracket probabilities) but does not expose the median temperature. Compute it from the returned probabilities:

```python
def compute_model_median(bracket_probs):
    """Compute median temp from 1°F bracket probabilities."""
    sorted_brackets = sorted(bracket_probs.items())
    cumulative = 0.0
    for temp, prob in sorted_brackets:
        cumulative += prob
        if cumulative >= 0.5:
            return float(temp)
    return float(sorted_brackets[-1][0])  # fallback
```

This is passed to `SignalFilter.filter()` for adjacency-based strategies.

### Computing `running_max` in Backtest Context

The live engine queries `MAX(temp_f)` across all observations for the date because it only runs at the current time. In the backtest, future observations exist in the DB for past dates. Filter to observations at or before the simulated hour:

```python
def get_running_max(con, event_date, update_hour):
    """Max observed temp up to update_hour ET on event_date."""
    row = con.execute("""
        SELECT MAX(temp_f) FROM observations
        WHERE station_id = 'KNYC' AND observed_at::DATE = ?
          AND EXTRACT(HOUR FROM observed_at AT TIME ZONE 'UTC' AT TIME ZONE 'EST') <= ?
          AND temp_f IS NOT NULL
    """, [event_date, update_hour]).fetchone()
    return row[0] if row and row[0] is not None else None
```

### Optimization: Model Caching

The training data only changes when `run_hour` changes (because `get_training_data` filters all historical dates by the selected run_hour). Between run_hour changes, skip `get_training_data` + `model.fit` and reuse the cached model coefficients.

**Important:** `build_features()` and `predict_bracket_probs()` must still run every simulated hour, because the live feature vector changes (different `update_hour` → different divergence features, different `running_max`). Only the training/fitting step is cached.

This reduces ~24 model fits per day to ~2-4 (when run_hour actually shifts), cutting compute significantly.

## Data Sources

| Data | Table | Range | Resolution |
|------|-------|-------|-----------|
| HRRR forecasts | `forecasts` (model_name='hrrr') | 2020-12-12 → present | Hourly runs (0-23z) |
| GFS forecasts | `forecasts` (model_name='gfs') | 2021-01-01 → present | 00z |
| ECMWF forecasts | `forecasts` (model_name='ecmwf') | 2021-03-23 → present | 00z |
| Extended features | `forecast_extended` | Same as above | ECMWF/GFS 00z, hours 10-22 ET |
| Observations | `observations` (station_id='KNYC') | Full history | Sub-hourly |
| Market prices | `kalshi_candlesticks` | 2021-08-05 → 2026-02-23 | 1-minute candles |
| Market prices (recent) | `market_ticks` | 2026-02-23 → present | Tick-level |
| Settlement truth | `kalshi_settlements` | 2021-08-06 → 2026-02-23 | Daily |
| NWS actuals | `nws_daily` | Full history | Daily |

### Backtest Date Range

- **Full mode:** All dates with both candlestick data AND settlement truth: 2021-08-06 → 2026-02-23 (~1,659 days)
- **Quick mode:** 100 season-stratified random days from the same range (25 per season: DJF=Dec/Jan/Feb, MAM=Mar/Apr/May, JJA=Jun/Jul/Aug, SON=Sep/Oct/Nov)

### Market Price Lookup

For a given `(event_date, update_hour)`, query `kalshi_candlesticks`:

```sql
SELECT market_ticker, yes_bid_close, yes_ask_close
FROM kalshi_candlesticks
WHERE market_ticker LIKE 'KXHIGHNY-{event_ticker}-%'
  AND end_period_ts <= {event_date at update_hour UTC}
ORDER BY end_period_ts DESC
LIMIT 1 PER market_ticker
```

Parse `floor_strike` and `cap_strike` from the ticker string. Return prices in cents.

**NO-side price derivation:** The candlestick table only stores `yes_bid_close` and `yes_ask_close`. Derive NO prices from the binary contract identity:
- `no_ask = 100 - yes_bid` (to buy NO, you sell YES at the bid)
- `no_bid = 100 - yes_ask` (to sell NO, you buy YES at the ask)

Getting this backwards would silently invert all NO-side edge calculations.

### Settlement Resolution

Use `kalshi_settlements.settled_yes` as ground truth (what Kalshi actually paid). This avoids any bracket boundary logic disagreements — we use Kalshi's own answer.

Fallback for dates not in `kalshi_settlements`: apply bracket logic against `nws_daily.max_temp_f`:

```python
if floor is None:    return actual < cap        # lower tail
elif cap is None:    return actual > floor       # upper tail
else:                return floor <= actual <= cap  # interior (both inclusive)
```

NWS CLI always reports integer degrees, so boundary ambiguity cannot occur.

## Signal Generation

Mirrors `StrategyEngine._generate_signals()` exactly:

1. Aggregate 1°F model probs to Kalshi bracket format:
   - Lower tail `(None, cap)`: sum probs where `k < cap`
   - Upper tail `(floor, None)`: sum probs where `k > floor`
   - Interior `(floor, cap)`: sum probs where `floor <= k <= cap`

2. For each bracket, check YES edge first:
   - `yes_edge = (model_prob - yes_ask/100) * 100`
   - Pass if `yes_edge >= min_edge_pct` AND `model_prob >= min_model_prob`
   - Fee filter (per-contract, for threshold check only): `fee_cents = max(ceil(round(0.07 * P * (1-P) * 100, 10)), 1)`
   - EV filter: `yes_edge - fee_cents >= min_ev_cents`
   - Note: The full position fee (with `contracts` multiplier) is computed later for P&L — see Fee Model section

3. If YES fails, check NO edge:
   - `no_prob = 1.0 - model_prob`
   - Same filters applied to NO side

### Configuration Constants (matching live engine)

| Parameter | Value | Source |
|-----------|-------|--------|
| `min_edge_pct` | 5.0% | `paper_config` table default |
| `min_ev_cents` | 2 | `paper_config` table default |
| `min_model_prob` | 0.15 | `paper_config` table default |
| `starting_capital` | $100.00 | `paper_config` table default |
| `max_per_bracket` | 10 contracts | `paper_config` table default |

## Position Sizing: Half-Kelly

Mirrors `StrategyEngine._compute_contracts()`:

```python
def kelly_size(edge_decimal, price_cents, bankroll, max_per_bracket):
    half_kelly = edge_decimal / 2.0
    raw = half_kelly * bankroll / (price_cents / 100.0)
    contracts = max(1, min(int(raw), max_per_bracket))
    return contracts
```

- `edge_decimal` = model_prob - price for the traded side
- Bankroll updates after each settled day (starting_capital + cumulative realized P&L)
- Capped at `max_per_bracket` (10) contracts
- If bankroll <= 0 after a bad streak, Kelly clamps to floor (1 contract)

## Fee Model

Mirrors `PaperTrader._compute_fee()`:

```python
def compute_fee(contracts, price_cents):
    p = price_cents / 100.0
    raw = 0.07 * contracts * p * (1.0 - p)
    fee = max(math.ceil(round(raw * 100, 10)) / 100.0, contracts * 0.01)
    return round(fee, 2)  # dollars
```

- Entry fee charged at position open
- No settlement fee
- Exit fee charged only on early exits (edge reversals) — not modeled in this backtest since we hold to settlement

## P&L Calculation

All positions hold to settlement (no intra-day exit modeling for the strategy comparison):

```python
if settled_yes and direction == 'yes':
    gross = (100 - entry_price_cents) * contracts / 100.0   # won
elif not settled_yes and direction == 'no':
    gross = (100 - entry_price_cents) * contracts / 100.0   # won (NO side)
else:
    gross = -(entry_price_cents * contracts / 100.0)         # lost

net = gross - entry_fee
```

## Strategy Configurations

12 configurations. All share the same model, features, edge calculation, Kelly sizing, and fee model. They differ ONLY in which signals pass the filter at step 7.

### SignalFilter Interface

```python
class SignalFilter:
    def filter(self, signals, bracket_probs, model_median):
        """
        Args:
            signals: list of signal dicts with keys:
                floor, cap, direction, model_prob, market_price,
                edge_pct, ev_cents, contracts
            bracket_probs: dict {int: float} — 1°F model probabilities
            model_median: float — median temperature from QR model
        Returns:
            filtered list of signal dicts
        """
```

### The 12 Strategies

| # | Name | Logic |
|---|------|-------|
| 1 | `baseline` | No filter — pass all signals (current live behavior) |
| 2 | `adjacency_2` | Only brackets where `floor` or `cap` is within 2°F of `model_median` |
| 3 | `adjacency_4` | Within 4°F |
| 4 | `adjacency_6` | Within 6°F |
| 5 | `adjacency_8` | Within 8°F |
| 6 | `top_k_1` | Keep only the 1 signal with highest `edge_pct` |
| 7 | `top_k_2` | Top 2 by `edge_pct` |
| 8 | `top_k_3` | Top 3 |
| 9 | `top_k_5` | Top 5 |
| 10 | `portfolio_ev` | Mutually-exclusive EV optimizer: accounts for the fact that if bracket J wins, all others lose. Greedy addition — add signal if portfolio EV improves. Uses `EventPortfolio` math from old backtester. |
| 11 | `hybrid_4_2` | Adjacency(4) then Top-K(2) |
| 12 | `hybrid_6_3` | Adjacency(6) then Top-K(3) |

### Adjacency Filter Detail

```python
def adjacency_filter(signals, model_median, max_distance):
    return [s for s in signals
            if bracket_distance(s['floor'], s['cap'], model_median) <= max_distance]

def bracket_distance(floor, cap, median):
    if floor is None: return max(0, median - cap)     # lower tail
    if cap is None:   return max(0, floor - median)    # upper tail
    # interior: distance from median to nearest bracket edge
    if floor <= median <= cap: return 0
    return min(abs(median - floor), abs(median - cap))
```

### Dedup Rule

Per the live engine: track `{(floor, cap, direction)}` for all positions entered on this event_date. Skip any signal matching an already-held key. This prevents re-entering the same bracket at different hours.

## Output

### Quick Mode (100 sample days)

Print a ranked comparison table to stdout:

```
Strategy Comparison (100 sample days, season-stratified)
═══════════════════════════════════════════════════════════
Strategy         P&L($)  ROI(%)  Trades  WinRate  MaxDD  Sharpe  PF
─────────────────────────────────────────────────────────────────────
top_k_2          +14.30   14.3%     187   58.3%   -8.20   1.42  1.85
hybrid_4_2       +12.80   12.8%     156   61.2%   -6.10   1.55  1.92
adjacency_4       +9.50    9.5%     203   55.7%   -9.40   1.18  1.61
baseline          +3.20    3.2%     412   51.4%  -14.80   0.38  1.12
...
```

Also save detailed per-trade ledger to `data/backtest_results/quick_{timestamp}.json`.

### Full Mode (all 1,659 days)

Same table plus:
- 1000 bootstrap iterations for 95% confidence intervals on P&L and Sharpe
- P&L by season (DJF, MAM, JJA, SON)
- Cumulative P&L curve data (for plotting)
- Max drawdown duration
- Monthly P&L breakdown

Save to `data/backtest_results/full_{timestamp}.json`.

### Metrics

| Metric | Formula |
|--------|---------|
| Total P&L | Sum of net P&L across all trades |
| ROI | Total P&L / starting_capital * 100 |
| Win Rate | Winning trades / total trades |
| Avg Edge | Mean `edge_pct` at entry across all trades |
| Trade Count | Total positions entered |
| Max Drawdown | Largest peak-to-trough in cumulative P&L |
| Sharpe Ratio | Mean daily P&L / StdDev daily P&L * sqrt(252) |
| Profit Factor | Gross winning P&L / abs(Gross losing P&L) |

### Disqualification

Any strategy with max drawdown > $20 (20% of starting capital) is flagged as DISQUALIFIED in the output.

## Bug Fixes (separate commits)

These are fixed in the existing codebase alongside the new runner:

### Fix 1: `paper_trader.py` settlement boundary

Current (line ~221-225): interior bracket uses `floor <= actual < cap` (cap exclusive).
Fix: change to `floor <= actual <= cap` (both inclusive), matching Kalshi's CFTC filing.

### Fix 2: `services/settlement.py` settlement boundary

Same bug at line ~125. Also missing tail bracket handling (`floor is None` / `cap is None` cases), which would crash with `TypeError`. Apply same inclusive logic with proper tail handling.

## CLI Interface

```bash
# Quick mode (100 sample days, local)
python scripts/strategy_comparison.py --quick

# Full mode (all days, for Mac Mini)
python scripts/strategy_comparison.py --full

# Specific strategies only
python scripts/strategy_comparison.py --quick --strategies baseline,top_k_2,hybrid_4_2

# Custom sample size
python scripts/strategy_comparison.py --quick --sample-size 200

# Reproducible random seed
python scripts/strategy_comparison.py --quick --seed 42
```

## Performance Considerations

- **Model caching:** Only retrain when `run_hour` changes (~2-4 times/day, not 24)
- **Batch candlestick queries:** Pre-load all candlestick data for the event date in one query, index by hour
- **Strategy parallelism:** After signal generation (shared), run all 12 filters independently — could parallelize with multiprocessing
- **Quick mode first:** 100 days locally to get directional answers, full run on Mac Mini overnight
- **Progress bar:** `tqdm` with ETA for both modes

## Compute Estimate

- Quick mode (100 days): ~100 days x ~3 model fits/day x 7 quantile LPs x ~0.5s each = ~15-20 minutes locally
- Full mode (1,659 days): ~14x quick = ~4-5 hours on Mac Mini
- Strategy filtering adds negligible overhead (all 12 run on the same model output)
