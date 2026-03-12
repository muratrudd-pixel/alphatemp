# Paper Trading System Design

**Date:** 2026-03-11
**Status:** Approved
**Branch:** `autoresearch/run-2026-03-10`

## Goal

Build a fully autonomous paper trading system that continuously evaluates edge in Kalshi KXHIGHNY temperature markets, places simulated trades, and tracks P&L — all net of fees, with circuit breakers and a dashboard kill switch.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Existing Services (unchanged)                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ forecast.py  │ │ bias.py      │ │ market_fetcher.py│ │
│  │ (HRRR/GFS/   │ │ (drift       │ │ (Kalshi prices   │ │
│  │  ECMWF ingest)│ │  signals)    │ │  every 60s)      │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────────┘ │
│         │                │                │             │
│         ▼                ▼                ▼             │
│  ┌─────────────────────────────────────────────────┐    │
│  │              DuckDB (alphatemp.duckdb)           │    │
│  └─────────────────────┬───────────────────────────┘    │
└────────────────────────┼────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  strategy_engine.py │  ← NEW (runs every 5 min)
              │                     │
              │  1. Check for new   │
              │     data since last │
              │     run             │
              │  2. Build 23        │
              │     features        │
              │  3. Fit QR model    │
              │     (walk-forward)  │
              │  4. Predict         │
              │     quantiles       │
              │  5. Build CDF →     │
              │     bracket probs   │
              │  6. Compare to      │
              │     market prices   │
              │  7. Apply circuit   │
              │     breakers        │
              │  8. Send decisions  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  paper_trader.py    │  ← ENHANCED
              │                     │
              │  - Enter/exit       │
              │  - Track P&L        │
              │  - Log to DB        │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  settlement.py      │  ← NEW
              │                     │
              │  - Auto-settle at   │
              │    ~4:30 PM EST     │
              │  - Final settle at  │
              │    ~1:30 AM EST     │
              └─────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Dashboard (8050)   │  ← ENHANCED
              │                     │
              │  - Live positions   │
              │  - P&L tracker      │
              │  - Circuit breakers │
              │  - Kill switch      │
              └─────────────────────┘
```

The strategy engine reads from the DB rather than subscribing to events. Stateless — if it crashes and restarts, it reads current state and picks up where it left off.

## Components

### 1. Strategy Engine (`services/strategy_engine.py`) — NEW

Async loop running every 5 minutes inside `main.py`.

**Each cycle:**

1. **Staleness check** — Query latest timestamps for HRRR, GFS, ECMWF, observations. Skip cycle if nothing new since last run.

2. **Feature construction** — Same 23 features as experiment.py, built from live DB state:
   - Forecast features: `fcst_high`, `diurnal_range`, `solar_rad`, `dp_depression`, `humidity`, `max_gusts`, `cape`, `total_precip`, `gfs_precip`
   - Cross-model spreads: `ecmwf_spread`, `gfs_spread`, `dp_spread`, `solar_spread`, `precip_agree`
   - Drift signals: `running_max_div`, `slope_div`, `cum_div` (from bias.py)
   - Error features: `lag_error`, `abs(lag_error)` (yesterday's forecast error vs NWS settlement)
   - Time features: `update_hour`, `sin_month`, `cos_month`
   - Binary: `rain_day`

3. **Model fit** — Walk-forward: fit QR on trailing 180 days, standardize features (zero mean, unit variance), solve 7 quantile LPs via HiGHS. Cache coefficients — only refit when training window changes.

4. **Predict & build CDF** — 7 quantile predictions → piecewise-linear CDF with exp tails (lambda_upper=0.29/spread, lambda_lower=0.36/spread, radius=18) → bracket probabilities for each 2°F Kalshi bracket.

5. **Edge calculation** — For each bracket:
   ```
   edge = model_prob - market_implied_prob
   ```
   Only act when `|edge| >= 0.05` (5% threshold).

6. **Decision output** — List of `(bracket, direction, edge, size)` tuples sent to paper_trader.

### 2. Model (`services/model.py`) — NEW

Extracted from `experiment.py`. Two public functions:

- `fit(training_data) → coefficients` — Standardize, solve 7 quantile LPs
- `predict(features, coefficients) → bracket_probabilities` — Forward predict, build CDF, compute bracket probs

Same 23 features, same standardization, same CDF construction. No model changes.

### 3. Paper Trader (`services/paper_trader.py`) — ENHANCED

**Circuit breakers** (pre-trade gate):

| Breaker | Default | Behavior |
|---|---|---|
| Max daily loss | -$10 | No new entries for rest of day |
| Max open positions | 5 | No new entries until one settles |
| Max per bracket | 2 contracts | No additional on that bracket |
| Min edge | 5% | Below threshold = no trade |
| Cooldown | 30 min | No re-entry on same bracket after exit |
| Kill switch | ON/OFF | Dashboard toggle, OFF = no new entries |

Thresholds stored in `paper_config` table, adjustable from dashboard.

**Position lifecycle:**

```
SIGNAL → VALIDATE → ENTER → MONITOR → EXIT/SETTLE
```

- **ENTER**: Record entry price, timestamp, direction (YES/NO), bracket, edge, model probs
- **MONITOR**: Each cycle, re-evaluate edge. If edge flips sign, mark for exit.
- **EXIT**: Edge reversal or settlement
- **SETTLE**: Compare bracket outcome to position direction. Record P&L net of fees.

**Schema:**

```sql
paper_positions (
  id            INTEGER PRIMARY KEY,
  market_date   DATE,
  bracket       VARCHAR,
  direction     VARCHAR,
  entry_price   DECIMAL,
  entry_time    TIMESTAMP,
  exit_price    DECIMAL,
  exit_time     TIMESTAMP,
  exit_reason   VARCHAR,
  quantity      INTEGER,
  model_prob    DECIMAL,
  market_prob   DECIMAL,
  edge          DECIMAL,
  pnl_cents     INTEGER,
  settled       BOOLEAN DEFAULT FALSE
)
```

P&L net of Kalshi taker fee: `max(ceil(0.07 * contracts * price * (1-price)), contracts * 1)`.

### 4. Settlement (`services/settlement.py`) — NEW

- Polls `nws_daily` table every 10 minutes between 4-6 PM EST for preliminary CLI
- When found: determine bracket, settle all positions for that market date
- Secondary check at 1-2 AM EST for final CLI — re-settle if temperature changed
- Uses existing `nws_fetcher.py` for data ingestion
- Unsettled positions flagged as `PENDING_SETTLEMENT` on dashboard

### 5. Dashboard (`ui/web_dashboard.py`) — ENHANCED

New **Trading** tab with three panels:

**Live Positions (top)**

| Bracket | Dir | Qty | Entry | Current | Edge | Unrealized | Time Held |
|---|---|---|---|---|---|---|---|
| 72-74 | YES | 2 | 34¢ | 41¢ | +8.2% | +$0.14 | 2h 15m |

Auto-refreshes with market_fetcher data. Green = profit, red = underwater.

**P&L Summary (middle)**
- Today's realized / unrealized / total
- Running cumulative P&L chart (daily resolution)
- Win rate, average edge at entry, average P&L per trade

**Circuit Breaker Status (bottom)**
- Each breaker: current value vs threshold
- Kill switch toggle
- Cooldown timers

Existing tabs unchanged.

## Daily Lifecycle

```
  6 AM ET ─── Strategy engine evaluating tomorrow's market
              (GFS 00z available, HRRR rolling in)
    │
  Hourly ──── Re-evaluate as new HRRR/obs/GFS/ECMWF arrive
              Enter/exit positions based on edge
    │
  4:30 PM ─── NWS preliminary CLI → auto-settle positions
    │
  1:30 AM ─── NWS final CLI → adjust if different
    │
  Next AM ─── Dashboard shows yesterday's results
```

## Observation Source

AWC (Aviation Weather Center) as primary — Synoptic API trial expired. AWC provides same KNYC METAR data, free, already integrated.

## Design Constraints

- Same 23 features, same QR model, same CDF — no model changes
- Tomorrow's market only (today's market added later)
- Fully autonomous with circuit breakers
- All P&L net of simulated Kalshi fees
- Unrealized P&L marked to live market prices
- $100 starting capital
