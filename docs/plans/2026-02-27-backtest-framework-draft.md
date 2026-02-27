# AlphaTemp Kalshi Backtest Framework — Design Draft (for review)

## Project Context

AlphaTemp predicts daily high temperature settlement brackets for Kalshi KXHIGHNY markets (NYC, Central Park / KNYC). We have a 3-model ensemble (HRRR + GFS + ECMWF) with walk-forward adaptive weights that produces 1°F bracket probabilities, mapped to Kalshi's 2°F brackets. Current best Brier score: 0.7705 at 18 ET.

We need a backtest framework to:
1. **Optimize the model** — compare our probability accuracy vs Kalshi market implied probabilities by hour to find where we have edge
2. **Develop trading strategy** — determine entry thresholds, selective situations, and bet sizing rules net of fees
3. **Simulate P&L** — run practice simulations across historical data with bootstrap resampling and regime splits

## Available Data

### Model Side
- **Forecasts**: HRRR, GFS, ECMWF hourly temperature forecasts (backfilled via Open-Meteo)
- **Observations**: Real-time METAR from KNYC, KLGA, KEWR (Synoptic API + AWC)
- **Settlement truth**: NWS Daily Climate Report (CLI) for KNYC, stored in `nws_daily` table
- **Model output**: 1°F bracket probabilities via Gaussian mixture ensemble, evaluated at each ET hour using Phase 2B intraday mechanism (incorporates new obs each hour)

### Market Side
- **kalshi_settlements**: Every settled KXHIGHNY market — bracket definitions, settlement outcome, volume (~5K+ rows, Aug 2021 – Feb 2026)
- **kalshi_candlesticks**: 1-minute OHLCV price history with bid/ask — yes_bid_close, yes_ask_close, price OHLC, volume, open interest (~500K+ rows)
- **kalshi_trades**: Individual trade records — price, count, taker side, timestamp (~2M+ rows)
- **market_ticks**: Live polling snapshots (60-sec intervals) — bid/ask/last/volume per bracket

### Key Data Characteristics
- **Liquidity inflection: November 2024** — Volume jumped 18x (668K → 11.9M/month), spreads collapsed from 31¢ to 8¢. Pre-Nov 2024 data represents a fundamentally different (illiquid) market.
- **Current spreads**: Sub-3¢ and tightening (Feb 2026: 1.82¢ avg)
- **24-hour trading**: Candlesticks exist at every UTC hour. Market opens 10 AM ET the prior day.
- **Volume profile**: Peak at 10-11 AM ET (45 avg/candle), trough at 10 PM-1 AM ET (6-9 avg/candle)

## Architecture: Three-Layer Unified Pipeline

One module (`services/strategy_backtester.py`) with three composable layers:

```
Layer 1: Edge Analysis        → "Where does my model beat the market?"
Layer 2: Strategy Engine       → "Given edge, when and what do I trade?"
Layer 3: P&L Simulator         → "What would my returns look like?"
```

Each layer consumes the output of the one above it. Can run Layer 1 standalone for model optimization, or all three for full simulation.

## Layer 0: Data Foundation

### Time Window
- **Primary backtest period**: November 2024 → present (~16 months of liquid data)
- **Configurable**: can extend to earlier data for robustness checks, but calibration/strategy tuning uses liquid era only

### Market Price Reconstruction
- For each `(event_date, bracket, minute_timestamp)`, pull Kalshi midpoint from `kalshi_candlesticks`
- **Entry price assumption**: Midpoint of bid/ask (conservative enough to be useful, can stress-test by shifting to ask-side later)
- Covers full market window: **prior-day 10 AM ET through settlement** (~28+ hours)
- When no candle exists for a minute, forward-fill from last available price with `is_stale_price = True` flag
- Spread (ask - bid) tracked per minute for liquidity filtering

### Model Probability Reconstruction
- Evaluate model at **each ET hour** using Phase 2B intraday mechanism (incorporates all observations available up to that hour)
- Each hourly model output holds constant for 60 minutes until next update
- On the prior day (10 AM ET → midnight), model uses previous day's forecast runs (lower quality but market is thinner)
- On settlement day (midnight → settlement), model uses day-of forecast runs + accumulating observations
- Output: 1°F bracket probabilities mapped to Kalshi 2°F brackets

### Joined Output Per Minute
```python
TimeStep = {
    "event_date": date,           # settlement date
    "bracket": (floor, cap),      # Kalshi 2°F bracket
    "minute_ts": datetime,        # UTC timestamp
    "model_prob": float,          # our probability (updates hourly)
    "market_mid": float,          # Kalshi midpoint (updates per minute)
    "market_bid": float,          # for conservative fill estimate
    "market_ask": float,          # for spread/liquidity check
    "displacement": float,        # model_prob - market_mid
    "spread": float,              # ask - bid (cents)
    "volume": int,                # candle volume
    "is_stale_price": bool,       # True if forward-filled (no trade this minute)
    "settled_yes": int,           # ground truth (1 or 0)
}
```

### Walk-Forward Discipline (Critical)
Two independent walk-forward layers to prevent overfitting:

1. **Model walk-forward** (already built): Expanding-window bias/regression. Each day's prediction uses only data before that day. Per-run-hour bias estimation. Minimum 90 days of paired data before any prediction.

2. **Strategy walk-forward** (new): Displacement thresholds, edge filters, and any tuned parameters are learned from an expanding window of *prior strategy outcomes*. The strategy never sees future performance when setting current parameters.

- **Burn-in period**: ~90 days of liquid data (Nov 2024 – Jan 2025) used to seed the strategy walk-forward. No P&L reported from this period.
- **Evaluation period**: Feb 2025 onward — this is where P&L results are measured.

### Anomaly Detection
- **Market-shift detector**: Flag minutes where market midpoint moves >10% within a 30-min window while model probability is stable. These are "market knows something we don't" events — logged for manual review to identify missing data sources.
- **Rounding edge tracker**: For settlement temps on bracket boundaries (temp == floor_strike or temp == cap_strike), track whether the market priced the correct bracket. Identify systematic rounding errors as a potential edge source.
- **Stale price windows**: Flag stretches where the same price persists >30 min with no trades — potential entry opportunities if our model disagrees.

## Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entry price | Midpoint of bid/ask | Standard backtesting assumption; can stress-test at ask-side later |
| Trade direction | Buy YES only (NO trades deferred) | Validate core strategy first; model calibrated for YES direction |
| Bet sizing | Fixed size initially, Kelly criterion later | Prove edge exists before optimizing position sizing |
| Time granularity | Hourly model checkpoints, 1-min market scanning | Model updates are real (not interpolated); market scanning finds optimal entry within each model window |
| Backtest window | Nov 2024+ (liquid era only) | Pre-Nov data is fundamentally different market (18x volume gap, 10x spread gap) |
| Overfitting protection | Walk-forward at both model AND strategy layers | Only 16 months of data; can't afford static train/test split |
| P&L simulation | Full history + bootstrap resampling + regime splits | Full history for baseline, bootstrap for confidence intervals, regime splits for conditional analysis |
| Multi-bracket | Supported | Per-bracket evaluation; can buy 2+ brackets per day if edge exists on each independently |
| Market hours | Full window: prior-day 10 AM ET → settlement | Leave no stone unturned; overnight/early AM may have most edge due to thin crowd |

## Fee Structure (Hard Constraint)
- Trading fee: 1% of contract value
- Settlement fee: 10% of profit (only on winning trades)
- Withdrawal fee: 2%
- **Effective hurdle rate: ~11%** — displacement must exceed this to be profitable in expectation
- ALL strategy evaluations MUST be net of fees — no exceptions

## Layers 1-3: Design TBD
Edge analysis, strategy engine, and P&L simulator designs are in progress. This document covers the data foundation only.

## Open Questions for Review
1. Is the walk-forward approach for strategy parameters the right choice given only 16 months of liquid data, or would a different cross-validation scheme be more robust?
2. Is hourly model granularity sufficient, or should we consider sub-hourly model updates (e.g., re-evaluating whenever a new observation arrives)?
3. For the anomaly detector, what threshold should trigger a "market knows something" flag? We proposed >10% move in 30 min — is this appropriate for weather prediction markets?
4. Should we account for market microstructure effects (e.g., bid/ask bounce, adverse selection from informed traders) in the entry price assumption?
5. Is the 90-day burn-in period for strategy walk-forward appropriate, or should it be longer/shorter?
6. For bracket boundary rounding: Kalshi brackets appear to use inclusive bounds on both sides (floor <= temp <= cap). This means a temp of exactly 72°F would settle YES for BOTH the [71,72] and [72,73] brackets — is that correct, or does one side take priority?
