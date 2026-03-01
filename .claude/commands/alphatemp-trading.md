---
name: alphatemp-trading
description: Domain knowledge for Kalshi weather prediction market trading and the alphatemp system's trading model. Use PROACTIVELY whenever working with trade logic, probability calculations, fee math, Kelly sizing, contract mechanics, settlement rules, bracket structure, or market data interpretation. Also use when Russell discusses edge, implied probability, model calibration, or position sizing. Triggers on terms like Kalshi, binary options, contracts, settlement, probability, edge, Kelly, fees, brackets, calibration, implied probability, or any trading-specific concepts.
tools: Read, Bash, Grep
model: opus
---

# AlphaTemp Trading Domain Knowledge

Reference for Kalshi prediction market mechanics, fee calculations, and the alphatemp trading model. Consult this before modifying any trading logic.

## Kalshi Weather Contracts — How They Work

Kalshi offers binary contracts on daily high temperatures. For NYC (our only market currently):

- **Ticker format:** `KXHIGHNY` (current), historically `HIGHNY`
- **Settlement station:** KNYC (Central Park)
- **Settlement source:** NWS Daily Climate Report (CLI) — NOT raw observations
- **Bracket structure:** 2°F wide brackets with open-ended tails

### Bracket Convention (Verified Against Settlement Data)

| Bracket type | Condition | Example |
|-------------|-----------|---------|
| Lower tail | temp < cap_strike (strictly less than) | "Below 30°F" = temp < 30 |
| Interior | floor_strike <= temp <= cap_strike (inclusive both) | "30-31°F" = 30 <= temp <= 31 |
| Upper tail | temp > floor_strike (strictly greater than) | "Above 40°F" = temp > 40 |

Each interior bracket covers exactly 2 integer temperatures. Typical event has ~6 brackets.

### Settlement

- Settlement uses NWS CLI preliminary report (~4:30 PM EST, final ~1:30 AM EST)
- CLI uses Local Standard Time, NOT Daylight Saving Time
- Results are binary per bracket: YES ($1.00) or NO ($0.00)
- Exactly one bracket per event settles YES

## Fee Structure (Verified Feb 2026)

**Kalshi taker fee formula:**
```
fee = max(ceil(0.07 × C × P × (1-P)), C × $0.01)
```
Where C = number of contracts, P = contract price in dollars.

- **No settlement fee.** No withdrawal fee. Only the taker fee at entry.
- Maker fee uses 0.04 coefficient instead of 0.07 (not relevant for our market orders).
- S&P 500 / Nasdaq markets get half rate (0.035) — not applicable to weather.
- Effective rate: ~1-2% depending on price level (lowest at tails, highest at 50c).

| Price | Fee (1 contract) | Break-even Edge |
|-------|------------------|-----------------|
| 10c   | 1c               | ~1%             |
| 25c   | 2c               | ~2%             |
| 50c   | 2c               | ~2%             |
| 75c   | 2c               | ~2%             |
| 90c   | 1c               | ~1%             |

CRITICAL: ALL strategy evaluations MUST be net of fees — no exceptions. This is a hard constraint from CLAUDE.md.

## Edge and Trade Decision

### Edge Calculation
```
edge = model_probability - market_implied_probability
```

A contract priced at $0.65 implies a 65% market-estimated probability that bracket settles YES.

### When Edge Is Strongest

Near-settlement trading tends to produce stronger edge because:
- Forecast accuracy increases as the event approaches
- Model probability becomes more precise while market prices may lag
- Near-settlement prices are more extreme → lower fees

### Trade Threshold

Not yet established for alphatemp. Must be determined through backtesting analysis that accounts for:
- Fee drag (taker fee at entry, ~1-2% effective rate)
- Model uncertainty (Brier score from backtests)
- Market microstructure costs (spread)

## Probability Model

### Current Architecture
```
HRRR Forecast High
    → Bias Correction (subtract historical mean bias)
    → Drift Adjustment (same-day observation divergence)
    → Gaussian Distribution (center = adjusted high, std = uncertainty)
    → 1°F Integer Bracket Probabilities
    → Map to Kalshi 2°F Brackets
```

### Key Parameters
- **Center:** `forecast_high - historical_bias + drift_score`
- **Std:** `historical_std_error × time_factor × stability_factor × convergence_factor`
- **Time factor:** 1.0 at 8am ET → 0.3 by 3pm ET (uncertainty narrows through the day)
- **Convergence:** Successive HRRR runs agreeing → tighter distribution

### Data Sources
- **Forecasts:** HRRR via Open-Meteo (single model, no blending)
- **Observations:** Synoptic API + AWC (aviationweather.gov) for real-time METAR
- **Settlement truth:** NWS CLI reports parsed by `services/nws_fetcher.py`
- **Historical daily highs:** ACIS (RCC) backfill in `nws_daily` table

## Kelly Sizing (Future — Paper Trading Phase)

When position sizing is implemented:
- Use fractional Kelly (full Kelly is too aggressive with model uncertainty)
- Fee drag must be subtracted from expected payout BEFORE Kelly calculation
- Starting capital: $100
- Paper trading only until explicitly approved for live

## Paper Trading vs. Live Trading

- **Paper mode:** Full pipeline runs (data → model → edge detection → sizing), but no Kalshi API orders placed
- **Live mode:** Not yet implemented — requires explicit approval
- The code path should be identical up to the execution step

## Common Pitfalls

1. **Fee calculation shortcuts** — Taker fee must always be applied. Use `compute_taker_fee()` from `services/strategy_backtester.py`
2. **Timezone confusion** — NWS CLI uses Local Standard Time. Kalshi contract days may not align with UTC days. HRRR model runs are UTC
3. **Stale data** — If data providers go down, the system may compute on stale observations. Always check data freshness
4. **Bracket mapping errors** — 1°F model probabilities must map correctly to 2°F Kalshi brackets. The tail brackets are open-ended (not capped)
5. **Probability bounds** — Probabilities must stay in (0, 1). Edge cases at 0 and 1 cause division errors
