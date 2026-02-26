# AlphaTemp Master Plan — Prediction Engine v2

**Created:** 2026-02-24
**Status:** Active
**Goal:** Build a backtested probability model that predicts daily high temperature brackets for Kalshi KXHIGHNY markets, proving value at each step before adding complexity.

---

## Guiding Principles

1. **Prove it or kill it.** Every phase has a clear metric and a kill condition. No building on assumptions.
2. **Brackets are the product.** The model outputs probability distributions over Kalshi's 2°F brackets. Evaluate on bracket accuracy (Brier score) from day one.
3. **Same code path.** Backtest and live mode use identical logic via a DataProvider interface. No separate implementations that drift apart.
4. **Sequential gates.** Each phase depends on the previous one proving effective. Don't start Phase N+1 until Phase N passes its gate.

---

## Phase 0: Data Foundation

**Status:** COMPLETE (2026-02-25)
**Depends on:** Nothing

### What
Complete the historical dataset needed for backtesting:
- HRRR forecast backfill for all 4 run hours (00z, 06z, 12z, 18z) across ~1,900 days
- Verify observations (KNYC) and nws_daily (settlement truth) align with forecast dates
- Build the backtesting harness — a `Backtester` class that replays any model function against history

### Data Inventory
| Dataset | Target | Current | Status |
|---------|--------|---------|--------|
| Forecasts 06z | 1,900 days | 1,900 | DONE |
| Forecasts 12z | 1,900 days | 1,900 | DONE |
| Forecasts 00z | 1,900 days | 1,900 | DONE |
| Forecasts 18z | 1,900 days | 1,900 | DONE |
| NWS Daily (settlement) | 1,900 days | 1,900 | DONE |
| KNYC Observations | 1,900 days | 1,900 days (57K rows) | DONE |
| Market Ticks | Ongoing | 1 day | COLLECTING (start ASAP) |

### Deliverables
1. All 4 HRRR run hours backfilled and merged into main DB
2. `services/data_provider.py` — DataProvider ABC + LiveDataProvider + BacktestDataProvider
3. `services/backtester.py` — Backtester class that:
   - Iterates nws_daily dates
   - Feeds historical data through any model function via DataProvider
   - Outputs: Brier score, bracket accuracy, top-1/top-2 hit rate
4. Refactor `ProbabilityEngine` to accept a DataProvider

### Gate
Backtester runs end-to-end on historical data with a dummy model (e.g., uniform distribution). Infrastructure works.

---

## Phase 1: HRRR Error Analysis ("Is there signal?")

**Status:** COMPLETE (2026-02-25)
**Depends on:** Phase 0 complete

### The Question
For each historical day, the HRRR predicted some max temperature. NWS settled on an actual high. What does the error distribution look like, and can we use it to build a better probability distribution over 2°F brackets than just trusting HRRR's raw number?

### What We Built
Walk-forward bias correction with per-run-hour stats:
- For each evaluation, computes expanding-window mean bias and std from ALL prior dates only (no data leakage)
- Each run hour (00z, 06z, 12z, 18z) gets its own bias/std correction
- Uses NWS settlement truth (`nws_daily.max_temp_f`), not observations
- Minimum 90-day training window before scoring
- Also tested Student-t distribution for heavy tails (excess kurtosis up to +13.9 at 18z)

### Results
| Model | Brier Score | vs Uniform Baseline |
|-------|------------|-------------------|
| Uniform (clueless) | 1.0196 | — |
| Single-bias (old) | 0.9582 | +6.0% |
| **Walk-forward per-hour** | **0.8356** | **+18.0%** |
| Walk-forward Student-t | 0.8394 | +17.7% |

Per-run-hour bias discovered:
- 00z: -0.9°F (runs coldest), 06z: -0.2°F (most accurate), 12z: -0.3°F, 18z: -0.7°F

Market comparison (Kalshi's Brier score at same prediction times):
- After 00z: 0.63, After 06z: 0.62, After 12z: 0.44, After 18z: 0.14
- We need to roughly halve our score to compete with the market

### Key Findings
- Per-run-hour bias correction is 3x the improvement of single-bias
- The old single-bias model made 00z WORSE than uniform (1.10 vs 1.02)
- Student-t doesn't beat Gaussian despite heavy tails — killed
- Only 4.7% evaluations lost to the 90-day minimum training window
- Overnight hours (00z, 06z) have thinnest market — most likely edge opportunity

### Gate
PASSED. Walk-forward per-run-hour Gaussian beats all baselines.

---

## Phase 2: Enhanced Variables ("Can I explain more of the error?")

**Status:** COMPLETE (2026-02-26)
**Depends on:** Phase 1 passes its gate

### The Question
The HRRR misses by some amount each day. Can we predict HOW MUCH it will miss using additional variables available at forecast time?

### Variables Tested
| Feature | Source | Signal? | Details |
|---------|--------|---------|---------|
| **Forecast high** | `MAX(f.temp_f)` | **YES — r=0.39-0.52** | HRRR warm bias scales with temperature |
| **Month (season)** | `EXTRACT(MONTH)` | **YES — F-test p≈0** | Summer overpredict +1-2.3°F, winter underpredict -1.5-2.5°F |
| Day-over-day delta | `LAG(actual, 1) - LAG(actual, 2)` | **NO — r≈0** | Only 00z shows marginal signal (r=0.11), dead elsewhere |

### HRRR Bias Pattern Discovered
HRRR has a **temperature-dependent bias** — it systematically overpredicts more on hot days:
- Regression coefficient: ~+0.09-0.12°F per 1°F of forecast temperature
- Intercept: ~-7 to -8°F (overcorrects cold, undercorrects hot)
- This means on a 30°F day, predicted bias ≈ -4.3°F. On an 80°F day, predicted bias ≈ +0.2°F.
- Seasonal effect partially overlaps with temperature (hot days = summer) but sin/cos month encoding captures the remaining independent seasonal pattern.

### Approach
Walk-forward expanding-window OLS regression on HRRR error, conditioned on features available at prediction time:
- `error = β₀ + β₁·fcst_high + β₂·sin(2π·month/12) + β₃·cos(2π·month/12) + β₄·delta_temp`
- Predicted error replaces flat mean_bias; residual std replaces flat std_error
- Month encoded as sin/cos pair (2 continuous features instead of 11 dummies — well-conditioned with ~400+ samples per run hour)
- Manual OLS via `scipy.linalg.lstsq` — no sklearn dependency needed
- Same walk-forward discipline as Phase 1: strict `obs_date < current_date`, 90-day minimum

### Results
| Model | Brier | vs Phase 1 |
|-------|-------|-----------|
| walk_forward (P1 baseline) | 0.8356 | — |
| **wf_regression_full** (fcst + month + delta) | **0.7979** | **+4.5%** |
| wf_regression_fcst (fcst only) | 0.7998 | +4.3% |
| wf_regression_month (month only) | 0.8139 | +2.6% |
| wf_regression_delta (delta only) | 0.8356 | +0.0% |

Improvement consistent across all run hours (06z and 12z benefit most).

### Key Findings
- **fcst_high is the dominant feature** — alone gets 4.3% of the 4.5% total improvement
- Month adds 0.2% on top (partial overlap with temperature)
- delta_temp is worthless — regime transitions don't predict HRRR error
- Regression coefficients stable as training window grows (564→1,813 rows)
- Residual std: 2.4-3.0°F (down from flat std of 2.8-3.4°F in Phase 1)

### Gate
PASSED. +4.5% clears the >2% threshold. `wf_regression_full` is the new champion model.

### Possible Phase 2B: Intra-Day Observation Updates
Not yet started. The biggest remaining edge: update predictions as real-time observations arrive during the day. Kalshi market improves from 0.63→0.44→0.14 Brier through the day because traders watch the thermometer. Phase 2B would replay historical observations and shift predictions accordingly.

---

## Phase 3: Multi-Model Ensemble

**Status:** NOT STARTED
**Depends on:** Phase 1 or 2 produces a model worth ensembling with
**Prerequisite:** Backfill historical GFS/NAM/ECMWF data via Open-Meteo historical API

### The Question
Do other models' errors have different patterns than HRRR's? If so, combining bias-corrected forecasts from multiple models should reduce overall error.

### Concrete Steps
1. Backfill historical forecasts for GFS, NAM, ECMWF via Open-Meteo
2. Schema migration: add `model_name` column to forecasts table (`ALTER TABLE forecasts ADD COLUMN model_name VARCHAR DEFAULT 'HRRR'`)
3. Run Phase 1's analysis on each model independently — bias, std, Brier score
4. Test naive average of bias-corrected models vs best single model
5. If naive average wins: test time-weighted ensemble
   - Hypothesis: weight HRRR more at 12z/18z (fresher data), weight GFS/ECMWF more at 00z/06z
   - Let the backtester discover the optimal weights, don't assume them
6. Bias-correct each model individually FIRST, then ensemble the corrected values

### Evaluation Metric
Ensemble Brier score vs best single-model Brier score.

### Kill Condition
If naive average of all models doesn't beat the best single model. In that case, stick with the single-model approach from Phase 1/2.

---

## Phase 4: Kalshi Strategy Layer

**Status:** NOT STARTED
**Depends on:** A model from Phases 1-3 that beats naive baselines

### What
Map the model's bracket probabilities to actual Kalshi trading decisions.

### Concrete Steps
1. Compare model bracket probabilities vs Kalshi market prices
2. Identify brackets where `model_prob - market_prob > fee_threshold`
3. Fee structure: 1% trading fee, 10% settlement fee, 2% withdrawal fee
4. Simulate historical P&L net of all fees (requires market tick history)
5. Define position sizing rules (Kelly criterion or fixed-fraction)
6. Paper trade before going live

### Evaluation Metric
Simulated P&L net of fees over historical period.

### Kill Condition
If edge < fee drag consistently. Model is accurate but not profitable.

### Note
Market tick collection is a silent blocker. Only 1 day of history exists. Every day without ticks is a day we can't backtest strategy. Ticks should be collecting continuously from now on.

---

## Key Architecture Decisions

### DataProvider Interface
```
DataProvider (ABC)
├── LiveDataProvider    — queries DB for latest data
└── BacktestDataProvider — queries DB with time-fencing (only data available at ref_time)
```
ProbabilityEngine accepts a DataProvider. Same code for live and backtest.

### Schema Migration (Phase 3)
```sql
ALTER TABLE forecasts ADD COLUMN model_name VARCHAR DEFAULT 'HRRR';
-- Recreate unique constraint to include model_name
```
Non-breaking. All existing HRRR rows get default value.

### Bracket Definition
Kalshi uses 2°F brackets with open-ended tails:
- "X° or below" (lower tail)
- "X° to X+1°" (2°F bins, interior)
- "X° or above" (upper tail)

Brackets change daily based on the forecast. Model must handle variable bracket boundaries.

---

## Data Constraints & Guardrails

- **Conditional bias (Phase 2):** If sample size per regime drops below ~30 observations, fall back to flat bias
- **Multi-model (Phase 3):** Need at least 6 months of overlapping historical data across all models before ensembling
- **All timestamps are UTC** unless explicitly stated otherwise
- **Settlement uses Local Standard Time** (UTC-5 year-round, never EDT)
- **All strategy evaluations MUST be net of fees** — no exceptions
