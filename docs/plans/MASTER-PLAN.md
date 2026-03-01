# SUPERSEDED — See 2026-02-27-rebuild-design.md

This document is historical only. Do not use for implementation. The rebuild design doc (2026-02-27-rebuild-design.md) replaces all phases and decisions below.

---

# AlphaTemp Master Plan — Prediction Engine v2 (HISTORICAL)

**Created:** 2026-02-24
**Status:** Active
**Fee Notice (2026-02-28):** All fee references in this document (1% trading, 10% settlement, 11% hurdle) are WRONG. Actual Kalshi fees: taker fee = max(ceil(0.07*C*P*(1-P)), C*$0.01). No settlement fee. Real hurdle ~1-2%. See `services/strategy_backtester.py`.
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

**Status:** COMPLETE (2026-02-26)
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

### Results
- HRRR-GFS error correlation: 0.5965, HRRR-ECMWF: 0.5684, GFS-ECMWF: 0.6421 (all < 0.7 — ensemble viable)
- Equal-weight ensemble + Phase 2B at 18 ET: Brier 0.7808
- Phase 3.5 walk-forward adaptive weights (inverse-Brier, 90-day expanding window): Brier 0.7705 (+1.31%)
- GFS gets highest weight (~35%), ECMWF (~34%), HRRR (~32%)
- Extended weather variables (dewpoint, humidity, wind, pressure, cloud, radiation, CAPE) all killed — none cleared 2% gate

### Gate
PASSED. Ensemble beats best single model. Adaptive weights beat equal weights.

---

## Phase 3.6: Neighbor Station Observations

**Status:** KILLED (2026-02-27)
**Depends on:** Phase 3.5 complete
**Design doc:** [2026-02-26-neighbor-obs-design.md](2026-02-26-neighbor-obs-design.md)

### The Question
Can 1-minute ASOS data from KLGA (LaGuardia) and KEWR (Newark) — already in the DB with 583K obs each — improve the Phase 2B divergence features and push the obs crossover earlier than 14 ET?

### Results
6 ablation variants tested (3 station-mapping × 2 integration). All blended curve variants (x2) killed immediately — actively worse than baseline. Feature variants (x1) showed:
- Best: A1 (raw + features) at +1.80% at 18 ET, but only +0.2% to +1.2% during the tradeable window (13-16 ET)
- Crossover at 13 ET (1 hour earlier than KNYC-only), but improvement too small to clear gate
- Before 13 ET (the high-edge overnight/morning hours), neighbor features slightly hurt (-0.1% to -0.4%)

### Why Killed
The +1.80% headline number at 18 ET is a vanity metric — by 6 PM the daily high has long passed and the market has already collapsed uncertainty. During the hours that matter for trading (13-16 ET), improvement is +0.2% to +1.2%, well below the 2% gate. The real problem is fixed std, not obs data source.

### Revisit Later
The neighbor station data (KLGA/KEWR, 583K obs each, 1-minute resolution) and all code (services/neighbor_obs.py, 17 tests) remain in the codebase. The data may prove useful as:
- Variance predictors in Phase 3.7 (neighbor_peak_signal is a Tier 1 candidate)
- Input to a future feature we haven't conceived yet
- The infrastructure is built, tested, and ready to reuse

### Gate
- PASS: >2% Brier improvement across any sustained block of tradeable hours
- **Result: KILL** — no variant cleared 2% during 13-16 ET tradeable window

---

## Phase 3.7: Dynamic Uncertainty

**Status:** DESIGNED (2026-02-27)
**Depends on:** Phase 3.6 (more obs data = better confidence estimation)
**Design doc:** [2026-02-27-dynamic-uncertainty-design.md](2026-02-27-dynamic-uncertainty-design.md)

### The Problem
The model outputs a Gaussian with roughly fixed std (2.4-3.0°F) regardless of evidence. At midnight with no obs, std ≈ 2.5°F. At 5 PM with the high clearly locked in, std ≈ 2.3°F. The market's Brier drops from 0.63 to 0.14 through the day because traders collapse uncertainty. The model can't.

### Approach
Parallel variance regression at the Phase 2B layer: `log(residual² + 1e-6) ~ variance_features`. Same walk-forward OLS, same expanding window. Each NWP model gets its own variance regression.

Key safeguards:
- Winsorization at [2nd, 98th] percentile before squaring (outlier explosion prevention)
- Std floored at 0.3°F, capped at 5.0°F
- Per-model regression (error profiles differ across HRRR/GFS/ECMWF)

### Variance Feature Candidates (Full Ablation)
**Tier 1:** update_hour, divergence_slope, neighbor_peak_signal, forecast_spread (model disagreement)
**Tier 2:** run_to_run_convergence (spread of HRRR 00z/06z/12z/18z), hours_until_sunset (season×time interaction), cumulative_divergence, running_max_divergence
**Tier 3 (re-evaluated killed features):** fcst_high, dewpoint_depression, cloud_cover, max_wind, mean_pressure
**Tier 4 (urban micro-climate):** wind_direction_categorical (onshore vs offshore — sea breeze caps diurnal heating in NYC), hours_until_precip (early rain arrival locks in daily high)

### Gate (Dual)
Must pass BOTH:
1. Brier improvement >2% vs current best
2. Improved calibration (lower Expected Calibration Error)

Secondary metrics tracked: log-loss (catches overconfidence), Brier decomposition Reliability term, std trajectory by hour.

### Future: Quantile Regression (Escape Hatch)
If OLS variance regression passes but Gaussian tails assign probability to physically impossible temperatures (e.g., high < current obs late in the day), quantile regression is the documented next step. Predicts error percentiles directly, handles asymmetric risk without Gaussian assumption.

### Investigation Complete: Concept Drift from NWP Model Upgrades (2026-02-27)
Open-Meteo serves **actual historical operational forecasts**, not reforecasts. Model version transitions exist in our data:
- HRRR v4 operational Dec 2020 — our data starts Jun 2021, so **all v4, no transition**
- GFS v16 operational Mar 2021, Open-Meteo archive starts Mar 2021 — **all v16, no transition**
- ECMWF IFS: 5 transitions in window (47r2 May '21, 47r3 Oct '21, Atos Oct '22, 48r1 Jun '23, 49r1 Nov '24)

**Diagnostic results:** Plotted 6-month bias windows for all 3 models. All show strong seasonal oscillation (H1 winter/spring positive bias, H2 summer/fall negative) but **no discontinuities at version transition dates**. ECMWF bias swings (+2.5 to -0.2) align with seasonal pattern, not IFS cycle changes.

**Conclusion:** Expanding window is appropriate. Seasonal bias is already captured by Phase 2's sin/cos month features. No need to switch to rolling window.

---

## Phase 3.8: Probability Calibration Layer

**Status:** NOT STARTED
**Depends on:** Phase 3.7 (dynamic uncertainty must be in place first)

### The Problem
Even with perfectly calibrated individual models, a Gaussian mixture ensemble is not mathematically guaranteed to remain calibrated in the tails. Before feeding probabilities into Kelly criterion sizing, predictions must map to empirical hit rates.

### Approach
Apply Isotonic Regression (or Platt Scaling) to the ensemble's bracket probabilities:
- Walk-forward: train on prior predictions vs outcomes, apply to today's predictions
- Maps "when model says 85%, historically it hits X%" → adjusted probability = X%
- Prevents Kelly from overbetting on miscalibrated confidence

### Gate
- Reliability term of Brier decomposition must decrease
- ECE must decrease vs uncalibrated ensemble
- No degradation of overall Brier score (calibration shouldn't hurt accuracy)

---

## Phase 4: Kalshi Strategy Layer

**Status:** IN PROGRESS — Dashboard V2 complete (2026-03-01), PaperTrader shell live with placeholder strategy. Strategy brainstorm session pending.
**Depends on:** A calibrated model from Phases 1-3.8

### What
Map the model's bracket probabilities to actual Kalshi trading decisions.

### Concrete Steps
1. Compare model bracket probabilities vs Kalshi market prices
2. Identify brackets where `model_prob - market_prob > fee_threshold`
3. Fee structure: taker fee = max(ceil(0.07*C*P*(1-P)), C*$0.01). No settlement fee. ~1-2% effective rate.
4. Simulate historical P&L net of all fees (requires market tick history)
5. Define position sizing rules (Fractional Kelly — Kelly/4 with hard cap)
6. Paper trade before going live
7. **Dashboard V2 (DONE 2026-03-01):** KPI header, health page, blotter, paper trader shell, polish. Supports YES/NO positions and early exit.

### Edge Threshold
With ~1-2% effective fee drag, edge threshold is much lower than originally estimated. Strategy brainstorm session needed to define exact entry/exit rules, position sizing, and NO-side logic.

### Position Sizing
- **Fractional Kelly (Kelly/4):** Full Kelly is optimal but assumes infinite divisibility and no model error. With $100, a single bad beat on full Kelly could wipe 40%+ of the account.
- **Hard daily cap:** Never risk more than 10% of bankroll on a single day's event.
- **Scale up gradually:** As bankroll grows and model track record lengthens, fractional Kelly divisor can decrease.

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
