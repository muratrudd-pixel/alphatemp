# AlphaTemp Rebuild Design — 2026-02-27

**Status:** Approved for implementation
**Fee Notice (2026-02-28):** Fee references in this document (1% trading, 10% settlement, 12% min displacement) are SUPERSEDED. Actual: taker fee = max(ceil(0.07*C*P*(1-P)), C*$0.01). No settlement fee. Real hurdle ~1-2%.
**Authors:** Russell Rudd + Claude (with Gemini Deep Research consultation)
**Replaces:** MASTER-PLAN.md (phases 0–4), all prior phase design docs
**Next step:** Invoke writing-plans skill to generate implementation plan

---

## Executive Summary

The AlphaTemp prediction pipeline requires a full rebuild. Three foundational data gaps — HRRR (4 of 24 hourly runs), GFS (1 of 4 daily runs), ECMWF (1 of 4 daily runs) — cascaded through every subsequent modeling phase. The existing Open-Meteo dependency for multi-run data is a dealbreaker: Open-Meteo serves composites that destroy Point-in-Time integrity.

The rebuild is organized in four sequential phases. Phase 1 fixes the data foundation entirely before any model work resumes. Phases 2–3 rebuild bias correction and uncertainty with rigorous head-to-head ablation. Phase 4 implements the strategy layer across the full Kalshi trading window. A concurrent evaluation discipline runs the strategy backtester at every phase so P&L trajectory is visible throughout.

**The baseline for comparison:** -44.7% return, 4.4% win rate. Root cause: static std leaks probability mass into sub-10¢ tail brackets. The path to profitability requires (1) complete data, (2) calibrated bias correction, and (3) dynamic uncertainty collapse.

---

## Background: What Failed and Why

### Data Gaps

| Source | We Had | We Need | Impact |
|--------|--------|---------|--------|
| HRRR | 4 of 24 hourly runs | All 24 | 20 intraday update signals missing |
| GFS | 1 of 4 runs (midnight composite) | 4 runs/day | 06z/12z/18z signals missing |
| ECMWF | 1 of 4 runs (midnight composite) | 00z/12z full runs | Highest-skill afternoon run missing |
| NBM | 16 months only | 5+ years | Archive too shallow for walk-forward |

### The Open-Meteo Problem

Open-Meteo's Historical Forecast API assembles the "most recent available run" at each timestep into a composite. It cannot expose individual model runs. Using it for multi-run HRRR/GFS/ECMWF destroys Point-in-Time integrity — the composite does not reflect what any single model run actually predicted. Open-Meteo is safe to use only for the midnight UTC 00z run, where the composite behavior accidentally preserves PiT.

### Why NBM is Killed

NBM is available on Open-Meteo only from October 2024 (16 months of history). The walk-forward framework requires a minimum 90-day training window plus meaningful evaluation periods. 16 months is insufficient for any statistical conclusion. Additionally, NBM's output is already a multi-model blend — we cannot extract independent model errors to exploit. Killed.

### Why AI Models (GraphCast et al.) are Killed

GraphCast, Pangu-Weather, Aurora, and similar AI weather models operate at 25km+ resolution. KNYC (Central Park) is defined by urban heat island effects and sea-breeze dynamics that operate at sub-kilometer scales. A 25km grid cell cannot represent Central Park. Additionally, these models are trained on ERA5 reanalysis which itself uses future observations — reforecast skill estimates are optimistic. Killed.

### The Static Std Problem

The existing model outputs a Gaussian distribution with approximately fixed std (2.4–3.0°F) regardless of evidence. The Kalshi market's Brier score collapses from 0.63 at midnight to 0.14 by 6 PM ET because human traders watch the thermometer and collapse their uncertainty. The model cannot do this — hence -44.7% returns. Fixing dynamic uncertainty is Phase 3's primary goal.

---

## What's Preserved

The following infrastructure is well-tested and load-bearing. It must not be discarded.

| Component | Location | Why Preserved |
|-----------|----------|---------------|
| `DataProvider` ABC + `BacktestDataProvider` | `services/data_provider.py` | 7-layer lookahead prevention, 189 passing tests |
| Walk-forward expanding window | throughout `backtester.py` | Empirically validated: no concept drift across 5 ECMWF IFS transitions |
| Strategy backtester | `services/strategy_backtester.py` | TriggerDetector, EventPortfolio, PnLSimulator, sanity filters |
| Phase 2 OLS formulas | `services/backtester.py` | Candidate A in bias ablation; champion until beaten |
| Phase 2B divergence features | `services/divergence.py` | running_max alone = 87% of Phase 2B improvement; stateless, clean |
| Fee math | `services/strategy_backtester.py` | Correct; never skip fees |
| Kalshi data pipeline | `scripts/backfill_kalshi_history.py` | Settlements, candlesticks, trades — all historical |
| Neighbor station data | `services/neighbor_obs.py` | KLGA/KEWR 583K obs each; useful as variance predictor candidates |

**Technical debt to fix in Phase 1 schema work:**
- `market_ticks` table has no UNIQUE constraint — add it
- `forecast_extended` table (65K rows) has no consumer — evaluate for medallion integration or drop
- `drift_signals.drift_score` hardcoded to 0.0 in `BacktestDataProvider` — fix before Phase 4

---

## Guiding Principles

1. **Data integrity before modeling.** No model work until Phase 1 gate is passed.
2. **Prove it or kill it.** Every phase has a clear metric and kill condition.
3. **Concurrent P&L tracking.** The strategy backtester runs at every phase from Phase 2 onward.
4. **Edge-agnostic discovery.** No time-of-day filtering until empirical evidence identifies where edge exists.
5. **Full trading window.** Evaluate from Kalshi market open (10 AM ET D-1) through settlement, not just same-day.
6. **Separate prediction/decision layers.** Market prices do not appear in model features.
7. **Dual metrics.** Log-loss for training (penalizes overconfident misses), Brier for market comparison continuity, net-of-fees P&L as the ultimate objective.
8. **Expanding window over rolling.** We empirically found no detectable bias discontinuity across 5 ECMWF IFS transitions. Expanding window retains all signal.
9. **Head-to-head ablation.** No blind replacement of one method with another. May the best Brier score win.

---

## Data Architecture: Medallion Schema

The current single-layer schema mixes raw ingested data with processed features. The rebuild introduces a three-tier medallion architecture.

### Bronze (Raw)
The exact data as received from external sources, with minimal transformation. Purpose: reproducibility and audit trail.

| Table | Contents |
|-------|----------|
| `bronze_hrrr_grib_meta` | GRIB file metadata (model_run, fxx, s3_path, ingested_at, is_spinup) |
| `bronze_gfs_grib_meta` | GFS GRIB metadata per run |
| `bronze_ecmwf_grib_meta` | ECMWF GRIB metadata per run |
| `bronze_metar_raw` | Raw METAR strings before parsing |
| `bronze_api_responses` | Raw JSON from Synoptic, NWS, Kalshi (for debug) |

### Silver (Cleaned)
Parsed, validated, normalized data. All temperatures in °F. All timestamps naive UTC. Deduped via UNIQUE constraints.

| Table | Contents | Key Change |
|-------|----------|-----------|
| `forecasts` | Existing table, extend to 24 HRRR run hours | Add `is_spinup BOOLEAN` (fxx ≤ 3) |
| `forecast_extended` | Keep or integrate into `forecasts` | Evaluate usage before Phase 2 |
| `observations` | Existing table, add KJFK and DSM | Add `obs_type VARCHAR` (metar/dsm) |
| `nws_daily` | Existing — no change needed | Already authoritative |
| `kalshi_*` tables | Existing — add UNIQUE to market_ticks | Minor fix |

### Gold (Features, Materialized)
Pre-computed, phase-specific feature tables for fast backtesting. Updated incrementally as new data arrives.

| Table | Contents |
|-------|----------|
| `gold_hrrr_bias_features` | Per (date, run_hour): fcst_high, sin/cos month, delta_temp |
| `gold_obs_divergence` | Per (date, update_hour_et): 4 divergence features (running_max, instantaneous, cumulative, slope) |
| `gold_multi_model_features` | Per (date, run_hour): GFS/ECMWF/HRRR aligned forecasts + ensemble spread |
| `gold_market_features` | Per (date, hour_et): yes_ask, spread, volume — forward-filled |

**Gold tables are views or materialized to time-fence automatically.** The `BacktestDataProvider` reads from gold tables with `WHERE date < :current_date`.

---

## Phase 1: Data Foundation

**Goal:** Fix all data gaps. Pass gate before any model work resumes.
**Gate:** All data loaded, schema supports multi-run queries, no PiT violations, gap audit shows no missing runs.

### 1.1 HRRR Complete Backfill (24 Hourly Runs)

**Target:** All 24 run hours × ~1,900 days (Jun 2021–present) × fxx 1–18
**Source:** AWS S3 `s3://noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2`
**Tool:** Herbie (Python)
**Infrastructure:** AWS EC2 (us-east-1) for co-location with bucket — reduces latency and egress cost significantly
**Scale:** ~820,000 GRIB fetches (24 runs × 1,900 days × ~18 fxx). Estimated 2–5 days wall-clock with parallelized script.

**Extended runs** (00z, 06z, 12z, 18z): fetch fxx 1–48 (48-hour horizon)
**Standard runs** (all others): fetch fxx 1–18 (18-hour horizon)

**HRRR spin-up flag:**
Forecast hours fxx=1–3 have initialization artifacts from radar and surface observation assimilation. The skill peak is fxx=9–12. In the schema, tag `is_spinup = TRUE` for fxx ≤ 3. The Phase 2 ablation will test whether excluding spin-up hours improves bias correction.

**Backfill script design:**
- Temp DB per day (not per session) — never long-write to main DB (DuckDB segfault history)
- Resume support: check `MAX(model_run)` per run_hour before starting
- Idempotent: UNIQUE constraint on (station_id, model_run, valid_at, model_name)
- Continue on error — log gaps, don't abort
- Merge to main DB after each day completes

### 1.2 GFS Multi-Run Backfill

**Current state:** Only 00z via Open-Meteo midnight composite (~4,000 days, from 2021-03-23)
**Target:** Add 06z, 12z, 18z from UCAR archive
**Priority run:** 12z (settlement-day coverage, highest value for afternoon predictions)

**⚠️ TIME-SENSITIVE: UCAR RDA ds084.1 archive is shutting down early 2026. Download 12z data first.**

| Run | Source | Archive Depth | Priority |
|-----|--------|--------------|---------|
| 00z | Open-Meteo (keep — PiT preserved) | 2021+ | Done |
| 12z | UCAR RDA ds084.1 | 2015+ | FIRST — time-sensitive |
| 06z | UCAR RDA ds084.1 | 2015+ | Second |
| 18z | UCAR RDA ds084.1 | 2015+ | Third |

**Format:** GRIB2, 0.25° resolution. Herbie can access UCAR with appropriate credentials.
**Data latency note:** GFS takes ~3.5–4h after init time. The 12z run is not available until ~15:30 UTC (~11:30 AM ET).

### 1.3 ECMWF Multi-Run Backfill

ECMWF opened all data free under CC-BY-4.0 since October 2025. Herbie supports it. AWS archive starts January 2023.

| Run | Type | Horizon | Source | Priority |
|-----|------|---------|--------|---------|
| 00z | Full | 240h | AWS (Jan 2023+) | Second |
| 12z | Full | 240h | AWS (Jan 2023+) | FIRST |
| 06z | Short-cutoff | 90h | AWS (Jan 2023+) | Third |
| 18z | Short-cutoff | 90h | AWS (Jan 2023+) | Third |

**Data latency note:** ECMWF takes ~6.75–8.5h after init. The 12z run is not available until ~20:30 UTC (~3:30 PM ET).

**Note on archive depth:** ECMWF archive via Herbie/AWS starts Jan 2023. This gives ~3 years of 12z data — sufficient for Phase 2 ablation but shorter than the HRRR/GFS window. Phase 2 will use dates with all three models available.

### 1.4 Observation Sources

**Keep (existing):**
- KNYC — hourly METAR via Synoptic API + IEM/AWC (SPECI-critical). Settlement station.
- KLGA, KEWR — 1-min ASOS, 583K obs each, already in DB. Neighbor stations.

**Add: KJFK**
KJFK (JFK International) is a full ASOS station with heavy SPECI traffic — more frequent sub-hourly reports than KLGA or KEWR. It provides an additional independent NYC-area observation for divergence features. Access via Synoptic API (same pattern as existing stations) and IEM historical backfill.

**Add: DSM (Daily Summary Message)**
The NWS Daily Summary Message arrives in the afternoon (typically 4–5 PM ET) on the settlement day, before the CLI final report (~1:30 AM ET next day). DSM contains the authoritative daily max and min temperature. Ingesting DSM provides an early settlement truth proxy that can be used to validate model predictions and inform live trading decisions before final CLI arrives.
Source: NWS text products (same KOKX office as CLI). Parse alongside existing `nws_fetcher.py`.

**Investigate: NYC Micronet**
The NYC Micronet is a 29-station network with 5-minute reporting frequency across all five boroughs, operated by SUNY Albany. It could provide sub-grid urban microclimate signals not visible in the standard ASOS network.
Action: Email `mesonet@albany.edu` to request access. Evaluate access terms and latency before committing to integration. If access is granted, run an ablation test on divergence features before adding to production.

### 1.5 Schema Changes Summary

| Change | Why |
|--------|-----|
| `forecasts` + `is_spinup BOOLEAN` | Flag fxx ≤ 3 for ablation testing |
| `forecasts` + `fxx INTEGER` | Enable per-fxx analysis and filtering |
| `market_ticks` UNIQUE(market_id, captured_at) | Fix missing constraint — currently accumulates duplicates |
| `observations` + `obs_type VARCHAR` | Distinguish METAR, DSM, Micronet |
| `observations` + `kjfk` rows | Add new station |
| Bronze tables | Metadata provenance for GRIB files |
| Gold feature tables | Pre-computed feature tables for fast backtesting |

---

## Phase 2: Bias Correction — Head-to-Head Ablation

**Goal:** Find the best bias correction method for 24 HRRR runs + multi-model.
**Gate:** Brier improvement > 2% vs Phase 1 baseline. P&L tracked concurrently (diagnostic only here).

### Setup

- Training data: all 24 HRRR run hours × all dates with data
- Expanding window, 90-day minimum (unchanged)
- Settlement truth: `nws_daily.max_temp_f` (unchanged)
- Evaluation period: dates with all three models available (i.e., from ECMWF archive start Jan 2023)

**Cross-hour training option:**
Phase 2 will test whether training all run hours together (with `update_hour` as a feature) outperforms separate per-hour models. The Phase 3.7 failure was in the variance layer; the bias layer uses a simpler architecture that may benefit from cross-hour pooling.

### Candidate A: OLS (Current Champion)

```
error = β₀ + β₁·fcst_high + β₂·sin(2π·month/12) + β₃·cos(2π·month/12) + β₄·delta_temp
center = fcst_high - predicted_bias
std    = max(0.3, residual_std)
```

Phase 2 champion at Brier 0.7979 on 4 HRRR runs. Baseline for all comparisons. Stable coefficients as training window grows. This is Candidate A — it must be beaten, not assumed to remain champion.

**Ablation sub-tests:** With and without is_spinup exclusion (fxx ≤ 3).

### Candidate B: EMOS (Non-Homogeneous Gaussian Regression)

EMOS (Ensemble Model Output Statistics, Gneiting et al. 2005) is the academic gold standard for NWP post-processing. It jointly estimates mean and variance from ensemble output:

```
μ = a + b·fcst
σ² = c + d·ensemble_spread²
```

Where `ensemble_spread` is the standard deviation across available HRRR runs for that initialization time, or across HRRR/GFS/ECMWF forecasts. The variance equation is physically motivated: when models disagree, uncertainty is higher.

**Key advantage over OLS:** EMOS produces calibrated mean *and* variance in one step. It may solve Phase 3's static-std problem without requiring a separate uncertainty layer.
**Key risk:** Requires sufficient ensemble spread signal. With only 1–4 models and limited NWP variety, spread may not be informative enough.

### Candidate C: XGBoost

XGBoost gradient-boosted regressor on all available features:

```
features: fcst_high, sin/cos month, delta_temp, run_hour,
          ensemble_spread, fxx, is_spinup, obs_features (if post-14 ET)
```

Run in two modes: (1) standard regression for point estimate, (2) quantile regression with pinball loss for direct uncertainty estimation.

**Key risk:** Overfitting with ~1,900 days × 24 run hours ≈ 45,600 training rows across all features. Mandatory 5-fold time-series cross-validation. If out-of-fold Brier degrades vs OLS, kill it.

### Phase 2B: Observation-Based Residual Correction

The existing Phase 2B divergence feature layer is preserved and applied after whichever candidate wins the bias ablation. The architecture (residual regression on Phase 2's unexplained error, activated only after 14 ET) is validated and does not need to be re-run in ablation.

---

## Phase 3: Uncertainty and Calibration

**Goal:** Dynamic uncertainty collapse — the model must narrow its std as evidence accumulates.
**Gate:** Brier improvement > 2% AND P&L trajectory improves (co-equal gate for the first time).

### The Root Problem

Market Brier collapses 0.63 → 0.14 through the day. The model barely moves. Until `std` can collapse as observations confirm the high, the strategy is structurally unable to beat the market in the afternoon — which is exactly when edge should be highest.

**Why Phase 3.7 failed:** The two-level cache was keyed by `update_hour_et`, so all training rows in a given batch had the same `update_hour` value. The primary variance feature was constant in every regression. The fix is cross-hour training.

### Candidate A: EMOS Variance Layer

If EMOS (Phase 2 Candidate B) won the bias ablation, it already outputs a calibrated variance. Test whether EMOS variance, conditioned on additional features (running_max, hours to settlement, ensemble spread), closes the dynamic uncertainty gap without a separate Phase 3 model.

### Candidate B: Quantile Regression (Primary Approach)

Linear quantile regression with pinball loss, 7 quantiles:

```
Quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Loss:      pinball(τ, ŷ, y) = τ·max(y - ŷ, 0) + (1-τ)·max(ŷ - y, 0)
```

**Features (6):**
1. `update_hour_et` — time of day as continuous feature (NOT a per-subset split — this is the Phase 3.7 fix)
2. `fcst_high` — temperature regime
3. `sin(2π·month/12)` — seasonal encoding
4. `cos(2π·month/12)` — seasonal encoding
5. `running_max_divergence` — hard floor on predicted high (87% of Phase 2B improvement)
6. `slope_divergence` — trend signal in obs vs forecast gap

**Training:** Cross-hour — all historical dates × all update hours in one training set. `update_hour_et` is a feature. Walk-forward: strictly `date < current_date`.

**Ablation variants:**
1. QR with all 6 features
2. QR without obs features (pre-14 ET compatibility)
3. QR with HRRR-only features (no obs)

### CDF Construction

**Interior:** Piecewise-linear interpolation between the 7 quantile points. Bracket probability = CDF(bracket_cap) − CDF(bracket_floor).

**Tails (exponential decay):**
```python
# Upper tail: beyond q95
λ_upper = 1.0 / (q95 - q50)   # tuned empirically
p(T > T0) = p_remaining_upper × exp(-λ_upper × (T - q95))  for T > q95

# Lower tail: beyond q05
λ_lower = 1.0 / (q50 - q05)
p(T < T0) = p_remaining_lower × exp(-λ_lower × (q05 - T))  for T < q05
```

Exponential decay (not flat clamping) prevents probability mass accumulation at physically impossible temperatures. For example, a 95°F high in February or a 15°F high in August are improbable but not impossible — exponential tails assign small but nonzero probability. Flat clamping assigns zero probability, which hurts log-loss catastrophically on rare events.

**λ tuning:** Fit empirically on training data to minimize ECE on tail brackets. Do not hand-tune.

### Calibration Verification

After CDF construction, apply Isotonic Regression as a calibration pass:
- Walk-forward: train on prior predictions vs outcomes
- Maps "when model says 85%, historically it hits X%" → adjusted probability
- Gate requirement: ECE must decrease vs uncalibrated distribution

---

## Phase 4: Strategy and Execution

**Goal:** Find profitable trading opportunities, net of fees, across the full Kalshi trading window.
**Gate:** Net P&L positive, bootstrap CI lower bound > 0, results hold in walk-forward, Profit Factor > 1.5.

### Full Trading Window

Kalshi KXHIGHNY markets open at approximately **10 AM ET on the day before settlement (D-1)**. The rebuild must evaluate model edge across the complete window, not just same-day afternoon hours.

| Period | Clock | Available NWP | Market State |
|--------|-------|--------------|--------------|
| Prior-day morning | 10 AM–2 PM ET (D-1) | GFS 06z (ready ~11:30 AM ET), HRRR hourly | Market freshly opened, thinnest liquidity |
| Prior-day afternoon | 2 PM–6 PM ET (D-1) | GFS 12z (ready ~3:30 PM ET), HRRR hourly | Pricing updating |
| Prior-day evening | 6 PM–midnight ET (D-1) | GFS 18z, ECMWF 12z (ready ~8:30 PM ET) | Settlement-day HRRR 00z starts at ~01:30 UTC |
| Overnight | Midnight–6 AM ET (D) | HRRR 00z–05z | Thinnest trading; model may have most edge |
| Morning | 6 AM–2 PM ET (D) | HRRR 06z–13z, GFS 06z | Market improving with obs |
| Afternoon | 2 PM–settlement | HRRR 14z–18z + obs | running_max converges; Phase 2B active |

The prior-day window (10 AM to midnight D-1) is expected to contain significant edge because:
1. Market prices are weakest when the settlement day is still 12–24 hours away
2. GFS/ECMWF long-range forecasts (30–48h horizon) are the primary signal — human traders are also using these
3. Model bias corrections for long-horizon forecasts may be more stable than market intuition

### Edge-Agnostic Discovery

**No pre-filtering by time of day.** Previous Phase 4 design specified "morning edge focus (06z-14z)" — this was premature and based on incomplete data (4 HRRR runs only, midnight-onward evaluation). The rebuild will:

1. Compute displacement (model_prob − market_implied_prob) at every available timestep from market open through settlement
2. Sweep displacement thresholds [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
3. Report P&L by: hour of day, hours-to-settlement, season, temperature range, bracket position
4. Let the data identify where edge exists before applying any filters

### Execution Model

- Entry pricing: `yes_ask` — always cross the spread (conservative; if profitable here, profitable live)
- Execution latency: 60 seconds after model evaluation
- Minimum displacement: 12% (11% fee hurdle + 1% noise buffer)
- Sanity filters (5, unchanged): model_prob < 0.05, post-peak detection, spread > 10¢, model_std > 3.5°F, churn prevention

### Position Sizing

- Fractional Kelly (Kelly/4): conservative for $100 starting capital with uncertain edge estimates
- 10% daily cap: never risk more than $10 on a single day's event
- Prediction/decision layer separation: market prices (yes_ask, spread) are inputs to the *execution layer* only, never to model features

### Fee Calculation (unchanged)

```
Entry cost:     price × qty × 1.01                        (1% trading fee)
Win payout:     (1 - price) × qty × 0.90                  (10% settlement fee on winnings only)
Net win P&L:    (1 - price) × qty × 0.90 - price × qty × 1.01
Net loss P&L:   -(price × qty × 1.01)
```

All strategy evaluations must be net of all fee layers. No exceptions.

---

## Concurrent Evaluation Framework

The strategy backtester runs at every phase. P&L trajectory is visible throughout the build, not just at Phase 4.

### Gate Structure

| Phase | Primary Gate | P&L Role | Kill Condition |
|-------|-------------|----------|---------------|
| Phase 1 (Data) | All data loaded, schema validated, gap audit passes, no PiT violations | Not tracked | Any PiT violation blocks Phase 2 |
| Phase 2 (Bias) | Brier > 2% over Phase 1 baseline | Tracked — diagnostic. Is P&L trending up? | Brier < 2% improvement: kill the candidate |
| Phase 3 (Uncertainty) | Brier > 2% AND P&L improves | **Co-equal gate.** Both must pass. | Brier improves but P&L flat: investigate before proceeding |
| Phase 4 (Strategy) | Net P&L positive, CI > 0, holds walk-forward | **Primary gate.** | P&L negative net of fees: kill strategy variant |

### Why P&L is Not a Phase 2 Kill Signal

The -44.7% baseline loss was caused by static std leaking into tail brackets — not by bad bias correction. If we gate on P&L in Phase 2, we would kill every feature, including foundationally correct ones, because profitability requires dynamic uncertainty (Phase 3) first. Tracking P&L in Phase 2 provides trajectory information: "Is the model trending toward profitable, even if not there yet?"

### Backtester Discipline

The `strategy_backtester.py` infrastructure already exists. At every phase gate evaluation, run:
```python
# Phase 2 gate evaluation
brier = run_brier_backtest(model_fn, data)
pnl_report = run_strategy_backtest(model_fn, market_data, fees)
print(f"Brier: {brier:.4f} | Net P&L: ${pnl_report.total_pnl:.2f} | "
      f"Win rate: {pnl_report.win_rate:.1%} | Trades: {pnl_report.n_trades}")
```

---

## HRRR Spin-Up Awareness

Forecast hours fxx=1–3 have initialization artifacts from the data assimilation cycle. The radar and surface observation assimilation introduces transient noise that takes ~3 forecast hours to spin out. The skill peak for HRRR is fxx=9–12.

**Schema:** `forecasts.is_spinup = TRUE` for `fxx ≤ 3`

**Phase 2 ablation:** Test with and without spin-up hours in bias correction training. If excluding is_spinup rows improves Brier, exclude them. If not, include them — more training data is better.

**Phase 3 feature:** `is_spinup` (or `fxx` directly) as a variance feature candidate. Higher variance expected for spin-up hours.

**Live use:** For intraday updates in the first 3 forecast hours of a new HRRR run, flag predictions as lower-confidence.

---

## Open Questions and Blockers

| Item | Status | Action |
|------|--------|--------|
| UCAR GFS archive shutdown | ⚠️ TIME-SENSITIVE | Download 12z data as first task in Phase 1 |
| NYC Micronet access | Unknown timeline | Email mesonet@albany.edu; no dependency on Phase 1 gate |
| Synoptic HF-ASOS (1-min KNYC) | Was down Oct 2023–Jan 2026 | Check if back up; evaluate if available |
| HRRR backfill scale | ~820K GRIB fetches | EC2 in us-east-1; estimate 2–5 days |
| ECMWF archive depth | Jan 2023+ only | Shorter than HRRR/GFS; Phase 2 uses intersection of available dates |
| DuckDB crash risk | Known issue, DuckDB 1.4.4 | Temp DB + merge pattern mandatory; never long-write to main |
| Drift score in backtest | Hardcoded to 0.0 | Fix `BacktestDataProvider.get_drift_score()` before Phase 4 |
| `forecast_extended` table | 65K rows, no consumer | Evaluate for medallion integration or drop in Phase 1 schema work |

---

## Implementation Order

Dependencies are strict. Do not begin a phase until its predecessor's gate is passed.

```
Phase 1 (Data Foundation)
├── 1a. UCAR GFS 12z download      ← FIRST, time-sensitive
├── 1b. HRRR 24-run backfill       ← AWS EC2, parallelized
├── 1c. ECMWF 12z backfill         ← Herbie, AWS
├── 1d. Medallion schema redesign  ← Bronze/Silver/Gold, fix market_ticks
├── 1e. Add KJFK + DSM ingestion   ← Synoptic API extension
├── 1f. NYC Micronet outreach      ← Email, no hard dependency
└── Gate: data loaded, no PiT violations, gap audit passes

Phase 2 (Bias Correction)
├── 2a. Phase 1 gate passed
├── 2b. Candidate A: OLS on 24-run HRRR    ← new baseline
├── 2c. Candidate B: EMOS                   ← ablation
├── 2d. Candidate C: XGBoost QR             ← ablation with cross-validation
├── 2e. Run strategy backtester (P&L diagnostic)
└── Gate: Brier > 2% over Phase 1

Phase 3 (Uncertainty)
├── 3a. Phase 2 gate passed
├── 3b. Candidate A: EMOS variance (if Phase 2 winner)
├── 3c. Candidate B: Linear QR (primary)    ← cross-hour training
├── 3d. CDF construction + exponential tails
├── 3e. Isotonic calibration pass
├── 3f. Run strategy backtester (P&L co-equal gate)
└── Gate: Brier > 2% AND P&L trajectory improves

Phase 4 (Strategy)
├── 4a. Phase 3 gate passed
├── 4b. Full trading window evaluation (10 AM ET D-1 onward)
├── 4c. Edge-agnostic displacement sweep (all hours)
├── 4d. Conditional analysis (hour, season, temp range, bracket)
├── 4e. Fractional Kelly sizing + 10% daily cap
└── Gate: Net P&L positive, bootstrap CI > 0, walk-forward validated
```

---

## Technical Constraints

These apply to all implementation work:

- **Python 3.9** — no subscripted builtins (`dict[str, ...]` → `Dict[str, ...]`)
- **DuckDB only** — not SQLite
- **Type hints via `typing` module** — `Dict`, `List`, `Optional`, `Tuple`
- **Tests with pytest** — all new code requires tests before the phase gate is evaluated
- **loguru for logging** — not print statements
- **All timestamps naive UTC in DB** — convert to ET only at display layer
- **`zoneinfo.ZoneInfo("America/New_York")`** for DST-aware local conversions
- **Temp DB + merge pattern** for all backfills — never long-write to main DB

---

## Decisions Not to Revisit

These decisions are closed. Do not reopen without new empirical evidence.

| Decision | Rationale |
|----------|-----------|
| Kill NBM | 16-month archive; insufficient for walk-forward |
| Kill AI models (GraphCast) | 25km+ resolution; can't see Central Park UHI |
| Expanding over rolling window | No concept drift in 5 ECMWF IFS transitions; empirical beats theoretical |
| Separate prediction/decision layers | No market prices in model features |
| Settlement truth = NWS CLI | Kalshi settles on CLI; optimize for the thing that matters |
| log-loss for training, Brier + P&L for evaluation | Log-loss prevents overconfident miss; Brier maintains comparability |
| yes_ask entry pricing | Conservative; if profitable here, profitable live |
