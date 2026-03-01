# Gemini Deep Research Brief — AlphaTemp Reassessment

**Date:** 2026-02-27
**Project:** AlphaTemp — Kalshi Weather Trading System
**Fee Notice (2026-02-28):** Fee references in this document (1% trading, 10% settlement, 11% hurdle) are WRONG. Actual: taker fee = max(ceil(0.07*C*P*(1-P)), C*$0.01). No settlement fee. Real hurdle ~1-2%.
**Goal:** Comprehensive research to inform a full rebuild of our temperature prediction pipeline for trading Kalshi KXHIGHNY daily high temperature bracket markets.

---

## Context

I'm building a system to predict daily high temperature settlement brackets (2°F wide) for Kalshi's KXHIGHNY market (NYC Central Park / KNYC). The goal is to produce better probability distributions than the market, finding edge where our model price > market price + fee hurdle (~11%).

### What We've Built So Far
- Walk-forward expanding-window OLS bias correction on HRRR forecasts
- Phase 2 regression: error ~ fcst_high + sin/cos(month) + delta_temp (fcst_high is dominant)
- Phase 2B: intraday observation divergence features (running_max_divergence is king, only useful after 14 ET)
- 3-model ensemble (HRRR + GFS + ECMWF) with adaptive inverse-Brier weights
- Full strategy backtester with Kalshi market data (settlements, candlesticks, trades)

### What Went Wrong
1. **Only backfilled 4 of 24 HRRR hourly runs** (00z/06z/12z/18z) — missed 20 standard runs that update hourly with fresh radar/surface assimilation
2. **GFS and ECMWF only have 1 run/day** via Open-Meteo (midnight UTC run)
3. **Static Gaussian std** couldn't collapse uncertainty through the day — market Brier goes from 0.63 to 0.14 while our model barely moves
4. **Dynamic variance OLS failed** — per-hour training made the primary feature (update_hour) constant
5. **No NBM, NAM, GraphCast, or AI models** evaluated
6. **Model is not yet profitable** — Brier ~0.70 vs Kalshi market ~0.63 at best hours

### Key Constraints
- NYC only (KNYC Central Park settlement station)
- Starting capital: $100 (paper trading phase)
- Fee structure: 1% trading + 10% settlement + 2% withdrawal = ~11% hurdle
- Settlement source: NWS Daily Climate Report (CLI) from Central Park
- 2°F bracket resolution
- Python 3.9, DuckDB, existing backtester infrastructure works well

---

## Research Questions

### Pillar 1: Model Architecture & Order of Operations

1. **NBM vs. Custom Ensemble:** The National Blend of Models already combines GFS, HRRR, NAM, ECMWF, and others with MOS (Model Output Statistics). It outputs percentiles and probabilistic guidance natively. For our specific use case (NYC daily high temperature brackets):
   - Does NBM already capture most of the signal we'd get from building our own ensemble?
   - Where does NBM fall short that a custom model could exploit?
   - Is it better to use NBM as our primary backbone and add corrections, or treat it as one input among many?

2. **Optimal Build Order:** We built single-model → bias correct → add models → ensemble. Should we instead:
   - Start with the best available blended forecast (NBM?) and correct that?
   - Get ALL data sources in first, then build modeling layers?
   - Or some other ordering?
   - What does the meteorological/statistical literature recommend?

3. **Multi-Frequency Temporal Alignment:** Our data sources update at very different cadences — HRRR every hour, GFS 4x/day, ECMWF 2x/day, NBM hourly, observations every 1-60 minutes, SPECI reports at irregular intervals. How should the system handle this?
   - When a new HRRR run arrives (hourly), should the ensemble re-weight using stale GFS/ECMWF data, or only update the HRRR component?
   - How should irregular observation events (SPECI reports triggered by significant weather changes) propagate through the model? Should they trigger a full model re-evaluation or just update divergence features?
   - What's the right "clock" for the system — event-driven (trigger on any new data arrival) vs. fixed-interval polling (check every N minutes)?

4. **Bias Correction Approach:** We used expanding-window OLS. Alternatives include:
   - MOS (Model Output Statistics) — what NOAA uses for NBM
   - Kalman filter / adaptive bias correction
   - Quantile mapping
   - Machine learning approaches (random forest, gradient boosting)
   - What works best for daily high temperature prediction specifically?

5. **Uncertainty Quantification:** Our Gaussian CDF with static std failed. We designed (but didn't implement) quantile regression. What approaches work best for producing calibrated probability distributions over temperature brackets?
   - Quantile regression (linear? GBM?)
   - Ensemble spread as uncertainty proxy
   - NBM's native percentile output
   - Bayesian approaches
   - Conformal prediction
   - EMOS (Ensemble Model Output Statistics)
   - **Tail bracket handling:** Kalshi has open-ended brackets (e.g., "≤30°F", "≥90°F"). Piecewise-linear CDFs and Gaussian tails both risk leaking probability mass into physically impossible ranges. What's the correct approach for assigning probability to open-ended tail brackets without distorting the rest of the distribution?

6. **Machine Learning Approaches:** Is there a viable ML approach for this problem that we should consider? Specifically:
   - Could gradient boosting (XGBoost/LightGBM) or random forests replace or augment OLS for bias correction?
   - Neural networks for probabilistic temperature forecasting — viable at our scale (single station, ~1,900 days of history)?
   - What about using ML at the ensemble layer (stacking) rather than inverse-Brier weighting?
   - Risk of overfitting with limited training data — how much history do ML approaches need vs. OLS?
   - Any ML approaches specifically designed for small-data probabilistic forecasting?
   - Key question: where in the pipeline does ML add value vs. add complexity? (feature engineering? bias correction? ensemble weighting? uncertainty estimation? all of the above?)

7. **Backtesting Methodology:** We use walk-forward expanding-window evaluation. Should we consider alternatives?
   - Walk-forward expanding window (current) vs. rolling window vs. sliding window
   - Cross-validation approaches for time series (blocked CV, purged CV)
   - How to handle concept drift from NWP model upgrades (we confirmed ECMWF has had 5 IFS transitions in our data window — no detectable bias discontinuity, but should we worry?)
   - **Historical data reconstruction:** For sources we haven't been collecting (NBM, GraphCast, etc.), what are the options for building historical archives for backtesting? Reanalysis data? Archived forecasts? How far back can we go, and how does archive depth affect backtest reliability?
   - Minimum sample size for reliable walk-forward evaluation — we have ~1,900 days for HRRR but may have less for newer sources
   - Should backtesting be purely model-layer, or should strategy-layer backtesting (P&L simulation) be integrated from the start?

### Pillar 2: Data Source Audit

For NYC daily high temperature prediction specifically, evaluate each of these data sources:

**NWP Models (Currently Used):**
- HRRR (3km, hourly runs, 18h/48h horizon) — we only use 4 of 24 runs
- GFS (0.25°, 4 runs/day, 16-day horizon) — we only use 1 run via Open-Meteo
- ECMWF IFS (0.1°, 2 runs/day, 15-day horizon) — we only use 1 run via Open-Meteo

**NWP Models (Not Yet Used):**
- NBM (National Blend of Models) — 2.5km, hourly updates, probabilistic output
- NAM (North American Mesoscale) — 3km nest, 4 runs/day
- ICON (German Weather Service) — global + EU nest
- UKMO (UK Met Office)
- Canadian GEM/GDPS
- JMA (Japan Meteorological Agency)

**AI/ML Weather Models:**
- Google DeepMind GraphCast — available via Open-Meteo
- Google WeatherNext 2
- Huawei Pangu-Weather
- NVIDIA FourCastNet
- Microsoft Aurora
- Are any of these useful for surface temperature at a single point?

**Specialized Sources:**
- Wethr.net — Kalshi-focused weather API, 16+ models, 30 stations. Worth the subscription?
- Acme Weather — just launched Feb 2026, Dark Sky founders, uncertainty-aware forecasts. Any API?
- MOS guidance (GFS MOS, NAM MOS) — statistical post-processing from NOAA
- LAMP (Localized Aviation MOS Program) — hourly updates

**Observation Sources (Currently Used):**
- Synoptic API (KNYC, KLGA, KEWR)
- AWC / Aviation Weather Center (METAR + SPECI)
- Iowa State Mesonet (historical backfill)
- NWS CLI (settlement truth)
- ACIS/RCC (historical daily backfill)

**Questions:**
- Which of these sources are redundant vs. complementary?
- Which have the most independent signal for NYC daily high temps?
- What's the availability/cost/latency/resolution of each?
- **For each source: what is the data volume (records/day), update frequency, and latency from model run to data availability?** We need to know when each source's data becomes available relative to Kalshi market hours (market opens ~10 AM ET the prior day, settles after midnight ET).
- **Historical archive depth:** How far back does each source's historical data go? Can we backfill for backtesting, or only collect going forward? What format (GRIB, API, CSV)?
- Are there important data sources we're not even considering?
- For Open-Meteo specifically: what model runs are available? We only used midnight UTC — are there more?

**Data Acquisition Strategy:**

Our current HRRR backfill approach (Herbie → AWS GRIB → extract single grid point) works but is painfully slow — each forecast hour is a separate GRIB download, and we hit rate limits. Backfilling all 24 hourly runs × ~1,900 days × 18 forecast hours = ~820,000 individual fetches. At our current throughput, that's days of wall-clock time. For the rebuild we need to potentially acquire data for multiple new models on top of that.

- **What are the fastest/most efficient ways to bulk-acquire historical NWP forecast data for a single grid point?**
  - GRIB file-by-file via Herbie/AWS (current approach) vs. bulk archive download vs. third-party APIs (Open-Meteo, GribStream, etc.)
  - Open-Meteo's Historical Forecast API returns pre-extracted point data as JSON — no GRIB parsing needed. Does it cover all run hours? All models? How far back?
  - AWS Open Data (NOAA) has full GRIB archives — is there a way to batch-extract a single lat/lon across thousands of runs without downloading full grids?
  - GribStream.com claims fast historical forecast delivery — is this viable? Cost?
  - Are there academic or research data repositories with pre-processed station-level NWP forecast archives?
- **Parallelization and infrastructure:**
  - Can we parallelize across multiple API endpoints simultaneously (e.g., Open-Meteo for GFS/ECMWF, Herbie for HRRR, AWS for NBM)?
  - Should we spin up a temporary cloud instance for the heavy backfill work rather than running on a MacBook?
  - Our DuckDB crashed on long write sessions — what's the best approach for ingesting hundreds of thousands of records? (We solved this with temp DBs + merge, but is there a better pattern?)
- **"Good enough" data shortcuts:**
  - For models where full historical GRIB backfill is impractical, can reanalysis datasets (ERA5, RAP analysis) serve as a proxy for backtesting?
  - Is there a minimum backfill depth that gives reliable walk-forward results? (e.g., 2 years vs. 5 years — diminishing returns?)
  - For AI models (GraphCast, Pangu) that are relatively new — how do we backtest without historical archives? Hindcast datasets?

**Data Architecture & Schema Design:**

Our current schema evolved organically — `forecasts` table got a `model_name` column bolted on in Phase 3, `forecast_extended` is a separate table for weather variables, observations and settlements live in separate tables. Before we add 5-10 more data sources, we should get the foundation right.

- **What's the optimal schema design for multi-model NWP forecast data at a single station?**
  - One wide table for all models vs. separate tables per model vs. a normalized design with model metadata?
  - How should we handle models with different temporal resolutions (hourly HRRR vs. 3-hourly GFS vs. daily NBM percentiles)?
  - Should raw forecasts and derived features (bias-corrected values, divergence signals, ensemble output) live in the same table or separate layers?
  - Columnar storage considerations for DuckDB — what partitioning/indexing strategy works best for walk-forward queries that repeatedly scan "all data before date X"?
- **Feature store pattern:** Should we pre-compute and store derived features (bias-corrected forecasts, divergence signals, ensemble probabilities) in the DB, or compute them on-the-fly during backtesting?
  - Our current backtester uses a two-level cache (Level 1 = bulk SQL per run_hour, Level 2 = divergence features per update_hour). This was fast enough but got complex. Is there a cleaner pattern?
  - Trade-off: pre-computed features are faster but create staleness risk when the model changes. On-the-fly is always fresh but slower.
- **Data pipeline architecture:** What's the best practice for organizing ingestion → storage → feature engineering → model training → backtesting?
  - Should we adopt a medallion architecture (bronze/raw → silver/cleaned → gold/features)?
  - How do production weather forecasting systems organize their data pipelines?
  - Any DuckDB-specific patterns or anti-patterns for time series forecast data at this scale (~millions of rows, single-digit GB)?
- **Observation-forecast alignment:** Our current system matches forecasts to observations by timestamp with interpolation. With 24 hourly HRRR runs + multiple other models, the join logic gets complex. What's the cleanest way to align multi-model forecasts with observations and settlement truth for training and evaluation?

### Pillar 3: Competitive Intelligence

1. **What tools and models do successful Kalshi weather traders use?** Check:
   - Kalshi Discord / community forums
   - Reddit (r/kalshi, r/predictionmarkets, r/weathertrading)
   - Twitter/X weather trading discussions
   - Any public strategies, blog posts, or research papers

2. **Wethr.net specifically:** They're built for Kalshi weather markets. What's their data offering? Is it worth subscribing? Do they provide edge or just convenience? What models do they aggregate?

3. **What does the academic literature say about post-processing NWP for surface temperature?** Specifically:
   - EMOS (Ensemble Model Output Statistics) — Gneiting et al.
   - BMA (Bayesian Model Averaging)
   - NGR (Non-homogeneous Gaussian Regression)
   - Modern ML post-processing approaches
   - Any papers specifically about probabilistic temperature forecasting for decision-making

4. **Are there other prediction markets for weather besides Kalshi?** Polymarket, Metaculus, etc.? What can we learn from how they price weather events?

### Pillar 4: Kalshi Strategy Integration

The model doesn't exist in a vacuum — it exists to make profitable trades on Kalshi. The strategy layer should inform how we build the model, not just consume its output.

1. **Should model optimization target Brier score or trading P&L?** We've been optimizing global Brier score, but:
   - Our early strategy backtest showed the only profitable zone was center brackets (35-50¢) with ~50% win rate
   - The 11% fee hurdle means we need large model-vs-market displacements (>10-15%) to profit
   - Should the model optimize for detecting large displacements rather than overall calibration?
   - Is there a loss function that better aligns with trading profitability than Brier score?

2. **Kalshi market microstructure:** How should market mechanics influence model design?
   - Market opens ~10 AM ET the prior day — the first ~18 hours have thin liquidity and wide spreads. Is this where edge lives (stale pricing) or where execution is worst?
   - Settlement is based on NWS CLI (reported after midnight ET). Does the model need to predict what NWS will report, or what the actual temperature is? (These can differ due to reporting methodology.)
   - Bracket structure changes daily based on the forecast center. How should the model handle variable bracket boundaries?
   - Liquidity is concentrated in center brackets. Should the model focus on where it can actually execute, or seek edge in illiquid tails?

3. **Position sizing and bankroll management:** Should these influence the model at all?
   - With $100 starting capital, minimum bet sizes constrain which brackets are even tradeable
   - Kelly criterion requires calibrated probabilities — does this mean calibration (ECE) matters more than raw accuracy (Brier)?
   - Should we be modeling expected P&L per bracket-hour combination rather than just probability accuracy?

4. **Timing of trades:** When should we enter and exit positions?
   - Our model improves through the day (more obs = better prediction), but the market also improves (less edge)
   - Is there an optimal entry window where model-improvement-rate exceeds market-improvement-rate?
   - Should the model produce a "confidence-adjusted edge" signal that accounts for both model quality and market staleness at each hour?
   - Early research suggests overnight/morning hours have thinnest market pricing — is this where the strategy should focus?

5. **End-to-end optimization:** In quantitative finance, the best systems optimize the full pipeline (signal → sizing → execution) jointly, not in isolation. Should we:
   - Build the model to directly output "edge per bracket" rather than "probability per bracket"?
   - Incorporate market prices as a model input (i.e., predict where the market is wrong, not just what the temperature will be)?
   - Use the strategy backtester's P&L as the training signal for the model, rather than Brier score?

---

## Desired Output Format

For each research question, please provide:
1. **Answer** — direct, concise
2. **Evidence** — sources, papers, data points
3. **Recommendation** — what we should do given our constraints ($100 capital, NYC only, Python/DuckDB stack)
4. **Priority** — high/medium/low for our specific use case

At the end, provide a **recommended order of operations** for rebuilding the pipeline, considering:
- What data to acquire first (and how to backfill historical archives for backtesting)
- What model architecture to use (OLS? ML? hybrid?)
- What order to build the layers (data → bias → ensemble → calibration → strategy, or something different?)
- Where the highest-ROI improvements are
- How the strategy layer should integrate with model development (separate phases or co-developed?)
- What backtesting methodology to use and minimum data requirements

---

## What I DON'T Need

- General weather API recommendations not relevant to daily high temperature
- Paid enterprise solutions beyond $50/month (we're bootstrapping with $100)
- Approaches requiring GPU compute (we're running on a DigitalOcean droplet + MacBook)
- Theory without practical implementation guidance

---

## Appendix: Current System State (Facts for Reference)

This is what actually exists today — not aspirational, not in-progress. Use this to ground your recommendations.

### Phase Results (What Worked, What Failed)
| Phase | What | Result | Brier | Key Finding |
|-------|------|--------|-------|-------------|
| 0 | Data foundation + backtester | PASS | — | Infrastructure works well |
| 1 | Walk-forward per-run-hour bias (4 HRRR runs) | PASS | 0.8356 | Per-hour correction 3x better than flat bias |
| 2 | OLS regression on error (fcst_high + month) | PASS (+4.5%) | 0.7979 | fcst_high is dominant feature |
| 2B | Intraday obs divergence features | PASS (-3.2%) | 0.7722 overall, 0.7034 at 18 ET | running_max does 87% of the work; obs only help after 14 ET |
| 3 | 3-model ensemble (HRRR+GFS+ECMWF) | PASS | 0.7808 (with P2B at 18 ET) | Error correlations all < 0.7; equal weights ≈ optimized |
| 3.5 | Adaptive inverse-Brier weights | PASS (+1.3%) | 0.7705 | GFS ~35%, ECMWF ~34%, HRRR ~32% |
| 3.6 | Neighbor station obs (KLGA/KEWR) | KILLED | — | Marginal improvement; problem is std, not obs coverage |
| 3.7 | Dynamic variance OLS | KILLED | — | Per-hour training made features constant; all variants 4-5% worse |
| 3.8 | Quantile regression | DESIGNED only | — | Never implemented; designed as Gaussian replacement |

### Kalshi Market Baseline (What We Need to Beat)
| Time | Kalshi Market Brier | Our Best Brier | Gap |
|------|-------------------|----------------|-----|
| After 00z (overnight) | 0.63 | 0.84 | -0.21 |
| After 06z (morning) | 0.62 | ~0.80 | -0.18 |
| After 12z (afternoon) | 0.44 | ~0.78 | -0.34 |
| After 18z (evening) | 0.14 | 0.70 | -0.56 |

### Strategy Backtest (Baseline Model — NOT Profitable)
- 1,626 trades, Nov 2024 → Feb 2026
- Return: -44.7%, Win rate: 4.4%
- 76% of bets on sub-10¢ tail brackets (fat Gaussian tails)
- One bright spot: 35-50¢ center brackets hit ~50% win rate
- Root cause: static std → model can't collapse uncertainty like the market does

### Current Data Inventory
| Dataset | Volume | Coverage |
|---------|--------|----------|
| HRRR forecasts (4 run hours) | ~7,600 run-days | Jun 2021 → present |
| GFS forecasts (1 run/day via Open-Meteo) | ~1,800 days | Mar 2021 → present |
| ECMWF IFS forecasts (1 run/day via Open-Meteo) | ~1,800 days | 2017 → present |
| KNYC observations (1-min ASOS) | ~57,000 rows | Jun 2021 → present |
| KLGA/KEWR observations | ~583,000 rows each | Jun 2021 → present |
| NWS daily settlement (CLI + ACIS) | ~1,900 days | Jun 2021 → present |
| Kalshi settlements | All historical | Aug 2021 → present |
| Kalshi candlesticks (1-min OHLCV) | All historical | Aug 2021 → present |
| Kalshi trades | All historical | Aug 2021 → present |

### Database Schema (Simplified)
- `forecasts` — station_id, model_run, valid_at, temp_f, model_name (UNIQUE on station+run+valid+model)
- `forecast_extended` — same keys + dewpoint, humidity, wind, pressure, cloud, precip, radiation, CAPE
- `observations` — station_id, observed_at, temp_f, raw_metar, ingest_source
- `nws_daily` — station_id, obs_date, max_temp_f, source ('ACIS' or 'NWS_CLI')
- `kalshi_settlements` — market_ticker, event_date, floor/cap_strike, settled_yes, volume
- `kalshi_candlesticks` — market_ticker, end_period_ts, OHLCV + open_interest
- `kalshi_trades` — trade_id, ticker, yes_price, count, created_time

### Tech Stack
- Python 3.9, DuckDB 1.4.4
- Herbie (HRRR GRIB access via AWS)
- Open-Meteo (GFS/ECMWF historical forecasts)
- Kalshi API v2 (RSA-PSS auth)
- Flask + Plotly (dashboard)
- Docker on DigitalOcean (deployment)
- MacBook for development, no GPU
