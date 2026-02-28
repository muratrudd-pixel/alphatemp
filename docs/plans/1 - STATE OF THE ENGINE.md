# AlphaTemp — State of the Engine

**Date:** 2026-02-27
**Purpose:** Complete technical summary of the current codebase for external review.
**What exists:** Everything below is implemented and functional unless explicitly marked otherwise.

---

## 1. Data Foundation & Ingestion

### Forecast Sources

**HRRR (live + backfill)** — `services/forecast.py`, `services/forecast_backfiller.py`
- Source: AWS via Herbie library, GRIB format
- Grid point extraction: nearest-neighbor using cosine-weighted L1 distance at Central Park (40.7789, -73.9692)
- **Only 4 of 24 hourly runs backfilled** (00z, 06z, 12z, 18z). The 20 standard runs (01z-05z, 07z-11z, 13z-17z, 19z-23z) were never fetched.
- Live fetcher: async loop, 2-min polling, DB-aware (checks existing runs, fetches missing ones newest-first)
- `fxx_range = 1..18` (18 forecast hours per run)
- Early exit on 2 consecutive fxx misses (run not published yet)
- Timestamps stored as naive UTC in DuckDB

**GFS & ECMWF (backfill only)** — `scripts/backfill_openmeteo.py`
- Source: Open-Meteo Historical Forecast API
- GFS: `gfs_seamless` model, data from 2021-03-23
- ECMWF: `ecmwf_ifs` model, data from 2017-01-01
- **Only midnight UTC run available** — `model_run` is hardcoded to `datetime(year, month, day, 0, 0)` regardless of actual GFS/ECMWF run times
- Batched in 60-day chunks, written to **temp DB** (`data/backfill_<model>.duckdb`), requires manual merge into main DB
- Resume support via `MAX(valid_at)` query
- Extended variables (dewpoint, humidity, wind, pressure, cloud, precip, radiation, CAPE) stored in separate `forecast_extended` table — **never used by any model**

### Observation Sources

**Synoptic API** — `services/ingestor.py` (SynopticIngestor)
- Stations: KNYC, KLGA, KEWR
- 90-minute lookback window, 60-second poll cycle
- Parses from METAR remarks: T-group (0.1°C precision), 6-hour synoptic max/min
- Stub METARs (missing Zulu timestamp) → all temperature fields nulled
- Retry: 3 attempts with 2s base delay

**AWC / Aviation Weather Center** — `services/iem_ingestor.py` (IEMIngestor)
- Only polls KNYC (Central Park)
- Critical for SPECI reports (special observations on significant weather changes) that IEM/Synoptic miss for non-airport stations
- Uses `obsTime` (Unix epoch), not `reportTime` (which rounds to :00)
- 60-second poll cycle, public API (no auth needed)

**Iowa State Mesonet** — `scripts/backfill_iem.py`
- Historical ASOS backfill for KNYC, KLGA, KEWR
- 30-day chunks, free API

### Settlement Truth

**NWS CLI** — `services/nws_fetcher.py`
- Source: `api.weather.gov/products/types/CLI` from KOKX (Upton, NY) office
- Parses TODAY and YESTERDAY sections for max/min temperatures
- Preliminary available ~4:30 PM ET, final ~1:30 AM ET
- **CLI always overwrites ACIS** via delete-then-insert pattern
- 30-minute poll cycle

**ACIS / RCC-ACIS** — same file
- Source: `data.rcc-acis.org/StnData`
- Backfill/catch-all, polled every 4 hours
- Standard INSERT with duplicate skip (does not overwrite CLI)
- `backfill(days_back=365)` for historical population

### Market Data

**Kalshi API v2** — `services/exchange.py` (KalshiClient)
- Auth: RSA-PSS request signing (`timestamp_ms + METHOD + path`)
- `TRADING_ENABLED = False` — hard code-level kill switch
- Available methods: `search_markets`, `get_orderbook`, `get_candlesticks`, `get_trades`, `get_settled_markets`, `get_balance`, `place_order` (blocked), `get_positions`

**Market tick collection** — `services/market_fetcher.py`
- 60-second poll cycle, appends to `market_ticks` (no dedup — no UNIQUE constraint)
- Converts cents (0-99) to decimal probability

**Historical backfill** — `scripts/backfill_kalshi_history.py`
- Three phases: (A) settlements, (B) 1-minute candlesticks, (C) individual trades
- Exponential backoff on HTTP 429
- Ticker date format: YYMMMDD (e.g., `KXHIGHNY-26FEB25`)
- Candlestick time window: `event_date - 2 days` to `close_time`

---

## 2. Database Schema

All in `core/db.py` → `init_db()`. DuckDB 1.4.4. All timestamps naive UTC.

| Table | Key Columns | UNIQUE Constraint | Row Count (approx) |
|-------|-------------|-------------------|--------------------|
| `observations` | station_id, observed_at, temp_f, temp_c_tenth, six_hr_max_c, six_hr_min_c, raw_metar, ingest_source | (station_id, observed_at) | ~1.2M |
| `forecasts` | station_id, model_run, valid_at, temp_f, temp_c, model_name | (station_id, model_run, valid_at, model_name) | ~140K |
| `forecast_extended` | station_id, model_run, valid_at, model_name, dewpoint/humidity/wind/pressure/cloud/precip/radiation/cape | (station_id, model_run, valid_at, model_name) | ~65K |
| `nws_daily` | station_id, obs_date, max_temp_f, min_temp_f, source ('ACIS'/'NWS_CLI') | (station_id, obs_date) | ~1,900 |
| `market_ticks` | market_id, city, captured_at, yes_bid/ask, volume, floor/cap_strike | None | Growing |
| `kalshi_settlements` | market_ticker, event_ticker, series_ticker, event_date, floor/cap_strike, settled_yes, volume | (market_ticker) | All historical |
| `kalshi_candlesticks` | market_ticker, end_period_ts, OHLCV, open_interest | (market_ticker, end_period_ts) | All historical |
| `kalshi_trades` | trade_id, market_ticker, yes_price, count, taker_side, created_time | (trade_id) | All historical |
| `drift_signals` | city, calculated_at, model_run, drift_score, slope_divergence, projected_high | (city, calculated_at, model_run) | — |
| `station_bias` | station_id, calculated_at, mean_bias, std_error, sample_days | (station_id, calculated_at) | — |
| `paper_positions` | id, city, event_date, bracket_floor/cap, model_prob, market_price, edge, net_pnl, status | (id) | — |

Three startup migrations handle schema evolution (6hr columns, ingest_source, model_name). 16 indexes on common access patterns.

---

## 3. The Predictive Model (Phases 1–3)

### DataProvider Interface — `services/data_provider.py`

ABC with two implementations: `LiveDataProvider` (queries DB directly) and `BacktestDataProvider` (time-fenced queries).

**BacktestDataProvider time-fencing (lookahead prevention):**

| Method | Fence |
|--------|-------|
| `get_forecast_high` | `WHERE model_run = self.model_run` (pinned to exact run) |
| `get_forecast_curve` | `WHERE model_run = self.model_run` (pinned) |
| `get_bias_stats` | `WHERE calculated_at <= self.ref_time` |
| `get_observations_in_range` | `effective_end = min(end_utc, self.ref_time)` |
| `get_recent_forecast_highs` | `WHERE model_run <= self.model_run` |
| `get_drift_score` | Returns `0.0` (no historical drift signals) |

Constructor takes `model_run` and `ref_time = model_run + 2h` (HRRR availability lag). Accepts optional shared DuckDB connection (one connection for entire backtest run = fast).

### Model Function Interface

```python
ModelFn = Callable[[BacktestDataProvider, datetime], Optional[Dict[int, float]]]
```

Every model returns `Dict[int, float]` — integer °F temperature → probability. Models also expose `.raw()` returning `Optional[Tuple[float, float]]` = `(center, std)` for ensemble composition.

### Phase 1: Walk-Forward Per-Run-Hour Bias Correction

`walk_forward_model` / `wf_bias_hrrr` / `wf_bias_gfs` / `wf_bias_ecmwf`

- For each evaluation date, queries ALL prior dates with same run_hour
- Computes expanding-window: `mean_bias = mean(fcst_high - actual_high)`, `std_error = std(errors, ddof=1)`
- Center: `fcst_high - mean_bias`, Std: `max(0.3, std_error)`
- Distribution: `N(center, std)`, bracket probs via CDF differences at ±0.5°F boundaries
- Minimum 90 training days before scoring
- Also tested Student-t (df estimated from excess kurtosis, clamped [3, 30]) — killed, no improvement

**Result: Brier 0.8356** (+18% over uniform, +13% over single-bias)

### Phase 2: Walk-Forward OLS Regression on Error

`_make_regression_model(name, feature_indices, model_name)` factory

- Training data: `(error, fcst_high, sin_month, cos_month, delta_temp, [extended_vars])`
- `delta_temp = actual(D-1) - actual(D-2)` using only prior settlement dates
- Month encoded as `(sin(2π·month/12), cos(2π·month/12))`
- OLS via `scipy.linalg.lstsq`
- `predicted_bias = X @ β`, `residual_std = std(residuals, ddof=1+n_features)`
- Center: `fcst_high - predicted_bias`, Std: `max(0.3, residual_std)`

Feature index mapping: `0=fcst_high, 1=sin_month, 2=cos_month, 3=delta_temp, 4-11=extended_vars`

Champion model: `wf_regression_full` = [fcst_high, sin_month, cos_month, delta_temp]

**Result: Brier 0.7979** (+4.5% over Phase 1). fcst_high alone = +4.3%. delta_temp = +0.0% (dead).

### Phase 2B: Observation-Based Residual Correction

Corrects the Phase 2 residual using intraday observations vs. the raw HRRR forecast curve.

**Two-Level Cache Architecture (module-level dicts):**

- **Level 1** — keyed by `(id(con), run_hour, station_id, model_name)`:
  - Bulk-fetches ALL forecast curves for that run_hour
  - Bulk-fetches ALL observations, grouped by ET date
  - Computes Phase 2 expanding-window predictions for ALL dates in one pass
  - Walk-forward: predicts for date D BEFORE adding D to training

- **Level 2** — keyed by `(id(con), run_hour, station_id, update_hour_et, model_name)`:
  - For each historical date: computes Phase 2 residual = `actual_error - phase2_predicted_bias`
  - Computes 4 divergence features at the given `update_hour_et` cutoff
  - Stores `(date, residual, feature_vector)` for regression training

**Divergence Features** — `services/divergence.py` (stateless, no DB):

| Feature | Computation |
|---------|-------------|
| `temp_divergence` | `obs[-1] - fcst_interp[-1]` (latest instantaneous gap) |
| `cumulative_divergence` | `mean(obs_i - fcst_i)` over all paired points |
| `running_max_divergence` | `max(obs) - max(forecast_curve_up_to_T)` |
| `slope_divergence` | OLS slope of `(obs_i - fcst_i)` vs time (≥3 obs, else 0.0) |

Forecast interpolation: linear between hourly HRRR points, clamped to endpoints.

**Inference (per date):**
1. Get Phase 2 bias from Level 1 cache
2. Compute today's 4 divergence features from obs up to `ref_time`
3. Filter Level 2 cache to `date < current_date` (strict walk-forward)
4. OLS on `residual ~ intercept + selected_features`
5. `center = fcst_high - (phase2_bias + phase2b_residual_correction)`

**Result: Brier 0.7722 overall (-3.2%), 0.7034 at 18 ET (-11.8%).** running_max alone does 87%. Obs only help after 14 ET.

### Phase 3: Multi-Model Ensemble — `services/ensemble.py`

**`combine_mixture_brackets(predictions, weights, radius=15)`:**

Each model provides `(mu, sigma)`. For each 1°F bracket k:
```
p_k = Σ_i [ w_i × (Φ((k+0.5-μ_i)/σ_i) - Φ((k-0.5-μ_i)/σ_i)) ]
```

Weighted Gaussian mixture, renormalized. Brackets with p < 0.0001 dropped.

**`make_ensemble_model_fn(model_fns, weights=None)`:**
- Calls `.raw()` on each model → `(center, std)` or `None`
- Drops models returning `None`, renormalizes weights
- Default: equal weights (`1/n`)

**Phase 3.5: Adaptive inverse-Brier weights** — 90-day expanding window. Settled weights: GFS ~35%, ECMWF ~34%, HRRR ~32%.

**Result: Brier 0.7808 (ensemble + P2B at 18 ET), 0.7705 with adaptive weights (+1.3%)**

Error correlations: HRRR-GFS 0.5965, HRRR-ECMWF 0.5684, GFS-ECMWF 0.6421 (all < 0.7).

### Phase 3.6: Neighbor Station Observations — KILLED

`services/neighbor_obs.py` — complete code exists but not wired into live pipeline.

- Walk-forward offset learning (mean KLGA/KEWR - KNYC difference)
- Neighbor divergence features (offset-corrected)
- Peak signal detection (declining temperature after running max)
- Blended KNYC+neighbor curve (fills Central Park's hourly gaps with 1-min neighbor data)

**Result: +1.80% at 18 ET but only +0.2-1.2% during tradeable window (13-16 ET). Below 2% gate. KILLED.**

### Phase 3.7: Dynamic Variance OLS — KILLED

Code removed entirely from codebase.

- Attempted: `log(residual² + 1e-6) ~ variance_features` per-update-hour OLS
- Fatal flaw: Level 2 cache is keyed by (run_hour, update_hour_et), so all training rows had the SAME update_hour → primary feature was constant → absorbed into intercept with zero signal
- Additionally: log(residual²) target inherently noisy — squaring amplifies noise
- **All 9 variants 4-5% WORSE than static-std baseline**

### Phase 3.8: Quantile Regression — DESIGNED BUT NOT IMPLEMENTED

Design doc committed. Key decisions:
- Linear QR with pinball loss, 7 quantiles [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
- Piecewise-linear CDF replaces Gaussian CDF entirely
- Cross-hour training (all update hours in one model — fixes Phase 3.7's fatal flaw)
- 6 features: update_hour, fcst_high, sin/cos_month, running_max, slope

---

## 4. The Kalshi Strategy Backtester — `services/strategy_backtester.py`

### Fee Math

```
TRADING_FEE_RATE = 0.01      # 1% of contract value on entry
SETTLEMENT_FEE_RATE = 0.10   # 10% of profit for winners only

Entry cost:       price × qty × 1.01
Win settlement:   (100 - price) × qty × 0.90 - price × qty × 0.01
Loss settlement:  -(price × qty × 1.01)
Early exit:       revenue × 0.99 - entry × 1.01
```

### Execution Model

- Entry pricing: `yes_ask` (crossing the spread — conservative)
- Execution latency: 60 seconds after model evaluation
- Market data: 1-minute candlesticks from `kalshi_candlesticks`, forward-filled with `is_stale=True` for gaps

### Trigger System — `TriggerDetector`

Generates `(model_time, execution_time, trigger_type)` tuples:
- **Observation triggers:** every distinct `observed_at` in the market window
- **Forecast run triggers:** each run_hour where `run_time + 2h` falls in market window
- Market window: prior day 10 AM ET → event day 11 PM ET

### Sanity Filter — 5 Pre-Trade Checks

1. `model_prob < 0.05` → reject
2. `is_post_peak` (≥3 obs, after 2 PM ET, last 2+ declining, peak >60 min ago) → reject
3. `spread > 10¢` → reject (illiquid)
4. `model_std > 3.5°F` → reject (too uncertain)
5. Recently exited bracket → reject (churn prevention)

### Portfolio Management — `EventPortfolio`

For mutually exclusive brackets, combined EV:
```
EV = Σ_j [ prob_j × (gain_j - Σ_{i≠j} loss_i) ] + (1 - Σ_prob) × (-total_loss)
```

Greedy `optimal_subset`: filter to positive individual EV, sort descending, add bracket if portfolio EV increases.

### Position Manager — `PositionManager`

- Tracks capital in dollars, converts to/from cents
- Rejects trades exceeding available capital
- `settle_day`: applies settlement P&L, returns `TradeRecord` list
- Tracks recently exited brackets for churn prevention

### P&L Analysis — `PnLSimulator`

- `aggregate`: total P&L, return %, Sharpe (daily), max drawdown, win rate, avg win/loss, profit factor
- `bootstrap(n=10000)`: resamples daily P&L → 95% CI on total P&L and Sharpe, `prob_profitable`
- `regime_split`: groups trades by any caller-supplied label function

### Main Loop — `StrategyBacktester.run(model_fn)`

```
For each settlement date:
  Get Kalshi brackets and settled_bracket
  Get triggers (obs + forecast arrivals)

  For each trigger:
    Find best model_run available at model_time
    Create BacktestDataProvider(model_run, ref_time=model_time)
    Call model_fn → bracket probs
    Map to Kalshi brackets
    Get market snapshots at execution_time

    EdgeAnalyzer.record() — always (even during burn-in)

    If before burn_in_end: skip trading

    For each bracket:
      SanityFilter.check()
      Check displacement > min_displacement (default 12%)
      Add to candidates
    EventPortfolio.optimal_subset(candidates)
    PositionManager.open_position() for each selected

  PositionManager.settle_day(event_date, settled_bracket)
```

### Strategy Backtest Results (Baseline Model)

- 1,626 trades, Nov 2024 → Feb 2026
- Return: **-44.7%**, Win rate: 4.4%
- 76% of bets on sub-10¢ tail brackets (fat Gaussian tails → spraying probability into impossible ranges)
- Center brackets (35-50¢): ~50% win rate — only bright spot
- Root cause: static std can't collapse uncertainty; model bleeds probability into tails

---

## 5. Live Probability Engine (Pinned) — `services/probability.py`

**Status: Pinned for rework — edge columns zeroed out.** Code is structurally intact but not producing actionable signals.

**Center:** `fcst_high - historical_bias + drift_score`

**Std:** `max(0.3, historical_std × time_factor × stability_factor × convergence_factor)`

| Factor | Logic |
|--------|-------|
| `time_factor` | Linear decay: 1.0 at ≤8 AM ET → 0.3 at ≥3 PM ET |
| `stability_factor` | `clamp(0.5, 0.5 + variance_of_recent_drift_scores, 1.5)` |
| `convergence_factor` | Spread of last 3 HRRR runs: <1°F → 0.6, >4°F → 1.3, linear between |

**Bracket probs:** Gaussian CDF, ±15 bracket radius, renormalized. Floor std at 0.3°F.

**Known issues:**
- `BacktestDataProvider` returns `drift_score = 0.0` always → backtests don't capture drift signal
- Default std is 2.0°F if no bias stats exist — never calibrated
- No seasonal adjustment on std scaling
- This is the OLD probability engine — the backtester's model functions bypass it entirely

---

## 6. Lookahead Prevention — Full Summary

| Layer | Mechanism |
|-------|-----------|
| Forecast data | `BacktestDataProvider` pins to exact `model_run` |
| Observations | `effective_end = min(end_utc, ref_time)` |
| Bias stats | `WHERE calculated_at <= ref_time` |
| Phase 2 regression training | `obs_date < current_date` (strict) |
| Phase 2B Level 1 (Phase 2 preds) | Sequential pass: predict BEFORE adding date to training |
| Phase 2B Level 2 filter | `d < current_date` enforced at inference time |
| Strategy model run selection | Only runs where `run_time + 2h <= trigger_time` |
| Strategy market prices | Snapshots at `execution_time = model_time + 60s` |

---

## 7. Scoring

**Primary: Multi-category Brier score**
```
BS = Σ_k (p_k - o_k)²
```
where `o_k = 1` if bracket k settled YES. Lower = better. Perfect = 0.0.

**Kalshi bracket mapping:** `map_probs_to_kalshi_brackets` sums 1°F integer probabilities into 2°F Kalshi brackets. Lower tail: `temp < cap_strike`. Interior: `floor ≤ temp ≤ cap`. Upper tail: `temp > floor`. Renormalized if total deviates >0.01 from 1.0.

**Fallback:** 1°F Brier for dates without Kalshi bracket data. +1.0 penalty if actual bracket missing from model output.

---

## 8. Current Performance vs. Market

| Time | Kalshi Brier | Our Best Brier | Gap |
|------|-------------|----------------|-----|
| After 00z (overnight) | 0.63 | 0.84 | -0.21 |
| After 06z (morning) | 0.62 | ~0.80 | -0.18 |
| After 12z (afternoon) | 0.44 | ~0.78 | -0.34 |
| After 18z (evening) | 0.14 | 0.70 | -0.56 |

The gap WIDENS through the day because the market collapses uncertainty (Brier 0.63→0.14) while our model barely moves (0.84→0.70). The static Gaussian std is the primary bottleneck.

---

## 9. Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `RUN_HOURS` | [0, 6, 12, 18] | backtester.py |
| `WALK_FORWARD_MIN_DAYS` | 90 | backtester.py |
| `HRRR_AVAILABILITY_LAG_HOURS` | 2 | backtester.py, data_provider.py |
| `TRADING_ENABLED` | False | exchange.py |
| `POLL_INTERVAL_SECONDS` | 60 | constants.py |
| `FETCH_INTERVAL_SECONDS` | 120 | forecast.py |
| `TRADING_FEE_RATE` | 0.01 | strategy_backtester.py |
| `SETTLEMENT_FEE_RATE` | 0.10 | strategy_backtester.py |
| Settlement station | KNYC (Central Park) | constants.py |
| Neighbor stations | KLGA, KEWR | constants.py |
| Central Park coords | (40.7789, -73.9692) | constants.py |

---

## 10. Tech Stack

- Python 3.9 (no subscripted builtins)
- DuckDB 1.4.4 (known crash on long write sessions — use temp DBs)
- Herbie (HRRR GRIB access via AWS)
- Open-Meteo (GFS/ECMWF historical forecasts)
- scipy.linalg.lstsq (OLS regression)
- scipy.stats.norm (Gaussian CDF)
- Kalshi API v2 (RSA-PSS auth)
- Flask + Plotly (dashboard, port 8050)
- Docker on DigitalOcean (deployment)
- loguru (logging)
- pytest + pytest-asyncio (testing)
- No GPU, no sklearn, no ML libraries currently
