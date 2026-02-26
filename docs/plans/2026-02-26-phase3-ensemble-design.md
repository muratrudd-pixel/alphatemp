# Phase 3 Design: Multi-Model Ensemble

**Created:** 2026-02-26
**Status:** Approved
**Author:** Russell + Claude
**Depends on:** Phase 2B complete (Brier 0.7722 overall, 0.7034 at 18 ET)

---

## Goal

Test whether combining bias-corrected forecasts from multiple weather models (HRRR + GFS + ECMWF) reduces Brier score compared to HRRR alone.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Models | GFS + ECMWF | Two independent global models. GFS is NOAA's workhorse, ECMWF is the gold standard. NAM deferred (too correlated with HRRR). |
| Data source | Open-Meteo Historical Forecast API | No GRIB infrastructure needed. Returns hourly temp_2m at lat/lon. If ensemble proves value, upgrade to Herbie/GRIB for per-run-hour GFS data. |
| Run hour handling | HRRR: per-run-hour (00z/06z/12z/18z). GFS/ECMWF: daily (one prediction per day) | Open-Meteo doesn't expose specific run hours. Assembles most-recent-run data per valid hour — same as querying live. |
| Bias correction | Per-model independent Phase 1+2+2B | Each model gets its own walk-forward bias, regression, and observation divergence. No assumptions carried from HRRR. |
| Variable ablation | Full ablation per model | delta_temp was dead for HRRR but may matter for GFS/ECMWF. Test all combos independently. |
| Ensemble method | Gaussian mixture, equal weights | P(bracket) = (1/N) × Σ [Φ_model(upper) - Φ_model(lower)]. Not a single Gaussian — preserves multi-modality. |
| Weighting | Equal first, then inverse-Brier if gate passes | Prove it or kill it. Don't optimize weights until equal weights beat single model. |
| Phase 2B in ensemble | Per-model Phase 2B, then ensemble | Each model's observation divergence is computed against its own forecast curve. Activate after 14 ET only. |
| Evaluation structure | Variable-composition ensemble at each HRRR run hour | At 00z: HRRR-00z + GFS-daily + ECMWF-daily. GFS/ECMWF are static within the day; only HRRR varies by run hour. |

## Architecture

### Data Layer

**Schema migration:**
```sql
ALTER TABLE forecasts ADD COLUMN model_name VARCHAR DEFAULT 'hrrr';
-- Recreate unique constraint to include model_name
-- New: UNIQUE (station_id, model_run, valid_at, model_name)
```

All existing queries get `AND model_name = 'hrrr'` filter to prevent silent corruption when multi-model data is present.

**Open-Meteo backfill:**
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Parameters: `latitude=40.78&longitude=-73.97&hourly=temperature_2m&temperature_unit=fahrenheit&models=gfs_seamless` (or `ecmwf_ifs`)
- Date range: overlap window with HRRR data (~1,900 days)
- Write to temp DuckDB files, merge (same pattern as HRRR backfill)
- For GFS/ECMWF rows: `model_run` = midnight UTC of the forecast day, `valid_at` = each hourly timestamp

**Important assumption:** Open-Meteo historical forecasts assemble the most recent model run for each valid hour — NOT the best-performing run. This mirrors live API behavior (you always get the freshest available data). Verify by spot-checking 10 random days against NOMADS archives.

### Per-Model Bias Pipeline

Each model independently goes through:

1. **Phase 1: Walk-forward bias** — expanding-window mean bias and std
   - HRRR: per-run-hour (4 sets of stats per day)
   - GFS/ECMWF: single daily stats
   - Minimum 90-day training window

2. **Phase 2: Feature-conditioned regression** — walk-forward OLS
   - `error = β₀ + β₁·fcst_high + β₂·sin(2π·month/12) + β₃·cos(2π·month/12) + β₄·delta_temp`
   - Full ablation per model:
     - fcst_high only
     - fcst_high + month
     - fcst_high + month + delta_temp
     - Each variable solo
   - Champion regression selected per model based on Brier improvement

3. **Phase 2B: Observation divergence** — residual regression on Phase 2 error
   - Divergence features computed against each model's forecast curve
   - `running_max_divergence`, `instantaneous_divergence` (and full set for ablation)
   - Separate OLS per update hour (14-18 ET)
   - Only activate after 14 ET crossover

### Ensemble Layer

**New service:** `services/ensemble.py`

**Gaussian mixture combination:**
```
P(bracket) = Σ_i  w_i × [Φ_i(upper) - Φ_i(lower)]

where:
  w_i = 1/N (equal weight)
  Φ_i = CDF of model i's corrected Gaussian N(μ_i, σ²_i)
  μ_i = fcst_high_i - predicted_bias_i (- phase2b_adjustment_i after 14 ET)
  σ_i = residual_std_i from walk-forward regression
```

This is a proper mixture distribution, not a single Gaussian. Preserves multi-modality when models disagree.

**EnsembleModelFn:** Wraps multiple `ModelFn` instances into a single `ModelFn` compatible with the backtester. At each evaluation point:
1. Calls each constituent model function
2. Collects (μ, σ) from each
3. Computes bracket probabilities from the mixture
4. Returns bracket probability dict

## Implementation Sequence

### Step 1: Schema Migration + Open-Meteo Backfill
- Add `model_name` column to `forecasts`
- Rebuild unique constraint: `(station_id, model_run, valid_at, model_name)`
- Add `AND model_name = 'hrrr'` to ALL existing forecast queries
- New: `scripts/backfill_openmeteo.py` — backfill GFS + ECMWF to temp DBs, merge
- **Gate:** All existing tests pass. Spot-check 10 days vs NOMADS.

### Step 2: Per-Model Phase 1 Analysis
- Run walk-forward bias on GFS and ECMWF independently
- Report: mean bias, std, Brier per model
- Compute error correlation between models (HRRR vs GFS, HRRR vs ECMWF, GFS vs ECMWF)
- **Gate:** At least one new model has error correlation < 0.7 with HRRR

### Step 3: Per-Model Phase 2 Regression + Ablation
- Full ablation per model (all variable combos)
- Select champion regression per model
- **Gate:** Each model's regression beats its own Phase 1 baseline

### Step 4: Equal-Weight Ensemble Evaluation
- Build `services/ensemble.py` with Gaussian mixture
- Backtest 1/3-weighted ensemble across all HRRR run hours
- **Gate (Phase 3 kill condition):** Ensemble Brier < best single-model Brier

### Step 5: Per-Model Phase 2B + Observation Divergence
- Apply Phase 2B independently per model
- Re-evaluate ensemble with Phase 2B corrections (14-18 ET)
- **Gate:** Ensemble + Phase 2B beats ensemble without Phase 2B

### Step 6: Learned Weights (conditional)
- Only if Step 4 gate passes
- Test inverse-Brier weighting (weight by recent walk-forward Brier performance)
- Evaluate whether learned weights beat equal weights
- If not, stick with equal weights

## Test Strategy

- Schema migration: existing tests must pass with `model_name` filter addition
- Backfill scripts: integration tests with small date range (7 days)
- Ensemble logic: unit tests with known Gaussian distributions
- Each step extends test coverage incrementally

## Kill Conditions

1. **Error correlation > 0.7** for all model pairs → ensemble can't add much, reconsider model selection
2. **Equal-weight ensemble doesn't beat best single model** → stick with HRRR, Phase 3 killed
3. **Phase 2B hurts ensemble** → use ensemble without Phase 2B for non-HRRR models

## Future Upgrade Path

If Phase 3 passes gates with Open-Meteo daily data:
- Upgrade GFS to Herbie/GRIB for per-run-hour granularity (Approach B)
- This would give GFS the same 4-run-hour structure as HRRR
- Expected to improve ensemble further since GFS temporal freshness would vary by run hour
- ECMWF stays on Open-Meteo (no free GRIB access)

## Data Constraints

- Need at least 90 days of overlapping data across all models before scoring
- Open-Meteo GFS data starts 2021-03-23, ECMWF starts 2017-01-01
- Our HRRR data covers ~1,900 days — overlap window is ~1,800 days for GFS
- All timestamps UTC unless explicitly stated
- Settlement uses Local Standard Time (UTC-5 year-round)
