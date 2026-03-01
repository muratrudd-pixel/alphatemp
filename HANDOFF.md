# Handoff - 2026-03-01 (18:30 UTC)

## Current State
- **Level 3 multimodel_full: CHAMPION** — Brier 0.6511, +14.25% over HRRR OLS (0.7593). Helps all 24 hours.
- **Level 2 learned weights: IMPLEMENTED BUT DISABLED** — too slow for full backtest without .raw() caching. Code is in ensemble.py, disabled in phase2_ensemble.py.
- **All prior uncommitted work from last session: COMMITTED** — settlement-day fxx filter, XGBoost extended features (commits 835a0de, a7f6d75, ce08354)
- **GFS 12z/18z backfill: STATUS UNKNOWN** — 6 EC2 instances were running last session. Check status.
- **Probability engine: DISABLED** — `PROBABILITY_ENGINE_ENABLED = False`

## What Was Done This Session

### Level 3: Multi-model OLS Stacking (backtester.py)
Added `_get_latest_model_fcst_highs_bulk`, `_fit_and_predict_multimodel`, `_make_multimodel_regression_model`, and two pre-built instances:
- `wf_multimodel_full` — HRRR + GFS + ECMWF, features [0-5]: hrrr_high, gfs_high, ecmwf_high, sin/cos month, delta_temp
- `wf_multimodel_hrrr_gfs` — HRRR + GFS only, features [0,1,3,4,5]

Training flow: uses `_walk_forward_regression_data` for HRRR base, merges secondary model highs via bulk SQL, imputes missing with HRRR value. Separate `_fit_and_predict_multimodel` to avoid touching existing OLS.

### Level 2: Learned Mixture Weights (ensemble.py)
Added `_generate_simplex_weights`, `_grid_search_simplex_weights`, `_compute_brier_for_weights`, `make_learned_weight_ensemble_fn`. Walk-forward weight optimization using grid search over weight simplex (231 points for 3 models at 0.05 step). Falls back to equal weights when <90 dates of aligned history.

**Disabled in phase2_ensemble.py** — each eval calls `.raw()` on 3 models × 365 lookback dates, each an expanding-window OLS fit. Total: ~hours for full backtest. Needs precomputation/caching of `.raw()` results per (run_hour, model, date) before enabling.

### Comparison Script (scripts/phase2_ensemble.py)
Added multimodel_full, multimodel_hg candidates. Learned_weights commented out with TODO.

### Tests (tests/test_ensemble.py)
5 new tests, all passing:
- `test_multimodel_uses_gfs_fcst_high` — bulk SQL returns GFS data, OLS gets both model highs
- `test_multimodel_imputes_missing` — ECMWF absent → imputed with HRRR
- `test_multimodel_returns_valid_probs` — bracket probs sum to ~1.0
- `test_grid_search_finds_optimal` — synthetic 2-model scenario, good model gets weight ≥ 0.7
- `test_learned_weights_no_data_returns_none` — empty DB returns None

### Diagnostics (scripts/multimodel_diagnostics.py)
New script with 4 diagnostic panels. Key findings:

**OLS Coefficients** — Two regimes:
- Hours 0-5z: HRRR coeff ~+0.9, GFS **negative** (~-0.65), ECMWF ~-0.2. GFS used as contrarian signal.
- Hours 6-23z: GFS coeff ~0.00 (ignored), ECMWF **strongly negative** (~-0.55 to -0.71). ECMWF is the contrarian anchor.
- Negative coefficients = OLS exploits model disagreement as a bias correction feature.

**Data Coverage** — ~95% for both GFS and ECMWF at all hours. Imputation to HRRR only ~5%.

**Seasonal Brier** — Winter +16.6%, Spring +14.2%, Summer +9.5%, Fall +16.5%. Summer least benefit.

**Error Distribution** — MAE 1.91→1.36°F. Within-2°F: 72%→85%. Std: 2.61→1.84 (30% tighter).

## Uncommitted Changes (5 files)
```
M services/backtester.py          — Level 3: _get_latest_model_fcst_highs_bulk, _fit_and_predict_multimodel, _make_multimodel_regression_model, wf_multimodel_full, wf_multimodel_hrrr_gfs + import _find_latest_run_hour from ensemble
M services/ensemble.py            — Level 2: _generate_simplex_weights, _grid_search_simplex_weights, _compute_brier_for_weights, make_learned_weight_ensemble_fn
M tests/test_ensemble.py          — 5 new tests for Level 2 + Level 3
A scripts/phase2_ensemble.py      — comparison script with 8 candidates (learned_weights disabled)
A scripts/multimodel_diagnostics.py — 4-panel diagnostic script
```

## What Needs to Happen Next
1. **Commit these changes** — multimodel_full champion + Level 2 code + tests + diagnostics
2. **Phase 3: Dynamic uncertainty** — static std is still the Brier bottleneck (MAE improved but std is fixed). Quantile regression or EMOS variance should further improve.
3. **Re-run strategy backtester with multimodel_full** — see if the 14% Brier improvement translates to P&L improvement
4. **Investigate summer gap** — +9.5% vs +14-17% other seasons. Models agree more in summer → less spread signal.
5. **Future: Level 2 learned weights optimization** — precompute `.raw()` results per (run_hour, model, date) to make grid search feasible. May not be worth it given Level 3's 14% already.
6. **Check EC2 fleet status** — GFS 12z/18z backfill may be done by now.

## Decisions Made This Session
- multimodel_full is Phase 2 champion (Brier 0.6511, +14.25% over HRRR OLS)
- ECMWF adds +9% beyond HRRR+GFS alone — worth keeping despite only 00z data
- Learned weights deprioritized — OLS coefficients already find optimal linear relationship, learned weights would need nonlinear interactions to improve further
- GFS standalone is terrible (-26.4%) but GFS as a *feature* in OLS is valuable — the OLS compensates for GFS biases

## Pre-existing Test Failures (9 total, not from this session)
- `test_db`: test_market_ticks_unique_constraint, test_migrate_forecasts_model_name_from_old_schema
- `test_phase1_gate`: all 5 tests (market_ticks schema mismatch)
- `test_forecast`: test_fetcher_deduplicates
- `test_web_dashboard`: test_brier_comparison_endpoint

## Key Context
- SSH key: ~/.ssh/alphatemp-hrrr.pem, user: ec2-user
- PYTHONPATH must be set: `PYTHONPATH=/home/ec2-user/alphatemp`
- **Authoritative plan:** `docs/plans/2026-02-27-rebuild-design.md`

## Verification Queries
```bash
# 1. Run ensemble tests (expect 25 pass: 18 ensemble + 7 multimodel_safety)
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. pytest tests/test_ensemble.py tests/test_multimodel_safety.py -v

# 2. Verify multimodel_full Brier (quick sanity — run on small date range)
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. python scripts/phase2_ensemble.py --start 2025-01-01 --end 2025-06-01

# 3. Full comparison (all 8 candidates, ~40 min)
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. python scripts/phase2_ensemble.py

# 4. Diagnostics (coefficients, coverage, seasonal, residuals — ~25 min)
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. python scripts/multimodel_diagnostics.py

# 5. EC2 fleet check
# ssh -i ~/.ssh/alphatemp-hrrr.pem ec2-user@<IP> "ps aux | grep backfill | grep -v grep; tail -3 ~/alphatemp/logs/gfs_*_gap_*.log"
```
