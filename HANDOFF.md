# Handoff - 2026-02-26

## Current State
- **Phase 3 analysis complete. All gates PASSED.** 10 commits on main. 213+ tests passing.
- **Equal-weight ensemble beats HRRR by 2.0% overall** (Brier 0.8606 vs 0.8779)
- GFS + ECMWF data backfilled and merged into main DB (43,224 rows each, 2021-03-23 to 2026-02-25)

## Phase 3 Results

### Per-Model Phase 1 (at 00z)
| Model | Brier |
|-------|-------|
| HRRR  | 0.8598 |
| GFS   | 0.7555 |
| ECMWF | 0.8074 |

### Regression Ablation — all models: `full` wins
| Model | Phase 1 | Best (full) | Improvement |
|-------|---------|-------------|-------------|
| HRRR  | 0.8356  | 0.7979      | +4.5%       |
| GFS   | 0.7555  | 0.7450      | +1.4%       |
| ECMWF | 0.8074  | 0.7917      | +1.9%       |

### Error Correlation (all < 0.7 — GATE PASS)
- HRRR-GFS: 0.5965
- HRRR-ECMWF: 0.5684
- GFS-ECMWF: 0.6421

### Ensemble (equal-weight, 1°F scoring — GATE PASS)
| Run Hour | Ensemble | HRRR Solo | Delta |
|----------|----------|-----------|-------|
| 00z      | 0.8685   | 0.8951    | +3.0% |
| 06z      | 0.8599   | 0.8778    | +2.0% |
| 12z      | 0.8595   | 0.8766    | +2.0% |
| 18z      | 0.8544   | 0.8621    | +0.9% |
| Overall  | 0.8606   | 0.8779    | +2.0% |

## Key Architecture
- `model_name` column on `forecasts` table (DEFAULT 'hrrr', UNIQUE includes model_name)
- GFS/ECMWF model_run = midnight UTC (run_hour=0 in queries)
- Model functions expose `.raw(provider, ref_time) -> Optional[(center, std)]` for ensemble
- `services/ensemble.py`: `combine_mixture_brackets()` and `make_ensemble_model_fn()`
- `Backtester.run()` accepts `model_name` param — threads to provider + has_forecast_data
- Phase 1 factories: `wf_bias_hrrr`, `wf_bias_gfs`, `wf_bias_ecmwf`
- Analysis script: `scripts/phase3_analysis.py` (Parts 1-4)

## What's Done (Tasks 1-11)
1. Schema migration (model_name column + rebuilt UNIQUE constraint)
2. Backtester query filters (4 functions)
3. DataProvider query filters (6 queries)
4. Remaining service filters (19 queries across 6 files)
5. Open-Meteo backfill script
6. Model function parameterization (20 new instances)
7. Ensemble service (Gaussian mixture)
8. Data backfill + merge (GFS 43K rows, ECMWF 43K rows)
9. Phase 1 per-model analysis + Phase 1 model factories
10. Per-model Phase 2 regression ablation (all pass)
11. Error correlation + equal-weight ensemble evaluation (all pass)

## Next Steps (Conditional — gates passed)
1. **Per-model Phase 2B** — apply obs divergence features per model independently
2. **Learned weights** — test inverse-Brier weighting (keep equal if no improvement)
3. **Phase 2B results from prior session:** HRRR Brier 0.7979 → 0.7722 overall (-3.2%), 0.7034 at 18 ET

## Design Doc
`docs/plans/2026-02-26-phase3-ensemble-design.md`

## Implementation Plan
`docs/plans/2026-02-26-phase3-implementation.md`

## Blockers / Open Questions
- The Open-Meteo Historical Forecast API assembles "most recent run" data per hour — NOT specific run hours. Verified this mirrors live API behavior (no hindsight advantage).
- Market tick collection still running (only ~2 days of history) — still a Phase 4 blocker.
- GFS is surprisingly strong at midnight (0.7450 regression_full vs HRRR's 0.8322) — consider higher GFS weight in learned-weights step.
