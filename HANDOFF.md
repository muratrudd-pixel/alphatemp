# Handoff - 2026-02-26

## Current State
- **Phase 3 COMPLETE.** 12 commits on main. All gates passed.
- **Best system: 3-model ensemble + Phase 2B at 18 ET → Brier 0.7808**
- Equal weights, running_max dominant feature across all models

## Phase 3 Results Summary

### Individual Model Performance (Phase 2B at 18 ET)
| Model | P1 Bias | P2 Regression | P2B at 18 ET |
|-------|---------|---------------|--------------|
| HRRR  | 0.8598  | 0.7979        | 0.7034       |
| GFS   | 0.7555  | 0.7450        | **0.6039**   |
| ECMWF | 0.8074  | 0.7917        | 0.6411       |

### Error Correlation (all < 0.7)
- HRRR-GFS: 0.5965, HRRR-ECMWF: 0.5684, GFS-ECMWF: 0.6421

### Ensemble Results (1°F scoring, equal weights)
| Configuration | Brier |
|---------------|-------|
| HRRR alone (P2) | 0.8779 |
| Ensemble (P2, no obs) | 0.8606 (+2.0%) |
| Ensemble + P2B at 18 ET | **0.7808** (+9.3% vs no-obs) |

### Ensemble+P2B by Update Hour
| Hour | Brier | vs No-Obs Ensemble |
|------|-------|--------------------|
| 14 ET | 0.8456 | +2.0% |
| 15 ET | 0.8345 | +3.3% |
| 16 ET | 0.8181 | +4.9% |
| 17 ET | 0.7976 | +7.3% |
| 18 ET | 0.7808 | +9.3% |

### Learned Weights: No improvement — keep equal 1/3 each

## Key Findings
1. GFS is surprisingly strong — best single model at midnight and at 18 ET
2. running_max_divergence dominates Phase 2B for all three models
3. Equal weights optimal — inverse-Brier weighting adds ~0.0%
4. Phase 2B crossover still at ~14 ET (same as HRRR-only finding)

## Architecture
- `model_name` column on `forecasts` table (DEFAULT 'hrrr', UNIQUE includes model_name)
- GFS/ECMWF model_run = midnight UTC (run_hour=0 in queries)
- `.raw(provider, ref_time) -> Optional[(center, std)]` interface on all model functions
- `Backtester.run()` accepts `model_name` param
- Phase 1 factories: `wf_bias_{hrrr,gfs,ecmwf}`
- Analysis scripts: `scripts/phase3_analysis.py` (Parts 1-4), `scripts/phase3b_analysis.py` (Parts 5-7)

## Completed Tasks
1. Schema migration (model_name column)
2-4. All forecast queries filtered by model_name (9 files, 19+ queries)
5. Open-Meteo backfill script
6. Model function parameterization (20 instances + 3 Phase 1 factories)
7. Ensemble service (Gaussian mixture)
8. Data backfill (GFS 43K rows, ECMWF 43K rows)
9-11. Phase 3 analysis: correlation, regression ablation, ensemble eval
12-13. Phase 3B: per-model Phase 2B, ensemble+P2B, learned weights

## Next Steps (Phase 4)
1. **Wire ensemble into live prediction pipeline** — currently only backtested
2. **Market tick collection** — still a blocker (~2 days of history only)
3. **Edge detection** — compare ensemble probabilities vs Kalshi market prices
4. **Paper trading** — deploy and run against live markets

## Blockers
- Market tick collection still running (only ~2 days of history) — Phase 4 blocker
- Open-Meteo Historical Forecast API assembles "most recent run" — mirrors live behavior

## Design Doc
`docs/plans/2026-02-26-phase3-ensemble-design.md`
