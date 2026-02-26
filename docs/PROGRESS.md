# AlphaTemp Progress Tracker

**Read this at the start of every session. Update before every compact or session end.**
**Master plan:** [docs/plans/MASTER-PLAN.md](plans/MASTER-PLAN.md)

---

## Current Phase: 2B — Intra-Day Observation Updates

Phase 2 is complete. Feature-conditioned regression (Brier 0.7979) beats Phase 1 flat bias (0.8356) by 4.5%. HRRR has a temperature-dependent bias that scales with forecast temp. Next: can real-time observations during the day shift our predictions?

### What's Proven
- Per-run-hour expanding-window bias correction works (Brier 0.84)
- Each run hour has distinct bias: 00z=-0.9°F, 06z=-0.2°F, 12z=-0.3°F, 18z=-0.7°F
- Student-t distribution doesn't help despite heavy tails — Gaussian wins
- **HRRR warm bias scales with temperature** — r=0.39-0.52, ~+0.1°F per 1°F of forecast
- **Clear seasonal pattern** — HRRR overpredicts summer (+1-2.3°F), underpredicts winter (-1.5-2.5°F)
- **Day-over-day temperature change is useless** — no predictive signal for HRRR error
- Feature-conditioned regression (fcst_high + month) is the champion model (Brier 0.7979)
- Kalshi market scores ~0.63 at overnight hours — gap narrowed from 0.21 to 0.17

### Key Files
- `services/backtester.py` — all model functions including Phase 2 regression variants
- `scripts/phase1_analysis.py` — Phase 1 model comparison
- `scripts/phase2_exploration.py` — feature signal detection (correlations, F-tests)
- `scripts/phase2_analysis.py` — Phase 2 model comparison + regression diagnostics
- `tests/test_data_provider.py` — 26 tests (Phase 1 walk-forward)
- `tests/test_phase2_models.py` — 13 tests (regression, walk-forward safety, bias detection)

### Next Steps (in order)
1. Phase 2B: intra-day observation updates (biggest remaining edge)
2. Phase 3: multi-model ensemble (GFS, NAM, ECMWF)
3. Continue market tick collection (silent blocker for Phase 4)

---

## Phase Gate Log

| Phase | Gate | Result | Date |
|-------|------|--------|------|
| 0 | Backtester runs end-to-end with dummy model | PASSED | 2026-02-25 |
| 1 | Walk-forward per-run-hour beats baselines (Brier 0.84 vs 1.02) | PASSED | 2026-02-25 |
| 2 | Enhanced variables improve Brier score over flat bias (0.7979 vs 0.8356, +4.5%) | PASSED | 2026-02-26 |
| 3 | Ensemble beats best single model | — | — |
| 4 | Simulated P&L positive net of fees | — | — |

---

## Session Log

### 2026-02-24
- Resumed from HANDOFF.md (HRRR backfill context)
- Kicked off 00z and 18z backfills in parallel
- 00z crashed at day 648 (DuckDB segfault, WAL corruption). Recovered DB by removing WAL. Rolled back to 463 days.
- 18z script deleted old temp DB (167 days) and restarted from scratch
- Restarted 00z using backfill_parallel.py (temp DB approach)
- Reviewed and revised master plan — shifted from theory-first to data-first approach
- Key decision: evaluate on Kalshi bracket Brier score from Phase 1 onward
- Created MASTER-PLAN.md and this PROGRESS.md

### 2026-02-25
- Phase 0 gate passed: backtester runs end-to-end (7,607 evaluations, ~3s)
- Built walk-forward per-run-hour bias correction (Phase 1)
- Results: Brier 0.8356 (18% over uniform, 13% over single-bias)
- Student-t tested and killed (doesn't beat Gaussian)
- Computed Kalshi market Brier scores by time of day — market scores 0.63 overnight, 0.14 by 18z
- Phase 1 gate passed

### 2026-02-26
- Phase 2 exploration: all 3 features tested against HRRR error per run hour
- fcst_high strongest signal (r=0.39-0.52), month significant (F-test p≈0), delta_temp dead
- Discovered HRRR temperature-dependent warm bias: ~+0.1°F per 1°F of forecast
- Built walk-forward OLS regression with 4 ablation variants
- Full model: Brier 0.7979, +4.5% over Phase 1 — clears >2% gate
- 13 new tests all pass, 26 existing tests still pass (zero regressions)
- Phase 2 gate passed
