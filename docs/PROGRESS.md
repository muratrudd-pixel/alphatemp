# AlphaTemp Progress Tracker

**Read this at the start of every session. Update before every compact or session end.**
**Master plan:** [docs/plans/MASTER-PLAN.md](plans/MASTER-PLAN.md)

---

## Current Phase: 2 — Enhanced Variables

Phase 1 is complete. The walk-forward per-run-hour bias correction model scores 0.8356 Brier (18% better than uniform). Next: can additional variables (temperature regime, season, etc.) explain more of the HRRR error?

### What's Proven
- Per-run-hour expanding-window bias correction works (Brier 0.84)
- Each run hour has distinct bias: 00z=-0.9°F, 06z=-0.2°F, 12z=-0.3°F, 18z=-0.7°F
- Student-t distribution doesn't help despite heavy tails — Gaussian wins
- Kalshi market scores ~0.63 at overnight hours — we need to roughly halve our score to compete

### Key Files
- `services/backtester.py` — walk_forward_model, walk_forward_t_model, all model functions
- `scripts/phase1_analysis.py` — runs all 4 models and outputs comparison table
- `tests/test_data_provider.py` — 26 tests including 6 walk-forward tests

### Next Steps (in order)
1. Decide whether to productionize walk-forward (schema migration) or move straight to Phase 2
2. Phase 2: test additional variables against HRRR error (season, temperature regime, etc.)
3. Continue market tick collection (silent blocker for Phase 4)

---

## Phase Gate Log

| Phase | Gate | Result | Date |
|-------|------|--------|------|
| 0 | Backtester runs end-to-end with dummy model | PASSED | 2026-02-25 |
| 1 | Walk-forward per-run-hour beats baselines (Brier 0.84 vs 1.02) | PASSED | 2026-02-25 |
| 2 | Enhanced variables improve Brier score over flat bias | — | — |
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
