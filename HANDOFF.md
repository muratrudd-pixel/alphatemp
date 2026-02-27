# Handoff - 2026-02-26

## Current State
- **Phase 3.5 COMPLETE.** Extended weather variables + adaptive weights.
- **Best system: 3-model ensemble + Phase 2B + WF adaptive weights at 18 ET → Brier 0.7705**
- Previous best: 0.7808 (equal weights). Improvement: +1.31%

## Phase 3.5 Results

### Extended Weather Variables — DEAD END
Backfilled 10 variables (dewpoint, humidity, wind, pressure, cloud, radiation, CAPE) for GFS/ECMWF into `forecast_extended` table (123,456 rows). Strong correlations in signal exploration but NO variable beat the 2% Brier gate in ablation. Best: GFS dewpoint at +1.3%. Signal already captured by existing features.

### Walk-Forward Adaptive Weights — PASS (+1.31%)
Inverse-Brier weighting with 90-day expanding window. Evaluated across full 0-18 ET window:

| Hour ET | Equal | WF Adaptive | Delta |
|---------|-------|-------------|-------|
| 0-5     | 0.868 | 0.866       | +0.2% |
| 6-12    | 0.854 | 0.852       | +0.3% |
| 13-14   | 0.845 | 0.842       | +0.4% |
| 15-16   | 0.826 | 0.821       | +0.6% |
| 17-18   | 0.789 | 0.780       | +1.2% |

GFS gets highest weight (~35%), ECMWF middle (~34%), HRRR lowest (~32%).

## Key Insight (Russell)
Kalshi KXHIGHNY opens at 10 AM ET the prior day. Market Brier is probably worst overnight/early AM when the crowd is thin. Our model may have the most edge at hours far from settlement, not near it. Need to compare our Brier vs Kalshi Brier by hour to find optimal trading windows.

## Phase 3 Results (prior, still valid)

### Individual Model Performance (Phase 2B at 18 ET)
| Model | P1 Bias | P2 Regression | P2B at 18 ET |
|-------|---------|---------------|--------------|
| HRRR  | 0.8598  | 0.7979        | 0.7034       |
| GFS   | 0.7555  | 0.7450        | **0.6039**   |
| ECMWF | 0.8074  | 0.7917        | 0.6411       |

### Error Correlation (all < 0.7)
- HRRR-GFS: 0.5965, HRRR-ECMWF: 0.5684, GFS-ECMWF: 0.6421

## Next Steps
1. Russell has "a few more things to assess" before Kalshi comparison
2. **TODO: Prior-day evaluation** — market opens 10 AM prior day, need model quality from evening before (requires code change for obs_date - 1)
3. **TODO: Our Brier vs Kalshi Brier by hour** — the real edge calculation
4. **TODO: Re-run Phase 3B GFS/ECMWF Phase 2B at [8..18]** — verify 14 ET crossover holds for global models (only validated for HRRR)
5. **Wire ensemble + adaptive weights into live pipeline**
6. **Edge detection / displacement P&L backtest**

## Architecture
- `forecast_extended` table: 10 weather vars, keyed by (station_id, model_run, valid_at, model_name)
- `services/backtester.py`: extended regression pipeline available but unused (no variable beat gate)
- Walk-forward weights: not yet wired into production — only backtested in `scripts/phase35b_weights.py`

## Kalshi Historical Data (READY)
- `kalshi_settlements`, `kalshi_candlesticks`, `kalshi_trades`, `market_ticks`
- Backfill script: `scripts/backfill_kalshi_history.py`

## Strategy Notes
- Fee hurdle: ~11% (1% trading + 10% settlement)
- Key hypothesis: model edge largest at hours when Kalshi crowd is thin (overnight, early AM)
- Potential edge vectors: large displacements, tail brackets, specific weather regimes, stale pricing windows

## Design Docs
- `docs/plans/2026-02-26-phase3-ensemble-design.md`
- Plan file: `~/.claude/plans/abstract-stirring-dijkstra.md` (Phase 3.5 plan)
