# Handoff - 2026-03-01 (Late Night)

## Current State
- **Dashboard V2: COMPLETE** — merged to main, 13 commits
- **Live data: FLOWING** — obs (AWC/Synoptic), HRRR (backfilling gap), market ticks (Kalshi)
- **Probability engine: DISABLED** — `PROBABILITY_ENGINE_ENABLED = False` in constants.py, awaiting Phase 2 rebuild
- **EC2 fleet: STATUS UNKNOWN** — check with fleet command below

## What Shipped This Session

### Dashboard V2 (10 tasks)
1. Schema + heartbeat infrastructure
2. KPI header bar (persistent across all pages)
3. System health page (/health)
4. Heartbeat integration (all 6 services)
5. PaperTrader service shell (placeholder strategy)
6. Settlement countdown + trade blotter (/blotter)
7. Polish operations (stale banner, error toasts, settlement labels)
8. Polish performance (real Brier comparison, paper P&L chart)
9. Polish review + mobile (missed_edge filter, settlement source, mobile blotter)
10. Loading skeletons (cross-cutting)

### Live Data Fixes (post-merge)
- `market_ticks` schema: added open_interest, liquidity, volume_24h columns
- `PROBABILITY_ENGINE_ENABLED` kill switch: suppresses broken model output
- Bracket spread: shows all Kalshi brackets including tails (≤X, ≥X)
- Forecast-curve: returns observations even when no forecasts available
- HRRR fetcher: fixed premature break — now continues to older runs when newest unavailable

## Key Files Changed
- `core/constants.py` — PROBABILITY_ENGINE_ENABLED flag
- `core/db.py` — market_ticks migration (3 new columns)
- `services/forecast.py` — HRRR fetcher gap recovery (consecutive miss logic)
- `ui/web_dashboard.py` — model suppression, tail brackets, obs-without-forecasts
- `static/js/operations.js` — market-only bracket chart, hide model trace
- `static/js/blotter.js` — tail bracket labels, null model handling
- Design docs: `docs/plans/2026-02-28-dashboard-v2-design.md`, `docs/plans/2026-02-28-dashboard-v2-implementation.md`

## EC2 Fleet
Check command:
```bash
for ip in 44.204.17.140 3.80.188.43 3.83.140.131 3.83.173.71 3.84.4.112 32.192.64.214 13.218.60.69 54.144.18.212; do
    result=$(ssh -i ~/.ssh/alphatemp-hrrr.pem -o ConnectTimeout=5 ec2-user@$ip \
        "procs=\$(pgrep -c python3 2>/dev/null || echo 0); mem=\$(free -h | awk '/Mem:/{print \$3\"/\"\$2}'); echo \"procs=\$procs RAM=\$mem\"" 2>/dev/null)
    echo "$ip: $result"
done
```

## Next Steps (in order)
1. **Check EC2 fleet** — may be done by now, harvest if complete
2. **Phase 2 bias correction** — per `docs/plans/2026-02-28-phase2-bias-correction-implementation.md`
3. **Paper trading strategy brainstorm** — entry/exit rules, edge thresholds, position sizing, NO-side logic
4. **Re-enable probability engine** — flip `PROBABILITY_ENGINE_ENABLED = True` after Phase 2 rebuild
5. **Fix pre-existing test failures** — test_fetcher_deduplicates, test_brier_comparison_endpoint

## Decisions This Session
1. Dashboard V2 design approved — KPI header, health page, blotter, polish, paper trader shell
2. Paper trader uses thin 60s polling loop (not event-driven)
3. Strategy logic is placeholder only — dedicated brainstorm session needed
4. Probability engine suppressed via kill switch until rebuilt
5. Kalshi brackets now include tails (≤X, ≥X) — market structure has changed from 2°F to variable widths
