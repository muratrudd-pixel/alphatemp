# Handoff - 2026-02-28 (Late Afternoon)

## Current State
- **Phase 1 code: COMPLETE** — all committed
- **EC2 fleet: RUNNING** — 8 instances, all 24 HRRR hours + GFS sequential
- **Gold feature views: DONE** — 4 views in `core/db.py`, tested
- **Bronze metadata table: DONE** — `bronze_grib_meta`
- **Phase 2 implementation plan: DONE** — `docs/plans/2026-02-28-phase2-bias-correction-implementation.md`
- **Compaction hooks: INSTALLED** — PreCompact + SessionStart in `~/.claude/settings.json`
- **Gemini research: ALL 3 COMPLETE** — EMOS (#1), HRRR bias (#2), market efficiency (#3) all logged
- **Compaction hook paths: FIXED** — changed from relative to `$HOME/.claude/hooks/`

## EC2 Fleet
All 8 instances running `scripts/batched_backfill.sh` with memory-leak fix. Memory stable at ~1-2 GB / 7.6 GB.

| IP | Date Range | Status |
|----|-----------|--------|
| 44.204.17.140 | 2021-06-01 → 2021-12-31 | All 24 HRRR hours running |
| 3.80.188.43 | 2022-01-01 → 2022-07-15 | All 24 HRRR hours running |
| 3.83.140.131 | 2022-07-16 → 2023-01-31 | All 24 HRRR hours running |
| 3.83.173.71 | 2023-02-01 → 2023-08-15 | All 24 HRRR hours running |
| 3.84.4.112 | 2023-08-16 → 2024-02-29 | All 24 HRRR hours running |
| 32.192.64.214 | 2024-03-01 → 2024-09-15 | All 24 HRRR hours running |
| 13.218.60.69 | 2024-09-16 → 2025-04-30 | All 24 HRRR hours running |
| 54.144.18.212 | 2025-05-01 → 2026-02-27 | All 24 HRRR hours running |

**ETA:** Most instances done early Sunday morning. Instance 8 by Sunday sunrise. Full pipeline (download + merge + gate) by Sunday mid-morning.

**Check fleet:**
```bash
for ip in 44.204.17.140 3.80.188.43 3.83.140.131 3.83.173.71 3.84.4.112 32.192.64.214 13.218.60.69 54.144.18.212; do
    result=$(ssh -i ~/.ssh/alphatemp-hrrr.pem -o ConnectTimeout=5 ec2-user@$ip \
        "procs=\$(pgrep -c python3 2>/dev/null || echo 0); mem=\$(free -h | awk '/Mem:/{print \$3\"/\"\$2}'); echo \"procs=\$procs RAM=\$mem\"" 2>/dev/null)
    echo "$ip: $result"
done
```

**AWS resources — TERMINATE WHEN DONE (~$0.72/hr total):**
- Key: `~/.ssh/alphatemp-hrrr.pem`
- Security group: `sg-0729bc1135388b966`
- Instance IDs: i-0cb60b4e2b3297fa8, i-0e6a0e80fd68a0112, i-09e2a3d495e9f7e37, i-0a510a018c58df2cf, i-05349669e69290a75, i-0c59a6e8826d05e43, i-017516e1962d87bce, i-0ba9ac13ea6b5837b

## Uncommitted Changes
```
M  HANDOFF.md
M  core/db.py                    — gold views, bronze table, indexes
M  scripts/backfill_gfs_ucar.py  — memory leak fix
M  scripts/backfill_hrrr_full.py — memory leak fix
M  scripts/phase1_gate_check.py  — relaxed ECMWF to 00z only
?? scripts/batched_backfill.sh   — new: memory-safe batched launcher
?? scripts/fleet_harvest.sh      — new: download/merge/gate/terminate
?? docs/plans/2026-02-28-phase2-bias-correction-implementation.md
```

Should be committed before any new work.

## Key Decisions This Session
1. **ECMWF 12z killed** — archive gaps, deprioritized
2. **NYC Micronet killed** — Russell won't get the data
3. **EMOS cannot skip Phase 3** — Gemini confirmed (see decisions.md 2026-02-28)
4. **Probability leakage insight** — Gaussian distributions leak mass into impossible brackets below observed running max. QR with floor constraint fixes this. Critical Phase 3 design input.
5. **Gold tables as SQL views** — auto-update as data lands, no materialization needed
6. **Reinstate extended features for XGBoost** — Gemini #2 found dewpoint/wind/humidity/etc. likely failed under OLS due to linear constraints, not lack of signal. XGBoost can capture non-linear interactions. Added as sub-ablation in Phase 2 Task 4.
7. **Compaction hooks installed** — PreCompact + SessionStart hooks in `~/.claude/settings.json` with scripts at `~/.claude/hooks/`
8. **Prediction market efficiency timing (Gemini #3)** — Kalshi Brier ~0.63 overnight → 0.14 at 18z. Weather markets structurally favor algorithmic traders (scheduled NWP releases). Overnight stale prices = potential displacement. **CAVEAT:** Brier figures are from old baseline model — need to re-measure in Phase 2 Task 8. Russell wants to be thoughtful about ALL hours, not write off late afternoon.
9. **Compaction hook paths fixed** — relative paths didn't resolve from project directory. Changed to `$HOME/.claude/hooks/`

## Next Steps (in order)
1. **Commit uncommitted changes**
2. **Check fleet Sunday morning** — run check command above
3. **When fleet done:** `./scripts/fleet_harvest.sh` (download, merge, gate-check, terminate)
4. **If Phase 1 gate passes:** Begin Phase 2 per `docs/plans/2026-02-28-phase2-bias-correction-implementation.md`
5. **Gemini research queue complete** — all 3 queries answered, findings logged to decisions.md + patterns.md
6. **Re-measure Kalshi Brier by hour** — during Phase 2 Task 8 backtester diagnostic, compute market Brier score by ET hour against new model. Gemini's 0.63→0.14 gradient needs validation with current data.

## Gemini Research Queue (ALL COMPLETE)
1. ~~EMOS vs Phase 3 shortcut~~ — DONE, cannot skip Phase 3
2. ~~HRRR 2m temp bias at urban stations~~ — DONE, key finding: extended features worth retesting under XGBoost
3. ~~Prediction market efficiency timing~~ — DONE, key finding: efficiency gradient 0.63→0.14 but numbers need re-measurement. Don't write off any hours pre-empirically.

## Files Changed Since Last Commit
- `core/db.py` — gold views, bronze table, indexes, fixed GROUP BY in sin/cos
- `scripts/backfill_hrrr_full.py` — memory leak fix (grbs.close, os.remove, gc.collect)
- `scripts/backfill_gfs_ucar.py` — same memory leak fix
- `scripts/phase1_gate_check.py` — relaxed ECMWF to 00z only
- `scripts/batched_backfill.sh` — NEW: memory-safe batched launcher
- `scripts/fleet_harvest.sh` — NEW: download/merge/gate/terminate pipeline
- `docs/plans/2026-02-28-phase2-bias-correction-implementation.md` — NEW: 9-task Phase 2 plan
- `HANDOFF.md` — this file
- `~/.claude/settings.json` — merged compaction hooks + deny list
- `~/.claude/hooks/pre-compact.sh` — NEW
- `~/.claude/hooks/post-compact-inject.sh` — NEW
- `~/.claude/projects/-Users-russellrudd/memory/decisions.md` — 2 new entries
- `~/.claude/projects/-Users-russellrudd/memory/patterns.md` — Phase 2/3 + EC2 learnings added
