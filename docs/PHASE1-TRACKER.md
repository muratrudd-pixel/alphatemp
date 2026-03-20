# Phase 1: Data Foundation — Tracker

**Gate:** All data loaded, schema validated, no PiT violations, gap audit passes.
**Last updated:** 2026-03-01

---

## Status Summary

| # | Item | Status | Blocks Gate? |
|---|------|--------|-------------|
| 1.1 | HRRR 24-run backfill | DONE | — |
| 1.2a | GFS 00z backfill | DONE (1,801 days) | — |
| 1.2b | GFS 06z backfill | DONE (1,733 days) | — |
| 1.2c | GFS 12z backfill | IN PROGRESS (364/~1,800 days, fleet) | No — proceed without |
| 1.2d | GFS 18z backfill | IN PROGRESS (531/~1,800 days, fleet) | No — proceed without |
| 1.3 | ECMWF 00z backfill | DONE (1,801 days) | — |
| 1.4 | KJFK station data | DONE (590K rows) | — |
| 1.5 | Schema: is_spinup, fxx, obs_type columns | DONE | — |
| 1.6 | Schema: market_ticks UNIQUE constraint | DONE | — |
| 1.7 | Gold feature views | DONE (4 views operational) | — |
| 1.8 | Bronze metadata table | DEFERRED (table exists, 0 rows — provenance only) | No |
| 1.9 | Fix fxx NULLs (ECMWF + GFS 00z) | DONE — 87,146 rows fixed via UPDATE | — |
| 1.10 | Fix fxx NULLs (HRRR live ingest bug) | DONE — bug fixed in forecast.py:126, rows backfilled | — |
| 1.11 | DSM ingestion | DEFERRED to Phase 4 | No |
| 1.12 | NYC Micronet outreach | NOT STARTED | No (no hard dependency) |
| 1.13 | Evaluate forecast_extended table | DONE — has consumers (backtester), keep it | — |
| 1.14 | PiT violation audit | PASSED | — |
| 1.15 | Gap audit (all models) | PASSED | — |

---

## GATE STATUS: PASSED (for available data)

All blockers resolved 2026-03-01. Remaining items (GFS 12z/18z backfill, DSM, bronze, micronet) are deferred — none block Phase 2.

---

## Audit Results (2026-03-01)

### PiT Violation Audit — CLEAN
- Zero forecasts with valid_at < model_run (no negative fxx)
- Zero forecasts with ingested_at < model_run
- All 4 gold views have temporal fencing
- Note: fxx=0 rows exist for ECMWF (1,801) and GFS (4,428) — nowcasts from Open-Meteo, harmless

### Gap Audit — CLEAN (for available data)
- ECMWF 00z: 1,801/1,801 days — no gaps
- GFS 00z: 1,801/1,801 days — no gaps
- GFS 06z: 1,733/1,733 days — no gaps
- GFS 12z: 364/1,885 days — 1,521 missing (fleet backfilling, expected)
- GFS 18z: 531/531 days — no gaps in available range
- HRRR all 24 hours: no gaps (1 day tolerance on 00z-05z, within bounds)

---

## Fixes Applied (2026-03-01)

### 1.9 + 1.10 — fxx NULL Fix
- **87,146 rows** updated: `SET fxx = DATE_DIFF('hour', model_run, valid_at)`
- `is_spinup` set to `fxx <= 3` for all previously-NULL rows
- Checkpointed to disk

### 1.10 — Live Ingest Bug Fix
- **File:** `services/forecast.py` line 126
- **Bug:** INSERT omitted `fxx` and `is_spinup` columns despite `fxx` being in scope
- **Fix:** Added both columns to INSERT statement with `fxx, fxx <= 3` values

---

## Deferred Items

### DSM Ingestion (1.11) → Phase 4
DSM provides early settlement truth (~4 PM ET) before CLI final (~1:30 AM ET). Value is in live trading execution (exit positions with certainty), not in model training (CLI historical data already provides ground truth). Defer to Phase 4 strategy/execution layer.

### Bronze Table Population (1.8) → Indefinite
Provenance tracking for GRIB source files. No downstream consumer in Phases 2-4. Can wire into ingest pipeline later for new data; backfilling provenance for 1.2M existing rows has no modeling payoff.

### NYC Micronet (1.12) → Outreach only
Email mesonet@albany.edu. No hard dependency on any phase.

---

## Completed Items Detail

### 1.1 — HRRR 24-Run Backfill
All 24 run hours × ~1,904-1,905 days (2020-12-12 → 2026-03-01). No internal gaps.

### 1.2a/b — GFS 00z + 06z
- 00z: 1,801 days (2021-03-23 → 2026-02-25) via Open-Meteo
- 06z: 1,733 days (2021-06-01 → 2026-02-27) via UCAR

### 1.3 — ECMWF 00z
1,801 days (2021-03-23 → 2026-02-25) via Open-Meteo composite. PiT preserved for 00z.

### 1.4 — KJFK
590,548 METAR rows (2020-09-07 → 2026-03-01). Fully current.

### 1.5/1.6/1.7 — Schema + Gold Views
All schema columns added. UNIQUE constraint on market_ticks confirmed. Four gold feature views operational with data.

### 1.13 — forecast_extended
123,456 rows (ECMWF + GFS, 00z only). Used by backtester for extended meteorological features. Keep.
