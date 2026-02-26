---
name: alphatemp-data-pipeline
description: Domain knowledge for the alphatemp data pipeline — HRRR forecast ingestion, NWS settlement data, observation collection, and DuckDB schema. Use PROACTIVELY whenever working with data ingestion, forecast fetching, observation handling, NWS CLI reports, backfill scripts, data freshness, schema changes, or timezone conversions. Also use when Russell asks about data sources, API endpoints, or why data looks stale or missing. Triggers on terms like HRRR, forecast, ingest, observation, NWS, CLI, backfill, mesonet, synoptic, AWC, METAR, Open-Meteo, data freshness, pipeline, or schema.
tools: Read, Bash, Grep, Edit
model: sonnet
---

# AlphaTemp Data Pipeline

Domain knowledge for data ingestion, storage, and the data flow through the alphatemp system.

## Data Flow Overview

```
Open-Meteo API (HRRR forecasts)
    → services/forecast.py → forecasts table

Synoptic API + AWC (real-time METAR obs)
    → services/ingestor.py → observations table

NWS CLI Report (settlement truth)
    → services/nws_fetcher.py → nws_daily table

ACIS/RCC (historical daily highs)
    → scripts/backfill scripts → nws_daily table

Kalshi API (market data)
    → services/market_fetcher.py → market_ticks table
    → kalshi_settlements table (historical)
```

## HRRR Forecasts

### Source
- **API:** Open-Meteo (free, no API key required)
- **Model:** HRRR (High-Resolution Rapid Refresh), 3km grid resolution
- **Run schedule:** Every hour; extended runs at 00z, 06z, 12z, 18z (48h horizon)
- **Standard horizon:** 18h for non-extended runs

### Key Rules
- All HRRR timestamps are UTC
- Model data lands ~45-90 min after the model run time (HRRR_AVAILABILITY_LAG_HOURS = 2 for backtest safety)
- Station mapping uses STATION_COORDS in `core/constants.py` — each station has a lat/lon that maps to the nearest HRRR grid point
- KNYC, KLGA, KJFK are different locations with different microclimates — never assume interchangeability

### Ingestion
- `services/forecast.py` — Polls Open-Meteo at FORECAST_POLL_INTERVAL_SECONDS
- Stores: station_id, model_run, valid_at, temp_f, temp_c, ingested_at
- DB-aware: skips model runs already in the database

### Backfill
- `scripts/backfill_parallel.py` — Multi-run-hour parallel HRRR backfill
- `scripts/backfill_all.py` — Orchestrates full backfill
- Coverage: ~144K forecast rows across 4 run hours

## Observations

### Sources
1. **Synoptic API** — Primary real-time source. Requires API key. 2-5 min latency.
2. **AWC (Aviation Weather Center)** — Backup. Parses METAR remarks for 6-hour synoptic max/min.
3. **Iowa State Mesonet** — Bulk historical backfill only (`scripts/backfill_iem.py`)

### Key Rules
- Parse 6-hour synoptic max/min from METAR remarks (T-group: `TXXXXxxxx` format)
- Settlement station is KNYC. Neighbors KLGA, KEWR are for cross-validation only
- Always clarify: surface temp, 2m temp, or feels-like
- observation timestamps are UTC

### Ingestion
- `services/ingestor.py` — Dual ingestion from Synoptic + AWC
- Stores: station_id, observed_at, temp_f, temp_c, source, raw

## NWS Settlement Data

### Source
- NWS Daily Climate Report (CLI) from Central Park (KNYC)
- Preliminary report: ~4:30 PM EST
- Final report: ~1:30 AM EST

### Critical Timezone Rule
**CLI uses Local Standard Time (EST), NOT Daylight Saving Time (EDT).** This means during summer, the CLI day does NOT match wall-clock day. Always handle this explicitly.

### Ingestion
- `services/nws_fetcher.py` — Parses CLI reports
- CLI values overwrite ACIS backfill values when both exist
- Stores: station_id, obs_date, max_temp_f, min_temp_f, source

### Backfill
- ACIS (RCC) provides historical daily highs: `scripts/backfill_acis.py` (if it exists) or inline in backfill scripts
- Coverage: ~1,902 settlement days

## Kalshi Market Data

### Settlement Data
- `kalshi_settlements` table — Historical bracket settlement outcomes
- Columns: event_date, floor_strike, cap_strike, settled_yes
- Coverage: ~1,659 dates with bracket data
- WARNING: ~1,418 records have NULL strikes (unparsed ticker data) — filter these out

### Live Market Data
- `services/market_fetcher.py` — Fetches bid/ask/volume from Kalshi API
- Stores in `market_ticks` table
- Rate limits apply — check current limits before designing polling loops

## DuckDB Schema (Key Tables)

| Table | Key columns | Notes |
|-------|------------|-------|
| `forecasts` | station_id, model_run, valid_at, temp_f | TIMESTAMP columns are naive UTC |
| `observations` | station_id, observed_at, temp_f | TIMESTAMP columns are naive UTC |
| `nws_daily` | station_id, obs_date, max_temp_f | DATE column, settlement truth |
| `station_bias` | station_id, calculated_at, mean_bias, std_error | Bias correction stats |
| `drift_signals` | city, calculated_at, drift_score | Same-day observation drift |
| `kalshi_settlements` | event_date, floor_strike, cap_strike, settled_yes | Bracket outcomes |
| `market_ticks` | ticker, timestamp, yes_bid, yes_ask | Live market prices |

## Timezone Rules (CRITICAL)

1. **All DuckDB TIMESTAMP columns store naive UTC** — never pass timezone-aware Python datetimes directly
2. Use `_strip_tz()` from `services/data_provider.py` to strip tzinfo before DB queries
3. **Never silently convert timezones** — always log or comment the conversion
4. HRRR model runs: UTC
5. NWS CLI: Local Standard Time (NOT DST)
6. Kalshi contract days: Eastern Time
7. When converting ET boundaries to UTC, use `core/timezone.py` helpers

## Data Freshness Checks

```sql
-- Latest HRRR model run
SELECT MAX(model_run), COUNT(*) FROM forecasts WHERE station_id = 'KNYC';

-- Latest observation
SELECT MAX(observed_at) FROM observations WHERE station_id = 'KNYC';

-- NWS settlement coverage
SELECT MIN(obs_date), MAX(obs_date), COUNT(*) FROM nws_daily WHERE station_id = 'KNYC';

-- Kalshi bracket coverage
SELECT MIN(event_date), MAX(event_date), COUNT(DISTINCT event_date) FROM kalshi_settlements;

-- Forecast coverage by run hour
SELECT EXTRACT(HOUR FROM model_run) as run_hour, COUNT(*), MIN(model_run), MAX(model_run)
FROM forecasts WHERE station_id = 'KNYC'
GROUP BY run_hour ORDER BY run_hour;
```
