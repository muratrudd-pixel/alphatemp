# AlphaTemp Phase 1 Design — The Plumbing

**Date:** 2026-02-22
**Status:** Approved

## Elevator Pitch

AlphaTemp is a high-frequency temperature arbitrage system for Kalshi markets. It exploits Official Record Arbitrage through three mechanisms:

1. **Magnet Numbers** — Certain Fahrenheit integers capture 6 Celsius tenths instead of 5 (repeating every 9°F), making them ~20% more likely as settlement highs.
2. **Solar Bias** — Real-time comparison of HRRR forecasts vs high-res T-group METAR remarks.
3. **FLB Filter** — Exploit Favorite-Longshot Bias by prioritizing favorites (>$0.50) and avoiding longshots (<$0.15).

## Architecture

Layered services pattern:

```
alphatemp/
├── core/
│   ├── constants.py      # Magnets, FLB thresholds, station configs
│   ├── db.py             # DuckDB connection + table init
│   └── models.py         # Polars schema definitions
├── services/
│   ├── ingestor.py       # Async Synoptic poller + METAR parser
│   └── __init__.py
├── data/
│   └── alphatemp.duckdb  # Created at runtime
├── main.py               # Async entrypoint
└── .gitignore
```

## Database Schema (DuckDB)

### observations
| Column | Type | Notes |
|---|---|---|
| station_id | VARCHAR | ICAO code |
| observed_at | TIMESTAMP | UTC observation time |
| temp_f | DOUBLE | Standard Fahrenheit reading |
| temp_c_tenth | DOUBLE | High-res Celsius from T-group |
| raw_metar | VARCHAR | Full METAR remark string |
| ingested_at | TIMESTAMP | Storage timestamp |

Primary key: (station_id, observed_at)

### forecasts (stub for Phase 2)
| Column | Type | Notes |
|---|---|---|
| station_id | VARCHAR | Target station |
| valid_at | TIMESTAMP | Forecast valid time |
| temp_f | DOUBLE | Forecast temperature |
| model_run | TIMESTAMP | Model initialization time |
| ingested_at | TIMESTAMP | Storage timestamp |

### market_ticks
| Column | Type | Notes |
|---|---|---|
| market_id | VARCHAR | Kalshi market identifier |
| city | VARCHAR | NYC/PHL/CHI/MIA/LA |
| captured_at | TIMESTAMP | Snapshot time |
| yes_bid | DOUBLE | Best yes bid |
| yes_ask | DOUBLE | Best yes ask |
| no_bid | DOUBLE | Best no bid |
| no_ask | DOUBLE | Best no ask |
| last_trade | DOUBLE | Last trade price |
| volume | INTEGER | Total volume |

## Core Constants

- **Magnet numbers:** Generated algorithmically by iterating 0.1°C increments, converting to °F, rounding, and counting tenths per integer. 6+ tenths = magnet.
- **FLB thresholds:** Favorite floor = $0.50, Longshot ceiling = $0.15
- **Polling interval:** 60 seconds

## Station Configuration

| City | Settlement | Neighbors |
|---|---|---|
| NYC | KNYC | KLGA, KEWR |
| PHL | KPHL | KPNE |
| CHI | KMDW | KORD |
| MIA | KMIA | KOPF |
| LA | KLAX | KHHR |

## Ingestor Design

- Async event loop with httpx.AsyncClient
- Batch Synoptic API call for all 11 stations
- `recent=5` parameter for overlap/deduplication
- T-group METAR parser: regex `T(\d)(\d{3})` extracts high-res Celsius
- Deduplication on (station_id, observed_at) before insert
- Error handling: log + skip on failure, store NULL temp_c_tenth on parse failure
- Logging via Loguru

## Tech Stack

- Python 3.11+
- DuckDB (embedded analytics database)
- Polars (DataFrames)
- httpx (async HTTP)
- Loguru (logging)
- asyncio (event loop)
