# AlphaTemp — Kalshi Weather Trading System

## What This Project Does
Predicts daily high temperature settlement brackets for Kalshi KXHIGHNY markets.
Compares our probability distribution against Kalshi market prices to find tradeable edge.
NYC only — settlement station is KNYC (Central Park).

## Current Architecture
- **Database:** DuckDB (`data/alphatemp.duckdb`)
- **Observations:** Dual ingestion — Synoptic API + AWC (Aviation Weather Center)
  - Settlement station: KNYC
  - Neighbor stations: KLGA, KEWR (for cross-validation)
  - Parses 6-hour synoptic max/min from METAR remarks
- **Forecasts:** HRRR only via `services/forecast.py` (2-min polling, DB-aware)
- **Settlement truth:** NWS CLI report parsed by `services/nws_fetcher.py`, stored in `nws_daily` table
  - CLI overwrites ACIS backfill values
- **Market data:** Kalshi bid/ask/volume via `services/market_fetcher.py`
- **Bias:** `services/bias_model.py` calculates flat mean bias per station
- **Drift:** `services/bias.py` generates drift signals (obs vs forecast divergence)
- **Dashboard:** Flask + Plotly at port 8050 (`ui/web_dashboard.py`)
  - Forecast + obs chart with neighbor toggle, sidebar tables, Today/Tomorrow
- **Deployment:** Docker on DigitalOcean droplet via `deploy.sh`

## What's Broken / In Progress
- Probability engine (`services/probability.py`) is pinned for rework — edge columns zeroed out
- CI ribbon and bias-adjusted forecast line removed in prep for overhaul
- Bias engine is functional but the probability model doesn't use it properly yet

## Key Constraints
- Starting capital: $100 (paper trading phase)
- Kalshi fees: taker fee = max(ceil(0.07 * C * P * (1-P)), C * $0.01). No settlement fee. ~1-2% effective rate.
- ALL strategy evaluations MUST be net of fees — no exceptions
- Settlement source: NWS Daily Climate Report (CLI) from Central Park (KNYC)
- CLI uses Local Standard Time, NOT Daylight Saving Time
- Temperature brackets are 2°F wide on Kalshi

## Data Sources
- Forecasts: HRRR via Open-Meteo (expanding to multi-model)
- Observations: Synoptic API + AWC (aviationweather.gov) for real-time METAR
- Historical observations: Iowa State Mesonet for bulk backfill
- Settlement: NWS CLI reports (preliminary ~4:30 PM EST, final ~1:30 AM EST)
- Historical daily highs: ACIS (RCC) backfill in `nws_daily` table
- Market prices: Kalshi API

## Coding Standards
- Python 3.9 compatible (no subscripted builtins like `dict[str, ...]`)
- DuckDB for all storage — NOT SQLite
- Type hints via `typing` module (Dict, List, Optional, etc.)
- Tests in `tests/` using pytest + pytest-asyncio
- loguru for logging
- Clear separation: core/ (db, constants, helpers) → services/ (business logic) → ui/ (dashboard)

## What NOT to Do
- Don't add complexity that can't prove its value in backtesting
- Don't use consumer weather apps as data sources
- Don't place live trades — paper trading only until explicitly approved
- Don't use SQLite — this project uses DuckDB
- Don't use Python 3.10+ syntax (must run on 3.9)
