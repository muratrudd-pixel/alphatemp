# AlphaTemp Phase 2 Design — The Bias Engine

**Date:** 2026-02-22
**Status:** Approved

## Goal

Build the real-time drift detection system that compares HRRR forecast curves against live ASOS observations to produce actionable trading signals for Kalshi temperature markets.

## Architecture

Two new services added to the existing async event loop:

- **HRRRFetcher** (`services/forecast.py`) — Polls HRRR model data via Herbie every 15 minutes. Extracts 2m temperature forecasts for the 5 settlement stations from the last 3 model runs.
- **BiasEngine** (`services/bias.py`) — Runs every 60 seconds after each ingestor cycle. Compares observed temperature trajectory against HRRR forecast curves and produces a full drift report per city.

## Data Flow

```
Every 60s:  Ingestor.poll_once()  →  observations table
                                          ↓
                                   BiasEngine.update()  →  drift_signals table
                                          ↑
Every 15m:  HRRRFetcher.fetch()  →  forecasts table
```

## Database Schema Changes

### forecasts (updated from Phase 1 stub)

| Column | Type | Notes |
|---|---|---|
| station_id | VARCHAR | ICAO code (settlement stations only) |
| model_run | TIMESTAMP | HRRR initialization time |
| valid_at | TIMESTAMP | Forecast valid time (each hour) |
| temp_f | DOUBLE | 2m temperature in °F |
| temp_c | DOUBLE | 2m temperature in °C (native HRRR) |
| ingested_at | TIMESTAMP | Storage timestamp |

UNIQUE: (station_id, model_run, valid_at)

### drift_signals (new)

| Column | Type | Notes |
|---|---|---|
| city | VARCHAR | NYC/PHL/CHI/MIA/LA |
| calculated_at | TIMESTAMP | Report generation time |
| model_run | TIMESTAMP | Which HRRR run this drift is against |
| drift_score | DOUBLE | Mean divergence: observed minus forecast (°F) |
| slope_divergence | DOUBLE | Rate of divergence (°F/hour) |
| forecast_trend | DOUBLE | How successive runs shift the high (°F/run) |
| magnet_proximity | INTEGER | Nearest magnet to projected high |
| magnet_distance | DOUBLE | Distance from projected high to nearest magnet (°F) |
| confidence | DOUBLE | 0.0–1.0 based on observation count and divergence stability |
| projected_high | DOUBLE | Corrected high estimate: HRRR high + drift + slope extrapolation |

UNIQUE: (city, calculated_at, model_run)

## Station Coordinates

Added to core/constants.py for HRRR grid point extraction:

| Station | Lat | Lon |
|---|---|---|
| KNYC | 40.7789 | -73.9692 |
| KPHL | 39.8721 | -75.2411 |
| KMDW | 41.7868 | -87.7522 |
| KMIA | 25.7959 | -80.2870 |
| KLAX | 33.9425 | -118.4081 |

## HRRRFetcher Design

- Uses Herbie library for GRIB2 subsetting (downloads only needed grid points)
- Fetches `TMP:2 m` variable for forecast hours 1–18
- Targets last 3 HRRR runs (accounting for ~90 min publication delay)
- Converts Kelvin → Celsius → Fahrenheit
- Deduplicates on (station_id, model_run, valid_at)
- 15-minute polling cadence
- Graceful skip if GRIB not yet available

## BiasEngine Calculations

1. **drift_score** — Interpolate HRRR hourly curve to observation timestamps, take mean of (observed - forecast)
2. **slope_divergence** — Linear regression on divergence time series (°F/hour)
3. **forecast_trend** — Delta in forecasted daily high across last 3 HRRR runs (°F/run)
4. **projected_high** — Latest HRRR high + drift_score + (slope_divergence × hours remaining)
5. **magnet_proximity/distance** — Nearest magnet number to projected high
6. **confidence** — Weighted: observation count (last 2h) + divergence variance stability, scaled 0.0–1.0

## New Dependencies

- herbie-data (HRRR access)
- xarray (GRIB data handling)
- cfgrib (GRIB2 backend)
- scipy (interpolation, linear regression)

## Tech Stack

Same as Phase 1 + above additions. Python 3.9+, DuckDB, Polars, httpx, Loguru, asyncio.
