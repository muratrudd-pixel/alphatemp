---
name: duckdb-explorer
description: Query patterns and schema reference for the alphatemp DuckDB database. Use PROACTIVELY whenever Russell asks to query the database, check data coverage, verify data freshness, explore table schemas, debug data issues, or write SQL against alphatemp.duckdb. Also use when writing or modifying any code that queries DuckDB. Triggers on terms like query, SQL, DuckDB, table, schema, data check, coverage, freshness, database, SELECT, or any mention of specific tables like forecasts, observations, nws_daily, kalshi_settlements, station_bias, drift_signals.
tools: Read, Bash, Grep
model: sonnet
---

# AlphaTemp DuckDB Explorer

Schema reference and query patterns for `data/alphatemp.duckdb`.

## Connection Patterns

### Read-only queries (safe, allows concurrent access):
```python
import duckdb
con = duckdb.connect("data/alphatemp.duckdb", read_only=True)
result = con.execute("SELECT ...").fetchall()
con.close()
```

### Write operations (exclusive lock):
```python
from core.db import get_connection
con = get_connection("data/alphatemp.duckdb")
con.execute("INSERT INTO ...")
con.close()
```

CRITICAL: Only one write connection at a time. Close connections promptly to avoid locks.

### Using the DuckDB MCP Server
The MCP server provides read-only access without file locking. Prefer this for exploration.

## Schema Reference

### forecasts
HRRR temperature forecasts by station and model run.
```sql
DESCRIBE forecasts;
-- station_id   VARCHAR    -- e.g., 'KNYC', 'KLGA'
-- model_run    TIMESTAMP  -- Naive UTC, e.g., '2025-01-15 12:00:00'
-- valid_at     TIMESTAMP  -- Naive UTC, forecast valid time
-- temp_f       DOUBLE     -- Temperature in Fahrenheit
-- temp_c       DOUBLE     -- Temperature in Celsius
-- ingested_at  TIMESTAMP  -- When this row was inserted
```

### observations
Real-time METAR observations from Synoptic API and AWC.
```sql
DESCRIBE observations;
-- station_id   VARCHAR
-- observed_at  TIMESTAMP  -- Naive UTC
-- temp_f       DOUBLE
-- temp_c       DOUBLE
-- source       VARCHAR    -- 'synoptic' or 'awc'
-- raw          VARCHAR    -- Raw METAR text (if available)
```

### nws_daily
NWS Daily Climate Report data — the settlement truth source.
```sql
DESCRIBE nws_daily;
-- station_id   VARCHAR
-- obs_date     DATE       -- The calendar date
-- max_temp_f   DOUBLE     -- Daily high temperature
-- min_temp_f   DOUBLE     -- Daily low temperature (may be NULL)
-- source       VARCHAR    -- 'cli' or 'acis'
```

### station_bias
Historical forecast bias statistics per station.
```sql
DESCRIBE station_bias;
-- station_id    VARCHAR
-- calculated_at TIMESTAMP
-- mean_bias     DOUBLE     -- Positive = forecast runs hot
-- std_error     DOUBLE     -- Standard deviation of forecast errors
-- sample_days   INTEGER    -- Number of days in the calculation
```

### drift_signals
Intraday observation vs forecast drift.
```sql
DESCRIBE drift_signals;
-- city           VARCHAR
-- calculated_at  TIMESTAMP
-- model_run      TIMESTAMP
-- drift_score    DOUBLE
-- slope_divergence DOUBLE
-- forecast_trend DOUBLE
-- confidence     DOUBLE
-- projected_high DOUBLE
```

### kalshi_settlements
Historical Kalshi bracket settlement outcomes.
```sql
DESCRIBE kalshi_settlements;
-- event_date    DATE
-- floor_strike  DOUBLE     -- NULL for lower tail
-- cap_strike    DOUBLE     -- NULL for upper tail
-- settled_yes   INTEGER    -- 1 if this bracket settled YES
```
WARNING: ~1,418 records have BOTH floor_strike and cap_strike as NULL (unparsed ticker data). Always filter: `WHERE floor_strike IS NOT NULL OR cap_strike IS NOT NULL`.

### market_ticks
Live Kalshi market price snapshots.
```sql
DESCRIBE market_ticks;
-- ticker     VARCHAR
-- timestamp  TIMESTAMP
-- yes_bid    DOUBLE
-- yes_ask    DOUBLE
-- volume     INTEGER
```

## Common Query Patterns

### Data Freshness
```sql
-- When was each data source last updated?
SELECT 'forecasts' as source, MAX(model_run) as latest FROM forecasts WHERE station_id = 'KNYC'
UNION ALL
SELECT 'observations', MAX(observed_at) FROM observations WHERE station_id = 'KNYC'
UNION ALL
SELECT 'nws_daily', MAX(obs_date)::TIMESTAMP FROM nws_daily WHERE station_id = 'KNYC'
UNION ALL
SELECT 'bias', MAX(calculated_at) FROM station_bias WHERE station_id = 'KNYC'
UNION ALL
SELECT 'drift', MAX(calculated_at) FROM drift_signals WHERE city = 'NYC';
```

### Forecast for a Specific Model Run
```sql
SELECT valid_at, temp_f
FROM forecasts
WHERE station_id = 'KNYC' AND model_run = '2025-06-15 12:00:00'
ORDER BY valid_at;

-- Forecast high for that run
SELECT MAX(temp_f) FROM forecasts
WHERE station_id = 'KNYC' AND model_run = '2025-06-15 12:00:00';
```

### Forecast vs Actual Comparison
```sql
SELECT
    n.obs_date,
    n.max_temp_f as actual_high,
    MAX(f.temp_f) as forecast_high,
    MAX(f.temp_f) - n.max_temp_f as error
FROM nws_daily n
JOIN forecasts f ON f.station_id = n.station_id
    AND f.model_run::DATE = n.obs_date
    AND EXTRACT(HOUR FROM f.model_run) = 12  -- 12z runs only
WHERE n.station_id = 'KNYC'
GROUP BY n.obs_date, n.max_temp_f
ORDER BY n.obs_date DESC
LIMIT 20;
```

### Coverage by Run Hour
```sql
SELECT
    EXTRACT(HOUR FROM model_run) as run_hour,
    COUNT(*) as n_forecasts,
    COUNT(DISTINCT model_run::DATE) as n_days,
    MIN(model_run)::DATE as first_date,
    MAX(model_run)::DATE as last_date
FROM forecasts
WHERE station_id = 'KNYC'
GROUP BY run_hour
ORDER BY run_hour;
```

### Kalshi Settlement Analysis
```sql
-- Brackets for a specific date
SELECT floor_strike, cap_strike, settled_yes
FROM kalshi_settlements
WHERE event_date = '2025-06-15'
AND (floor_strike IS NOT NULL OR cap_strike IS NOT NULL)
ORDER BY floor_strike NULLS FIRST;

-- Which bracket settled YES?
SELECT floor_strike, cap_strike
FROM kalshi_settlements
WHERE event_date = '2025-06-15' AND settled_yes = 1;
```

### Bias History
```sql
SELECT calculated_at, mean_bias, std_error, sample_days
FROM station_bias
WHERE station_id = 'KNYC'
ORDER BY calculated_at DESC
LIMIT 10;
```

### Gap Detection
```sql
-- Find dates missing from nws_daily
WITH date_range AS (
    SELECT UNNEST(generate_series(
        (SELECT MIN(obs_date) FROM nws_daily WHERE station_id = 'KNYC'),
        (SELECT MAX(obs_date) FROM nws_daily WHERE station_id = 'KNYC'),
        INTERVAL 1 DAY
    ))::DATE as d
)
SELECT d FROM date_range
WHERE d NOT IN (SELECT obs_date FROM nws_daily WHERE station_id = 'KNYC')
ORDER BY d;
```

## Timezone Rules for Queries

- All TIMESTAMP columns store **naive UTC** — never pass timezone-aware Python datetimes
- Use `_strip_tz()` from `services/data_provider.py` when querying from Python with aware datetimes
- When filtering by ET date, convert ET boundaries to UTC first using `core/timezone.py` helpers
- `obs_date` in `nws_daily` is a DATE (no timezone issue), but be aware CLI uses Local Standard Time
