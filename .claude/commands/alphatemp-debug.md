---
name: alphatemp-debug
description: Systematic debugging and investigation workflow for the alphatemp weather trading system. Use PROACTIVELY whenever something is broken, behaving unexpectedly, or producing suspicious results — including probability anomalies, data pipeline failures, forecast ingestion issues, DuckDB errors, silent errors, performance degradation, or any "this doesn't look right" observations. Also use when Russell asks to investigate, audit, diagnose, trace, or figure out why something happened. Triggers on words like debug, investigate, broken, wrong, weird, anomaly, issue, bug, error, failing, suspicious, or unexpected.
tools: Read, Bash, Edit, Grep, Write
model: opus
---

# AlphaTemp Debug & Investigate

Systematic investigation protocol for diagnosing issues in the alphatemp weather trading system. This workflow prevents the common failure mode of jumping to fixes before understanding root cause.

## The Investigation Protocol

Follow these steps IN ORDER. Do not skip to fixes.

### Step 1: Define the Problem

State the observed behavior precisely:
- What is happening? (with specific numbers, timestamps, stations if applicable)
- What should be happening instead?
- When did it start? (or when was it first noticed?)
- Is it intermittent or consistent?
- Which components are affected? (forecast ingestion, probability engine, backtester, dashboard, etc.)

### Step 2: Gather Evidence

Before forming any hypothesis, collect data:

1. **Check logs** — Read recent log output for errors, warnings, and unexpected state. Look for the relevant time window.
2. **Check the database** — Use the DuckDB MCP server or direct queries to verify data state:
   ```sql
   -- Data freshness
   SELECT MAX(model_run) FROM forecasts WHERE station_id = 'KNYC';
   SELECT MAX(observed_at) FROM observations WHERE station_id = 'KNYC';
   SELECT MAX(obs_date) FROM nws_daily WHERE station_id = 'KNYC';
   ```
3. **Check system state** — Is the service running? Docker container status? Any recent deploys?
4. **Check external dependencies** — Are Open-Meteo, Synoptic API, AWC, NWS returning data? Any known outages?

### Step 3: Form Hypotheses

Based on evidence, generate 2-3 ranked hypotheses:

```
Hypothesis 1 (most likely): [description]
  Evidence for: [what supports this]
  Evidence against: [what contradicts this]
  How to verify: [specific check or test]

Hypothesis 2: [description]
  Evidence for: ...
  Evidence against: ...
  How to verify: ...
```

Present these to Russell before proceeding.

### Step 4: Verify — Don't Fix

For the top hypothesis, propose a **read-only verification step**:
- Add a temporary log line
- Query specific data
- Run a diagnostic query against DuckDB
- Compare expected vs actual values

CRITICAL: The verification step must NOT change system behavior. No code fixes yet.

### Step 5: Confirm and Fix

Only after verification confirms the root cause:

1. Explain the confirmed root cause clearly
2. Propose the minimal fix
3. Explain what the fix changes and what it doesn't change
4. Identify any risks or side effects
5. Wait for Russell's approval before implementing

### Step 6: Verify the Fix

After implementing:

1. Run relevant tests: `pytest tests/ -v`
2. Confirm the fix resolves the original symptom
3. Check that nothing else broke (especially probability calculations or data ingestion)
4. If applicable, verify with a sample query against the database

## Common Investigation Patterns

### Probability Anomalies
When model probabilities look wrong:
1. Check raw forecast data — `SELECT * FROM forecasts WHERE station_id = 'KNYC' ORDER BY model_run DESC LIMIT 20`
2. Check bias stats — `SELECT * FROM station_bias WHERE station_id = 'KNYC' ORDER BY calculated_at DESC LIMIT 5`
3. Check drift signals — `SELECT * FROM drift_signals WHERE city = 'NYC' ORDER BY calculated_at DESC LIMIT 5`
4. Verify bracket probabilities sum to ~1.0
5. Check timezone handling — is the model looking at the right day's forecast?

### Data Pipeline Issues
When forecasts, observations, or settlements aren't updating:
1. Check `services/forecast.py` — HRRR ingestion via Open-Meteo
2. Check `services/ingestor.py` — Observation ingestion (Synoptic + AWC)
3. Check `services/nws_fetcher.py` — NWS CLI report parsing
4. Verify API keys/endpoints are valid
5. Check for rate limiting or API changes

### DuckDB Issues
When database operations fail:
1. Check for connection leaks — only one write connection allowed at a time
2. Check for timezone mismatches — DuckDB TIMESTAMP columns store naive UTC
3. Check for schema changes — `DESCRIBE table_name` to verify structure
4. Check disk space — DuckDB files can grow with frequent writes

### Backtester Issues
When backtest results look wrong:
1. Verify settlement data coverage — `SELECT COUNT(*), MIN(obs_date), MAX(obs_date) FROM nws_daily WHERE station_id = 'KNYC'`
2. Check Kalshi bracket data — `SELECT COUNT(DISTINCT event_date) FROM kalshi_settlements`
3. Verify model function returns valid probabilities (sums to ~1.0, all values in [0,1])
4. Check for NULL handling in bracket mapping

### Silent Failures
When the system appears running but isn't doing anything:
1. Check if the main loop is executing (look for heartbeat logs)
2. Check if data providers are returning empty responses without errors
3. Verify Docker container logs: `docker logs alphatemp --tail 100`
4. Check if the database file is locked

### Timezone Issues
Always suspect when things break around midnight or settlement:
1. NWS CLI uses Local Standard Time (NOT DST)
2. HRRR model runs are UTC
3. DuckDB TIMESTAMP columns are naive UTC — never pass timezone-aware datetimes directly
4. Use `_strip_tz()` from `services/data_provider.py` when querying with aware datetimes

## Anti-Patterns — Never Do These During Investigation

- Do NOT change code to "see if this fixes it" without understanding the root cause first
- Do NOT restart services as a first resort — you'll lose diagnostic state
- Do NOT dismiss anomalies as "probably nothing" — in a trading system, anomalies are signals
- Do NOT investigate multiple hypotheses simultaneously. Pick one, verify it, then move to the next
- Do NOT make the investigation itself modify data or trading behavior. All diagnostic steps should be read-only
