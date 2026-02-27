---
name: duckdb-data-pipeline
description: Operational patterns for BUILDING DuckDB data pipelines in alphatemp — temp-DB-then-merge, crash avoidance, connection lifecycle, idempotent writes, schema migrations. Use PROACTIVELY when writing or modifying any script that writes to DuckDB, creating new backfill scripts, adding tables or columns, or debugging write failures. NOT for querying (use duckdb-explorer) or data flow reference (use alphatemp-data-pipeline). Triggers on new backfill, write pipeline, schema change, migration, merge script, temp DB, WAL, connection error, constraint, ATTACH.
tools: Read, Write, Edit, Bash, Grep
model: opus
---

# DuckDB Data Pipeline — Operational Patterns

How to write, modify, and maintain DuckDB data pipelines in alphatemp. This skill is about BUILDING pipelines, not querying data.

## Critical Constraint: DuckDB 1.4.4 Crash Avoidance

DuckDB 1.4.4 has a known issue with long-running write sessions causing B-tree corruption and SIGSEGV crashes. This has corrupted the WAL and lost data in production.

**Rules:**
- NEVER run long write sessions (>500 rows) directly against the main DB
- Use the temp-DB-then-merge pattern for all backfills
- Do NOT use `CHECKPOINT` — it triggers the crash in this version
- If a WAL file exists after a crash, removing it recovers the DB (rolls back unflushed writes)

## Pattern 1: Temp-DB-Then-Merge

The core pattern for safe bulk writes. Each worker writes to a separate temp DB, then a merge script combines them.

### Writing to Temp DB

```python
# Each worker gets its own temp DB file
db_path = f"data/backfill_{hour:02d}z.duckdb"

# Initialize schema (idempotent — safe to call multiple times)
init_db(db_path)

# Write freely — no contention with main DB
con = duckdb.connect(db_path)
try:
    for row in data:
        try:
            con.execute("INSERT INTO table_name (...) VALUES (?, ...)", [row])
        except duckdb.ConstraintException:
            pass  # Duplicate — skip
finally:
    con.close()
```

### Merging to Main DB

```python
# ATTACH/DETACH pattern — read temp, write to main
con = duckdb.connect(MAIN_DB)

con.execute(f"ATTACH '{temp_path}' AS src (READ_ONLY)")

before = con.execute("SELECT COUNT(*) FROM table_name").fetchone()[0]

con.execute("""
    INSERT INTO table_name
    SELECT s.* FROM src.table_name s
    WHERE NOT EXISTS (
        SELECT 1 FROM table_name m
        WHERE m.key_col1 = s.key_col1
          AND m.key_col2 = s.key_col2
    )
""")

after = con.execute("SELECT COUNT(*) FROM table_name").fetchone()[0]
logger.info(f"  table_name: +{after - before:,} rows")

con.execute("DETACH src")
con.close()
```

### Cleanup After Merge

```python
for f in [temp_path, temp_path + ".wal"]:
    if os.path.exists(f):
        os.remove(f)
        logger.info(f"Removed {f}")
```

### When to Use

- Any backfill touching >100 rows
- Any operation running >30 seconds
- Any parallel ingestion (each worker MUST have its own temp DB)

## Pattern 2: Connection Lifecycle

### Short Operations (<1s)
```python
con = get_connection(db_path)
con.execute(...)
con.close()
```

### Long Operations (>1s)
```python
con = duckdb.connect(db_path)
try:
    while has_work:
        con.execute(...)
finally:
    con.close()  # Always close, even on error
```

### Read-Only (Audits, Checks, Source Reads)
```python
con = duckdb.connect(db_path, read_only=True)
result = con.execute(...).fetchall()
con.close()
```

- `read_only=True` never blocks writers
- Safe to open multiple read-only connections simultaneously
- Use for resumption checks, final audits, cross-DB reads

### Rules
- **One write connection at a time** — DuckDB is single-writer
- **Always use try/finally for operations >1s** — learned from test failures where assertions skipped `con.close()`
- **Close connections promptly** — leaked connections block other sessions
- **Clear module-level caches** between tests — `_p2b_level1.clear()` etc.

## Pattern 3: Idempotent Writes

### Single-Row: Catch ConstraintException
```python
try:
    con.execute("INSERT INTO ... VALUES (?, ...)", [row])
    inserted += 1
except duckdb.ConstraintException:
    skipped += 1  # Duplicate — skip silently
```

### Bulk: INSERT...SELECT WHERE NOT EXISTS
```python
con.execute("""
    INSERT INTO target_table
    SELECT s.* FROM src.source_table s
    WHERE NOT EXISTS (
        SELECT 1 FROM target_table t
        WHERE t.key1 = s.key1 AND t.key2 = s.key2
    )
""")
```

Use `WHERE NOT EXISTS` for bulk merges (more efficient than per-row try/except).

### Delete-Then-Insert Upsert
```python
# When authoritative source must overwrite existing data
con.execute("DELETE FROM nws_daily WHERE station_id = ? AND obs_date = ?", [sid, date])
con.execute("INSERT INTO nws_daily (...) VALUES (?, ...)", [row])
```

Used when NWS CLI (authoritative) overwrites ACIS backfill.

## Pattern 4: Resume Support

```python
def _get_resume_date(db_path, model):
    """Check what's already ingested — resume from there."""
    con = duckdb.connect(db_path, read_only=True)
    result = con.execute(
        "SELECT MAX(valid_at) FROM forecasts WHERE model_name = ?", [model]
    ).fetchone()
    con.close()
    if result and result[0]:
        return result[0].date() + timedelta(days=1)
    return None

# Then skip past already-ingested dates
resume_date = _get_resume_date(db_path, model)
if resume_date and resume_date > start_date:
    start_date = resume_date
```

Every backfill script should support resumption. Users WILL interrupt and restart.

## Pattern 5: Source DB Pattern

Read from main DB, write to temp DB. Prevents long write locks on main.

```python
def run_phase(client, db_path, source_db=None):
    con = duckdb.connect(db_path)  # Write target (temp DB)

    if source_db:
        source_con = duckdb.connect(source_db, read_only=True)
        markets = source_con.execute("SELECT ... FROM kalshi_settlements").fetchall()
        source_con.close()  # Close source immediately
    else:
        markets = con.execute("SELECT ... FROM kalshi_settlements").fetchall()

    # ... write to con (temp DB) ...
    con.close()
```

## Pattern 6: Schema Migrations

### Adding Columns (Idempotent)
```python
for col in ("new_col1", "new_col2"):
    try:
        con.execute(f"ALTER TABLE table_name ADD COLUMN {col} DOUBLE")
    except Exception:
        pass  # Column already exists
```

### Table Rebuild (When UNIQUE Constraint Changes)

DuckDB can't drop inline UNIQUE constraints. Must rebuild:

```python
# 1. Check if migration needed
has_col = con.execute("""
    SELECT COUNT(*) FROM information_schema.columns
    WHERE table_name = 'forecasts' AND column_name = 'model_name'
""").fetchone()[0]
if has_col > 0:
    return  # Already migrated

# 2. Create new table with updated schema
con.execute("DROP TABLE IF EXISTS forecasts_new")
con.execute("CREATE TABLE forecasts_new (...new schema...)")

# 3. Copy data
con.execute("INSERT INTO forecasts_new SELECT ..., 'hrrr' AS model_name FROM forecasts")

# 4. VERIFY row counts before destructive drop
old_count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
new_count = con.execute("SELECT COUNT(*) FROM forecasts_new").fetchone()[0]
if old_count != new_count:
    con.execute("DROP TABLE forecasts_new")
    raise RuntimeError(f"Migration mismatch: {old_count} vs {new_count}")

# 5. Swap
con.execute("DROP TABLE forecasts")
con.execute("ALTER TABLE forecasts_new RENAME TO forecasts")
```

**Always verify row counts before dropping the original table.**

## Pattern 7: Rate Limiting

```python
REQUEST_DELAY = 0.5  # Seconds between API calls
BACKOFF_BASE = 2.0   # For exponential backoff
MAX_RETRIES = 5

def api_call_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 429:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Rate limited, backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Max retries exceeded")
```

Standard delays by API:
- Synoptic: 0.5s
- IEM ASOS: 1.0s
- Open-Meteo: 1.0s
- Kalshi: 0.5s + exponential backoff on 429

## Checklist for New Pipeline Scripts

- [ ] Uses temp-DB-then-merge (not direct main DB writes)
- [ ] try/finally around all connections
- [ ] UNIQUE constraints for idempotency
- [ ] Resume support (MAX() query to find start point)
- [ ] Rate limiting between API calls
- [ ] Progress logging (every N rows/markets)
- [ ] Error handling that continues on single-row failures
- [ ] Merge function added to `backfill_merge.py`
- [ ] WAL cleanup in merge script

## Reference: Existing Backfill Scripts

| Script | Pattern | Temp DB |
|--------|---------|---------|
| `backfill_parallel.py` | Per-run-hour HRRR | `data/backfill_{hour}z.duckdb` |
| `backfill_openmeteo.py` | Batched GFS/ECMWF | `data/backfill_{model}.duckdb` |
| `backfill_kalshi_history.py` | 3-phase A/B/C | `data/backfill_kalshi_{type}.duckdb` |
| `backfill_iem.py` | Chunked historical obs | Per-batch connection |
| `backfill_merge.py` | ATTACH/DETACH merge | N/A (reads temp, writes main) |
