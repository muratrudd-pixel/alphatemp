# Phase 3: Multi-Model Ensemble Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GFS and ECMWF forecasts from Open-Meteo, apply independent bias correction per model, and combine into a Gaussian mixture ensemble that beats HRRR alone.

**Architecture:** Open-Meteo Historical Forecast API provides daily hourly temperature curves for GFS/ECMWF at NYC coords. Each model gets independent Phase 1+2+2B bias correction. Ensemble combines corrected Gaussians as a mixture distribution (not a single Gaussian). Equal weights first, learned weights only if gate passes.

**Tech Stack:** Python 3.9, DuckDB 1.4.4, scipy, requests (Open-Meteo HTTP API), pytest

**Design doc:** `docs/plans/2026-02-26-phase3-ensemble-design.md`

---

## Task 1: Schema Migration — Add `model_name` Column

**Files:**
- Modify: `core/db.py` (lines 27-36 CREATE TABLE, lines 169-170 indexes)
- Test: `tests/test_db.py`

**Step 1: Write failing test for model_name column**

In `tests/test_db.py`, add:

```python
def test_forecasts_has_model_name_column():
    """model_name column exists with default 'hrrr'."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""INSERT INTO forecasts
        (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
        VALUES ('KNYC', '2024-01-01 12:00', '2024-01-01 13:00', 50.0, 10.0, '2024-01-01 14:00')""")
    row = con.execute("SELECT model_name FROM forecasts").fetchone()
    assert row[0] == "hrrr"
    con.close()


def test_forecasts_unique_constraint_includes_model_name():
    """Same station/run/valid_at but different model_name should NOT conflict."""
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""INSERT INTO forecasts
        (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
        VALUES ('KNYC', '2024-01-01 00:00', '2024-01-01 01:00', 50.0, 10.0, '2024-01-01 02:00', 'hrrr')""")
    # Same station/run/valid_at but model_name='gfs' should succeed
    con.execute("""INSERT INTO forecasts
        (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
        VALUES ('KNYC', '2024-01-01 00:00', '2024-01-01 01:00', 48.0, 8.9, '2024-01-01 02:00', 'gfs')""")
    count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    assert count == 2
    # Same model_name + station/run/valid_at SHOULD conflict
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""INSERT INTO forecasts
            (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
            VALUES ('KNYC', '2024-01-01 00:00', '2024-01-01 01:00', 49.0, 9.4, '2024-01-01 02:00', 'gfs')""")
    con.close()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py::test_forecasts_has_model_name_column -v
pytest tests/test_db.py::test_forecasts_unique_constraint_includes_model_name -v
```
Expected: FAIL (no model_name column)

**Step 3: Implement schema migration in `core/db.py`**

The CREATE TABLE (line 27) stays as-is for fresh databases — update it to include model_name:

```python
con.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        station_id  VARCHAR NOT NULL,
        model_run   TIMESTAMP NOT NULL,
        valid_at    TIMESTAMP NOT NULL,
        temp_f      DOUBLE,
        temp_c      DOUBLE,
        ingested_at TIMESTAMP NOT NULL,
        model_name  VARCHAR NOT NULL DEFAULT 'hrrr',
        UNIQUE (station_id, model_run, valid_at, model_name)
    )
""")
```

For existing databases, add a migration in the migration section (after line ~72). DuckDB can't drop inline UNIQUE constraints, so we need the table-rebuild dance:

```python
# --- forecasts: add model_name column (Phase 3) ---
try:
    has_col = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = 'forecasts' AND column_name = 'model_name'"
    ).fetchone()[0]
    if not has_col:
        logger.info("Migrating forecasts table: adding model_name column...")
        con.execute("""
            CREATE TABLE forecasts_new (
                station_id  VARCHAR NOT NULL,
                model_run   TIMESTAMP NOT NULL,
                valid_at    TIMESTAMP NOT NULL,
                temp_f      DOUBLE,
                temp_c      DOUBLE,
                ingested_at TIMESTAMP NOT NULL,
                model_name  VARCHAR NOT NULL DEFAULT 'hrrr',
                UNIQUE (station_id, model_run, valid_at, model_name)
            )
        """)
        con.execute("""
            INSERT INTO forecasts_new
            SELECT station_id, model_run, valid_at, temp_f, temp_c,
                   ingested_at, 'hrrr' as model_name
            FROM forecasts
        """)
        con.execute("DROP TABLE forecasts")
        con.execute("ALTER TABLE forecasts_new RENAME TO forecasts")
        con.execute("CREATE INDEX IF NOT EXISTS idx_fcst_station_run ON forecasts (station_id, model_run)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_fcst_station_valid ON forecasts (station_id, valid_at)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_fcst_model ON forecasts (model_name)")
        logger.info("Migration complete: model_name column added to forecasts")
except Exception as e:
    logger.warning(f"forecasts model_name migration check: {e}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```
Expected: ALL PASS (including new tests)

**Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```
Expected: Some tests may fail if they insert into forecasts without model_name — fix any that break by adding the column to INSERT statements in test seed helpers.

**Step 6: Commit**

```bash
git add core/db.py tests/test_db.py
git commit -m "feat: add model_name column to forecasts table (Phase 3 schema migration)"
```

---

## Task 2: Add `model_name` Filters to Backtester Queries

**Files:**
- Modify: `services/backtester.py` (lines 245, 350, 601, 641, 947)
- Test: `tests/test_phase2_models.py` or new `tests/test_multimodel_safety.py`

**Step 1: Write failing test — multi-model data isolation**

Create `tests/test_multimodel_safety.py`:

```python
"""Verify that backtester queries filter by model_name and don't mix models."""
import duckdb
import pytest
from datetime import date, datetime, timezone
from core.db import init_db
from services.backtester import _walk_forward_bias_query

TEST_DB = "data/test_multimodel.duckdb"

@pytest.fixture(autouse=True)
def clean_db():
    import os
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _seed_multimodel(con):
    """Seed HRRR and GFS data with DIFFERENT biases."""
    for i in range(100):
        d = date(2023, 4, 1 + i % 28) if i < 28 else date(2023, 5, 1 + (i - 28) % 31)
        # Adjust to get 100 unique dates
        d = date(2023, 1, 1) + __import__('datetime').timedelta(days=i)
        actual = 50.0
        hrrr_fcst = 52.0  # HRRR bias = +2.0
        gfs_fcst = 46.0   # GFS bias = -4.0

        con.execute("INSERT INTO nws_daily (station_id, obs_date, max_temp_f) VALUES (?, ?, ?)",
                     ["KNYC", d, actual])
        # HRRR forecast
        mr = datetime(d.year, d.month, d.day, 12, 0)
        for fxx in range(1, 19):
            va = datetime(d.year, d.month, d.day, 12 + fxx, 0) if 12 + fxx < 24 else \
                 datetime(d.year, d.month, d.day + 1 if d.day < 28 else 1, (12 + fxx) % 24, 0)
            con.execute(
                "INSERT OR IGNORE INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ["KNYC", mr, mr + __import__('datetime').timedelta(hours=fxx), hrrr_fcst, 11.1, mr, "hrrr"])
        # GFS forecast (midnight run)
        mr_gfs = datetime(d.year, d.month, d.day, 0, 0)
        for h in range(24):
            con.execute(
                "INSERT OR IGNORE INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ["KNYC", mr_gfs, datetime(d.year, d.month, d.day, h, 0), gfs_fcst, 7.8, mr_gfs, "gfs"])


def test_walk_forward_bias_isolates_hrrr():
    """HRRR bias query should NOT be contaminated by GFS data."""
    con = duckdb.connect(TEST_DB)
    _seed_multimodel(con)
    result = _walk_forward_bias_query(con, 12, "KNYC", date(2023, 4, 11))
    assert result is not None
    mean_bias = result[0]
    # HRRR bias should be ~+2.0, NOT influenced by GFS's -4.0
    assert 1.5 < mean_bias < 2.5, f"HRRR bias {mean_bias} contaminated by GFS data"
    con.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_multimodel_safety.py::test_walk_forward_bias_isolates_hrrr -v
```
Expected: FAIL — bias will be wrong because GFS data mixes in (or test may need adjustments based on exact seed data)

**Step 3: Add model_name parameter to backtester query functions**

In `services/backtester.py`, modify each function:

**`_walk_forward_bias_query` (line ~226):**
Add `model_name='hrrr'` parameter. Add `AND f.model_name = ?` to WHERE clause. Add `model_name` to the parameter list.

```python
def _walk_forward_bias_query(con, run_hour, station_id, current_date, model_name='hrrr'):
```
SQL change: add `AND f.model_name = ?` after `AND EXTRACT(HOUR FROM f.model_run) = ?`
Params: `[run_hour, station_id, current_date]` → `[run_hour, model_name, station_id, current_date]`

**`_walk_forward_regression_data` (line ~334):**
Same pattern — add `model_name='hrrr'` param, `AND f.model_name = ?` to SQL.

**`_ensure_level1` (line ~577):**
Add `model_name='hrrr'` param. Update cache key to include model_name: `key = (id(con), run_hour, station_id, model_name)`. Add filter to both forecast queries (lines ~601 and ~641).

**`_has_forecast_data` (line ~947):**
Add `model_name='hrrr'` param, `AND model_name = ?` to SQL.

**Step 4: Run tests**

```bash
pytest tests/test_multimodel_safety.py -v
pytest tests/test_phase2_models.py -v
pytest tests/test_phase2b_models.py -v
```
Expected: ALL PASS — existing tests use default model_name='hrrr', new test verifies isolation.

**Step 5: Commit**

```bash
git add services/backtester.py tests/test_multimodel_safety.py
git commit -m "feat: add model_name filter to backtester queries (multi-model safety)"
```

---

## Task 3: Add `model_name` Filters to DataProvider

**Files:**
- Modify: `services/data_provider.py` (6 forecast queries)
- Test: `tests/test_data_provider.py`

**Step 1: Add model_name to BacktestDataProvider constructor**

```python
class BacktestDataProvider(DataProvider):
    def __init__(self, db_path, station_id, model_run, ref_time, connection=None, model_name='hrrr'):
        # ... existing init ...
        self.model_name = model_name
```

**Step 2: Add `AND model_name = ?` to all BacktestDataProvider forecast queries**

- `get_forecast_high` (line 294): add `AND model_name = ?` with `self.model_name`
- `get_recent_forecast_highs` (line 318): same
- `get_forecast_curve` (line 334): same

**Step 3: Add model_name to LiveDataProvider forecast queries**

Default to `'hrrr'` in all queries. The LiveDataProvider will need a `model_name` constructor param for future multi-model live mode, but for now default to `'hrrr'`.

- `get_forecast_high` (line 151): add `AND f.model_name = 'hrrr'`
- `get_recent_forecast_highs` (line 191): same
- `get_forecast_curve` (line 204): same

**Step 4: Update test seed helpers in `tests/test_data_provider.py`**

Add `model_name` to any INSERT INTO forecasts statements in seed functions. Verify existing tests still pass.

**Step 5: Run tests**

```bash
pytest tests/test_data_provider.py -v
```

**Step 6: Commit**

```bash
git add services/data_provider.py tests/test_data_provider.py
git commit -m "feat: add model_name filter to DataProvider forecast queries"
```

---

## Task 4: Add `model_name` Filters to Remaining Services

**Files:**
- Modify: `services/forecast.py` (line 53, line 125)
- Modify: `services/forecast_backfiller.py` (line 55)
- Modify: `services/bias_model.py` (lines 48, 63, 72)
- Modify: `services/bias.py` (lines 45, 59, 100, 171)
- Modify: `ui/web_dashboard.py` (lines 74, 233, 330, 333, 343, 345, 398, 405, 426, 432)
- Modify: `scripts/backfill_merge.py` — update dedup query to include model_name

**Approach:** Add `AND model_name = 'hrrr'` (or `AND f.model_name = 'hrrr'`) to every SELECT on the forecasts table. For INSERT statements in `forecast.py`, add `model_name` column with value `'hrrr'`.

**Key changes:**

`services/forecast.py` line 53 (get latest stored run):
```sql
SELECT MAX(model_run) FROM forecasts WHERE model_name = 'hrrr'
```

`services/forecast.py` line 125 (INSERT):
```sql
INSERT INTO forecasts
(station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
VALUES (?, ?, ?, ?, ?, ?, 'hrrr')
```

`scripts/backfill_merge.py` — update the NOT EXISTS dedup to include model_name:
```sql
INSERT INTO forecasts
SELECT s.* FROM src.forecasts s
WHERE NOT EXISTS (
    SELECT 1 FROM forecasts m
    WHERE m.station_id = s.station_id
      AND m.model_run  = s.model_run
      AND m.valid_at   = s.valid_at
      AND m.model_name = s.model_name
)
```

**Step: Run full test suite**

```bash
pytest tests/ -v
```
Expected: ALL 189+ tests pass.

**Commit:**

```bash
git add services/forecast.py services/forecast_backfiller.py services/bias_model.py services/bias.py ui/web_dashboard.py scripts/backfill_merge.py
git commit -m "feat: add model_name='hrrr' filter to all forecast queries (multi-model safe)"
```

---

## Task 5: Open-Meteo Backfill Script

**Files:**
- Create: `scripts/backfill_openmeteo.py`
- Test: `tests/test_openmeteo_backfill.py`

**Step 1: Write the backfill script**

```python
"""Backfill GFS and ECMWF historical forecasts from Open-Meteo API.

Fetches hourly temperature_2m for NYC (Central Park coords) and writes
to a temp DuckDB file for later merge into the main DB.

Usage:
    python scripts/backfill_openmeteo.py gfs       # backfill GFS
    python scripts/backfill_openmeteo.py ecmwf     # backfill ECMWF
    python scripts/backfill_openmeteo.py gfs --start 2023-01-01 --end 2023-06-30
"""
import sys
import time
from datetime import datetime, date, timedelta

import duckdb
import requests
from loguru import logger

from core.db import init_db

# Central Park coords (same as STATION_COORDS["KNYC"])
LAT = 40.7789
LON = -73.9692

# Open-Meteo model identifiers
MODELS = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs",
}

# Open-Meteo data availability start dates
MODEL_START_DATES = {
    "gfs": date(2021, 3, 23),
    "ecmwf": date(2017, 1, 1),
}

# Max days per API call (Open-Meteo handles large ranges but be polite)
BATCH_DAYS = 60

API_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_batch(model_key, start_date, end_date):
    """Fetch one batch from Open-Meteo. Returns list of (valid_at, temp_f) tuples."""
    model_id = MODELS[model_key]
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "models": model_id,
    }
    resp = requests.get(API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    results = []
    for t_str, temp_f in zip(times, temps):
        if temp_f is None:
            continue
        valid_at = datetime.fromisoformat(t_str)
        results.append((valid_at, round(temp_f, 1)))

    return results


def backfill_model(model_key, start_date=None, end_date=None, db_path=None):
    """Backfill a single model to a temp DuckDB file."""
    if start_date is None:
        start_date = MODEL_START_DATES[model_key]
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    if db_path is None:
        db_path = f"data/backfill_{model_key}.duckdb"

    logger.info(f"Backfilling {model_key} from {start_date} to {end_date} -> {db_path}")

    init_db(db_path)
    con = duckdb.connect(db_path)

    # Find latest date already in temp DB to resume
    try:
        row = con.execute(
            "SELECT MAX(valid_at)::DATE FROM forecasts WHERE model_name = ?",
            [model_key]
        ).fetchone()
        if row and row[0]:
            resume_date = row[0] + timedelta(days=1)
            if resume_date > start_date:
                logger.info(f"Resuming from {resume_date} (found existing data)")
                start_date = resume_date
    except Exception:
        pass

    total_inserted = 0
    current = start_date

    while current <= end_date:
        batch_end = min(current + timedelta(days=BATCH_DAYS - 1), end_date)
        logger.info(f"Fetching {model_key} {current} to {batch_end}...")

        try:
            rows = fetch_batch(model_key, current, batch_end)
        except Exception as e:
            logger.error(f"API error for {current} to {batch_end}: {e}")
            current = batch_end + timedelta(days=1)
            time.sleep(5)
            continue

        now = datetime.utcnow()
        batch_inserted = 0

        for valid_at, temp_f in rows:
            # model_run = midnight UTC of the forecast day
            model_run = datetime(valid_at.year, valid_at.month, valid_at.day, 0, 0)
            temp_c = round((temp_f - 32.0) * 5.0 / 9.0, 2)
            try:
                con.execute(
                    """INSERT INTO forecasts
                    (station_id, model_run, valid_at, temp_f, temp_c, ingested_at, model_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    ["KNYC", model_run, valid_at, temp_f, temp_c, now, model_key]
                )
                batch_inserted += 1
            except Exception:
                pass  # duplicate

        total_inserted += batch_inserted
        days_done = (batch_end - MODEL_START_DATES[model_key]).days
        days_total = (end_date - MODEL_START_DATES[model_key]).days
        pct = int(days_done / max(days_total, 1) * 100)
        logger.info(f"  {batch_inserted} rows inserted ({pct}% complete, {total_inserted} total)")

        current = batch_end + timedelta(days=1)
        time.sleep(1)  # rate limiting

    con.close()
    logger.info(f"Backfill complete: {total_inserted} rows for {model_key}")
    return total_inserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["gfs", "ecmwf"])
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    backfill_model(args.model, args.start, args.end, args.db)
```

**Step 2: Write integration test**

```python
"""Test Open-Meteo backfill with a small date range."""
import os
import duckdb
import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime
from scripts.backfill_openmeteo import fetch_batch, backfill_model
from core.db import init_db

TEST_DB = "data/test_openmeteo.duckdb"

@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _mock_response(times, temps):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "hourly": {"time": times, "temperature_2m": temps}
    }
    return resp


@patch("scripts.backfill_openmeteo.requests.get")
def test_backfill_gfs_inserts_rows(mock_get):
    """Backfill should insert hourly forecast rows with model_name='gfs'."""
    times = [f"2024-01-01T{h:02d}:00" for h in range(24)]
    temps = [float(30 + h) for h in range(24)]
    mock_get.return_value = _mock_response(times, temps)

    count = backfill_model("gfs", date(2024, 1, 1), date(2024, 1, 1), TEST_DB)
    assert count == 24

    con = duckdb.connect(TEST_DB, read_only=True)
    rows = con.execute("SELECT COUNT(*) FROM forecasts WHERE model_name = 'gfs'").fetchone()[0]
    assert rows == 24
    # model_run should be midnight
    mr = con.execute("SELECT DISTINCT model_run FROM forecasts").fetchone()[0]
    assert mr.hour == 0
    con.close()
```

**Step 3: Run tests**

```bash
pytest tests/test_openmeteo_backfill.py -v
```

**Step 4: Commit**

```bash
git add scripts/backfill_openmeteo.py tests/test_openmeteo_backfill.py
git commit -m "feat: Open-Meteo backfill script for GFS and ECMWF forecasts"
```

---

## Task 6: Parameterize Model Functions by `model_name`

**Files:**
- Modify: `services/backtester.py` — model factories accept model_name
- Test: `tests/test_phase2_models.py`, `tests/test_multimodel_safety.py`

**What changes:**

The key insight: GFS/ECMWF data uses `model_run = midnight UTC`, so `EXTRACT(HOUR FROM model_run) = 0` always. The existing per-run-hour query structure works if we pass `run_hour=0` for GFS/ECMWF.

**Step 1: Update `_make_regression_model` factory (line ~462)**

Add `model_name='hrrr'` parameter. Pass it through to `_walk_forward_regression_data` and `_walk_forward_bias_query`. The factory already accepts `features` — just add `model_name`:

```python
def _make_regression_model(features, label=None, model_name='hrrr'):
    """Factory for walk-forward regression models."""
    # ... existing code ...
    def model_fn(provider, ref_time):
        # ... existing code, but pass model_name to query functions ...
        training = _walk_forward_regression_data(con, run_hour, station_id, obs_date, model_name=model_name)
        # ...
    model_fn.__name__ = label or f"wf_regression_{'_'.join(features)}"
    return model_fn
```

For GFS/ECMWF: `run_hour` is extracted from `provider.model_run`. Since GFS model_run is midnight, run_hour=0. The query `EXTRACT(HOUR FROM f.model_run) = 0 AND f.model_name = 'gfs'` correctly returns only GFS data.

**Step 2: Update `_make_phase2b_model` factory (line ~767)**

Same pattern — add `model_name='hrrr'` parameter, pass to `_ensure_level1`.

**Step 3: Create named model instances for GFS/ECMWF**

After the existing HRRR model instances (lines ~504-507, ~858-863):

```python
# GFS models
wf_regression_full_gfs = _make_regression_model(_FEATURES_FULL, "wf_regression_full_gfs", model_name="gfs")
wf_regression_fcst_gfs = _make_regression_model(_FEATURES_FCST, "wf_regression_fcst_gfs", model_name="gfs")
# ... etc for each ablation variant ...

# ECMWF models
wf_regression_full_ecmwf = _make_regression_model(_FEATURES_FULL, "wf_regression_full_ecmwf", model_name="ecmwf")
# ... etc ...
```

**Step 4: Run tests**

```bash
pytest tests/test_phase2_models.py -v
pytest tests/test_multimodel_safety.py -v
```

**Step 5: Commit**

```bash
git add services/backtester.py
git commit -m "feat: parameterize model functions by model_name for multi-model support"
```

---

## Task 7: Ensemble Service

**Files:**
- Create: `services/ensemble.py`
- Test: `tests/test_ensemble.py`

**Step 1: Write failing tests**

```python
"""Test Gaussian mixture ensemble combination."""
import pytest
from services.ensemble import combine_mixture_brackets


def test_equal_weight_two_identical_models():
    """Two identical Gaussians should produce same brackets as single."""
    predictions = [(70.0, 3.0), (70.0, 3.0)]  # (mu, sigma)
    weights = [0.5, 0.5]
    brackets = combine_mixture_brackets(predictions, weights)
    # Should be centered around 70
    assert max(brackets, key=brackets.get) == 70
    # Should sum to ~1.0
    assert abs(sum(brackets.values()) - 1.0) < 0.01


def test_mixture_wider_than_components():
    """Two models disagreeing should produce wider distribution."""
    single = combine_mixture_brackets([(70.0, 2.5)], [1.0])
    mixture = combine_mixture_brackets([(68.0, 2.5), (72.0, 2.5)], [0.5, 0.5])
    # Mixture should have more probability spread across more brackets
    assert len(mixture) >= len(single)


def test_three_model_equal_weights():
    """HRRR + GFS + ECMWF equal weight ensemble."""
    predictions = [(70.0, 2.5), (68.0, 3.0), (71.0, 2.8)]
    weights = [1/3, 1/3, 1/3]
    brackets = combine_mixture_brackets(predictions, weights)
    assert abs(sum(brackets.values()) - 1.0) < 0.01
    # Peak should be between 68 and 71
    peak = max(brackets, key=brackets.get)
    assert 68 <= peak <= 71
```

**Step 2: Implement `services/ensemble.py`**

```python
"""Ensemble combination — Gaussian mixture over multiple model predictions.

Combines bias-corrected Gaussian distributions from multiple weather models
into bracket probabilities via a weighted mixture distribution.
"""
from typing import Dict, List, Optional, Tuple

from scipy.stats import norm


def combine_mixture_brackets(
    predictions: List[Tuple[float, float]],
    weights: List[float],
    radius: int = 15,
) -> Dict[int, float]:
    """Compute bracket probabilities from a weighted Gaussian mixture.

    Parameters
    ----------
    predictions : list of (mu, sigma)
        Each model's bias-corrected center and residual std.
    weights : list of float
        Per-model weights (must sum to ~1.0).
    radius : int
        Half-width of bracket range around the weighted mean center.

    Returns
    -------
    Dict mapping integer temperature to probability.
    """
    if not predictions:
        return {}

    # Weighted center for bracket range
    center = sum(w * mu for (mu, _), w in zip(predictions, weights))
    center_int = round(center)

    probs = {}  # type: Dict[int, float]
    for k in range(center_int - radius, center_int + radius + 1):
        p = 0.0
        for (mu, sigma), w in zip(predictions, weights):
            if sigma <= 0:
                continue
            p += w * (norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma))
        if p > 0.0001:
            probs[k] = round(p, 4)

    # Renormalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}
    return probs
```

**Step 3: Create `make_ensemble_model_fn` — wraps multiple ModelFns**

```python
from services.backtester import ModelFn
from services.data_provider import BacktestDataProvider


def make_ensemble_model_fn(
    model_configs: List[dict],
    weights: Optional[List[float]] = None,
) -> ModelFn:
    """Create an ensemble ModelFn from multiple model configurations.

    Parameters
    ----------
    model_configs : list of dict
        Each dict has:
          - 'model_fn': a ModelFn callable
          - 'model_name': str (for BacktestDataProvider)
          - 'run_hour': int or None (None = use provider's run_hour)
    weights : list of float or None
        Per-model weights. None = equal weights.

    Returns
    -------
    A ModelFn that returns ensemble bracket probabilities.
    """
    n = len(model_configs)
    if weights is None:
        weights = [1.0 / n] * n

    def ensemble_fn(provider, ref_time):
        # type: (BacktestDataProvider, datetime) -> Optional[Dict[int, float]]
        predictions = []
        active_weights = []

        for cfg, w in zip(model_configs, weights):
            # Create a provider scoped to this model
            model_provider = BacktestDataProvider(
                db_path=provider.db_path,
                station_id=provider.station_id,
                model_run=provider.model_run if cfg.get('run_hour') is None
                    else provider.model_run.replace(hour=0, minute=0),
                ref_time=provider.ref_time,
                connection=provider._shared_con,
                model_name=cfg['model_name'],
            )
            result = cfg['model_fn'](model_provider, ref_time)
            if result is not None:
                # Extract (mu, sigma) from bracket probs
                # Actually, we need the model to return (mu, sigma) not brackets
                # This requires a different interface...
                predictions.append(result)
                active_weights.append(w)

        if not predictions:
            return None

        # Renormalize weights for active models
        total_w = sum(active_weights)
        active_weights = [w / total_w for w in active_weights]

        return combine_mixture_brackets(predictions, active_weights)

    ensemble_fn.__name__ = "ensemble_equal"
    return ensemble_fn
```

**NOTE:** The ensemble needs (mu, sigma) from each model, not bracket probabilities. This requires the model functions to expose their intermediate (center, std) values. Two options:

1. **Add a return mode** to model functions that returns (mu, sigma) instead of brackets
2. **Create a wrapper** that intercepts the Gaussian parameters before bracket conversion

Option 2 is cleaner — create `_make_regression_model_raw` that returns (mu, sigma) tuple, then the ensemble converts to brackets. The existing bracket-returning version wraps this.

This detail will be refined during implementation, but the core `combine_mixture_brackets` function is correct.

**Step 4: Run tests**

```bash
pytest tests/test_ensemble.py -v
```

**Step 5: Commit**

```bash
git add services/ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble service with Gaussian mixture combination"
```

---

## Task 8: Run Open-Meteo Backfill

**Not a code task — operational step.**

```bash
# Backfill GFS to temp DB (uses our HRRR data date range)
python scripts/backfill_openmeteo.py gfs --start 2021-03-23

# Backfill ECMWF to temp DB
python scripts/backfill_openmeteo.py ecmwf --start 2021-03-23

# Merge into main DB (update backfill_merge.py first to include new temp DBs)
python scripts/backfill_merge.py
```

**Verification:** Spot-check 10 random days:
```sql
SELECT model_name, COUNT(*), MIN(valid_at::DATE), MAX(valid_at::DATE)
FROM forecasts GROUP BY model_name;
```

**Commit after merge:**
```bash
git commit -m "chore: backfill GFS and ECMWF forecasts from Open-Meteo"
```

---

## Task 9: Per-Model Phase 1 Analysis

**Files:**
- Create: `scripts/phase3_analysis.py`

**What it does:**
1. Run walk-forward bias on GFS (run_hour=0) and ECMWF (run_hour=0) independently
2. Report mean bias, std, Brier for each model
3. Compute error correlation matrix (HRRR vs GFS, HRRR vs ECMWF, GFS vs ECMWF)
4. **Gate check:** At least one error correlation < 0.7

Uses existing `Backtester.run()` with model_name-parameterized model functions.

**Commit:**
```bash
git add scripts/phase3_analysis.py
git commit -m "feat: Phase 3 per-model analysis script with error correlation"
```

---

## Task 10: Per-Model Phase 2 Regression + Ablation

**Files:**
- Modify: `scripts/phase3_analysis.py` (extend)

Run full ablation per model:
- `wf_regression_full_{model}` (fcst + month + delta)
- `wf_regression_fcst_{model}` (fcst only)
- `wf_regression_month_{model}` (month only)
- `wf_regression_delta_{model}` (delta only)

Compare each model's best regression vs its own Phase 1 baseline.
Select champion regression per model.

**Gate:** Each model's regression beats its own Phase 1.

---

## Task 11: Equal-Weight Ensemble Evaluation

**Files:**
- Modify: `scripts/phase3_analysis.py` (extend)

Backtest the 1/3-weighted ensemble at each HRRR run hour:
- At 00z: HRRR-00z + GFS-daily + ECMWF-daily
- At 06z: HRRR-06z + GFS-daily + ECMWF-daily
- At 12z: HRRR-12z + GFS-daily + ECMWF-daily
- At 18z: HRRR-18z + GFS-daily + ECMWF-daily

Report: ensemble Brier vs best single-model Brier, per run hour and overall.

**GATE (Phase 3 kill condition):** Ensemble Brier < best single-model Brier.

---

## Task 12: Per-Model Phase 2B (conditional — only if Task 11 gate passes)

Apply Phase 2B independently per model. Ensemble with Phase 2B active (14-18 ET).

**Gate:** Ensemble + Phase 2B beats ensemble without Phase 2B.

---

## Task 13: Learned Weights (conditional — only if Task 11 gate passes)

Test inverse-Brier weighting. If it doesn't beat equal weights, keep equal.

---

## Critical Files Reference

| File | Role |
|------|------|
| `core/db.py:27-36` | forecasts CREATE TABLE — add model_name |
| `core/db.py:169-170` | forecasts indexes |
| `services/backtester.py:226-257` | `_walk_forward_bias_query` — add model_name filter |
| `services/backtester.py:334-372` | `_walk_forward_regression_data` — add model_name filter |
| `services/backtester.py:577-685` | `_ensure_level1` — add model_name filter + cache key |
| `services/backtester.py:462-500` | `_make_regression_model` — add model_name param |
| `services/backtester.py:767-854` | `_make_phase2b_model` — add model_name param |
| `services/backtester.py:947` | `_has_forecast_data` — add model_name filter |
| `services/data_provider.py:151-211` | LiveDataProvider forecast queries — add 'hrrr' filter |
| `services/data_provider.py:290-340` | BacktestDataProvider forecast queries — add model_name filter |
| `services/forecast.py:53,125` | HRRR fetcher — tag inserts with 'hrrr' |
| `services/bias_model.py:48,63,72` | Bias calc — add 'hrrr' filter |
| `services/bias.py:45,59,100,171` | Drift signals — add 'hrrr' filter |
| `scripts/backfill_merge.py` | Dedup query — include model_name |
| `ui/web_dashboard.py` | 10+ dashboard queries — add 'hrrr' filter |

## Verification

After each task, run:
```bash
pytest tests/ -v  # all 189+ existing tests pass
```

After Task 8 (backfill), verify:
```bash
python -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
print(con.execute('SELECT model_name, COUNT(*), MIN(valid_at::DATE), MAX(valid_at::DATE) FROM forecasts GROUP BY model_name').fetchall())
con.close()
"
```

After Task 11 (ensemble evaluation), the analysis script prints the gate result:
```
ENSEMBLE BRIER: X.XXXX vs BEST SINGLE MODEL: X.XXXX
GATE: PASS/FAIL
```
