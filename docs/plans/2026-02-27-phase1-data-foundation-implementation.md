# Phase 1: Data Foundation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all data gaps (HRRR 24 hourly runs, GFS multi-run, ECMWF multi-run) and extend observations (KJFK, DSM) before any model work resumes.

**Architecture:** Temp-DB-per-backfill pattern (never long-write to main), Herbie for GRIB extraction, existing SynopticIngestor pattern for new observation sources. Schema migrations via `core/db.py` ALTER TABLE + catch pattern.

**Tech Stack:** Python 3.9, DuckDB, Herbie, pygrib, numpy, httpx, loguru

**Design doc:** `docs/plans/2026-02-27-rebuild-design.md`

**Prerequisite knowledge:**
- Read `docs/plans/2026-02-27-rebuild-design.md` for full context on WHY each task exists
- Read `CLAUDE.md` for coding standards
- Read `~/.claude/rules/domain-guardrails.md` for NWP data rules

---

## Task Dependencies

```
Task 1 (Schema) ──────┬── Task 2 (UCAR GFS) ← TIME-SENSITIVE, start immediately
                       ├── Task 3 (HRRR backfill script)
                       ├── Task 4 (ECMWF backfill)
                       ├── Task 5 (KJFK obs)
                       └── Task 6 (DSM ingestion)

Task 3 (HRRR script) ── Task 3b (EC2 setup + run) ── long-running, 2-5 days

All backfills complete ── Task 7 (Gap audit + gate validation)
```

Tasks 2–6 can run in parallel after Task 1 completes.

---

### Task 1: Schema Migrations

**Files:**
- Modify: `core/db.py`
- Modify: `core/constants.py`
- Test: `tests/test_db.py`

**Context:** The `forecasts` table needs two new columns (`fxx INTEGER`, `is_spinup BOOLEAN`). The `market_ticks` table needs a UNIQUE constraint. The `observations` table needs an `obs_type` column. KJFK needs to be added to `STATION_COORDS`. All migrations follow the existing pattern in `core/db.py`: `ALTER TABLE ... ADD COLUMN`, catch exception if already exists.

**Step 1: Write failing tests for new columns**

```python
# tests/test_db.py — add these tests

def test_forecasts_has_fxx_column(tmp_path):
    """forecasts table should have fxx INTEGER column."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path, read_only=True)
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'forecasts' AND column_name = 'fxx'"
    ).fetchall()
    con.close()
    assert len(cols) == 1


def test_forecasts_has_is_spinup_column(tmp_path):
    """forecasts table should have is_spinup BOOLEAN column."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path, read_only=True)
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'forecasts' AND column_name = 'is_spinup'"
    ).fetchall()
    con.close()
    assert len(cols) == 1


def test_observations_has_obs_type_column(tmp_path):
    """observations table should have obs_type VARCHAR column."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path, read_only=True)
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'observations' AND column_name = 'obs_type'"
    ).fetchall()
    con.close()
    assert len(cols) == 1


def test_market_ticks_unique_constraint(tmp_path):
    """market_ticks should reject duplicate (market_id, captured_at) pairs."""
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)
    now = datetime(2026, 1, 1, 12, 0, 0)
    row = ["MKT1", "NYC", now, 0.5, 0.6, 0.4, 0.5, 0.55, 100, 70.0, 72.0]
    con.execute(
        "INSERT INTO market_ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row
    )
    # Second insert should fail
    import pytest
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO market_ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row
        )
    con.close()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v -k "fxx or is_spinup or obs_type or market_ticks_unique"
```

Expected: 4 FAIL

**Step 3: Add migrations to `core/db.py`**

Add these migrations after the existing `_backfill_ingest_source(con)` call in `init_db()`:

```python
# Migration: add fxx and is_spinup to forecasts
for col, dtype in [("fxx", "INTEGER"), ("is_spinup", "BOOLEAN DEFAULT FALSE")]:
    try:
        con.execute(f"ALTER TABLE forecasts ADD COLUMN {col} {dtype}")
    except Exception:
        pass  # Column already exists

# Migration: add obs_type to observations (metar, dsm, micronet)
try:
    con.execute("ALTER TABLE observations ADD COLUMN obs_type VARCHAR DEFAULT 'metar'")
except Exception:
    pass

# Migration: add UNIQUE constraint to market_ticks
# DuckDB can't add constraints to existing tables, so rebuild
_migrate_market_ticks_unique(con)
```

Add the migration function:

```python
def _migrate_market_ticks_unique(con: duckdb.DuckDBPyConnection) -> None:
    """Add UNIQUE constraint to market_ticks on (market_id, captured_at).

    DuckDB can't add constraints to existing tables, so we rebuild.
    Only runs if the constraint is missing.
    """
    try:
        con.execute("DROP TABLE IF EXISTS _mt_migration_check")
        # Check if constraint exists by trying a probe
        has_unique = con.execute("""
            SELECT COUNT(*) FROM duckdb_constraints()
            WHERE table_name = 'market_ticks' AND constraint_type = 'UNIQUE'
        """).fetchone()[0]
        if has_unique > 0:
            return  # Already migrated

        logger.info("Migrating market_ticks: adding UNIQUE constraint")
        con.execute("DROP TABLE IF EXISTS market_ticks_new")

        # Dedup existing rows first (keep latest ingested)
        con.execute("""
            CREATE TABLE market_ticks_new (
                market_id    VARCHAR NOT NULL,
                city         VARCHAR NOT NULL,
                captured_at  TIMESTAMP NOT NULL,
                yes_bid      DOUBLE,
                yes_ask      DOUBLE,
                no_bid       DOUBLE,
                no_ask       DOUBLE,
                last_trade   DOUBLE,
                volume       INTEGER,
                floor_strike DOUBLE,
                cap_strike   DOUBLE,
                UNIQUE (market_id, captured_at)
            )
        """)
        con.execute("""
            INSERT INTO market_ticks_new
            SELECT DISTINCT ON (market_id, captured_at)
                market_id, city, captured_at, yes_bid, yes_ask,
                no_bid, no_ask, last_trade, volume, floor_strike, cap_strike
            FROM market_ticks
            ORDER BY market_id, captured_at
        """)

        old_count = con.execute("SELECT COUNT(*) FROM market_ticks").fetchone()[0]
        new_count = con.execute("SELECT COUNT(*) FROM market_ticks_new").fetchone()[0]
        dupes = old_count - new_count

        con.execute("DROP TABLE market_ticks")
        con.execute("ALTER TABLE market_ticks_new RENAME TO market_ticks")

        # Recreate indexes
        con.execute("CREATE INDEX IF NOT EXISTS idx_market_city_time ON market_ticks (city, captured_at)")

        logger.info(f"market_ticks migrated — UNIQUE added, {dupes} duplicates removed")
    except Exception:
        logger.warning("Failed to migrate market_ticks for UNIQUE constraint", exc_info=True)
```

**Step 4: Add KJFK to `core/constants.py`**

```python
# In CITIES dict, add KJFK to NYC neighbors
CITIES: dict = {
    "NYC": {"settlement": "KNYC", "neighbors": ["KLGA", "KEWR", "KJFK"]},
}

# In STATION_COORDS, add KJFK
STATION_COORDS: dict = {
    "KNYC": (40.7789, -73.9692),
    "KJFK": (40.6413, -73.7781),
}
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v -k "fxx or is_spinup or obs_type or market_ticks_unique"
```

Expected: 4 PASS

**Step 6: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: All existing tests still pass. The new columns have defaults so existing code won't break.

**Step 7: Commit**

```bash
git add core/db.py core/constants.py tests/test_db.py
git commit -m "feat: add fxx/is_spinup columns, market_ticks UNIQUE, KJFK station"
```

---

### Task 2: UCAR GFS 12z Backfill Script

**⚠️ TIME-SENSITIVE: UCAR archive shutting down early 2026. Start this immediately after Task 1.**

**Files:**
- Create: `scripts/backfill_gfs_ucar.py`
- Test: `tests/test_gfs_backfill.py`

**Context:** UCAR RDA dataset ds084.1 has GFS 0.25° back to Jan 2015. We need 12z (priority), then 06z, 18z. Herbie can access UCAR GFS data. The script follows the same temp-DB pattern as `scripts/backfill_openmeteo.py` but extracts from GRIB files instead of JSON APIs. Uses the existing `HRRRFetcher._extract_nearest()` pattern for grid point extraction.

**Step 1: Write failing tests**

```python
# tests/test_gfs_backfill.py

import duckdb
import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

from core.db import init_db


def test_backfill_gfs_creates_temp_db(tmp_path):
    """GFS backfill should create a temp DB and insert forecast rows."""
    db_path = str(tmp_path / "test_gfs.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path, read_only=True)
    # Verify forecasts table exists and accepts gfs model_name
    count = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE model_name = 'gfs'"
    ).fetchone()[0]
    con.close()
    assert count == 0  # Empty but table exists


def test_gfs_resume_point(tmp_path):
    """Resume should find the latest GFS date in the temp DB."""
    from scripts.backfill_gfs_ucar import get_resume_date

    db_path = str(tmp_path / "test_gfs.duckdb")
    init_db(db_path)

    # No data -> None
    assert get_resume_date(db_path, run_hour=12) is None

    # Insert a row
    con = duckdb.connect(db_path)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 12), datetime(2025, 6, 15, 13),
         75.0, 23.89, datetime(2026, 1, 1), "gfs", 1, False],
    )
    con.close()

    resume = get_resume_date(db_path, run_hour=12)
    assert resume == date(2025, 6, 16)


def test_gfs_extract_inserts_with_fxx(tmp_path):
    """Inserted GFS rows should have fxx and is_spinup populated."""
    db_path = str(tmp_path / "test_gfs.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 12), datetime(2025, 6, 15, 15),
         75.0, 23.89, datetime(2026, 1, 1), "gfs", 3, False],
    )
    row = con.execute(
        "SELECT fxx, is_spinup FROM forecasts WHERE model_name = 'gfs'"
    ).fetchone()
    con.close()
    assert row == (3, False)  # GFS doesn't have spin-up issues
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_gfs_backfill.py -v
```

Expected: 1 PASS (schema test), 2 FAIL (import errors — script doesn't exist yet)

**Step 3: Write the backfill script**

```python
# scripts/backfill_gfs_ucar.py
"""Backfill GFS 12z/06z/18z from UCAR RDA ds084.1 via Herbie.

Downloads GRIB files, extracts 2m temp at Central Park grid point,
writes to a temp DuckDB. Merge to main DB separately.

Usage:
    python scripts/backfill_gfs_ucar.py --run-hour 12
    python scripts/backfill_gfs_ucar.py --run-hour 12 --start 2023-01-01
"""

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS
from core.db import init_db

STATION_ID = "KNYC"
LAT, LON = STATION_COORDS[STATION_ID]

# GFS forecast hours to extract (short-range, relevant for day+1 high)
# 0.25° GFS at 12z: fxx 0-24 covers 12z today through 12z tomorrow
GFS_FXX_RANGE = range(0, 49)  # 0-48h covers 2 full days

DATA_START = date(2015, 1, 15)  # UCAR archive starts Jan 2015


def get_resume_date(db_path, run_hour):
    # type: (str, int) -> Optional[date]
    """Find the latest GFS date for this run_hour in the temp DB."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(model_run) FROM forecasts "
            "WHERE model_name = 'gfs' AND EXTRACT(HOUR FROM model_run) = ?",
            [run_hour],
        ).fetchone()
        con.close()
        if result and result[0]:
            latest = result[0]
            if isinstance(latest, datetime):
                return latest.date() + timedelta(days=1)
            return latest + timedelta(days=1)
    except Exception:
        pass
    return None


def extract_nearest(msg, lat, lon):
    # type: (object, float, float) -> float
    """Extract value at nearest grid point (same as HRRRFetcher)."""
    lat_grid, lon_grid = msg.latlons()
    cos_lat = np.cos(np.radians(lat))
    dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon) * cos_lat
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(msg.values[idx])


def backfill_gfs(
    run_hour=12,
    start_date=None,
    end_date=None,
    db_path=None,
    delay_seconds=1.0,
):
    # type: (int, Optional[date], Optional[date], Optional[str], float) -> int
    """Backfill GFS forecasts for one run hour from UCAR.

    Args:
        run_hour: UTC hour (0, 6, 12, 18)
        start_date: First date to fetch
        end_date: Last date to fetch
        db_path: Temp DB path
        delay_seconds: Delay between GRIB fetches

    Returns:
        Total rows inserted.
    """
    if run_hour not in (0, 6, 12, 18):
        raise ValueError("GFS run_hour must be 0, 6, 12, or 18")

    if db_path is None:
        db_path = "data/backfill_gfs_{:02d}z.duckdb".format(run_hour)

    if start_date is None:
        start_date = DATA_START
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    init_db(db_path)

    # Resume support
    resume = get_resume_date(db_path, run_hour)
    if resume and resume > start_date:
        logger.info("Resuming from {} (skipping already-ingested)", resume)
        start_date = resume

    if start_date > end_date:
        logger.info("Nothing to backfill")
        return 0

    con = duckdb.connect(db_path)
    total_inserted = 0
    total_days = (end_date - start_date).days + 1
    current = start_date

    logger.info(
        "Backfilling GFS {:02d}z from {} to {} ({} days) -> {}",
        run_hour, start_date, end_date, total_days, db_path,
    )

    day_num = 0
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    while current <= end_date:
        day_num += 1
        pct = int(day_num / total_days * 100)
        model_run = datetime(current.year, current.month, current.day, run_hour)
        day_inserted = 0

        for fxx in GFS_FXX_RANGE:
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="gfs",
                    product="pgrb2.0p25",
                    fxx=fxx,
                )
                grib_path = H.download("TMP:2 m above ground")
                grbs = pygrib.open(str(grib_path))
                msg = grbs.select(name="2 metre temperature")[0]
            except Exception as e:
                logger.debug(
                    "GFS {:02d}z {} fxx={}: not available — {}",
                    run_hour, current, fxx, e,
                )
                continue

            valid_at = model_run + timedelta(hours=fxx)

            try:
                temp_k = extract_nearest(msg, LAT, LON)
                temp_c = round(temp_k - 273.15, 2)
                temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'gfs', ?, FALSE)",
                    [STATION_ID, model_run, valid_at, temp_f, temp_c,
                     now_utc, fxx],
                )
                day_inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate
            except Exception as e:
                logger.warning("GFS extract failed fxx={}: {}", fxx, e)

            time.sleep(delay_seconds)

        total_inserted += day_inserted
        if day_inserted > 0:
            logger.info(
                "Day {}/{} ({:>3}%) — GFS {:02d}z {}: +{} rows (total: {})",
                day_num, total_days, pct, run_hour, current,
                day_inserted, total_inserted,
            )
        else:
            logger.debug(
                "Day {}/{} ({:>3}%) — GFS {:02d}z {}: archive gap",
                day_num, total_days, pct, run_hour, current,
            )

        current += timedelta(days=1)

    con.close()
    logger.info("GFS {:02d}z backfill complete: {} rows", run_hour, total_inserted)
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill GFS from UCAR RDA")
    parser.add_argument("--run-hour", type=int, default=12, choices=[0, 6, 12, 18])
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    backfill_gfs(
        run_hour=args.run_hour,
        start_date=args.start,
        end_date=args.end,
        db_path=args.db,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
pytest tests/test_gfs_backfill.py -v
```

Expected: 3 PASS

**Step 5: Smoke test with 2 days of data**

```bash
python scripts/backfill_gfs_ucar.py --run-hour 12 --start 2025-01-01 --end 2025-01-02 --db data/test_gfs.duckdb
```

Expected: GRIB downloads succeed, rows inserted. Then verify:
```bash
python -c "import duckdb; con=duckdb.connect('data/test_gfs.duckdb',read_only=True); print(con.sql('SELECT COUNT(*), MIN(model_run), MAX(model_run) FROM forecasts WHERE model_name=\\'gfs\\'').fetchone())"
```

**Step 6: Commit**

```bash
git add scripts/backfill_gfs_ucar.py tests/test_gfs_backfill.py
git commit -m "feat: add GFS UCAR backfill script with resume support"
```

**Step 7: Start the full 12z backfill (long-running)**

```bash
nohup python scripts/backfill_gfs_ucar.py --run-hour 12 --start 2015-01-15 > logs/gfs_12z_backfill.log 2>&1 &
```

Monitor: `tail -f logs/gfs_12z_backfill.log`

After 12z completes, run 06z and 18z:
```bash
nohup python scripts/backfill_gfs_ucar.py --run-hour 6 > logs/gfs_06z_backfill.log 2>&1 &
nohup python scripts/backfill_gfs_ucar.py --run-hour 18 > logs/gfs_18z_backfill.log 2>&1 &
```

---

### Task 3: HRRR 24-Run Backfill Script

**Files:**
- Create: `scripts/backfill_hrrr_full.py`
- Test: `tests/test_hrrr_full_backfill.py`

**Context:** The existing `forecast_backfiller.py` only handles one run_hour per invocation. We need a script that backfills ALL 24 hourly runs, distinguishes standard (fxx 1-18) from extended (fxx 1-48), populates `fxx` and `is_spinup`, and uses the temp-DB pattern. The heavy lifting (820K GRIB fetches) runs on an AWS EC2 instance in us-east-1.

**Step 1: Write failing tests**

```python
# tests/test_hrrr_full_backfill.py

import duckdb
import pytest
from datetime import datetime, date

from core.db import init_db


def test_extended_run_hours():
    """Extended runs (00z, 06z, 12z, 18z) should use fxx 1-48."""
    from scripts.backfill_hrrr_full import get_fxx_range
    assert get_fxx_range(0) == range(1, 49)
    assert get_fxx_range(6) == range(1, 49)
    assert get_fxx_range(12) == range(1, 49)
    assert get_fxx_range(18) == range(1, 49)


def test_standard_run_hours():
    """Standard runs should use fxx 1-18."""
    from scripts.backfill_hrrr_full import get_fxx_range
    assert get_fxx_range(1) == range(1, 19)
    assert get_fxx_range(7) == range(1, 19)
    assert get_fxx_range(23) == range(1, 19)


def test_spinup_flag():
    """fxx <= 3 should be flagged as spin-up."""
    from scripts.backfill_hrrr_full import is_spinup
    assert is_spinup(1) is True
    assert is_spinup(2) is True
    assert is_spinup(3) is True
    assert is_spinup(4) is False
    assert is_spinup(12) is False


def test_resume_per_run_hour(tmp_path):
    """Resume should be independent per run_hour."""
    from scripts.backfill_hrrr_full import get_resume_date

    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    con = duckdb.connect(db_path)

    # Insert run_hour=6 data
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2025, 6, 15, 6), datetime(2025, 6, 15, 7),
         75.0, 23.89, datetime(2026, 1, 1), "hrrr", 1, True],
    )
    con.close()

    # run_hour=6 should resume from 2025-06-16
    assert get_resume_date(db_path, run_hour=6) == date(2025, 6, 16)
    # run_hour=12 should have no resume point
    assert get_resume_date(db_path, run_hour=12) is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hrrr_full_backfill.py -v
```

Expected: 4 FAIL (import errors)

**Step 3: Write the backfill script**

```python
# scripts/backfill_hrrr_full.py
"""HRRR full 24-run backfill via AWS S3 / Herbie.

Backfills all 24 hourly HRRR runs. Extended runs (00z, 06z, 12z, 18z) fetch
fxx 1-48. Standard runs fetch fxx 1-18. Populates fxx and is_spinup columns.

Usage:
    python scripts/backfill_hrrr_full.py --run-hour 7
    python scripts/backfill_hrrr_full.py --run-hour 7 --start 2021-07-01
    python scripts/backfill_hrrr_full.py --all-hours --start 2021-07-01
"""

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import numpy as np
import pygrib
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS
from core.db import init_db

STATION_ID = "KNYC"
LAT, LON = STATION_COORDS[STATION_ID]

EXTENDED_HOURS = {0, 6, 12, 18}
DATA_START = date(2021, 6, 1)  # HRRR v4 archive on AWS starts ~Dec 2020


def get_fxx_range(run_hour):
    # type: (int) -> range
    """Return forecast hour range: extended (1-48) or standard (1-18)."""
    if run_hour in EXTENDED_HOURS:
        return range(1, 49)
    return range(1, 19)


def is_spinup(fxx):
    # type: (int) -> bool
    """True if forecast hour has spin-up artifacts (fxx <= 3)."""
    return fxx <= 3


def get_resume_date(db_path, run_hour):
    # type: (str, int) -> Optional[date]
    """Find latest HRRR date for this run_hour in the DB."""
    try:
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(
            "SELECT MAX(model_run) FROM forecasts "
            "WHERE model_name = 'hrrr' AND EXTRACT(HOUR FROM model_run) = ?",
            [run_hour],
        ).fetchone()
        con.close()
        if result and result[0]:
            latest = result[0]
            if isinstance(latest, datetime):
                return latest.date() + timedelta(days=1)
            return latest + timedelta(days=1)
    except Exception:
        pass
    return None


def extract_nearest(msg, lat, lon):
    # type: (object, float, float) -> float
    """Extract value at nearest grid point."""
    lat_grid, lon_grid = msg.latlons()
    cos_lat = np.cos(np.radians(lat))
    dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon) * cos_lat
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return float(msg.values[idx])


def backfill_hrrr_hour(
    run_hour,
    start_date=None,
    end_date=None,
    db_path=None,
    delay_seconds=0.5,
):
    # type: (int, Optional[date], Optional[date], Optional[str], float) -> int
    """Backfill one HRRR run hour across all dates.

    Returns total rows inserted.
    """
    if db_path is None:
        db_path = "data/backfill_hrrr_{:02d}z.duckdb".format(run_hour)

    if start_date is None:
        start_date = DATA_START
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    init_db(db_path)

    resume = get_resume_date(db_path, run_hour)
    if resume and resume > start_date:
        logger.info("Resuming {:02d}z from {}", run_hour, resume)
        start_date = resume

    if start_date > end_date:
        logger.info("Nothing to backfill for {:02d}z", run_hour)
        return 0

    con = duckdb.connect(db_path)
    fxx_range = get_fxx_range(run_hour)
    total_inserted = 0
    total_days = (end_date - start_date).days + 1
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    logger.info(
        "Backfilling HRRR {:02d}z fxx={}-{} from {} to {} ({} days) -> {}",
        run_hour, fxx_range.start, fxx_range.stop - 1,
        start_date, end_date, total_days, db_path,
    )

    current = start_date
    day_num = 0

    while current <= end_date:
        day_num += 1
        pct = int(day_num / total_days * 100)
        model_run = datetime(current.year, current.month, current.day, run_hour)
        day_inserted = 0
        consecutive_misses = 0

        for fxx in fxx_range:
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                )
                grib_path = H.download("TMP:2 m")
                grbs = pygrib.open(str(grib_path))
                msg = grbs.select(name="2 metre temperature")[0]
                consecutive_misses = 0
            except Exception:
                consecutive_misses += 1
                if consecutive_misses >= 3:
                    break  # Archive gap — stop trying this day
                continue

            valid_at = model_run + timedelta(hours=fxx)
            spinup = is_spinup(fxx)

            try:
                temp_k = extract_nearest(msg, LAT, LON)
                temp_c = round(temp_k - 273.15, 2)
                temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                con.execute(
                    "INSERT INTO forecasts "
                    "(station_id, model_run, valid_at, temp_f, temp_c, "
                    "ingested_at, model_name, fxx, is_spinup) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'hrrr', ?, ?)",
                    [STATION_ID, model_run, valid_at, temp_f, temp_c,
                     now_utc, fxx, spinup],
                )
                day_inserted += 1
            except duckdb.ConstraintException:
                pass
            except Exception as e:
                logger.warning("HRRR extract failed {:02d}z fxx={}: {}", run_hour, fxx, e)

            time.sleep(delay_seconds)

        total_inserted += day_inserted
        if day_num % 50 == 0 or day_inserted > 0:
            logger.info(
                "Day {}/{} ({:>3}%) — HRRR {:02d}z {}: +{} rows (total: {})",
                day_num, total_days, pct, run_hour, current,
                day_inserted, total_inserted,
            )

        current += timedelta(days=1)

    con.close()
    logger.info("HRRR {:02d}z backfill complete: {} rows", run_hour, total_inserted)
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill HRRR from AWS S3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-hour", type=int, choices=range(24),
                       metavar="0-23")
    group.add_argument("--all-hours", action="store_true",
                       help="Run all 24 hours sequentially")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s),
                        default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s),
                        default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.all_hours:
        total = 0
        for hour in range(24):
            total += backfill_hrrr_hour(
                run_hour=hour,
                start_date=args.start,
                end_date=args.end,
                db_path=args.db,
                delay_seconds=args.delay,
            )
        logger.info("All hours complete: {} total rows", total)
    else:
        backfill_hrrr_hour(
            run_hour=args.run_hour,
            start_date=args.start,
            end_date=args.end,
            db_path=args.db,
            delay_seconds=args.delay,
        )


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
pytest tests/test_hrrr_full_backfill.py -v
```

Expected: 4 PASS

**Step 5: Smoke test locally (1 day, 1 hour)**

```bash
python scripts/backfill_hrrr_full.py --run-hour 7 --start 2025-01-01 --end 2025-01-01 --db data/test_hrrr.duckdb
```

**Step 6: Commit**

```bash
git add scripts/backfill_hrrr_full.py tests/test_hrrr_full_backfill.py
git commit -m "feat: add HRRR 24-run backfill script with fxx/spinup tracking"
```

**Step 7: EC2 deployment (operational — not code)**

1. Launch EC2 `t3.large` in us-east-1 (co-located with S3 bucket)
2. Install Python 3.9, pip install herbie pygrib duckdb numpy loguru
3. Clone repo, copy .env
4. Run all 24 hours in parallel using tmux/screen:
   ```bash
   for hour in $(seq 0 23); do
     tmux new-session -d -s "hrrr_${hour}" \
       "python scripts/backfill_hrrr_full.py --run-hour $hour --delay 0.3 2>&1 | tee logs/hrrr_${hour}z.log"
   done
   ```
5. Monitor: `for h in $(seq 0 23); do echo -n "$h: "; tail -1 logs/hrrr_${h}z.log; done`

---

### Task 4: ECMWF Backfill Script

**Files:**
- Create: `scripts/backfill_ecmwf.py`
- Test: `tests/test_ecmwf_backfill.py`

**Context:** ECMWF opened all data free (CC-BY-4.0) since Oct 2025. AWS archive from Jan 2023. Herbie supports ECMWF open data. We need 12z (priority) and 00z. 06z/18z are short-cutoff (90h) — lower priority. Same temp-DB + Herbie pattern.

**Step 1: Write failing tests**

```python
# tests/test_ecmwf_backfill.py

from datetime import date, datetime
import duckdb
from core.db import init_db


def test_ecmwf_run_config():
    """ECMWF run hours should have correct horizon."""
    from scripts.backfill_ecmwf import get_fxx_range
    # Full runs: 240h
    assert get_fxx_range(0) == range(0, 73)   # 0-72h (3-day coverage, 3h steps)
    assert get_fxx_range(12) == range(0, 73)
    # Short-cutoff: 90h
    assert get_fxx_range(6) == range(0, 31)   # 0-30h
    assert get_fxx_range(18) == range(0, 31)


def test_ecmwf_resume(tmp_path):
    """Resume should work per run_hour."""
    from scripts.backfill_ecmwf import get_resume_date
    db_path = str(tmp_path / "test.duckdb")
    init_db(db_path)
    assert get_resume_date(db_path, 12) is None

    con = duckdb.connect(db_path)
    con.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KNYC", datetime(2024, 3, 1, 12), datetime(2024, 3, 1, 15),
         65.0, 18.33, datetime(2026, 1, 1), "ecmwf", 3, False],
    )
    con.close()
    assert get_resume_date(db_path, 12) == date(2024, 3, 2)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ecmwf_backfill.py -v
```

**Step 3: Write the script**

The script follows the identical pattern as `backfill_gfs_ucar.py` with these differences:
- `model="ecmwf"` in Herbie
- Product: `"oper"` or similar Herbie identifier for ECMWF open data
- ECMWF GRIB uses 3-hourly steps for longer horizons
- `DATA_START = date(2023, 1, 1)` (AWS archive)
- fxx range is 0-72 for full runs (3h steps = 24 fxx values), 0-30 for short-cutoff
- Model name: `"ecmwf"`
- No spin-up flag (ECMWF doesn't have the HRRR spin-up issue)

**Note:** The exact Herbie model/product strings for ECMWF open data should be verified at implementation time:
```python
H = Herbie("2024-01-01 12:00", model="ecmwf", product="oper", fxx=6)
```

**Step 4: Run tests, smoke test, commit**

```bash
pytest tests/test_ecmwf_backfill.py -v
python scripts/backfill_ecmwf.py --run-hour 12 --start 2024-01-01 --end 2024-01-02 --db data/test_ecmwf.duckdb
git add scripts/backfill_ecmwf.py tests/test_ecmwf_backfill.py
git commit -m "feat: add ECMWF open data backfill script"
```

---

### Task 5: KJFK Observation Ingestion

**Files:**
- Modify: `core/constants.py` (done in Task 1)
- Modify: `services/ingestor.py` (KJFK is auto-included via `get_all_station_ids()`)
- Create: `scripts/backfill_kjfk.py`
- Test: `tests/test_kjfk_backfill.py`

**Context:** KJFK was added to `CITIES["NYC"]["neighbors"]` in Task 1. The `SynopticIngestor` already uses `get_all_station_ids()` to poll all configured stations. So live ingestion of KJFK happens automatically after Task 1.

For historical data, we need an IEM ASOS backfill for KJFK. The existing `scripts/backfill_iem.py` already does this for KNYC/KLGA/KEWR — we just need to add KJFK.

**Step 1: Check if existing backfill script handles KJFK**

Read `scripts/backfill_iem.py` to see if it uses `get_all_station_ids()` or has hardcoded stations.

**Step 2: If hardcoded, add KJFK. If dynamic, just run it.**

The `SynopticIngestor` uses `get_all_station_ids()` which now includes KJFK. If the IEM backfill script also uses it, no code change needed — just run:

```bash
python scripts/backfill_iem.py KJFK --days 2000
```

If it has hardcoded stations, update it to use `get_all_station_ids()` or add KJFK explicitly.

**Step 3: Verify KJFK data**

```bash
python -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
print(con.sql(\"SELECT COUNT(*), MIN(observed_at), MAX(observed_at) FROM observations WHERE station_id='KJFK'\").fetchone())
"
```

Expected: >500K rows, spanning 2019+ (similar to KLGA/KEWR)

**Step 4: Commit (if code changed)**

```bash
git add scripts/backfill_iem.py
git commit -m "feat: add KJFK to IEM historical backfill"
```

---

### Task 6: DSM Ingestion

**Files:**
- Modify: `services/nws_fetcher.py`
- Test: `tests/test_nws_fetcher.py`

**Context:** The Daily Summary Message (DSM) is issued by NWS offices and contains the official daily max/min temperature. It arrives in the afternoon (typically 4-5 PM ET) on the settlement day — before the CLI final report (~1:30 AM ET next day). Ingesting DSM provides an early settlement proxy.

DSM text products are available at `https://api.weather.gov/products/types/DSM`.

**Step 1: Write failing test**

```python
# Add to tests/test_nws_fetcher.py

def test_parse_dsm_product():
    """DSM parser should extract max/min temps."""
    from services.nws_fetcher import parse_dsm_product

    # DSM format: station-specific, contains lines like:
    # NYC  27/ 15/...  (max/min in F)
    sample_dsm = {
        "productText": "DSMNYC\n...\nNYC  85/ 62/ ...\n...",
        "issuanceTime": "2026-02-27T22:00:00+00:00",
        "issuingOffice": "KOKX",
    }
    rows = parse_dsm_product(sample_dsm)
    assert len(rows) >= 1
    assert rows[0]["max_temp_f"] == 85.0
    assert rows[0]["min_temp_f"] == 62.0
    assert rows[0]["source"] == "DSM"
```

**Step 2: Research exact DSM format**

Before implementing, fetch a real DSM product to understand the exact format:

```python
import httpx
resp = httpx.get("https://api.weather.gov/products/types/DSM",
                 headers={"User-Agent": "(alphatemp)"})
products = resp.json().get("@graph", [])
# Find a KOKX DSM and examine its productText
```

**Step 3: Implement `parse_dsm_product()` and add DSM polling**

Follow the same pattern as `poll_cli()` — list DSM products, filter by issuing office, parse, upsert. DSM should NOT overwrite CLI (CLI is settlement truth), but should overwrite ACIS. Source hierarchy: CLI > DSM > ACIS.

**Step 4: Update `_upsert_row()` to respect source hierarchy**

```python
def _upsert_row(self, row: dict) -> bool:
    """Upsert with source hierarchy: CLI > DSM > ACIS."""
    con = get_connection(self.db_path)
    existing = con.execute(
        "SELECT source FROM nws_daily WHERE station_id = ? AND obs_date = ?",
        [row["station_id"], row["obs_date"]],
    ).fetchone()

    # Don't overwrite higher-priority sources
    PRIORITY = {"NWS_CLI": 3, "DSM": 2, "ACIS": 1}
    if existing:
        existing_priority = PRIORITY.get(existing[0], 0)
        new_priority = PRIORITY.get(row["source"], 0)
        if new_priority <= existing_priority:
            con.close()
            return False  # Don't overwrite

    con.execute("DELETE FROM nws_daily WHERE station_id = ? AND obs_date = ?",
                [row["station_id"], row["obs_date"]])
    con.execute(
        "INSERT INTO nws_daily (...) VALUES (...)",
        [...]
    )
    con.close()
    return True
```

**Step 5: Run tests, commit**

```bash
pytest tests/test_nws_fetcher.py -v
git add services/nws_fetcher.py tests/test_nws_fetcher.py
git commit -m "feat: add DSM ingestion with source hierarchy"
```

---

### Task 7: Backfill Merge Script

**Files:**
- Create: `scripts/merge_backfill.py`
- Test: `tests/test_merge_backfill.py`

**Context:** All backfills write to temp DBs. We need a merge script that safely copies rows from temp DB to main DB. Existing approach was manual — now we formalize it.

**Step 1: Write the merge script**

```python
# scripts/merge_backfill.py
"""Merge rows from a temp backfill DB into the main DB.

Usage:
    python scripts/merge_backfill.py data/backfill_gfs_12z.duckdb
    python scripts/merge_backfill.py data/backfill_hrrr_07z.duckdb --table forecasts
"""

import argparse
import duckdb
from loguru import logger


def merge_table(main_db, temp_db, table="forecasts"):
    # type: (str, str, str) -> int
    """Copy rows from temp DB to main DB, skipping duplicates.

    Returns number of rows inserted.
    """
    main_con = duckdb.connect(main_db)

    # Attach temp DB
    main_con.execute(f"ATTACH '{temp_db}' AS temp_db (READ_ONLY)")

    before = main_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # Insert with duplicate skip
    main_con.execute(f"""
        INSERT OR IGNORE INTO {table}
        SELECT * FROM temp_db.{table}
    """)

    after = main_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    inserted = after - before

    main_con.execute("DETACH temp_db")
    main_con.close()

    logger.info(f"Merged {inserted} rows into {table} from {temp_db}")
    return inserted
```

**Note:** DuckDB's `INSERT OR IGNORE` behavior with UNIQUE constraints needs verification at implementation time. If not supported, use the `INSERT ... WHERE NOT EXISTS` pattern:

```sql
INSERT INTO main.forecasts
SELECT * FROM temp_db.forecasts t
WHERE NOT EXISTS (
    SELECT 1 FROM main.forecasts m
    WHERE m.station_id = t.station_id
      AND m.model_run = t.model_run
      AND m.valid_at = t.valid_at
      AND m.model_name = t.model_name
)
```

**Step 2: Test, commit**

```bash
pytest tests/test_merge_backfill.py -v
git add scripts/merge_backfill.py tests/test_merge_backfill.py
git commit -m "feat: add backfill merge script"
```

---

### Task 8: Backfill Existing HRRR Data (fxx + is_spinup)

**Files:**
- Create: `scripts/migrate_hrrr_fxx.py`

**Context:** The existing ~140K HRRR forecast rows in the main DB don't have `fxx` or `is_spinup` populated. We need to compute these from `model_run` and `valid_at`:

```python
fxx = (valid_at - model_run).total_seconds() / 3600
is_spinup = fxx <= 3
```

**Step 1: Write the migration**

```python
# scripts/migrate_hrrr_fxx.py
"""Backfill fxx and is_spinup for existing HRRR forecast rows."""

import duckdb
from loguru import logger
from core.db import get_connection


def migrate():
    con = get_connection()

    # Count rows needing migration
    need_update = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE model_name = 'hrrr' AND fxx IS NULL"
    ).fetchone()[0]

    if need_update == 0:
        logger.info("All HRRR rows already have fxx populated")
        con.close()
        return

    logger.info(f"Backfilling fxx/is_spinup for {need_update} HRRR rows")

    con.execute("""
        UPDATE forecasts
        SET fxx = CAST(EXTRACT(EPOCH FROM (valid_at - model_run)) / 3600 AS INTEGER),
            is_spinup = (EXTRACT(EPOCH FROM (valid_at - model_run)) / 3600) <= 3
        WHERE model_name = 'hrrr' AND fxx IS NULL
    """)

    updated = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE model_name = 'hrrr' AND fxx IS NOT NULL"
    ).fetchone()[0]
    logger.info(f"Updated {updated} rows with fxx/is_spinup")

    con.close()


if __name__ == "__main__":
    migrate()
```

**Step 2: Run and verify**

```bash
python scripts/migrate_hrrr_fxx.py
python -c "
import duckdb
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
print('Spinup:', con.sql('SELECT COUNT(*) FROM forecasts WHERE is_spinup = TRUE').fetchone())
print('Non-spinup:', con.sql('SELECT COUNT(*) FROM forecasts WHERE is_spinup = FALSE').fetchone())
print('NULL fxx:', con.sql('SELECT COUNT(*) FROM forecasts WHERE fxx IS NULL').fetchone())
"
```

**Step 3: Commit**

```bash
git add scripts/migrate_hrrr_fxx.py
git commit -m "feat: backfill fxx/is_spinup for existing HRRR rows"
```

---

### Task 9: Gap Audit and Phase 1 Gate Validation

**Files:**
- Create: `scripts/phase1_gate_check.py`

**Context:** Before moving to Phase 2, validate that all data gaps are filled and no PiT violations exist.

**Step 1: Write the gate check script**

```python
# scripts/phase1_gate_check.py
"""Phase 1 gate validation — run after all backfills complete."""

import duckdb
from datetime import date
from loguru import logger

from core.db import get_connection


def check_gate():
    con = get_connection()
    all_pass = True

    # --- Check 1: HRRR run hour coverage ---
    print("\n=== HRRR Run Hour Coverage ===")
    hrrr_hours = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first_day,
               MAX(model_run::DATE) AS last_day
        FROM forecasts WHERE model_name = 'hrrr'
        GROUP BY run_hour ORDER BY run_hour
    """).fetchall()
    print(f"{'Hour':>4} | {'Days':>6} | {'First':>12} | {'Last':>12}")
    print("-" * 45)
    for row in hrrr_hours:
        print(f"{int(row[0]):4d} | {row[1]:6d} | {row[2]} | {row[3]}")
    if len(hrrr_hours) < 24:
        print(f"FAIL: Only {len(hrrr_hours)} of 24 run hours present")
        all_pass = False

    # --- Check 2: GFS run hour coverage ---
    print("\n=== GFS Run Hour Coverage ===")
    gfs_hours = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first_day,
               MAX(model_run::DATE) AS last_day
        FROM forecasts WHERE model_name = 'gfs'
        GROUP BY run_hour ORDER BY run_hour
    """).fetchall()
    for row in gfs_hours:
        print(f"{int(row[0]):4d}z | {row[1]:6d} days | {row[2]} to {row[3]}")
    if len(gfs_hours) < 4:
        print(f"FAIL: Only {len(gfs_hours)} of 4 GFS run hours")
        all_pass = False

    # --- Check 3: ECMWF run hour coverage ---
    print("\n=== ECMWF Run Hour Coverage ===")
    ecmwf_hours = con.execute("""
        SELECT EXTRACT(HOUR FROM model_run) AS run_hour,
               COUNT(DISTINCT model_run::DATE) AS days,
               MIN(model_run::DATE) AS first_day,
               MAX(model_run::DATE) AS last_day
        FROM forecasts WHERE model_name = 'ecmwf'
        GROUP BY run_hour ORDER BY run_hour
    """).fetchall()
    for row in ecmwf_hours:
        print(f"{int(row[0]):4d}z | {row[1]:6d} days | {row[2]} to {row[3]}")
    if len(ecmwf_hours) < 2:
        print(f"FAIL: Need at least 00z and 12z ECMWF")
        all_pass = False

    # --- Check 4: KJFK observations ---
    print("\n=== KJFK Observations ===")
    kjfk = con.execute("""
        SELECT COUNT(*), MIN(observed_at)::DATE, MAX(observed_at)::DATE
        FROM observations WHERE station_id = 'KJFK'
    """).fetchone()
    print(f"Rows: {kjfk[0]:,} | Range: {kjfk[1]} to {kjfk[2]}")
    if kjfk[0] == 0:
        print("FAIL: No KJFK observations")
        all_pass = False

    # --- Check 5: fxx/is_spinup populated ---
    print("\n=== fxx/is_spinup Population ===")
    null_fxx = con.execute(
        "SELECT COUNT(*) FROM forecasts WHERE fxx IS NULL"
    ).fetchone()[0]
    print(f"Rows with NULL fxx: {null_fxx}")
    if null_fxx > 0:
        print("FAIL: fxx not fully populated")
        all_pass = False

    # --- Check 6: market_ticks UNIQUE ---
    print("\n=== Market Ticks Dedup ===")
    dupe_check = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT (market_id, captured_at))
        FROM market_ticks
    """).fetchone()[0]
    print(f"Duplicate market_ticks rows: {dupe_check}")
    if dupe_check > 0:
        print("FAIL: market_ticks still has duplicates")
        all_pass = False

    # --- Check 7: Gap analysis (HRRR) ---
    print("\n=== HRRR Gap Analysis (sample: 12z) ===")
    gaps = con.execute("""
        WITH dates AS (
            SELECT UNNEST(generate_series(
                (SELECT MIN(model_run::DATE) FROM forecasts WHERE model_name='hrrr'),
                (SELECT MAX(model_run::DATE) FROM forecasts WHERE model_name='hrrr'),
                INTERVAL 1 DAY
            ))::DATE AS d
        ),
        present AS (
            SELECT DISTINCT model_run::DATE AS d
            FROM forecasts
            WHERE model_name = 'hrrr'
              AND EXTRACT(HOUR FROM model_run) = 12
        )
        SELECT COUNT(*) FROM dates WHERE d NOT IN (SELECT d FROM present)
    """).fetchone()[0]
    print(f"Missing 12z days: {gaps}")
    if gaps > 30:
        print(f"WARNING: {gaps} missing days for HRRR 12z")

    # --- Summary ---
    print("\n" + "=" * 50)
    if all_pass:
        print("PHASE 1 GATE: PASSED")
    else:
        print("PHASE 1 GATE: FAILED — fix issues above before Phase 2")

    con.close()
    return all_pass


if __name__ == "__main__":
    check_gate()
```

**Step 2: Run and evaluate**

```bash
python scripts/phase1_gate_check.py
```

**Step 3: Commit**

```bash
git add scripts/phase1_gate_check.py
git commit -m "feat: add Phase 1 gate validation script"
```

---

## Phases 2–4: Outlined (Detailed Plans After Phase 1 Gate)

These phases will receive their own implementation plans after Phase 1 passes its gate. The design doc (`2026-02-27-rebuild-design.md`) contains full specifications.

### Phase 2: Bias Correction Head-to-Head

- Rerun OLS baseline on 24-run HRRR (Candidate A)
- Implement EMOS (Candidate B)
- Implement XGBoost QR (Candidate C)
- Cross-hour training ablation
- Strategy backtester P&L tracking (diagnostic)
- Gate: Brier > 2% improvement

### Phase 3: Uncertainty and Calibration

- EMOS variance layer OR linear quantile regression
- Cross-hour training (Phase 3.7 fix)
- Piecewise-linear CDF with exponential tail decay
- Isotonic calibration pass
- Strategy backtester P&L (co-equal gate)
- Gate: Brier > 2% AND P&L trajectory improves

### Phase 4: Strategy and Execution

- Full trading window evaluation (10 AM ET D-1)
- Edge-agnostic displacement sweep
- Conditional analysis by hour/season/temp/bracket
- Fractional Kelly sizing
- Gate: Net P&L positive, bootstrap CI > 0

---

## NYC Micronet Investigation (Parallel Track — No Dependency)

**Action:** Email `mesonet@albany.edu` requesting access to the NYC Micronet data feed.
**Template:**

> Subject: Data Access Request — NYC Micronet for Temperature Research
>
> We are building a probabilistic temperature forecasting system for New York City
> and are interested in accessing the NYC Micronet 5-minute observation data to
> improve our real-time observation-based model adjustments. Could you provide
> information on data access options, formats, and any applicable terms?

**If access granted:** Create `services/micronet_ingestor.py` following the `SynopticIngestor` pattern, run a Phase 2B-style ablation test on divergence features.
