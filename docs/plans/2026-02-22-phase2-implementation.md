# Phase 2: Bias Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the HRRR forecast ingestion and real-time drift detection system that compares forecast curves against live observations and produces full drift reports per city.

**Architecture:** Two new services join the async event loop. `HRRRFetcher` polls HRRR model data via Herbie every 15 minutes. `BiasEngine` runs every 60 seconds, comparing observed temperature trajectories against HRRR forecast curves to produce drift reports with magnet proximity signals.

**Tech Stack:** Python 3.9+, DuckDB, Polars, Herbie, xarray, cfgrib, scipy, Loguru, asyncio

---

### Task 1: Install Dependencies

**Step 1: Install new packages**

```bash
source venv/bin/activate && pip install herbie-data xarray cfgrib scipy
```

**Step 2: Verify imports work**

```bash
source venv/bin/activate && python -c "from herbie import Herbie; import xarray; import cfgrib; import scipy; print('All imports OK')"
```

**Step 3: Commit** — No code changes, skip commit.

---

### Task 2: Add Station Coordinates and Forecast Polling Interval

**Files:**
- Modify: `core/constants.py:41-44`
- Modify: `tests/test_constants.py`

**Step 1: Write the failing tests**

Append to `tests/test_constants.py`:

```python
from core.constants import (
    generate_magnets,
    get_all_station_ids,
    MAGNETS,
    FLB_FAVORITE_FLOOR,
    FLB_LONGSHOT_CEILING,
    CITIES,
    POLL_INTERVAL_SECONDS,
    STATION_COORDS,
    FORECAST_POLL_INTERVAL_SECONDS,
)


def test_station_coords_exist_for_all_settlements():
    """Every settlement station must have coordinates."""
    for city_cfg in CITIES.values():
        stid = city_cfg["settlement"]
        assert stid in STATION_COORDS, f"Missing coords for {stid}"
        lat, lon = STATION_COORDS[stid]
        assert -90 <= lat <= 90, f"Invalid lat for {stid}"
        assert -180 <= lon <= 180, f"Invalid lon for {stid}"


def test_station_coords_only_settlements():
    """Coords should only be for settlement stations, not neighbors."""
    assert len(STATION_COORDS) == 5


def test_forecast_poll_interval():
    assert FORECAST_POLL_INTERVAL_SECONDS == 900
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_constants.py -v
```
Expected: FAIL — `ImportError: cannot import name 'STATION_COORDS'`

**Step 3: Write the implementation**

Add to `core/constants.py` after `POLL_INTERVAL_SECONDS` (after line 44):

```python
# Forecast polling interval (15 minutes — HRRR updates hourly, this catches trickle-in)
FORECAST_POLL_INTERVAL_SECONDS: int = 900

# Settlement station coordinates for HRRR grid point extraction
STATION_COORDS: dict = {
    "KNYC": (40.7789, -73.9692),
    "KPHL": (39.8721, -75.2411),
    "KMDW": (41.7868, -87.7522),
    "KMIA": (25.7959, -80.2870),
    "KLAX": (33.9425, -118.4081),
}
```

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_constants.py -v
```
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add core/constants.py tests/test_constants.py
git commit -m "feat: add station coordinates and forecast poll interval"
```

---

### Task 3: Update Database Schema

**Files:**
- Modify: `core/db.py:26-34` (replace forecasts table)
- Modify: `core/db.py` (add drift_signals table)
- Modify: `tests/test_db.py`

**Step 1: Write the failing tests**

Append to `tests/test_db.py`:

```python
def test_forecasts_schema_updated():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE forecasts").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["station_id", "model_run", "valid_at", "temp_f", "temp_c", "ingested_at"]:
        assert expected in col_names, f"Missing column: {expected}"
    con.close()


def test_forecasts_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
        VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 45.0, 7.2, CURRENT_TIMESTAMP)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO forecasts (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
            VALUES ('KNYC', '2026-02-22 12:00:00', '2026-02-22 13:00:00', 46.0, 7.8, CURRENT_TIMESTAMP)
        """)
    con.close()


def test_drift_signals_table_exists():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = {t[0] for t in tables}
    assert "drift_signals" in table_names
    con.close()


def test_drift_signals_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE drift_signals").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["city", "calculated_at", "model_run", "drift_score",
                     "slope_divergence", "forecast_trend", "magnet_proximity",
                     "magnet_distance", "confidence", "projected_high"]:
        assert expected in col_names, f"Missing column: {expected}"
    con.close()


def test_drift_signals_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO drift_signals (city, calculated_at, model_run, drift_score,
            slope_divergence, forecast_trend, magnet_proximity, magnet_distance,
            confidence, projected_high)
        VALUES ('NYC', '2026-02-22 12:00:00', '2026-02-22 06:00:00', 1.2,
            0.3, 0.5, 31, -0.3, 0.85, 31.3)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO drift_signals (city, calculated_at, model_run, drift_score,
                slope_divergence, forecast_trend, magnet_proximity, magnet_distance,
                confidence, projected_high)
            VALUES ('NYC', '2026-02-22 12:00:00', '2026-02-22 06:00:00', 2.0,
                0.1, 0.2, 32, 0.7, 0.90, 32.7)
        """)
    con.close()
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_db.py -v
```
Expected: FAIL — missing columns in forecasts, missing drift_signals table

**Step 3: Write the implementation**

Replace the forecasts table DDL in `core/db.py` (lines 26-34) with:

```python
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            station_id  VARCHAR NOT NULL,
            model_run   TIMESTAMP NOT NULL,
            valid_at    TIMESTAMP NOT NULL,
            temp_f      DOUBLE,
            temp_c      DOUBLE,
            ingested_at TIMESTAMP NOT NULL,
            UNIQUE (station_id, model_run, valid_at)
        )
    """)
```

Add the drift_signals table after the market_ticks DDL (after line 48):

```python
    con.execute("""
        CREATE TABLE IF NOT EXISTS drift_signals (
            city             VARCHAR NOT NULL,
            calculated_at    TIMESTAMP NOT NULL,
            model_run        TIMESTAMP NOT NULL,
            drift_score      DOUBLE,
            slope_divergence DOUBLE,
            forecast_trend   DOUBLE,
            magnet_proximity INTEGER,
            magnet_distance  DOUBLE,
            confidence       DOUBLE,
            projected_high   DOUBLE,
            UNIQUE (city, calculated_at, model_run)
        )
    """)
```

IMPORTANT: Since the existing `data/alphatemp.duckdb` already has the old forecasts table, you must DROP and recreate it. Add this line right before the forecasts CREATE:

```python
    con.execute("DROP TABLE IF EXISTS forecasts")
```

This is safe — the forecasts table is still a stub with no real data. Remove this DROP line after the migration.

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_db.py -v
```
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add core/db.py tests/test_db.py
git commit -m "feat: updated forecasts schema and new drift_signals table"
```

---

### Task 4: HRRRFetcher Service

**Files:**
- Create: `tests/test_forecast.py`
- Create: `services/forecast.py`

**Step 1: Write the failing tests**

```python
# tests/test_forecast.py
import os
import pytest
import duckdb
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from core.db import init_db
from services.forecast import HRRRFetcher, get_recent_model_runs

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_get_recent_model_runs():
    """Should return 3 model run datetimes, each 1 hour apart, offset by publication delay."""
    ref_time = datetime(2026, 2, 22, 15, 30, tzinfo=timezone.utc)
    runs = get_recent_model_runs(ref_time, count=3, delay_hours=2)
    assert len(runs) == 3
    # At 15:30 UTC with 2h delay, latest available run is 13z
    assert runs[0] == datetime(2026, 2, 22, 13, 0, tzinfo=timezone.utc)
    assert runs[1] == datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    assert runs[2] == datetime(2026, 2, 22, 11, 0, tzinfo=timezone.utc)


def test_get_recent_model_runs_crosses_midnight():
    """Model runs should cross midnight correctly."""
    ref_time = datetime(2026, 2, 22, 2, 30, tzinfo=timezone.utc)
    runs = get_recent_model_runs(ref_time, count=3, delay_hours=2)
    assert runs[0] == datetime(2026, 2, 22, 0, 0, tzinfo=timezone.utc)
    assert runs[1] == datetime(2026, 2, 21, 23, 0, tzinfo=timezone.utc)
    assert runs[2] == datetime(2026, 2, 21, 22, 0, tzinfo=timezone.utc)


def test_fetcher_stores_forecasts(test_db):
    """Mock Herbie to verify forecasts land in DuckDB."""
    import numpy as np
    import xarray as xr

    # Create a mock xarray dataset that Herbie.xarray() would return
    mock_ds = xr.Dataset(
        {"t2m": xr.DataArray(
            data=np.array([[280.0]]),  # 280K = 6.85°C = 44.33°F
            dims=["y", "x"],
            coords={"latitude": (["y", "x"], [[40.78]]),
                     "longitude": (["y", "x"], [[-73.97]])},
        )}
    )

    mock_herbie = MagicMock()
    mock_herbie.xarray.return_value = mock_ds

    with patch("services.forecast.Herbie", return_value=mock_herbie):
        fetcher = HRRRFetcher(db_path=test_db)
        model_run = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
        inserted = fetcher.fetch_run(model_run, fxx_range=range(1, 3))

    assert inserted > 0

    con = duckdb.connect(test_db)
    rows = con.execute("SELECT station_id, temp_f, temp_c FROM forecasts").fetchall()
    con.close()

    assert len(rows) > 0
    # Verify Kelvin conversion: 280K = 6.85°C
    for row in rows:
        assert row[2] is not None  # temp_c exists
        assert row[1] is not None  # temp_f exists


def test_fetcher_deduplicates(test_db):
    """Running fetch_run twice with same data should not create duplicates."""
    import numpy as np
    import xarray as xr

    mock_ds = xr.Dataset(
        {"t2m": xr.DataArray(
            data=np.array([[280.0]]),
            dims=["y", "x"],
            coords={"latitude": (["y", "x"], [[40.78]]),
                     "longitude": (["y", "x"], [[-73.97]])},
        )}
    )

    mock_herbie = MagicMock()
    mock_herbie.xarray.return_value = mock_ds

    with patch("services.forecast.Herbie", return_value=mock_herbie):
        fetcher = HRRRFetcher(db_path=test_db)
        model_run = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
        fetcher.fetch_run(model_run, fxx_range=range(1, 3))
        fetcher.fetch_run(model_run, fxx_range=range(1, 3))

    con = duckdb.connect(test_db)
    count = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    con.close()

    # 5 stations x 2 forecast hours = 10 rows, no duplicates
    assert count == 10
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_forecast.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'services.forecast'`

**Step 3: Write the implementation**

```python
# services/forecast.py
"""HRRR forecast fetcher via Herbie library."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

import duckdb
from herbie import Herbie
from loguru import logger

from core.constants import STATION_COORDS, FORECAST_POLL_INTERVAL_SECONDS
from core.db import get_connection


def get_recent_model_runs(
    ref_time: datetime, count: int = 3, delay_hours: int = 2
) -> List[datetime]:
    """Return the N most recent HRRR model run times, accounting for publication delay.

    HRRR runs every hour but takes ~90 min to fully publish.
    delay_hours provides a buffer to ensure data availability.
    """
    latest_hour = ref_time - timedelta(hours=delay_hours)
    latest_hour = latest_hour.replace(minute=0, second=0, microsecond=0)
    return [latest_hour - timedelta(hours=i) for i in range(count)]


class HRRRFetcher:
    """Fetches HRRR 2m temperature forecasts for settlement stations."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path
        self.stations = STATION_COORDS

    def fetch_run(self, model_run: datetime, fxx_range: range = range(1, 19)) -> int:
        """Fetch a single HRRR run for all stations. Returns rows inserted."""
        inserted = 0
        con = get_connection(self.db_path)
        now = datetime.now(timezone.utc)

        for fxx in fxx_range:
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                )
                ds = H.xarray("TMP:2 m")
            except Exception as e:
                logger.debug(f"HRRR fxx={fxx} not available for {model_run}: {e}")
                continue

            valid_at = model_run + timedelta(hours=fxx)

            for stid, (lat, lon) in self.stations.items():
                try:
                    point = ds["t2m"].sel(
                        y=lat, x=lon, method="nearest"
                    )
                    temp_k = float(point.values)
                    temp_c = round(temp_k - 273.15, 2)
                    temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                    con.execute(
                        """INSERT INTO forecasts
                           (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        [stid, model_run.replace(tzinfo=None), valid_at.replace(tzinfo=None),
                         temp_f, temp_c, now.replace(tzinfo=None)],
                    )
                    inserted += 1
                except duckdb.ConstraintException:
                    pass  # Duplicate
                except Exception as e:
                    logger.warning(f"Failed to extract point for {stid} fxx={fxx}: {e}")

        con.close()
        logger.info(f"HRRR {model_run.strftime('%Y-%m-%d %Hz')}: inserted {inserted} forecast points")
        return inserted

    async def fetch_latest(self) -> int:
        """Fetch the last 3 HRRR runs. Runs synchronous Herbie in executor."""
        runs = get_recent_model_runs(datetime.now(timezone.utc))
        total = 0
        loop = asyncio.get_event_loop()
        for run in runs:
            count = await loop.run_in_executor(None, self.fetch_run, run)
            total += count
        return total

    async def run(self) -> None:
        """Run the forecast fetcher loop indefinitely."""
        logger.info(f"Starting HRRR fetcher — polling every {FORECAST_POLL_INTERVAL_SECONDS}s")
        while True:
            try:
                await self.fetch_latest()
            except Exception as e:
                logger.error(f"HRRR fetch cycle failed: {e}")
            await asyncio.sleep(FORECAST_POLL_INTERVAL_SECONDS)
```

NOTE: Herbie's I/O is synchronous (GRIB downloads). We wrap `fetch_run` in `run_in_executor` to avoid blocking the async event loop. The `ds["t2m"].sel(y=lat, x=lon, method="nearest")` call depends on how Herbie names its coordinates — the implementer should verify this against a real Herbie download and adjust the coordinate dimension names if needed (could be `latitude`/`longitude` instead of `y`/`x`).

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_forecast.py -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add services/forecast.py tests/test_forecast.py
git commit -m "feat: HRRR forecast fetcher via Herbie"
```

---

### Task 5: BiasEngine — Drift Calculation

**Files:**
- Create: `tests/test_bias.py`
- Create: `services/bias.py`

**Step 1: Write the failing tests**

```python
# tests/test_bias.py
import os
import pytest
import duckdb
from datetime import datetime, timezone

from core.db import init_db, get_connection
from services.bias import BiasEngine, compute_drift, compute_slope_divergence

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def seeded_db(test_db):
    """Seed observations and forecasts for testing."""
    con = get_connection(test_db)

    # Seed 5 observations for KNYC over 5 hours
    # Observations show temps slightly above forecast
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO observations
               (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", f"2026-02-22 {hour}:00:00", 44.0 + i * 1.5 + 0.8,
             None, "", "2026-02-22 15:00:00"],
        )

    # Seed HRRR forecast curve (model run 06z)
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO forecasts
               (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", "2026-02-22 06:00:00", f"2026-02-22 {hour}:00:00",
             44.0 + i * 1.5, (44.0 + i * 1.5 - 32) * 5 / 9, "2026-02-22 07:00:00"],
        )

    # Seed second HRRR run (09z) with slightly higher temps
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO forecasts
               (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", "2026-02-22 09:00:00", f"2026-02-22 {hour}:00:00",
             44.0 + i * 1.5 + 0.3, (44.3 + i * 1.5 - 32) * 5 / 9, "2026-02-22 10:00:00"],
        )

    # Seed third HRRR run (12z) with even higher temps
    for i in range(5):
        hour = 10 + i
        con.execute(
            """INSERT INTO forecasts
               (station_id, model_run, valid_at, temp_f, temp_c, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ["KNYC", "2026-02-22 12:00:00", f"2026-02-22 {hour}:00:00",
             44.0 + i * 1.5 + 0.5, (44.5 + i * 1.5 - 32) * 5 / 9, "2026-02-22 13:00:00"],
        )

    con.close()
    return test_db


def test_compute_drift_positive_when_warmer():
    """Positive drift = observations warmer than forecast."""
    obs_temps = [45.0, 46.5, 48.0]
    fcst_temps = [44.0, 45.5, 47.0]
    drift = compute_drift(obs_temps, fcst_temps)
    assert drift == pytest.approx(1.0, abs=0.01)


def test_compute_drift_negative_when_cooler():
    obs_temps = [43.0, 44.5, 46.0]
    fcst_temps = [44.0, 45.5, 47.0]
    drift = compute_drift(obs_temps, fcst_temps)
    assert drift == pytest.approx(-1.0, abs=0.01)


def test_compute_drift_zero_when_matching():
    obs_temps = [44.0, 45.5, 47.0]
    fcst_temps = [44.0, 45.5, 47.0]
    drift = compute_drift(obs_temps, fcst_temps)
    assert drift == pytest.approx(0.0, abs=0.01)


def test_compute_slope_divergence_widening():
    """Positive slope = gap is widening (obs pulling further above forecast)."""
    # Divergences: 0.5, 1.0, 1.5 — linearly increasing
    obs_temps = [44.5, 46.5, 48.5]
    fcst_temps = [44.0, 45.5, 47.0]
    hours = [0.0, 1.0, 2.0]
    slope = compute_slope_divergence(obs_temps, fcst_temps, hours)
    assert slope > 0


def test_compute_slope_divergence_closing():
    """Negative slope = gap is closing."""
    # Divergences: 1.5, 1.0, 0.5 — linearly decreasing
    obs_temps = [45.5, 46.5, 47.5]
    fcst_temps = [44.0, 45.5, 47.0]
    hours = [0.0, 1.0, 2.0]
    slope = compute_slope_divergence(obs_temps, fcst_temps, hours)
    assert slope < 0


def test_bias_engine_produces_drift_report(seeded_db):
    """Full integration: seeded data should produce a drift signal for NYC."""
    engine = BiasEngine(db_path=seeded_db)
    reports = engine.calculate_all(
        ref_time=datetime(2026, 2, 22, 15, 0, tzinfo=timezone.utc)
    )

    assert "NYC" in reports
    report = reports["NYC"]
    assert report["drift_score"] > 0  # Obs are warmer
    assert report["projected_high"] is not None
    assert report["magnet_proximity"] is not None
    assert 0.0 <= report["confidence"] <= 1.0


def test_bias_engine_stores_signals(seeded_db):
    """Drift signals should be written to the drift_signals table."""
    engine = BiasEngine(db_path=seeded_db)
    engine.calculate_and_store(
        ref_time=datetime(2026, 2, 22, 15, 0, tzinfo=timezone.utc)
    )

    con = get_connection(seeded_db)
    count = con.execute("SELECT COUNT(*) FROM drift_signals").fetchone()[0]
    con.close()

    assert count > 0
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_bias.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'services.bias'`

**Step 3: Write the implementation**

```python
# services/bias.py
"""Bias Engine — real-time drift detection between observations and HRRR forecasts."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import duckdb
from loguru import logger
from scipy import stats

from core.constants import CITIES, MAGNETS, STATION_COORDS
from core.db import get_connection


def compute_drift(obs_temps: List[float], fcst_temps: List[float]) -> float:
    """Mean divergence between observed and forecast temperatures.

    Positive = observations warmer than forecast.
    """
    if not obs_temps or not fcst_temps or len(obs_temps) != len(fcst_temps):
        return 0.0
    diffs = [o - f for o, f in zip(obs_temps, fcst_temps)]
    return sum(diffs) / len(diffs)


def compute_slope_divergence(
    obs_temps: List[float], fcst_temps: List[float], hours: List[float]
) -> float:
    """Linear regression slope of the divergence time series.

    Positive slope = divergence is widening (obs pulling further above forecast).
    Units: degrees F per hour.
    """
    if len(obs_temps) < 2:
        return 0.0
    diffs = [o - f for o, f in zip(obs_temps, fcst_temps)]
    slope, _, _, _, _ = stats.linregress(hours, diffs)
    return float(slope)


def find_nearest_magnet(temp_f: float) -> tuple:
    """Find the nearest magnet number and distance from a temperature.

    Returns (magnet_int, distance) where distance is temp_f - magnet.
    """
    nearest = min(MAGNETS, key=lambda m: abs(temp_f - m))
    return nearest, round(temp_f - nearest, 1)


class BiasEngine:
    """Compares observed trajectories against HRRR forecast curves."""

    def __init__(self, db_path: str = "data/alphatemp.duckdb"):
        self.db_path = db_path

    def _get_forecast_curve(
        self, con: duckdb.DuckDBPyConnection, station_id: str, model_run: datetime
    ) -> List[tuple]:
        """Get forecast curve as [(valid_at, temp_f), ...] sorted by time."""
        rows = con.execute(
            """SELECT valid_at, temp_f FROM forecasts
               WHERE station_id = ? AND model_run = ?
               ORDER BY valid_at""",
            [model_run, model_run],
        ).fetchall()
        return rows

    def _get_observations_in_range(
        self, con: duckdb.DuckDBPyConnection, station_id: str,
        start: datetime, end: datetime
    ) -> List[tuple]:
        """Get observations as [(observed_at, temp_f), ...] sorted by time."""
        rows = con.execute(
            """SELECT observed_at, temp_f FROM observations
               WHERE station_id = ? AND observed_at BETWEEN ? AND ?
               ORDER BY observed_at""",
            [station_id, start, end],
        ).fetchall()
        return rows

    def _get_model_runs(self, con: duckdb.DuckDBPyConnection, station_id: str) -> List[datetime]:
        """Get the distinct model runs available for a station, most recent first."""
        rows = con.execute(
            """SELECT DISTINCT model_run FROM forecasts
               WHERE station_id = ?
               ORDER BY model_run DESC
               LIMIT 3""",
            [station_id],
        ).fetchall()
        return [r[0] for r in rows]

    def _interpolate_forecast_to_obs(
        self, forecast_curve: List[tuple], obs_times: List[datetime]
    ) -> List[Optional[float]]:
        """Linearly interpolate forecast curve to match observation timestamps.

        Returns list of interpolated forecast temps aligned with obs_times.
        """
        if len(forecast_curve) < 2:
            return [None] * len(obs_times)

        fc_times = [fc[0].timestamp() if hasattr(fc[0], 'timestamp') else fc[0] for fc in forecast_curve]
        fc_temps = [fc[1] for fc in forecast_curve]

        results = []
        for obs_t in obs_times:
            t = obs_t.timestamp() if hasattr(obs_t, 'timestamp') else obs_t
            # Clamp to forecast range
            if t <= fc_times[0]:
                results.append(fc_temps[0])
            elif t >= fc_times[-1]:
                results.append(fc_temps[-1])
            else:
                # Linear interpolation
                for i in range(len(fc_times) - 1):
                    if fc_times[i] <= t <= fc_times[i + 1]:
                        frac = (t - fc_times[i]) / (fc_times[i + 1] - fc_times[i])
                        interp = fc_temps[i] + frac * (fc_temps[i + 1] - fc_temps[i])
                        results.append(interp)
                        break
                else:
                    results.append(None)
        return results

    def _compute_forecast_trend(
        self, con: duckdb.DuckDBPyConnection, station_id: str,
        model_runs: List[datetime]
    ) -> float:
        """Compute how the forecasted daily high shifts across successive HRRR runs.

        Returns degrees F per run. Positive = successive runs forecasting higher.
        """
        if len(model_runs) < 2:
            return 0.0

        highs = []
        for run in model_runs:
            row = con.execute(
                """SELECT MAX(temp_f) FROM forecasts
                   WHERE station_id = ? AND model_run = ?""",
                [station_id, run],
            ).fetchone()
            if row and row[0] is not None:
                highs.append(row[0])

        if len(highs) < 2:
            return 0.0

        # Runs are ordered most recent first, so reverse for chronological
        highs.reverse()
        deltas = [highs[i + 1] - highs[i] for i in range(len(highs) - 1)]
        return sum(deltas) / len(deltas)

    def _compute_confidence(
        self, obs_count: int, drift_values: List[float]
    ) -> float:
        """Compute confidence score 0.0–1.0.

        Based on observation count and divergence stability.
        """
        # Observation count factor: more obs = more confidence, saturates at 20
        obs_factor = min(obs_count / 20.0, 1.0)

        # Stability factor: low variance in drift = high confidence
        if len(drift_values) < 2:
            stability_factor = 0.5
        else:
            mean_drift = sum(drift_values) / len(drift_values)
            variance = sum((d - mean_drift) ** 2 for d in drift_values) / len(drift_values)
            # Normalize: variance of 0 = 1.0, variance of 4+ = 0.0
            stability_factor = max(0.0, 1.0 - variance / 4.0)

        return round(obs_factor * 0.4 + stability_factor * 0.6, 3)

    def calculate_all(self, ref_time: Optional[datetime] = None) -> Dict[str, dict]:
        """Calculate drift reports for all cities. Returns dict of city -> report."""
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)

        con = get_connection(self.db_path)
        reports = {}

        for city_name, city_cfg in CITIES.items():
            stid = city_cfg["settlement"]

            model_runs = self._get_model_runs(con, stid)
            if not model_runs:
                logger.debug(f"No forecast data for {city_name}/{stid}, skipping")
                continue

            latest_run = model_runs[0]
            forecast_curve = self._get_forecast_curve(con, stid, latest_run)
            if not forecast_curve:
                continue

            # Get observations from the forecast's valid time range
            fc_start = forecast_curve[0][0]
            fc_end = forecast_curve[-1][0]
            observations = self._get_observations_in_range(con, stid, fc_start, fc_end)

            if not observations:
                logger.debug(f"No observations in forecast range for {city_name}")
                continue

            obs_times = [o[0] for o in observations]
            obs_temps = [o[1] for o in observations]

            # Interpolate forecast to observation timestamps
            fcst_interp = self._interpolate_forecast_to_obs(forecast_curve, obs_times)

            # Filter out None interpolations
            paired = [(o, f) for o, f in zip(obs_temps, fcst_interp) if f is not None]
            if not paired:
                continue

            obs_paired = [p[0] for p in paired]
            fcst_paired = [p[1] for p in paired]

            # Calculate hours from first observation
            t0 = obs_times[0].timestamp() if hasattr(obs_times[0], 'timestamp') else obs_times[0]
            hours = []
            for ot in obs_times[:len(paired)]:
                t = ot.timestamp() if hasattr(ot, 'timestamp') else ot
                hours.append((t - t0) / 3600.0)

            drift_score = compute_drift(obs_paired, fcst_paired)
            slope_div = compute_slope_divergence(obs_paired, fcst_paired, hours)
            forecast_trend = self._compute_forecast_trend(con, stid, model_runs)

            # Project high: latest forecast max + drift + slope extrapolation
            fcst_max_row = con.execute(
                "SELECT MAX(temp_f) FROM forecasts WHERE station_id = ? AND model_run = ?",
                [stid, latest_run],
            ).fetchone()
            fcst_high = fcst_max_row[0] if fcst_max_row else 0.0

            hours_remaining = max(0, (fc_end.timestamp() - ref_time.replace(tzinfo=None).timestamp()) / 3600.0) if hasattr(fc_end, 'timestamp') else 0.0
            projected_high = round(fcst_high + drift_score + slope_div * hours_remaining, 1)

            magnet, magnet_dist = find_nearest_magnet(projected_high)

            # Per-observation drift values for confidence
            drift_values = [o - f for o, f in paired]
            confidence = self._compute_confidence(len(observations), drift_values)

            reports[city_name] = {
                "model_run": latest_run,
                "drift_score": round(drift_score, 2),
                "slope_divergence": round(slope_div, 3),
                "forecast_trend": round(forecast_trend, 2),
                "magnet_proximity": magnet,
                "magnet_distance": magnet_dist,
                "confidence": confidence,
                "projected_high": projected_high,
            }

        con.close()
        return reports

    def calculate_and_store(self, ref_time: Optional[datetime] = None) -> int:
        """Calculate drift for all cities and store in drift_signals table."""
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)

        reports = self.calculate_all(ref_time)
        if not reports:
            return 0

        con = get_connection(self.db_path)
        stored = 0

        for city_name, report in reports.items():
            try:
                con.execute(
                    """INSERT INTO drift_signals
                       (city, calculated_at, model_run, drift_score, slope_divergence,
                        forecast_trend, magnet_proximity, magnet_distance, confidence,
                        projected_high)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [city_name, ref_time.replace(tzinfo=None), report["model_run"],
                     report["drift_score"], report["slope_divergence"],
                     report["forecast_trend"], report["magnet_proximity"],
                     report["magnet_distance"], report["confidence"],
                     report["projected_high"]],
                )
                stored += 1
            except duckdb.ConstraintException:
                pass

        con.close()
        logger.info(f"Stored {stored} drift signals")
        return stored

    async def run(self) -> None:
        """Run bias engine on a 60-second loop."""
        from core.constants import POLL_INTERVAL_SECONDS
        logger.info("Starting Bias Engine — calculating drift every 60s")
        import asyncio
        while True:
            try:
                self.calculate_and_store()
            except Exception as e:
                logger.error(f"Bias engine cycle failed: {e}")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
```

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_bias.py -v
```
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add services/bias.py tests/test_bias.py
git commit -m "feat: bias engine with drift calculation and magnet proximity"
```

---

### Task 6: Wire Everything Into main.py

**Files:**
- Modify: `main.py`

**Step 1: Update main.py to run all three services**

```python
# main.py
"""AlphaTemp — async entrypoint."""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from core.db import init_db
from services.ingestor import SynopticIngestor
from services.forecast import HRRRFetcher
from services.bias import BiasEngine


async def main():
    load_dotenv()
    token = os.getenv("SYNOPTIC_TOKEN")
    if not token:
        logger.error("SYNOPTIC_TOKEN not found in environment")
        return

    init_db()

    ingestor = SynopticIngestor(token=token)
    fetcher = HRRRFetcher()
    engine = BiasEngine()

    await asyncio.gather(
        ingestor.run(),
        fetcher.run(),
        engine.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Run full test suite**

```bash
source venv/bin/activate && python -m pytest tests/ -v
```
Expected: All tests PASS (no regressions)

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: wire forecast fetcher and bias engine into event loop"
```

---

### Task 7: Live Integration Test

**Step 1: Run a manual integration test**

This verifies the full pipeline end-to-end. Run from the project root:

```bash
cd /Users/russellrudd/alphatemp/alphatemp && source venv/bin/activate && python -c "
import asyncio, os
from dotenv import load_dotenv
from core.db import init_db
from services.ingestor import SynopticIngestor
from services.forecast import HRRRFetcher
from services.bias import BiasEngine

load_dotenv()
init_db()

# Step 1: Pull live observations
ingestor = SynopticIngestor(token=os.getenv('SYNOPTIC_TOKEN'))
obs_count = asyncio.run(ingestor.poll_once())
print(f'Observations inserted: {obs_count}')

# Step 2: Fetch HRRR forecasts
fetcher = HRRRFetcher()
fcst_count = fetcher.fetch_run(
    __import__('services.forecast', fromlist=['get_recent_model_runs']).get_recent_model_runs(
        __import__('datetime').datetime.now(__import__('datetime').timezone.utc)
    )[0],
    fxx_range=range(1, 4),  # Just 3 hours for speed
)
print(f'Forecast points inserted: {fcst_count}')

# Step 3: Calculate drift
engine = BiasEngine()
reports = engine.calculate_all()
for city, report in reports.items():
    print(f'{city}: drift={report[\"drift_score\"]:+.2f}F  slope={report[\"slope_divergence\"]:+.3f}F/hr  '
          f'projected_high={report[\"projected_high\"]}F  '
          f'nearest_magnet={report[\"magnet_proximity\"]}  confidence={report[\"confidence\"]}')
"
```

Report the actual output. If HRRR data is unavailable (common for very recent runs), note what failed and adjust the model run offset.

**Step 2: No commit needed — this is a verification step.**

---

### Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | Install deps | — | — |
| 2 | Station coords + forecast interval | core/constants.py | 3 tests |
| 3 | DB schema updates | core/db.py | 5 tests |
| 4 | HRRRFetcher | services/forecast.py | 4 tests |
| 5 | BiasEngine | services/bias.py | 7 tests |
| 6 | Wire into main.py | main.py | — |
| 7 | Live integration test | — | Live verification |
