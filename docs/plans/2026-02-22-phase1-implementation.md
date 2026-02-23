# AlphaTemp Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the data plumbing — DuckDB schema, magnet number generator, METAR T-group parser, and async Synoptic ingestor that polls 11 ASOS stations every 60 seconds.

**Architecture:** Layered services pattern. `core/` holds pure logic (constants, DB init, schemas). `services/` handles I/O (Synoptic API polling). `main.py` runs the async event loop. All data flows through Polars DataFrames into DuckDB.

**Tech Stack:** Python 3.9+, DuckDB, Polars, httpx, Loguru, asyncio, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `core/__init__.py`
- Create: `services/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create .gitignore**

```
.env
*.duckdb
*.duckdb.wal
__pycache__/
*.pyc
venv/
.pytest_cache/
data/
```

**Step 2: Create empty package init files**

Create empty `__init__.py` in `core/`, `services/`, and `tests/`.

**Step 3: Create data directory**

```bash
mkdir -p data
```

**Step 4: Install pytest**

```bash
source venv/bin/activate && pip install pytest pytest-asyncio
```

**Step 5: Commit**

```bash
git add .gitignore core/__init__.py services/__init__.py tests/__init__.py
git commit -m "scaffold: project structure with .gitignore"
```

---

### Task 2: Core Constants — Magnet Number Generator

**Files:**
- Create: `tests/test_constants.py`
- Create: `core/constants.py`

**Step 1: Write the failing tests**

```python
# tests/test_constants.py
from core.constants import (
    generate_magnets,
    MAGNETS,
    FLB_FAVORITE_FLOOR,
    FLB_LONGSHOT_CEILING,
    CITIES,
    POLL_INTERVAL_SECONDS,
)


def test_generate_magnets_returns_set():
    result = generate_magnets(30, 110)
    assert isinstance(result, set)
    assert len(result) > 0


def test_known_magnets_present():
    """40, 50, 60 are known magnet numbers in the 9°F repeating pattern."""
    result = generate_magnets(30, 110)
    for known in [40, 50, 60, 70]:
        assert known in result, f"{known} should be a magnet"


def test_magnet_captures_six_tenths():
    """Each magnet integer should map to 6+ Celsius tenths when rounded."""
    result = generate_magnets(30, 110)
    for mag in result:
        count = 0
        for tenths in range(10):
            c = mag  # we need to check: for what C values does round(C*9/5+32) == mag
        # Verify via reverse: count how many 0.1C values round to this F integer
        count = 0
        for c_int in range(max(-400, int((mag - 35) / 1.8 * 10)), int((mag - 28) / 1.8 * 10)):
            c_val = c_int / 10.0
            f_val = round(c_val * 9.0 / 5.0 + 32.0)
            if f_val == mag:
                count += 1
        assert count >= 6, f"Magnet {mag} only captures {count} tenths"


def test_non_magnets_capture_five_or_fewer():
    """Non-magnet integers should capture 5 or fewer Celsius tenths."""
    magnets = generate_magnets(30, 110)
    for f_int in range(30, 111):
        if f_int not in magnets:
            count = 0
            for c_int in range(int((f_int - 35) / 1.8 * 10), int((f_int - 28) / 1.8 * 10)):
                c_val = c_int / 10.0
                f_val = round(c_val * 9.0 / 5.0 + 32.0)
                if f_val == f_int:
                    count += 1
            assert count <= 5, f"Non-magnet {f_int} captures {count} tenths"


def test_flb_thresholds():
    assert FLB_FAVORITE_FLOOR == 0.50
    assert FLB_LONGSHOT_CEILING == 0.15


def test_cities_config():
    assert "CHI" in CITIES
    assert CITIES["CHI"]["settlement"] == "KMDW"
    assert "KORD" in CITIES["CHI"]["neighbors"]
    assert len(CITIES) == 5


def test_all_stations_list():
    """All 11 stations should be extractable from CITIES config."""
    all_stations = []
    for city in CITIES.values():
        all_stations.append(city["settlement"])
        all_stations.extend(city["neighbors"])
    assert len(all_stations) == 11


def test_poll_interval():
    assert POLL_INTERVAL_SECONDS == 60
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_constants.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core.constants'`

**Step 3: Write the implementation**

```python
# core/constants.py
"""AlphaTemp core constants — magnet numbers, FLB thresholds, station config."""

from typing import Set


def generate_magnets(f_min: int = 0, f_max: int = 130) -> Set[int]:
    """Generate magnet Fahrenheit integers.

    ASOS sensors report in 0.1°C. When converting each tenth to °F and rounding,
    certain integers capture 6 tenths instead of 5. These repeat every 9°F.
    """
    counts: dict[int, int] = {}
    # Sweep through a wide Celsius range that covers f_min to f_max
    c_min_raw = (f_min - 32) * 5.0 / 9.0
    c_max_raw = (f_max - 32) * 5.0 / 9.0
    c_start = int(c_min_raw * 10) - 10
    c_end = int(c_max_raw * 10) + 10

    for c_tenth in range(c_start, c_end + 1):
        c_val = c_tenth / 10.0
        f_rounded = round(c_val * 9.0 / 5.0 + 32.0)
        if f_min <= f_rounded <= f_max:
            counts[f_rounded] = counts.get(f_rounded, 0) + 1

    return {f_int for f_int, count in counts.items() if count >= 6}


# Pre-computed magnets for the typical settlement range
MAGNETS: Set[int] = generate_magnets(0, 130)

# Favorite-Longshot Bias thresholds
FLB_FAVORITE_FLOOR: float = 0.50
FLB_LONGSHOT_CEILING: float = 0.15

# Station configuration: settlement station + neighbors
CITIES: dict = {
    "NYC": {"settlement": "KNYC", "neighbors": ["KLGA", "KEWR"]},
    "PHL": {"settlement": "KPHL", "neighbors": ["KPNE"]},
    "CHI": {"settlement": "KMDW", "neighbors": ["KORD"]},
    "MIA": {"settlement": "KMIA", "neighbors": ["KOPF"]},
    "LA":  {"settlement": "KLAX", "neighbors": ["KHHR"]},
}

# Polling interval in seconds
POLL_INTERVAL_SECONDS: int = 60


def get_all_station_ids() -> list[str]:
    """Return flat list of all 11 station ICAO codes."""
    stations = []
    for city in CITIES.values():
        stations.append(city["settlement"])
        stations.extend(city["neighbors"])
    return stations
```

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_constants.py -v
```
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add core/constants.py tests/test_constants.py
git commit -m "feat: magnet number generator and core constants"
```

---

### Task 3: Database Layer

**Files:**
- Create: `tests/test_db.py`
- Create: `core/db.py`

**Step 1: Write the failing tests**

```python
# tests/test_db.py
import os
import duckdb
import pytest
from core.db import init_db, get_connection

TEST_DB = "data/test_alphatemp.duckdb"


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test DB before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_init_db_creates_tables():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = {t[0] for t in tables}
    assert "observations" in table_names
    assert "forecasts" in table_names
    assert "market_ticks" in table_names
    con.close()


def test_observations_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE observations").fetchall()
    col_names = {c[0] for c in cols}
    assert "station_id" in col_names
    assert "observed_at" in col_names
    assert "temp_f" in col_names
    assert "temp_c_tenth" in col_names
    assert "raw_metar" in col_names
    assert "ingested_at" in col_names
    con.close()


def test_observations_unique_constraint():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    con.execute("""
        INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
        VALUES ('KNYC', '2026-02-22 12:00:00', 45.0, 7.2, 'test', CURRENT_TIMESTAMP)
    """)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
            VALUES ('KNYC', '2026-02-22 12:00:00', 45.0, 7.2, 'test', CURRENT_TIMESTAMP)
        """)
    con.close()


def test_market_ticks_schema():
    init_db(TEST_DB)
    con = duckdb.connect(TEST_DB)
    cols = con.execute("DESCRIBE market_ticks").fetchall()
    col_names = {c[0] for c in cols}
    for expected in ["market_id", "city", "captured_at", "yes_bid", "yes_ask",
                     "no_bid", "no_ask", "last_trade", "volume"]:
        assert expected in col_names
    con.close()


def test_get_connection():
    init_db(TEST_DB)
    con = get_connection(TEST_DB)
    assert con is not None
    result = con.execute("SELECT 1").fetchone()
    assert result[0] == 1
    con.close()
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_db.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core.db'`

**Step 3: Write the implementation**

```python
# core/db.py
"""AlphaTemp database initialization and connection management."""

import duckdb
from loguru import logger

DEFAULT_DB_PATH = "data/alphatemp.duckdb"


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize DuckDB with all required tables."""
    con = duckdb.connect(db_path)
    logger.info(f"Initializing database at {db_path}")

    con.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            station_id   VARCHAR NOT NULL,
            observed_at  TIMESTAMP NOT NULL,
            temp_f       DOUBLE,
            temp_c_tenth DOUBLE,
            raw_metar    VARCHAR,
            ingested_at  TIMESTAMP NOT NULL,
            UNIQUE (station_id, observed_at)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            station_id  VARCHAR NOT NULL,
            valid_at    TIMESTAMP NOT NULL,
            temp_f      DOUBLE,
            model_run   TIMESTAMP,
            ingested_at TIMESTAMP NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS market_ticks (
            market_id   VARCHAR NOT NULL,
            city        VARCHAR NOT NULL,
            captured_at TIMESTAMP NOT NULL,
            yes_bid     DOUBLE,
            yes_ask     DOUBLE,
            no_bid      DOUBLE,
            no_ask      DOUBLE,
            last_trade  DOUBLE,
            volume      INTEGER
        )
    """)

    logger.info("Database tables initialized")
    con.close()


def get_connection(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection."""
    return duckdb.connect(db_path)
```

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_db.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add core/db.py tests/test_db.py
git commit -m "feat: DuckDB schema with observations, forecasts, market_ticks"
```

---

### Task 4: METAR T-Group Parser

**Files:**
- Create: `tests/test_ingestor.py`
- Create: `services/ingestor.py` (parser portion only)

**Step 1: Write the failing tests for the parser**

```python
# tests/test_ingestor.py
from services.ingestor import parse_t_group


def test_parse_positive_temp():
    assert parse_t_group("T02280167") == 22.8


def test_parse_negative_temp():
    assert parse_t_group("T10051012") == -0.5


def test_parse_zero():
    assert parse_t_group("T00000000") == 0.0


def test_parse_no_t_group():
    assert parse_t_group("RMK AO2 SLP135") is None


def test_parse_embedded_in_metar():
    metar = "RMK AO2 SLP135 T02280167 10272 20228 53012"
    assert parse_t_group(metar) == 22.8


def test_parse_negative_freezing():
    assert parse_t_group("T10171028") == -1.7


def test_parse_hot_day():
    assert parse_t_group("T03890350") == 38.9
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_ingestor.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'services.ingestor'`

**Step 3: Write the parser implementation**

```python
# services/ingestor.py
"""Synoptic API ingestor with METAR T-group parser."""

import re
from typing import Optional

T_GROUP_PATTERN = re.compile(r"\bT(\d)(\d{3})")


def parse_t_group(metar_remarks: str) -> Optional[float]:
    """Extract high-resolution Celsius temperature from METAR T-group.

    The T-group in METAR remarks encodes temperature to 0.1°C precision.
    Format: T[sign][temp_tenths][sign][dewpoint_tenths]
    Sign: 0 = positive, 1 = negative
    Example: T0228 -> +22.8°C, T1005 -> -0.5°C
    """
    match = T_GROUP_PATTERN.search(metar_remarks)
    if not match:
        return None
    sign = -1 if match.group(1) == "1" else 1
    temp = int(match.group(2)) / 10.0
    return sign * temp
```

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_ingestor.py -v
```
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add services/ingestor.py tests/test_ingestor.py
git commit -m "feat: METAR T-group parser for high-res Celsius extraction"
```

---

### Task 5: Async Synoptic Ingestor

**Files:**
- Modify: `services/ingestor.py` (add SynopticIngestor class)
- Modify: `tests/test_ingestor.py` (add ingestor integration tests)

**Step 1: Add integration test**

Append to `tests/test_ingestor.py`:

```python
import os
import pytest
import duckdb
from unittest.mock import AsyncMock, patch, MagicMock
from services.ingestor import SynopticIngestor, parse_t_group
from core.db import init_db

TEST_DB = "data/test_alphatemp.duckdb"

MOCK_SYNOPTIC_RESPONSE = {
    "STATION": [
        {
            "STID": "KNYC",
            "OBSERVATIONS": {
                "date_time": ["2026-02-22T12:00:00Z", "2026-02-22T12:01:00Z"],
                "air_temp_value_1": {"values": [7.2, 7.3]},
                "metar": {"values": [
                    "METAR KNYC 221200Z RMK AO2 T00720056",
                    "METAR KNYC 221201Z RMK AO2 T00730058",
                ]},
            },
        },
        {
            "STID": "KMDW",
            "OBSERVATIONS": {
                "date_time": ["2026-02-22T12:00:00Z"],
                "air_temp_value_1": {"values": [-2.5]},
                "metar": {"values": [
                    "METAR KMDW 221200Z RMK AO2 T10251033",
                ]},
            },
        },
    ]
}


@pytest.fixture
def test_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)
    yield TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.mark.asyncio
async def test_ingestor_stores_observations(test_db):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SYNOPTIC_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("services.ingestor.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        ingestor = SynopticIngestor(token="test_token", db_path=test_db)
        await ingestor.poll_once()

    con = duckdb.connect(test_db)
    rows = con.execute("SELECT station_id, temp_c_tenth FROM observations ORDER BY station_id, observed_at").fetchall()
    con.close()

    # KMDW: -2.5, KNYC: 7.2, 7.3
    assert len(rows) == 3
    station_ids = [r[0] for r in rows]
    assert "KNYC" in station_ids
    assert "KMDW" in station_ids

    # Check T-group parsing worked
    kmdw_row = [r for r in rows if r[0] == "KMDW"][0]
    assert kmdw_row[1] == -2.5


@pytest.mark.asyncio
async def test_ingestor_deduplicates(test_db):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SYNOPTIC_RESPONSE
    mock_response.raise_for_status = MagicMock()

    with patch("services.ingestor.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        ingestor = SynopticIngestor(token="test_token", db_path=test_db)
        await ingestor.poll_once()
        await ingestor.poll_once()  # Second poll — same data

    con = duckdb.connect(test_db)
    count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    con.close()
    assert count == 3  # No duplicates
```

**Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_ingestor.py -v
```
Expected: FAIL — `ImportError: cannot import name 'SynopticIngestor'`

**Step 3: Write the SynopticIngestor class**

Add to `services/ingestor.py`:

```python
import asyncio
from datetime import datetime, timezone

import duckdb
import httpx
import polars as pl
from loguru import logger

from core.constants import get_all_station_ids, POLL_INTERVAL_SECONDS
from core.db import get_connection

SYNOPTIC_BASE_URL = "https://api.synopticdata.com/v2/stations/timeseries"


class SynopticIngestor:
    """Polls Synoptic API for 1-minute ASOS observations and stores in DuckDB."""

    def __init__(self, token: str, db_path: str = "data/alphatemp.duckdb"):
        self.token = token
        self.db_path = db_path
        self.stations = get_all_station_ids()

    async def poll_once(self) -> int:
        """Execute a single poll cycle. Returns number of new rows inserted."""
        params = {
            "stid": ",".join(self.stations),
            "recent": 5,
            "vars": "air_temp",
            "obtimezone": "UTC",
            "token": self.token,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(SYNOPTIC_BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Synoptic API request failed: {e}")
            return 0

        if "STATION" not in data:
            logger.warning("No STATION data in Synoptic response")
            return 0

        rows = []
        now = datetime.now(timezone.utc)

        for station in data["STATION"]:
            stid = station["STID"]
            obs = station.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            temps = obs.get("air_temp_value_1", {}).get("values", [])
            metars = obs.get("metar", {}).get("values", [])

            for i, dt_str in enumerate(times):
                temp_f_val = temps[i] if i < len(temps) else None
                metar_str = metars[i] if i < len(metars) else ""

                # Convert Celsius API value to Fahrenheit
                temp_f = round(temp_f_val * 9.0 / 5.0 + 32.0, 1) if temp_f_val is not None else None

                # Parse T-group for high-res Celsius
                temp_c_tenth = parse_t_group(metar_str)

                rows.append({
                    "station_id": stid,
                    "observed_at": dt_str,
                    "temp_f": temp_f,
                    "temp_c_tenth": temp_c_tenth,
                    "raw_metar": metar_str,
                    "ingested_at": now.isoformat(),
                })

        if not rows:
            logger.debug("No observations to insert")
            return 0

        df = pl.DataFrame(rows).with_columns(
            pl.col("observed_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
            pl.col("ingested_at").str.to_datetime(),
        )

        con = get_connection(self.db_path)
        inserted = 0
        for row in df.iter_rows(named=True):
            try:
                con.execute(
                    """INSERT INTO observations (station_id, observed_at, temp_f, temp_c_tenth, raw_metar, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [row["station_id"], row["observed_at"], row["temp_f"],
                     row["temp_c_tenth"], row["raw_metar"], row["ingested_at"]],
                )
                inserted += 1
            except duckdb.ConstraintException:
                pass  # Duplicate — skip silently
        con.close()

        logger.info(f"Inserted {inserted} new observations ({len(rows) - inserted} duplicates skipped)")
        return inserted

    async def run(self) -> None:
        """Run the ingestor loop indefinitely."""
        logger.info(f"Starting Synoptic ingestor — polling {len(self.stations)} stations every {POLL_INTERVAL_SECONDS}s")
        while True:
            await self.poll_once()
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
```

**Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate && python -m pytest tests/test_ingestor.py -v
```
Expected: All 9 tests PASS (7 parser + 2 ingestor)

**Step 5: Commit**

```bash
git add services/ingestor.py tests/test_ingestor.py
git commit -m "feat: async Synoptic ingestor with deduplication"
```

---

### Task 6: Main Entrypoint + Live Test

**Files:**
- Create: `main.py`

**Step 1: Write main.py**

```python
# main.py
"""AlphaTemp — async entrypoint."""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from core.db import init_db
from services.ingestor import SynopticIngestor


async def main():
    load_dotenv()
    token = os.getenv("SYNOPTIC_TOKEN")
    if not token:
        logger.error("SYNOPTIC_TOKEN not found in environment")
        return

    init_db()
    ingestor = SynopticIngestor(token=token)
    await ingestor.run()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Install python-dotenv**

```bash
source venv/bin/activate && pip install python-dotenv
```

**Step 3: Run live test — single poll**

```bash
source venv/bin/activate && python -c "
import asyncio, os
from dotenv import load_dotenv
from core.db import init_db
from services.ingestor import SynopticIngestor

load_dotenv()
init_db()
ingestor = SynopticIngestor(token=os.getenv('SYNOPTIC_TOKEN'))
result = asyncio.run(ingestor.poll_once())
print(f'Inserted {result} observations')

import duckdb
con = duckdb.connect('data/alphatemp.duckdb')
rows = con.execute('''
    SELECT station_id, observed_at, temp_f, temp_c_tenth
    FROM observations
    WHERE station_id IN (\'KNYC\', \'KMDW\')
    ORDER BY station_id, observed_at DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(f'  {r[0]} | {r[1]} | {r[2]}°F | T-group: {r[3]}°C')
con.close()
"
```

Expected: Log output showing inserted observations with temp_c_tenth values for KNYC and KMDW.

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: async entrypoint with live Synoptic polling"
```

---

### Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | Scaffolding | .gitignore, __init__.py files | — |
| 2 | Constants | core/constants.py | 8 tests |
| 3 | Database | core/db.py | 5 tests |
| 4 | T-Group Parser | services/ingestor.py (parser) | 7 tests |
| 5 | Async Ingestor | services/ingestor.py (class) | 2 tests |
| 6 | Entrypoint + Live | main.py | Live verification |
