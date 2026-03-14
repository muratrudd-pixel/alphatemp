# Latest HRRR Run Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the feature builder to use the latest available HRRR run with afternoon coverage instead of hardcoded 00z, matching backtester behavior.

**Architecture:** Add `_find_best_hrrr_run_hour()` to FeatureBuilder, update both training and prediction to use it, and thread the selected run_hour through to the model's cache key. GFS/ECMWF stay at 00z.

**Tech Stack:** Python 3.9, DuckDB, pytest, numpy

**Spec:** `docs/superpowers/specs/2026-03-14-latest-hrrr-run-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/feature_builder.py` | Modify | Add `_find_best_hrrr_run_hour`, update training + prediction, update return types |
| `services/strategy_engine.py` | Modify | Unpack run_hour, pass to model.fit() and model.predict_bracket_probs() |
| `tests/test_feature_builder.py` | Modify | Add multi-run-hour test data, test run hour selection, update tuple unpacking |
| `tests/test_strategy_engine.py` | Modify | Update mock return values for new tuple shapes |

### Existing Patterns

- `tests/test_feature_builder.py`: Uses `test_db(tmp_path)` fixture that creates tables via `_create_tables(con)` and inserts 90 days of synthetic data. Helper `_insert_hrrr_forecasts(con, station, date, run_hour, temps_by_fxx)` already supports arbitrary run hours. Fixture `builder(test_db)` returns `FeatureBuilder` instance.
- `tests/test_strategy_engine.py`: Uses `engine()` fixture with `StrategyEngine.__new__` + mocked deps. Feature builder and model are mocked.
- Feature builder returns: `build_features()` → `Optional[Tuple[np.ndarray, float]]`, `get_training_data()` → `Optional[Tuple[np.ndarray, np.ndarray, List[date]]]`

---

## Chunk 1: Feature Builder — Run Hour Selection

### Task 1: Add `_find_best_hrrr_run_hour` with tests

**Files:**
- Modify: `services/feature_builder.py`
- Modify: `tests/test_feature_builder.py`

- [ ] **Step 1: Write tests for run hour selection**

Append to `tests/test_feature_builder.py`. Uses a dedicated lightweight fixture (not the full 90-day `test_db`) so tests are fast and isolated:

```python
# ---------------------------------------------------------------------------
# _find_best_hrrr_run_hour
# ---------------------------------------------------------------------------

@pytest.fixture
def run_hour_db(tmp_path):
    """Minimal DB for testing run hour selection only."""
    db_path = str(tmp_path / "rh_test.duckdb")
    con = duckdb.connect(db_path)
    _create_tables(con)
    con.close()
    return db_path


class TestFindBestHrrrRunHour:
    """Test _find_best_hrrr_run_hour selection logic."""

    def test_selects_latest_with_afternoon_coverage(self, run_hour_db):
        """12z run with fxx reaching 18z should be selected over 00z."""
        from services.feature_builder import FeatureBuilder
        con = duckdb.connect(run_hour_db)
        d = date(2026, 3, 14)
        station = "KNYC"

        # 00z run: fxx 1-10 only (no afternoon)
        _insert_hrrr_forecasts(con, station, d, 0,
                               {fxx: 38.0 for fxx in range(1, 11)})
        # 12z run: fxx 1-18 (reaches 18+12=30 > 18, has afternoon)
        _insert_hrrr_forecasts(con, station, d, 12,
                               {fxx: 47.0 for fxx in range(1, 19)})
        con.close()

        builder = FeatureBuilder(run_hour_db)
        con = duckdb.connect(run_hour_db)
        try:
            rh = builder._find_best_hrrr_run_hour(con, d)
        finally:
            con.close()
        assert rh == 12

    def test_selects_latest_when_multiple_have_afternoon(self, run_hour_db):
        """When 06z and 12z both cover afternoon, pick 12z (latest)."""
        from services.feature_builder import FeatureBuilder
        con = duckdb.connect(run_hour_db)
        d = date(2026, 3, 14)
        station = "KNYC"

        # 06z: fxx 1-18 (6+12=18, covers afternoon)
        _insert_hrrr_forecasts(con, station, d, 6,
                               {fxx: 45.0 for fxx in range(1, 19)})
        # 12z: fxx 1-18 (12+6=18, covers afternoon)
        _insert_hrrr_forecasts(con, station, d, 12,
                               {fxx: 47.0 for fxx in range(1, 19)})
        con.close()

        builder = FeatureBuilder(run_hour_db)
        con = duckdb.connect(run_hour_db)
        try:
            rh = builder._find_best_hrrr_run_hour(con, d)
        finally:
            con.close()
        assert rh == 12

    def test_falls_back_to_latest_when_none_reach_afternoon(self, run_hour_db):
        """Early morning: only 00z and 03z exist, neither reaches 18z."""
        from services.feature_builder import FeatureBuilder
        con = duckdb.connect(run_hour_db)
        d = date(2026, 3, 14)
        station = "KNYC"

        # 00z: fxx 1-10 (max valid = 10z)
        _insert_hrrr_forecasts(con, station, d, 0,
                               {fxx: 38.0 for fxx in range(1, 11)})
        # 03z: fxx 1-10 (max valid = 13z)
        _insert_hrrr_forecasts(con, station, d, 3,
                               {fxx: 39.0 for fxx in range(1, 11)})
        con.close()

        builder = FeatureBuilder(run_hour_db)
        con = duckdb.connect(run_hour_db)
        try:
            rh = builder._find_best_hrrr_run_hour(con, d)
        finally:
            con.close()
        assert rh == 3  # latest available, even without afternoon

    def test_returns_zero_when_no_data(self, run_hour_db):
        """No HRRR data at all -> falls back to 0."""
        from services.feature_builder import FeatureBuilder
        builder = FeatureBuilder(run_hour_db)
        con = duckdb.connect(run_hour_db)
        try:
            rh = builder._find_best_hrrr_run_hour(con, date(2026, 3, 14))
        finally:
            con.close()
        assert rh == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_feature_builder.py::TestFindBestHrrrRunHour -v`
Expected: FAIL — `_find_best_hrrr_run_hour` doesn't exist.

- [ ] **Step 3: Implement `_find_best_hrrr_run_hour`**

Add to `services/feature_builder.py`, inside the `FeatureBuilder` class, before `_build_features_impl`:

```python
    def _find_best_hrrr_run_hour(self, con, target_date):
        # type: (duckdb.DuckDBPyConnection, date) -> int
        """Find the latest HRRR run hour with afternoon forecast coverage.

        Returns the hour (0-23) of the latest HRRR run on target_date
        whose forecast hours extend to at least 18z UTC (1 PM ET).
        Falls back to the latest available run if none reach 18z.
        """
        station_id = STATION_ID

        # Best: latest run that covers afternoon (18z UTC)
        row = con.execute("""
            SELECT EXTRACT(HOUR FROM model_run)::INTEGER as rh
            FROM forecasts
            WHERE model_name = 'hrrr'
              AND model_run::DATE = ?
              AND station_id = ?
              AND (EXTRACT(HOUR FROM model_run) + fxx) >= 18
            GROUP BY rh
            ORDER BY rh DESC
            LIMIT 1
        """, [target_date, station_id]).fetchone()
        if row is not None:
            return int(row[0])

        # Fallback: latest run available (early morning, no afternoon data yet)
        row = con.execute("""
            SELECT MAX(EXTRACT(HOUR FROM model_run))::INTEGER
            FROM forecasts
            WHERE model_name = 'hrrr'
              AND model_run::DATE = ?
              AND station_id = ?
        """, [target_date, station_id]).fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])

        return 0  # absolute fallback
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_feature_builder.py::TestFindBestHrrrRunHour -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/feature_builder.py tests/test_feature_builder.py
git commit -m "feat: add _find_best_hrrr_run_hour to FeatureBuilder"
```

---

## Chunk 2: Feature Builder — Wire Up Run Hour + Return Types

### Task 2: Update `_build_features_impl` and `_get_training_data_impl`

**Files:**
- Modify: `services/feature_builder.py`

- [ ] **Step 6: Update `_build_features_impl` to use best run hour**

In `_build_features_impl` (line 476), replace:
```python
        run_hour = 0  # HRRR 00z run
```
With:
```python
        run_hour = self._find_best_hrrr_run_hour(con, target_date)
        logger.info("Using HRRR {}z run for {}", run_hour, target_date)
```

Add loguru import at top of file if not present:
```python
from loguru import logger
```

- [ ] **Step 7: Update `_build_features_impl` return to include run_hour**

At the end of `_build_features_impl` (line 656), replace:
```python
        return features, fcst_high
```
With:
```python
        return features, fcst_high, run_hour
```

- [ ] **Step 8: Update `build_features` return type comment**

In `build_features` (around line 148), update the return type docstring:
```python
        Returns
        -------
        (features_array, fcst_high, run_hour) or None
            features_array: ndarray shape (23,) — raw feature vector
            fcst_high: float — HRRR forecast high temperature
            run_hour: int — HRRR run hour used (0-23)
            Returns None if HRRR data missing for the date.
```

- [ ] **Step 9: Update `_get_training_data_impl` to use best run hour**

In `_get_training_data_impl` (line 183), replace:
```python
        run_hour = 0
```
With:
```python
        run_hour = self._find_best_hrrr_run_hour(con, current_date)
        logger.info("Training with HRRR {}z data (window ending {})", run_hour, current_date)
```

- [ ] **Step 10: Update `_get_training_data_impl` return to include run_hour**

Find the return statement at the end of `_get_training_data_impl` (returns `(X, y, dates)`) and add `run_hour`:

```python
        return np.array(X_rows), np.array(y_rows), d_rows, run_hour
```

- [ ] **Step 11: Update `get_training_data` return type comment**

In `get_training_data` (around line 120), update the return type docstring:
```python
        Returns
        -------
        (X, y, dates, run_hour) or None
            X: ndarray shape (n_samples, 23) — feature matrix
            y: ndarray (n_samples,) — actual errors (fcst_high - actual_high)
            dates: list of dates for each row
            run_hour: int — HRRR run hour used (0-23)
            Returns None if insufficient data (< MIN_SAMPLES daily rows).
```

- [ ] **Step 12: Verify existing feature builder tests still pass**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_feature_builder.py -v 2>&1 | tail -20`

Expected: FAIL — tuple unpacking errors at lines 320, 356, 379, 393, 458, 541 (old 2-tuple and 3-tuple unpacking).

### Task 3: Fix existing test unpacking

**Files:**
- Modify: `tests/test_feature_builder.py`

- [ ] **Step 13: Update all build_features unpacking to 3 values**

There are 9 sites that unpack `build_features()` results. Update ALL of them:

| Line | Old | New |
|------|-----|-----|
| 320 | `features, fcst_high = result` | `features, fcst_high, _run_hour = result` |
| 356 | `features, fcst_high = result` | `features, fcst_high, _run_hour = result` |
| 366 | `features, _ = result` | `features, _, _run_hour = result` |
| 379 | `features, fcst_high = result` | `features, fcst_high, _run_hour = result` |
| 393 | `features, fcst_high = result` | `features, fcst_high, _run_hour = result` |
| 427 | `features, _ = result` | `features, _, _run_hour = result` |
| 479 | `live_features, _ = live_result` | `live_features, _, _run_hour = live_result` |
| 519 | `features, _ = result` | `features, _, _run_hour = result` |
| 541 | `features, fcst_high = result` | `features, fcst_high, _run_hour = result` |

- [ ] **Step 14: Update all get_training_data unpacking to 4 values**

There are 4 sites that unpack `get_training_data()` results. Update ALL:

| Line | Old | New |
|------|-----|-----|
| 292 | `X, y, dates = result` | `X, y, dates, _run_hour = result` |
| 458 | `X, y, dates = train_result` | `X, y, dates, _run_hour = train_result` |
| 552 | `X, y, dates = result` | `X, y, dates, _run_hour = result` |
| 567 | `X, y, dates = result` | `X, y, dates, _run_hour = result` |

- [ ] **Step 15: Run all feature builder tests**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/test_feature_builder.py -v`
Expected: All tests PASS.

- [ ] **Step 16: Commit**

```bash
git add services/feature_builder.py tests/test_feature_builder.py
git commit -m "feat: feature builder uses latest HRRR run instead of hardcoded 00z"
```

---

## Chunk 3: Strategy Engine — Thread Run Hour Through

### Task 4: Update strategy engine to use run_hour

**Files:**
- Modify: `services/strategy_engine.py`

- [ ] **Step 17: Update training data unpacking**

In `_cycle()` (line 104), replace:
```python
        X_train, y_train, train_dates = train_result
```
With:
```python
        X_train, y_train, train_dates, run_hour = train_result
```

- [ ] **Step 18: Pass run_hour to model.fit()**

In `_cycle()` (line 107), replace:
```python
        coefficients = self.model.fit(
            X_train, y_train, run_hour=0, date_key=date_key
        )
```
With:
```python
        coefficients = self.model.fit(
            X_train, y_train, run_hour=run_hour, date_key=date_key
        )
```

- [ ] **Step 19: Update feature unpacking**

In `_cycle()` (line 124), replace:
```python
        features, fcst_high = feat_result
```
With:
```python
        features, fcst_high, _feat_run_hour = feat_result
```

- [ ] **Step 20: Pass run_hour to model.predict_bracket_probs()**

In `_cycle()` (line 128), replace:
```python
        bracket_probs = self.model.predict_bracket_probs(
            features, fcst_high, run_hour=0, date_key=date_key
        )
```
With:
```python
        bracket_probs = self.model.predict_bracket_probs(
            features, fcst_high, run_hour=run_hour, date_key=date_key
        )
```

- [ ] **Step 21: Add run_hour to the strategy cycle log**

Update the log message at line 83:
```python
        logger.info(
            "Strategy cycle: target={}, update_hour={}, now_et={}",
            target_date, update_hour, now_et.strftime("%H:%M"),
        )
```
This will naturally show the run_hour after the feature builder logs it. No change needed here — the feature builder's log message handles it.

### Task 5: Update strategy engine test mocks

**Files:**
- Modify: `tests/test_strategy_engine.py`

- [ ] **Step 22: Verify no strategy engine test changes needed**

The existing `tests/test_strategy_engine.py` tests only pure logic methods (`_compute_edge`, `_aggregate_to_kalshi_brackets`, `_generate_signals`, `_check_edge_reversals`). None of them mock `feature_builder.build_features` or `feature_builder.get_training_data`. No test file changes needed — just verify they still pass.

- [ ] **Step 23: Run all tests for regression**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/ -v --ignore=venv 2>&1 | tail -30`
Expected: All tests PASS.

- [ ] **Step 24: Commit**

```bash
git add services/strategy_engine.py tests/test_strategy_engine.py
git commit -m "feat: strategy engine passes actual HRRR run_hour to model"
```

---

## Chunk 4: Smoke Test

### Task 6: Verify against live data

- [ ] **Step 25: Check what run hour gets selected**

```bash
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -c "
import duckdb
from datetime import date
from services.feature_builder import FeatureBuilder

fb = FeatureBuilder('data/alphatemp.duckdb')
con = duckdb.connect('data/alphatemp.duckdb', read_only=True)
rh = fb._find_best_hrrr_run_hour(con, date(2026, 3, 14))
con.close()
print('Selected run hour: %02dz' % rh)
"
```
Expected: Should print a recent run hour (e.g., 12z or 13z), NOT 00z.

- [ ] **Step 26: Compare fcst_high with old vs new**

```bash
cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -c "
from datetime import date
from services.feature_builder import FeatureBuilder

fb = FeatureBuilder('data/alphatemp.duckdb')

# New: uses latest run
result = fb.build_features(date(2026, 3, 14), 9)
if result:
    features, fcst_high, run_hour = result
    print('New: fcst_high=%.1f F (using HRRR %02dz)' % (fcst_high, run_hour))
else:
    print('New: no result')
"
```
Expected: `fcst_high` should be ~47°F (matching HRRR runs), NOT 38.1°F.

- [ ] **Step 27: Run full test suite one final time**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. venv/bin/python -m pytest tests/ -v --ignore=venv`
Expected: All tests PASS.
