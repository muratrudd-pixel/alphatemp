# Use Latest HRRR Run in Feature Builder

**Date:** 2026-03-14
**Status:** Approved
**Context:** The feature builder hardcodes `run_hour = 0` (HRRR 00z) for both training and prediction. Backtest across all 24 HRRR run hours shows later runs are dramatically better: Brier 0.58 at 16z vs 0.70 at 00z. The live system is stuck on the worst available data. On March 14, the 00z run was also truncated (only fxx 1-10, no afternoon coverage), causing `fcst_high = 38.1°F` instead of the actual HRRR forecast of 47°F.

---

## Problem

1. `feature_builder._build_features_impl()` hardcodes `run_hour = 0` (line 476)
2. `feature_builder._get_training_data_impl()` hardcodes `run_hour = 0` (line 183)
3. `strategy_engine._cycle()` passes `run_hour=0` to `model.fit()` (line 107)

All three should use the latest available HRRR run that has afternoon forecast coverage.

### Evidence

Backtest over 90 days, 88 settlement dates, 6 update hours per day:

| Run Hours | Brier | Hit Rate | P&L |
|-----------|-------|----------|-----|
| [0,6,12,18] (current) | 0.6667 | 44.6% | 14,248c |
| [0..23] (all) | 0.6597 | 43.6% | 64,108c |

Per-run-hour Brier (EDT):
- 7-10 PM (00z-03z): 0.70-0.74 — worst (this is what the live system uses)
- 3-6 AM (08z-11z): 0.63-0.65 — decent
- 7-10 AM (12z-15z): 0.61-0.62 — good
- 11 AM-2 PM (16z-19z): 0.58-0.62 — best

---

## Design

### New method: `_find_best_hrrr_run_hour`

Added to `FeatureBuilder`. Queries the DB for the latest HRRR run on the target date that has forecast data extending past 18z UTC (1 PM ET), guaranteeing afternoon coverage.

```python
def _find_best_hrrr_run_hour(self, con, target_date):
    # type: (duckdb.DuckDBPyConnection, date) -> int
    """Find the latest HRRR run hour with afternoon forecast coverage.

    Returns the hour (0-23) of the latest HRRR run on target_date
    whose forecast hours extend to at least 18z UTC (1 PM ET).
    Falls back to the latest available run if none reach 18z.
    """
    # Best: latest run that covers afternoon
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
    """, [target_date, STATION_ID]).fetchone()
    if row is not None:
        return int(row[0])

    # Fallback: latest run available (early morning, no afternoon coverage yet)
    row = con.execute("""
        SELECT MAX(EXTRACT(HOUR FROM model_run))::INTEGER
        FROM forecasts
        WHERE model_name = 'hrrr'
          AND model_run::DATE = ?
          AND station_id = ?
    """, [target_date, STATION_ID]).fetchone()
    if row is not None and row[0] is not None:
        return int(row[0])

    return 0  # absolute fallback
```

### Call site 1: `_build_features_impl` (prediction)

Replace line 476:
```python
run_hour = 0  # HRRR 00z run
```
With:
```python
run_hour = self._find_best_hrrr_run_hour(con, target_date)
```

All downstream queries in this method that use `run_hour` (HRRR forecast high, HRRR forecast curve for divergence features, HRRR diurnal range, lag_error) automatically pick up the new run hour. No other changes needed in this method.

### Call site 2: `_get_training_data_impl` (training)

Replace line 183:
```python
run_hour = 0
```
With:
```python
run_hour = self._find_best_hrrr_run_hour(con, current_date)
```

This means: if predicting from the 12z run today, train on historical 12z data. Matches the backtester behavior — when it evaluates at run_hour=12, it calls `get_training_data(con, run_hour=12, ...)`.

Note: for historical dates where a specific run_hour has no data, the training query returns fewer rows (that date is simply absent from training). This is the same behavior as the backtester — it doesn't synthesize data for missing runs.

### Call site 3: `strategy_engine._cycle` (cache key)

The strategy engine needs to pass the actual run_hour to `model.fit()` so the coefficient cache keys correctly. Currently line 107:
```python
coefficients = self.model.fit(
    X_train, y_train, run_hour=0, date_key=date_key
)
```

Change to:
```python
coefficients = self.model.fit(
    X_train, y_train, run_hour=run_hour, date_key=date_key
)
```

This requires the feature builder to return the run_hour it selected, so the strategy engine can pass it through. Two options:

**Option A:** `build_features()` returns `(features, fcst_high, run_hour)` instead of `(features, fcst_high)`.

**Option B:** `get_training_data()` returns `(X, y, dates, run_hour)` instead of `(X, y, dates)`.

**Chosen: Both.** Both methods return the run_hour they used. The strategy engine extracts it from either and passes to `model.fit()` and `model.predict_bracket_probs()`.

The return type changes:
- `build_features()`: `Optional[Tuple[np.ndarray, float]]` → `Optional[Tuple[np.ndarray, float, int]]`
- `get_training_data()`: `Optional[Tuple[np.ndarray, np.ndarray, List[date]]]` → `Optional[Tuple[np.ndarray, np.ndarray, List[date], int]]`

### What doesn't change

- GFS and ECMWF queries stay pinned to 00z — the backtester uses them this way
- forecast_extended queries (solar, dewpoint, precip, humidity, gusts, cape) stay 00z — same reason
- The 23 features and their computation — unchanged
- Model architecture (QR, quantiles, LP solver) — unchanged
- Divergence features already use `update_hour` correctly
- Dashboard — no changes

---

### Logging

The feature builder should log the selected run_hour each time it's called, so we can verify it's picking up fresh runs:
```
logger.info("Using HRRR {}z run for {}", run_hour, target_date)
```

---

## Files Changed

| File | Change |
|------|--------|
| `services/feature_builder.py` | Add `_find_best_hrrr_run_hour`, update `_build_features_impl` and `_get_training_data_impl` to use it, update return types to include run_hour |
| `services/strategy_engine.py` | Unpack run_hour from feature builder returns, pass to `model.fit()` and `model.predict_bracket_probs()` |
| `tests/test_feature_builder.py` | Test `_find_best_hrrr_run_hour` with various data scenarios. **Note:** ~12 existing tuple unpacking sites must be updated for the new return shape. Test fixture needs HRRR data at non-zero run hours (e.g., 12z, 16z). |
| `tests/test_strategy_engine.py` | Update mocks for new return types |
