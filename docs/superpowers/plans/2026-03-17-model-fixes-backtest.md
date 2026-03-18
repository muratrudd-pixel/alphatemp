# Model Fixes & Strategy Engine Improvements — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 known issues (Herbie timeout, running high feature, GFS/ECMWF time-decay, strategy engine dedup/tail/EV), validated by before/after backtests on Mar 12-17.

**Architecture:** Baseline backtest → fix Herbie timeout → add running_max as model feature → decay stale forecast spreads → execute approved strategy engine spec → re-backtest to verify.

**Tech Stack:** Python 3.9, DuckDB, scipy (LP), Herbie, asyncio, pytest

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `scripts/diagnostic_backtest.py` | Create | Day-by-day model accuracy + P&L for date range |
| `services/forecast.py:143-198` | Modify | Wrap Herbie download in timeout |
| `services/feature_builder.py` | Modify | Add running_max feature (#23), decay GFS/ECMWF spreads |
| `services/strategy_engine.py` | Modify | Tail brackets, dedup guard, EV filter (per approved spec) |
| `services/circuit_breakers.py` | Modify | Optional[int] params, NULL-safe SQL |
| `services/paper_trader.py` | Modify | Tail settlement, tail unrealized lookup |
| `tests/test_feature_builder.py` | Modify | Test new feature + decay |
| `tests/test_forecast_timeout.py` | Create | Test Herbie timeout behavior |
| `tests/test_strategy_engine.py` | Modify | Tail, dedup, EV filter tests |

---

## Chunk 1: Baseline Diagnostic Backtest

### Task 1: Write Diagnostic Backtest Script

**Files:**
- Create: `scripts/diagnostic_backtest.py`

- [ ] **Step 1: Write the diagnostic script**

This script runs the walk-forward QR model for each day in a date range and prints a day-by-day report: predicted high, actual high, error, Brier score, trade count from paper_positions.

```python
"""Diagnostic backtest — day-by-day model accuracy report.

Usage:
    PYTHONPATH=. python scripts/diagnostic_backtest.py --start 2026-03-12 --end 2026-03-17
"""

import argparse
import math
from datetime import date, timedelta

import duckdb
import numpy as np
from loguru import logger

from services.feature_builder import FeatureBuilder
from services.model import QRModel


def parse_args():
    p = argparse.ArgumentParser(description="Day-by-day model diagnostic")
    p.add_argument("--db", default="data/alphatemp.duckdb")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    return p.parse_args()


def run_diagnostic(db_path, start_date, end_date):
    # type: (str, date, date) -> None
    fb = FeatureBuilder(db_path)
    model = QRModel()

    con = duckdb.connect(db_path, read_only=True)

    # Get actual highs from nws_daily
    actuals = {}
    rows = con.execute("""
        SELECT obs_date, max_temp_f FROM nws_daily
        WHERE station_id = 'KNYC'
          AND obs_date >= ? AND obs_date <= ?
        ORDER BY obs_date
    """, [start_date, end_date]).fetchall()
    for r in rows:
        actuals[r[0]] = r[1]

    # Get paper trading summary per day
    trade_summary = {}
    try:
        trade_rows = con.execute("""
            SELECT event_date::DATE as d,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(COALESCE(pnl_cents, 0)) as total_pnl
            FROM paper_positions
            WHERE event_date::DATE >= ? AND event_date::DATE <= ?
              AND status = 'settled'
            GROUP BY 1
        """, [start_date, end_date]).fetchall()
        for r in trade_rows:
            trade_summary[r[0]] = {"trades": r[1], "wins": r[2], "pnl_cents": r[3]}
    except Exception:
        pass  # table may not exist or be empty

    con.close()

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC BACKTEST: {} to {}".format(start_date, end_date))
    print("=" * 80)
    print("{:>10} {:>8} {:>8} {:>6} {:>8} {:>8} {:>8} {:>10}".format(
        "Date", "Predict", "Actual", "Error", "Trades", "Wins", "PnL($)", "RunHour"))
    print("-" * 80)

    current = start_date
    total_abs_error = 0.0
    days_counted = 0

    while current <= end_date:
        actual = actuals.get(current)
        if actual is None:
            current += timedelta(days=1)
            continue

        # Run model at update_hour=12 (midday — representative)
        train_result = fb.get_training_data(current, 12)
        if train_result is None:
            print("{:>10} {:>8} {:>8} {:>6} — insufficient training data".format(
                current.isoformat(), "N/A", actual, "N/A"))
            current += timedelta(days=1)
            continue

        X_train, y_train, _, run_hour = train_result
        date_key = current.isoformat()
        coefficients = model.fit(X_train, y_train, run_hour=run_hour, date_key=date_key)
        if coefficients is None:
            current += timedelta(days=1)
            continue

        feat_result = fb.build_features(current, 12)
        if feat_result is None:
            current += timedelta(days=1)
            continue

        features, fcst_high, _ = feat_result
        bracket_probs = model.predict_bracket_probs(
            features, fcst_high, run_hour=run_hour, date_key=date_key
        )

        # Predicted high = weighted sum of bracket centers
        if bracket_probs:
            predicted = sum(k * p for k, p in bracket_probs.items())
        else:
            predicted = fcst_high

        error = predicted - actual
        total_abs_error += abs(error)
        days_counted += 1

        ts = trade_summary.get(current, {"trades": 0, "wins": 0, "pnl_cents": 0})
        pnl_dollars = ts["pnl_cents"] / 100.0

        print("{:>10} {:>8.1f} {:>8.1f} {:>+6.1f} {:>8} {:>8} {:>8.2f} {:>10}z".format(
            current.isoformat(), predicted, actual, error,
            ts["trades"], ts["wins"], pnl_dollars, run_hour))

        current += timedelta(days=1)

    if days_counted > 0:
        mae = total_abs_error / days_counted
        print("-" * 80)
        print("  MAE: {:.2f}°F over {} days".format(mae, days_counted))
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    run_diagnostic(args.db, start, end)
```

- [ ] **Step 2: Run baseline diagnostic**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python scripts/diagnostic_backtest.py --start 2026-03-12 --end 2026-03-17`

Save the output — this is the "before" baseline.

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnostic_backtest.py
git commit -m "feat: add diagnostic backtest script for day-by-day model accuracy"
```

---

## Chunk 2: Herbie Timeout Fix

### Task 2: Add Timeout to Herbie Download

**Files:**
- Modify: `services/forecast.py:143-198`
- Create: `tests/test_forecast_timeout.py`

The problem: `Herbie.download()` hangs indefinitely on missing GRIB files, blocking the async event loop and freezing all services.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forecast_timeout.py
"""Test Herbie timeout wrapper."""

import signal
import time
from unittest.mock import patch, MagicMock

import pytest

from services.forecast import HRRRFetcher, HERBIE_TIMEOUT_SECONDS


def test_herbie_timeout_constant_exists():
    """Timeout constant should be defined and reasonable."""
    assert HERBIE_TIMEOUT_SECONDS >= 15
    assert HERBIE_TIMEOUT_SECONDS <= 120


def test_fetch_run_skips_on_timeout():
    """fetch_run should skip a forecast hour if Herbie.download hangs."""
    fetcher = HRRRFetcher(db_path=":memory:")

    def slow_download(*a, **kw):
        time.sleep(HERBIE_TIMEOUT_SECONDS + 5)

    from datetime import datetime, timezone
    model_run = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)

    with patch.object(fetcher, '_get_stored_fxx', return_value=set()), \
         patch('services.forecast.Herbie') as MockHerbie, \
         patch('services.forecast.get_connection') as mock_conn:
        mock_instance = MagicMock()
        mock_instance.download.side_effect = slow_download
        MockHerbie.return_value = mock_instance
        mock_conn.return_value = MagicMock()

        # Should complete in ~HERBIE_TIMEOUT_SECONDS, not hang
        start = time.monotonic()
        fetcher.fetch_run(model_run, fxx_range=range(1, 3))
        elapsed = time.monotonic() - start

        # Should have timed out quickly, not waited for slow_download
        assert elapsed < HERBIE_TIMEOUT_SECONDS + 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_forecast_timeout.py -v`

Expected: FAIL (HERBIE_TIMEOUT_SECONDS not defined, test hangs or fails)

- [ ] **Step 3: Implement the timeout wrapper**

In `services/forecast.py`, add the timeout constant and a wrapper function. The approach uses `signal.alarm` on Unix (macOS/Linux) since `Herbie.download()` is a synchronous call running in a thread via `run_in_executor`.

At the top of the file (after imports, before constants):
```python
import signal as _signal

# Timeout for individual Herbie.download() calls — prevents hanging on
# missing GRIB files that block the entire async event loop.
HERBIE_TIMEOUT_SECONDS = 45


class _HerbieTimeout(Exception):
    """Raised when Herbie.download() exceeds timeout."""
    pass


def _alarm_handler(signum, frame):
    raise _HerbieTimeout("Herbie download timed out")
```

Replace the try block in `fetch_run()` (lines 155-172) with:

```python
            try:
                H = Herbie(
                    model_run.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                )
                # Set alarm to prevent indefinite hang on missing GRIB
                old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
                _signal.alarm(HERBIE_TIMEOUT_SECONDS)
                try:
                    grib_path = H.download("TMP:2 m")
                finally:
                    _signal.alarm(0)  # cancel alarm
                    _signal.signal(_signal.SIGALRM, old_handler)
                grbs = pygrib.open(str(grib_path))
                msg = grbs.select(name="2 metre temperature")[0]
                consecutive_misses = 0
            except _HerbieTimeout:
                consecutive_misses += 1
                logger.warning(f"HRRR fxx={fxx} timed out after {HERBIE_TIMEOUT_SECONDS}s for {model_run}")
                if consecutive_misses >= 2:
                    break
                continue
            except Exception as e:
                consecutive_misses += 1
                if consecutive_misses >= 2:
                    logger.debug(f"HRRR {model_run.strftime('%Hz')} not published yet (fxx={fxx}), skipping run")
                    break
                logger.debug(f"HRRR fxx={fxx} not available for {model_run}: {e}")
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_forecast_timeout.py -v`

Expected: PASS

- [ ] **Step 5: Run existing forecast tests**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_forecast.py -v`

Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add services/forecast.py tests/test_forecast_timeout.py
git commit -m "fix: add 45s timeout to Herbie download to prevent event loop freeze"
```

---

## Chunk 3: Running High as QR Feature

### Task 3: Add Running Max Temperature as Model Feature

**Files:**
- Modify: `services/feature_builder.py`
- Modify: `tests/test_feature_builder.py`

Currently `running_max` is only used as a CDF floor in `quantile_model.py`. Adding it as a direct feature lets the QR model learn to shift its entire distribution based on observed temps, not just clamp impossible values.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_feature_builder.py`:

```python
def test_feature_vector_includes_running_max():
    """Feature vector should have 24 elements with running_max as feature #23."""
    fb = FeatureBuilder("data/alphatemp.duckdb")
    assert len(fb.FEATURE_NAMES) == 24
    assert fb.FEATURE_NAMES[23] == 'running_max'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_feature_builder.py::test_feature_vector_includes_running_max -v`

Expected: FAIL (FEATURE_NAMES has 23 elements)

- [ ] **Step 3: Add running_max feature to FeatureBuilder**

In `services/feature_builder.py`:

**Update FEATURE_NAMES** (line 72-96) — add `'running_max'` at the end:
```python
    FEATURE_NAMES = [
        'update_hour',
        'fcst_high',
        'sin_month',
        'cos_month',
        'running_max_div',
        'slope_div',
        'cum_div',
        'ecmwf_spread',
        'diurnal_range',
        'gfs_spread',
        'solar_rad',
        'dp_depression',
        'total_precip',
        'humidity',
        'max_gusts',
        'lag_error',
        'cape',
        'dp_spread',
        'gfs_precip',
        'rain_day',
        'precip_agree',
        'solar_spread',
        'abs(lag_error)',
        'running_max',
    ]
```

**Update training data feature construction** (line 423-447, inside the `for uh in TRAIN_UPDATE_HOURS` loop). After computing `running_max_div`, the raw `running_max` value is already available as `rm`. Add it to the features list:

Replace the features list (lines 423-447):
```python
                # Raw running max — 0 if no observations yet
                raw_running_max = rm_by_hour.get(uh, 0.0) if len(obs_up_to) >= 2 else 0.0

                features = [
                    float(uh),
                    float(fcst_high),
                    sin_m,
                    cos_m,
                    running_max_div,
                    slope_div,
                    cum_div,
                    ecmwf_spread,
                    diurnal_range,
                    gfs_spread,
                    solar_rad,
                    dp_depression,
                    total_precip,
                    humidity,
                    max_gusts,
                    lag_error,
                    cape,
                    dp_spread,
                    gfs_precip,
                    rain_day,
                    precip_agree,
                    solar_spread,
                    abs(lag_error),
                    raw_running_max,
                ]
```

**Update live prediction** (line 664-688 in `_build_features_impl`). The running_max is already computed in this method. Add it to the features array:

Replace the features array construction:
```python
        # Raw running max for the model
        raw_running_max = max(t for _, t in obs_by_hour_et) if len(obs_by_hour_et) >= 2 else 0.0

        features = np.array([
            float(update_hour),
            float(fcst_high),
            sin_m,
            cos_m,
            running_max_div,
            slope_div,
            cum_div,
            ecmwf_spread,
            diurnal_range,
            gfs_spread,
            solar_rad,
            dp_depression,
            total_precip,
            humidity,
            max_gusts,
            lag_error,
            cape,
            dp_spread,
            gfs_precip,
            rain_day,
            precip_agree,
            solar_spread,
            abs(lag_error),
            raw_running_max,
        ], dtype=np.float64)
```

**Update docstring** at top of file — add feature #23 to the list:
```
23. running_max            — observed daily high so far (0 if no obs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_feature_builder.py::test_feature_vector_includes_running_max -v`

Expected: PASS

- [ ] **Step 5: Run all feature builder tests**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_feature_builder.py -v`

Expected: All pass (some may need feature count assertions updated from 23 to 24).

- [ ] **Step 6: Commit**

```bash
git add services/feature_builder.py tests/test_feature_builder.py
git commit -m "feat: add running_max as direct QR model feature (#23)"
```

---

## Chunk 4: Time-Decay GFS/ECMWF Spreads

### Task 4: Apply Time Decay to Stale Forecast Spreads

**Files:**
- Modify: `services/feature_builder.py`
- Modify: `tests/test_feature_builder.py`

GFS and ECMWF 00z forecasts are 14+ hours stale by afternoon. The model should trust observations more as the day progresses. Apply a decay factor to ecmwf_spread (feature #7) and gfs_spread (feature #9) based on update_hour.

Decay formula: `decay = max(0.1, 1.0 - update_hour / 24.0)`
- At hour 0: decay = 1.0 (full weight)
- At hour 6: decay = 0.75
- At hour 12: decay = 0.50
- At hour 18: decay = 0.25
- Floor of 0.1 prevents complete zeroing

- [ ] **Step 1: Write the failing test**

Add to `tests/test_feature_builder.py`:

```python
def test_spread_decay_at_hour_18():
    """GFS/ECMWF spreads should be decayed at later hours."""
    from services.feature_builder import _compute_spread_decay
    # At hour 18, decay should be ~0.25
    assert abs(_compute_spread_decay(18) - 0.25) < 0.01
    # At hour 0, decay should be 1.0
    assert abs(_compute_spread_decay(0) - 1.0) < 0.01
    # Floor at 0.1
    assert _compute_spread_decay(24) >= 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_feature_builder.py::test_spread_decay_at_hour_18 -v`

Expected: FAIL (function doesn't exist)

- [ ] **Step 3: Implement decay**

In `services/feature_builder.py`:

**Add helper function** after the constants block (~line 56):
```python
def _compute_spread_decay(update_hour):
    # type: (int) -> float
    """Decay factor for GFS/ECMWF spread features.

    00z forecasts become stale as the day progresses.
    Returns 1.0 at hour 0, decaying to 0.1 at hour 24.
    """
    return max(0.1, 1.0 - update_hour / 24.0)
```

**Apply decay in training loop** (inside `for uh in TRAIN_UPDATE_HOURS`). After computing `ecmwf_spread` and `gfs_spread`, multiply by decay:

```python
            # Decay stale 00z forecast spreads by time of day
            decay = _compute_spread_decay(uh)
            ecmwf_spread = ecmwf_spread * decay
            gfs_spread = gfs_spread * decay
```

Insert this right before building the `features` list (before line 423).

**Apply decay in live prediction** (`_build_features_impl`). After computing `ecmwf_spread` (line 542) and `gfs_spread` (line 554):

```python
        # Decay stale 00z forecast spreads by time of day
        decay = _compute_spread_decay(update_hour)
        ecmwf_spread = ecmwf_spread * decay
        gfs_spread = gfs_spread * decay
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_feature_builder.py::test_spread_decay_at_hour_18 -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/feature_builder.py tests/test_feature_builder.py
git commit -m "feat: apply time-decay to GFS/ECMWF spread features"
```

---

## Chunk 5: Strategy Engine Fixes

### Task 5: Execute Approved Strategy Engine Spec

**Files:**
- Modify: `services/strategy_engine.py`
- Modify: `services/circuit_breakers.py`
- Modify: `services/paper_trader.py`
- Modify: `tests/test_strategy_engine.py`

This task implements the approved spec at `docs/superpowers/specs/2026-03-14-strategy-engine-fixes-design.md`. Three fixes:
1. Enable tail bracket trading (tuple keys)
2. Dedup guard (filter signals matching open positions)
3. Minimum EV filter (block penny bets)

The spec contains exact code for each change — follow it precisely. Key changes:

- [ ] **Step 1: Write failing tests for all three fixes**

Add to `tests/test_strategy_engine.py`:

```python
def test_tail_bracket_aggregation():
    """Lower and upper tail brackets should aggregate correctly."""
    bracket_probs = {40: 0.05, 41: 0.05, 42: 0.10, 50: 0.20, 51: 0.15, 60: 0.05}
    market_prices = {
        (None, 43): {"yes_ask": 10, "no_ask": 90, "yes_bid": 8, "no_bid": 88},
        (50, 51): {"yes_ask": 30, "no_ask": 70, "yes_bid": 28, "no_bid": 68},
        (60, None): {"yes_ask": 5, "no_ask": 95, "yes_bid": 3, "no_bid": 93},
    }
    result = StrategyEngine._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
    # Lower tail (None, 43): probs for 40, 41, 42 = 0.05 + 0.05 + 0.10 = 0.20
    assert abs(result[(None, 43)] - 0.20) < 0.01
    # Upper tail (60, None): probs for k > 60 = 0
    assert result[(60, None)] < 0.01


def test_dedup_guard_filters_existing_positions():
    """Signals matching open positions should be filtered out."""
    signals = [
        {"bracket_floor": 50, "bracket_cap": 52, "direction": "YES",
         "model_prob": 0.4, "market_price": 30, "edge_pct": 10.0},
        {"bracket_floor": 54, "bracket_cap": 56, "direction": "NO",
         "model_prob": 0.1, "market_price": 85, "edge_pct": 5.0},
    ]
    open_positions = [
        {"id": 1, "bracket_floor": 50, "bracket_cap": 52, "direction": "YES", "entry_price": 28},
    ]
    held = {(p["bracket_floor"], p["bracket_cap"], p["direction"]) for p in open_positions}
    filtered = [s for s in signals if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held]
    assert len(filtered) == 1
    assert filtered[0]["bracket_floor"] == 54


def test_ev_filter_blocks_penny_bets():
    """Penny YES bets with negative fee-adjusted EV should be blocked."""
    import math
    # 3c YES, 2% edge
    market_ask = 3
    edge_pct = 2.0
    price_decimal = market_ask / 100.0
    fee_cents = max(math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)), 1)
    ev_cents = edge_pct - fee_cents
    min_ev_cents = 2
    # EV = 2 - 1 = 1c, below min_ev_cents=2 -> blocked
    assert ev_cents < min_ev_cents
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_strategy_engine.py::test_tail_bracket_aggregation tests/test_strategy_engine.py::test_dedup_guard_filters_existing_positions tests/test_strategy_engine.py::test_ev_filter_blocks_penny_bets -v`

Expected: FAIL

- [ ] **Step 3: Implement all three fixes per the approved spec**

Follow the spec exactly at `docs/superpowers/specs/2026-03-14-strategy-engine-fixes-design.md`.

**strategy_engine.py changes:**
1. `_get_market_prices()` — Allow NULL floor/cap, tuple keys
2. `_aggregate_to_kalshi_brackets()` — Tail-aware aggregation (spec has exact code)
3. `_generate_signals()` — Add EV filter, carry bracket_cap in signals
4. `_cycle()` — Add dedup guard between steps 7 and 9
5. `_check_edge_reversals()` — Tuple key lookups
6. Add `import math` at top
7. Add `_load_config()` — also load `min_ev_cents` (default 2)

**circuit_breakers.py changes:**
1. `check()` — Optional[int] params, NULL-safe SQL

**paper_trader.py changes:**
1. `enter_position()` / `_record_entry()` — Accept Optional[int]
2. `_settle_positions()` — Tail bracket settlement logic
3. `_update_unrealized()` — Tail bracket market tick lookup

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/test_strategy_engine.py -v`

Expected: All pass

- [ ] **Step 5: Run full test suite**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All pass (update any assertions broken by feature count change 23→24)

- [ ] **Step 6: Commit**

```bash
git add services/strategy_engine.py services/circuit_breakers.py services/paper_trader.py tests/test_strategy_engine.py
git commit -m "feat: enable tail brackets, dedup guard, and EV filter (spec 2026-03-14)"
```

---

## Chunk 6: Verification Backtest

### Task 6: Re-run Diagnostic and Compare

- [ ] **Step 1: Run the same diagnostic backtest**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python scripts/diagnostic_backtest.py --start 2026-03-12 --end 2026-03-17`

Compare MAE and per-day errors against the baseline from Task 1. The running_max feature and time-decay should improve afternoon predictions.

- [ ] **Step 2: Run the strategy backtest**

Run: `cd ~/Projects/alphatemp/alphatemp && PYTHONPATH=. ../venv/bin/python scripts/run_strategy_backtest.py --start 2026-03-12 --end 2026-03-17 --model cross_hour`

Compare P&L, trade count, and Sharpe against baseline. Trade count should drop dramatically (dedup guard). Tail bracket trades should appear.

- [ ] **Step 3: Document results in HANDOFF.md**

Update `HANDOFF.md` with:
- Before/after MAE comparison
- Before/after trade count and P&L
- Any remaining issues discovered
- Next steps for tomorrow's live run

- [ ] **Step 4: Restart services for live trading**

If results look good, restart the dashboard and services:
```bash
cd ~/Projects/alphatemp/alphatemp
kill $(pgrep -f "python.*main.py") 2>/dev/null
PYTHONPATH=. ../venv/bin/python main.py &
```

- [ ] **Step 5: Final commit**

```bash
git add HANDOFF.md
git commit -m "docs: update handoff with model fix results and verification"
```
