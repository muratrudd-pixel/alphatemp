# Phase 3.6: Neighbor Station Observations — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Use 1-minute ASOS data from KLGA/KEWR to improve Phase 2B divergence features and push the obs crossover earlier than 14 ET.

**Architecture:** New `services/neighbor_obs.py` module provides neighbor data processing (offset learning, feature computation, blended curves, peak signal). Backtester's level-1 cache is extended to include neighbor obs. Six new model factories (3 station-mapping x 2 integration) plug into the existing backtester via `ModelFn`. Analysis script runs all variants across 0-18 ET.

**Tech Stack:** Python 3.9, DuckDB, scipy (lstsq), numpy, existing backtester infrastructure

**Design doc:** `docs/plans/2026-02-26-neighbor-obs-design.md`

---

## Task 1: Neighbor Obs Module — Station Offset Learning

Create the foundation module with walk-forward offset computation between neighbor and settlement stations.

**Files:**
- Create: `services/neighbor_obs.py`
- Create: `tests/test_neighbor_obs.py`

**Step 1: Write the failing tests**

```python
# tests/test_neighbor_obs.py
"""Tests for neighbor station observation processing."""
import pytest
from datetime import datetime, timedelta
from services.neighbor_obs import compute_walk_forward_offset


class TestStationOffset:
    """Walk-forward expanding-window offset between neighbor and KNYC."""

    def test_constant_offset(self):
        """If neighbor is always 2F warmer, offset should converge to 2.0."""
        # 100 matching hourly observations over ~4 days
        base = datetime(2023, 6, 1, 12, 0)
        knyc_obs = [(base + timedelta(hours=i), 70.0 + i * 0.1) for i in range(100)]
        neighbor_obs = [(base + timedelta(hours=i), 72.0 + i * 0.1) for i in range(100)]
        offset = compute_walk_forward_offset(knyc_obs, neighbor_obs, min_pairs=30)
        assert offset is not None
        assert abs(offset - 2.0) < 0.01

    def test_insufficient_pairs_returns_none(self):
        """If fewer than min_pairs overlapping obs, return None."""
        base = datetime(2023, 6, 1, 12, 0)
        knyc_obs = [(base + timedelta(hours=i), 70.0) for i in range(10)]
        neighbor_obs = [(base + timedelta(hours=i), 72.0) for i in range(10)]
        offset = compute_walk_forward_offset(knyc_obs, neighbor_obs, min_pairs=30)
        assert offset is None

    def test_no_overlapping_times(self):
        """If obs don't overlap in time, return None."""
        base = datetime(2023, 6, 1, 12, 0)
        knyc_obs = [(base + timedelta(hours=i), 70.0) for i in range(50)]
        neighbor_obs = [(base + timedelta(hours=100 + i), 72.0) for i in range(50)]
        offset = compute_walk_forward_offset(knyc_obs, neighbor_obs, min_pairs=30)
        assert offset is None

    def test_time_matching_tolerance(self):
        """Obs within 5 minutes should be paired."""
        base = datetime(2023, 6, 1, 12, 0)
        knyc_obs = [(base + timedelta(hours=i), 70.0) for i in range(50)]
        # Neighbor reports 3 minutes after KNYC
        neighbor_obs = [(base + timedelta(hours=i, minutes=3), 73.0) for i in range(50)]
        offset = compute_walk_forward_offset(knyc_obs, neighbor_obs, min_pairs=30)
        assert offset is not None
        assert abs(offset - 3.0) < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.neighbor_obs'`

**Step 3: Write minimal implementation**

```python
# services/neighbor_obs.py
"""Neighbor station observation processing for Phase 3.6.

Provides offset learning, divergence features, peak signal detection,
blended curve construction, and trend extraction from KLGA/KEWR data
to enhance Phase 2B predictions.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# Type alias: list of (timestamp, temp_f) tuples
ObsSeries = List[Tuple[datetime, float]]

# Maximum time difference (seconds) to consider two obs as simultaneous
_MATCH_TOLERANCE_S = 300  # 5 minutes


def compute_walk_forward_offset(
    knyc_obs: ObsSeries,
    neighbor_obs: ObsSeries,
    min_pairs: int = 30,
) -> Optional[float]:
    """Compute expanding-window mean offset: neighbor - KNYC.

    Pairs observations by timestamp (within 5-minute tolerance).
    Returns mean(neighbor_temp - knyc_temp) or None if insufficient pairs.
    """
    if not knyc_obs or not neighbor_obs:
        return None

    # Build lookup: round neighbor timestamps to nearest minute
    neighbor_by_minute = {}
    for ts, temp in neighbor_obs:
        key = ts.replace(second=0, microsecond=0)
        neighbor_by_minute[key] = temp

    diffs = []
    for knyc_ts, knyc_temp in knyc_obs:
        knyc_rounded = knyc_ts.replace(second=0, microsecond=0)
        # Check +/- tolerance window
        for offset_min in range(-5, 6):
            candidate = knyc_rounded + timedelta(minutes=offset_min)
            if candidate in neighbor_by_minute:
                diffs.append(neighbor_by_minute[candidate] - knyc_temp)
                break

    if len(diffs) < min_pairs:
        return None

    return sum(diffs) / len(diffs)
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add alphatemp/services/neighbor_obs.py alphatemp/tests/test_neighbor_obs.py
git commit -m "feat(phase3.6): add neighbor_obs module with walk-forward offset"
```

---

## Task 2: Neighbor Divergence Features

Add functions to compute divergence features from neighbor station obs, with optional offset correction.

**Files:**
- Modify: `services/neighbor_obs.py`
- Modify: `tests/test_neighbor_obs.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_neighbor_obs.py
from services.neighbor_obs import compute_neighbor_divergence


class TestNeighborDivergence:
    """Divergence features from neighbor obs vs forecast."""

    def test_raw_divergence_no_offset(self):
        """A variant: use raw neighbor temps, no offset correction."""
        # Neighbor obs: 1-minute data, 60 readings over 1 hour
        base_ts = 1000000.0
        obs_temps = [72.0 + i * 0.05 for i in range(60)]
        obs_timestamps = [base_ts + i * 60 for i in range(60)]
        # Forecast curve: hourly, 70F flat
        fcst_timestamps = [base_ts, base_ts + 3600]
        fcst_temps = [70.0, 70.0]

        result = compute_neighbor_divergence(
            obs_temps, obs_timestamps, fcst_timestamps, fcst_temps, offset=0.0
        )
        assert result is not None
        assert "neighbor_running_max" in result
        assert "neighbor_instant" in result
        assert "neighbor_cumul" in result
        assert "neighbor_slope" in result
        assert "n_neighbor_obs" in result
        # Running max: max(obs) = 72.0 + 59*0.05 = 74.95, max(fcst) = 70.0
        assert abs(result["neighbor_running_max"] - 4.95) < 0.1

    def test_with_offset_correction(self):
        """B variant: apply offset before computing divergence."""
        base_ts = 1000000.0
        # Neighbor reads 2F warmer than KNYC systematically
        obs_temps = [72.0] * 60
        obs_timestamps = [base_ts + i * 60 for i in range(60)]
        fcst_timestamps = [base_ts, base_ts + 3600]
        fcst_temps = [70.0, 70.0]

        result = compute_neighbor_divergence(
            obs_temps, obs_timestamps, fcst_timestamps, fcst_temps, offset=2.0
        )
        # After offset correction: obs become 70.0, forecast is 70.0
        # running_max should be near 0
        assert result is not None
        assert abs(result["neighbor_running_max"]) < 0.1

    def test_insufficient_obs_returns_none(self):
        """Need at least 2 obs for divergence features."""
        result = compute_neighbor_divergence(
            [72.0], [1000000.0], [1000000.0], [70.0], offset=0.0
        )
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py::TestNeighborDivergence -v`
Expected: FAIL — `ImportError: cannot import name 'compute_neighbor_divergence'`

**Step 3: Write implementation**

```python
# Add to services/neighbor_obs.py
from services.divergence import interpolate_forecast, compute_divergence_features


def compute_neighbor_divergence(
    obs_temps: List[float],
    obs_timestamps: List[float],
    fcst_timestamps: List[float],
    fcst_temps: List[float],
    offset: float = 0.0,
) -> Optional[dict]:
    """Compute divergence features from neighbor obs vs forecast.

    Args:
        obs_temps: Neighbor observed temperatures (F).
        obs_timestamps: Epoch seconds for each obs.
        fcst_timestamps: Forecast curve timestamps (epoch seconds).
        fcst_temps: Forecast curve temperatures (F).
        offset: Station offset to subtract from neighbor temps (neighbor - KNYC).
                 Pass 0.0 for raw (A variants), learned offset for (B variants).

    Returns:
        Dict with neighbor_ prefixed divergence features, or None if < 2 obs.
    """
    if len(obs_temps) < 2:
        return None

    # Apply offset correction: shift neighbor temps toward KNYC
    corrected_temps = [t - offset for t in obs_temps]

    # Interpolate forecast to obs timestamps
    fcst_interp = interpolate_forecast(fcst_timestamps, fcst_temps, obs_timestamps)

    # Compute hours since first obs (for slope regression)
    t0 = obs_timestamps[0]
    obs_hours = [(t - t0) / 3600.0 for t in obs_timestamps]

    # Forecast curve up to last obs timestamp
    last_obs_ts = obs_timestamps[-1]
    fcst_up_to_t = [
        fcst_temps[i]
        for i in range(len(fcst_timestamps))
        if fcst_timestamps[i] <= last_obs_ts
    ]
    if not fcst_up_to_t:
        fcst_up_to_t = fcst_temps[:1]

    raw = compute_divergence_features(corrected_temps, fcst_interp, obs_hours, fcst_up_to_t)
    if raw is None:
        return None

    # Prefix keys with neighbor_
    return {
        "neighbor_running_max": raw["running_max_divergence"],
        "neighbor_instant": raw["temp_divergence"],
        "neighbor_cumul": raw["cumulative_divergence"],
        "neighbor_slope": raw["slope_divergence"],
        "n_neighbor_obs": raw["n_obs"],
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add alphatemp/services/neighbor_obs.py alphatemp/tests/test_neighbor_obs.py
git commit -m "feat(phase3.6): add neighbor divergence feature computation"
```

---

## Task 3: Neighbor Peak Signal

Detect when neighbor stations start declining — leading indicator that the daily high has passed.

**Files:**
- Modify: `services/neighbor_obs.py`
- Modify: `tests/test_neighbor_obs.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_neighbor_obs.py
from services.neighbor_obs import compute_peak_signal


class TestPeakSignal:
    """Detect when neighbor temps start declining."""

    def test_still_rising(self):
        """Temps still going up — peak not yet reached."""
        # Steadily rising: 70, 71, 72, 73, 74
        temps = [70.0 + i for i in range(5)]
        timestamps = [1000000.0 + i * 300 for i in range(5)]  # 5-min intervals
        result = compute_peak_signal(temps, timestamps)
        assert result["peak_passed"] is False
        assert result["minutes_since_peak_update"] == 0.0

    def test_declining(self):
        """Temps peaked and now falling — peak has passed."""
        # Rise then fall: 70, 72, 74, 73, 71
        temps = [70.0, 72.0, 74.0, 73.0, 71.0]
        timestamps = [1000000.0 + i * 300 for i in range(5)]
        result = compute_peak_signal(temps, timestamps)
        assert result["peak_passed"] is True
        # Peak was at index 2 (timestamp 1000600), last obs at index 4 (1001200)
        assert abs(result["minutes_since_peak_update"] - 10.0) < 0.1
        assert result["decline_rate"] < 0  # negative = cooling

    def test_plateau(self):
        """Temps flat at the top — ambiguous, not declining."""
        temps = [70.0, 72.0, 74.0, 74.0, 74.0]
        timestamps = [1000000.0 + i * 300 for i in range(5)]
        result = compute_peak_signal(temps, timestamps)
        assert result["peak_passed"] is False
        assert result["minutes_since_peak_update"] == 0.0

    def test_insufficient_data(self):
        """Need at least 3 observations."""
        result = compute_peak_signal([70.0, 71.0], [1000000.0, 1000300.0])
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py::TestPeakSignal -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# Add to services/neighbor_obs.py

def compute_peak_signal(
    temps: List[float],
    timestamps: List[float],
    decline_threshold: float = 0.0,
) -> Optional[dict]:
    """Detect whether neighbor station temps have started declining.

    Args:
        temps: Observed temperatures (F), chronological.
        timestamps: Epoch seconds for each observation.
        decline_threshold: Temp must drop this much below peak to count as declining.
                           Default 0.0 means any drop triggers.

    Returns:
        Dict with:
            peak_passed (bool): True if temps are declining from peak.
            minutes_since_peak_update (float): Minutes since running_max last increased.
                                                0.0 if still rising or at peak.
            decline_rate (float): Rate of temp decline (F/min) from peak to latest obs.
                                   Negative means cooling. 0.0 if not declining.
        Returns None if < 3 observations.
    """
    if len(temps) < 3:
        return None

    # Find when the running max last increased
    running_max = temps[0]
    last_peak_idx = 0
    for i in range(1, len(temps)):
        if temps[i] > running_max:
            running_max = temps[i]
            last_peak_idx = i

    latest_idx = len(temps) - 1
    latest_temp = temps[latest_idx]

    # Peak has passed if: we're past the peak index AND temp has dropped below peak
    is_declining = (
        last_peak_idx < latest_idx
        and (running_max - latest_temp) > decline_threshold
    )

    if is_declining:
        elapsed_s = timestamps[latest_idx] - timestamps[last_peak_idx]
        minutes_since = elapsed_s / 60.0
        decline_rate = (latest_temp - running_max) / max(minutes_since, 0.01)
    else:
        minutes_since = 0.0
        decline_rate = 0.0

    return {
        "peak_passed": is_declining,
        "minutes_since_peak_update": minutes_since,
        "decline_rate": decline_rate,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py -v`
Expected: 11 passed

**Step 5: Commit**

```bash
git add alphatemp/services/neighbor_obs.py alphatemp/tests/test_neighbor_obs.py
git commit -m "feat(phase3.6): add neighbor peak signal detection"
```

---

## Task 4: Blended Curve Construction

Build a single enhanced temperature curve from KNYC + neighbor obs for the x2 integration variants.

**Files:**
- Modify: `services/neighbor_obs.py`
- Modify: `tests/test_neighbor_obs.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_neighbor_obs.py
from services.neighbor_obs import build_blended_curve


class TestBlendedCurve:
    """Build a blended temperature curve from KNYC + neighbor obs."""

    def test_fills_gaps_between_knyc(self):
        """Neighbor obs fill the gap between hourly KNYC reports."""
        base = 1000000.0
        # KNYC reports at :00 and :60 (hourly)
        knyc = [(base, 70.0), (base + 3600, 71.0)]
        # KLGA reports every minute between, reads 2F warmer
        klga = [(base + i * 60, 72.0 + i * (1.0 / 60)) for i in range(1, 60)]

        result = build_blended_curve(knyc, klga, offset=2.0)
        # Should have KNYC endpoints + interpolated neighbor data between
        assert len(result) > 2
        # First and last should be KNYC values (anchored)
        assert result[0] == (base, 70.0)
        assert result[-1] == (base + 3600, 71.0)
        # Middle points should be offset-corrected neighbor temps
        mid = result[len(result) // 2]
        assert mid[0] > base
        assert mid[0] < base + 3600

    def test_knyc_only_when_no_neighbor(self):
        """If no neighbor obs, return KNYC obs unchanged."""
        knyc = [(1000000.0, 70.0), (1003600.0, 71.0)]
        result = build_blended_curve(knyc, [], offset=0.0)
        assert len(result) == 2
        assert result[0] == knyc[0]
        assert result[1] == knyc[1]

    def test_anchors_on_knyc_reports(self):
        """At KNYC report times, blended curve uses KNYC temp, not neighbor."""
        base = 1000000.0
        knyc = [(base, 70.0), (base + 3600, 71.0)]
        # Neighbor reads 5F warmer at the same time as KNYC
        klga = [(base, 75.0), (base + 1800, 76.0), (base + 3600, 76.0)]
        result = build_blended_curve(knyc, klga, offset=0.0)
        # Endpoints should be KNYC, not neighbor
        assert result[0] == (base, 70.0)
        assert result[-1] == (base + 3600, 71.0)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py::TestBlendedCurve -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# Add to services/neighbor_obs.py

def build_blended_curve(
    knyc_obs: List[Tuple[float, float]],
    neighbor_obs: List[Tuple[float, float]],
    offset: float = 0.0,
) -> List[Tuple[float, float]]:
    """Build a blended temperature curve anchored on KNYC.

    At KNYC report times, uses KNYC temperature exactly.
    Between KNYC reports, fills in with offset-corrected neighbor data.

    Args:
        knyc_obs: [(epoch_seconds, temp_f), ...] sorted by time.
        neighbor_obs: [(epoch_seconds, temp_f), ...] sorted by time.
        offset: Neighbor-KNYC offset to subtract from neighbor temps.

    Returns:
        [(epoch_seconds, temp_f), ...] sorted by time. KNYC values at anchor
        points, offset-corrected neighbor values between.
    """
    if not knyc_obs:
        return []
    if not neighbor_obs:
        return list(knyc_obs)

    knyc_times = set(ts for ts, _ in knyc_obs)
    combined = []

    # Add all KNYC obs as anchors
    for ts, temp in knyc_obs:
        combined.append((ts, temp))

    # Add offset-corrected neighbor obs that fall between KNYC reports
    first_knyc = knyc_obs[0][0]
    last_knyc = knyc_obs[-1][0]
    for ts, temp in neighbor_obs:
        if ts < first_knyc or ts > last_knyc:
            continue
        # Skip if within tolerance of a KNYC report time
        is_knyc_time = any(abs(ts - kt) < _MATCH_TOLERANCE_S for kt in knyc_times)
        if is_knyc_time:
            continue
        combined.append((ts, temp - offset))

    combined.sort(key=lambda x: x[0])
    return combined
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py -v`
Expected: 14 passed

**Step 5: Commit**

```bash
git add alphatemp/services/neighbor_obs.py alphatemp/tests/test_neighbor_obs.py
git commit -m "feat(phase3.6): add blended curve construction"
```

---

## Task 5: Trend-Only Extraction (C Variants)

Extract slope and acceleration from neighbor obs between KNYC reports.

**Files:**
- Modify: `services/neighbor_obs.py`
- Modify: `tests/test_neighbor_obs.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_neighbor_obs.py
from services.neighbor_obs import compute_neighbor_trend


class TestNeighborTrend:
    """Extract rate-of-change from neighbor obs between KNYC reports."""

    def test_rising_trend(self):
        """Neighbor temps rising steadily -> positive slope."""
        # 30 minutes of 1-min data, rising 0.1F/min
        timestamps = [1000000.0 + i * 60 for i in range(30)]
        temps = [70.0 + i * 0.1 for i in range(30)]
        result = compute_neighbor_trend(temps, timestamps)
        assert result is not None
        # slope ~0.1 F/min = 6 F/hr
        assert abs(result["trend_slope_f_per_hr"] - 6.0) < 0.5

    def test_accelerating(self):
        """Temps rising faster and faster -> positive acceleration."""
        timestamps = [1000000.0 + i * 60 for i in range(30)]
        # Quadratic rise: acceleration
        temps = [70.0 + 0.01 * i * i for i in range(30)]
        result = compute_neighbor_trend(temps, timestamps)
        assert result is not None
        assert result["trend_accel_f_per_hr2"] > 0

    def test_insufficient_data(self):
        """Need at least 3 obs for slope, 5 for acceleration."""
        result = compute_neighbor_trend([70.0, 71.0], [1000000.0, 1000060.0])
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py::TestNeighborTrend -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

```python
# Add to services/neighbor_obs.py
import numpy as np


def compute_neighbor_trend(
    temps: List[float],
    timestamps: List[float],
) -> Optional[dict]:
    """Compute rate-of-change features from neighbor observations.

    For C variants: uses only the trajectory of neighbor temps,
    not their absolute values. Sidesteps the station offset problem.

    Args:
        temps: Neighbor temperatures (F), chronological.
        timestamps: Epoch seconds for each observation.

    Returns:
        Dict with:
            trend_slope_f_per_hr: Linear slope in F/hour.
            trend_accel_f_per_hr2: Acceleration in F/hour^2 (from quadratic fit).
        Returns None if < 3 observations.
    """
    if len(temps) < 3:
        return None

    # Convert timestamps to hours from first obs
    t0 = timestamps[0]
    hours = np.array([(t - t0) / 3600.0 for t in timestamps])
    y = np.array(temps)

    # Linear fit for slope
    A_lin = np.column_stack([np.ones(len(hours)), hours])
    result_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
    slope = result_lin[1]  # F/hour

    # Quadratic fit for acceleration (if enough data)
    accel = 0.0
    if len(temps) >= 5:
        A_quad = np.column_stack([np.ones(len(hours)), hours, hours ** 2])
        result_quad, _, _, _ = np.linalg.lstsq(A_quad, y, rcond=None)
        accel = 2.0 * result_quad[2]  # second derivative of ax^2 = 2a

    return {
        "trend_slope_f_per_hr": float(slope),
        "trend_accel_f_per_hr2": float(accel),
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_neighbor_obs.py -v`
Expected: 17 passed

**Step 5: Commit**

```bash
git add alphatemp/services/neighbor_obs.py alphatemp/tests/test_neighbor_obs.py
git commit -m "feat(phase3.6): add trend-only extraction for C variants"
```

---

## Task 6: Extend Backtester Level-1 Cache for Neighbor Obs

Add neighbor observation data to the bulk cache so Phase 2B neighbor models can access it efficiently.

**Files:**
- Modify: `services/backtester.py` (extend `_ensure_level1`)
- Modify: `tests/test_phase2b_models.py` (add neighbor obs to test seed data)

**Step 1: Write the failing test**

```python
# Add to tests/test_phase2b_models.py
from services.backtester import _ensure_level1, _p2b_level1


class TestNeighborObsCache:
    """Level-1 cache includes neighbor station observations."""

    def test_level1_includes_neighbor_obs(self, test_db):
        """Level-1 cache should have neighbor_obs key with KLGA and KEWR data."""
        _seed_phase2b_data(TEST_DB, n_days=100, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=100)  # new helper

        con = duckdb.connect(TEST_DB, read_only=True)
        try:
            data = _ensure_level1(con, 12, "KNYC", "hrrr")
            assert "neighbor_obs" in data
            assert "KLGA" in data["neighbor_obs"]
            assert "KEWR" in data["neighbor_obs"]
            # Neighbor obs should be organized by date
            some_date = list(data["neighbor_obs"]["KLGA"].keys())[0]
            assert len(data["neighbor_obs"]["KLGA"][some_date]) > 0
        finally:
            con.close()
```

**Step 2: Run test to verify it fails**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_phase2b_models.py::TestNeighborObsCache -v`
Expected: FAIL — `NameError: name '_seed_neighbor_obs' is not defined` or missing key

**Step 3: Add test helper to seed neighbor obs**

Add to `tests/test_phase2b_models.py`:
```python
def _seed_neighbor_obs(db_path, n_days=200, run_hour=12):
    """Seed KLGA and KEWR observations mirroring KNYC but at 1-min frequency."""
    con = duckdb.connect(db_path)
    base_date = datetime(2023, 1, 1)
    rows = []
    for station, offset in [("KLGA", 1.5), ("KEWR", 2.0)]:
        for d in range(n_days):
            obs_date = base_date + timedelta(days=d)
            day_of_year = obs_date.timetuple().tm_yday
            actual = 55 + 25 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            # 1-min obs from 05z to 23z (every 10 min to keep test data small)
            for hour in range(5, 24):
                for minute in [0, 10, 20, 30, 40, 50]:
                    obs_time = obs_date.replace(hour=hour, minute=minute)
                    # Temp follows diurnal cycle + station offset
                    hour_frac = hour + minute / 60.0
                    diurnal = -3.0 * abs(hour_frac - 19.0) + actual + offset
                    rows.append((station, obs_time, diurnal, obs_time, "test"))
    con.executemany(
        "INSERT OR IGNORE INTO observations (station_id, observed_at, temp_f, ingested_at, ingest_source) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    con.close()
```

**Step 4: Extend `_ensure_level1` in `services/backtester.py`**

After the existing obs fetch (around line 804), add neighbor obs fetch:

```python
# --- Neighbor observations (Phase 3.6) ---
neighbor_obs = {}  # station_id -> date -> [(ts, temp_f)]
neighbor_stations = ["KLGA", "KEWR"]
for nbr_station in neighbor_stations:
    nbr_rows = con.execute(
        """
        SELECT observed_at, temp_f
        FROM observations
        WHERE station_id = ?
          AND observed_at BETWEEN ? AND ?
          AND temp_f IS NOT NULL
        ORDER BY observed_at
        """,
        [nbr_station, min_date_utc, max_date_utc],
    ).fetchall()

    nbr_by_date = {}  # type: dict
    for obs_at, temp_f in nbr_rows:
        # Convert to ET date for grouping
        et_dt = obs_at - timedelta(hours=5)
        d = et_dt.date()
        nbr_by_date.setdefault(d, []).append(
            (obs_at.timestamp(), temp_f)
        )
    neighbor_obs[nbr_station] = nbr_by_date
```

Add `"neighbor_obs": neighbor_obs` to the returned dict.

**Step 5: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_phase2b_models.py -v`
Expected: All existing tests pass + new test passes

**Step 6: Commit**

```bash
git add alphatemp/services/backtester.py alphatemp/tests/test_phase2b_models.py
git commit -m "feat(phase3.6): extend level-1 cache with neighbor obs"
```

---

## Task 7: Neighbor Model Factories — "New Features" Variants (A1, B1, C1)

Create model factories that add neighbor-derived features alongside existing KNYC features.

**Files:**
- Modify: `services/backtester.py` (add new model factories)
- Modify: `tests/test_phase2b_models.py` (test new models return valid probs)

**Step 1: Write the failing tests**

```python
# Add to tests/test_phase2b_models.py

class TestNeighborModelsNewFeatures:
    """A1/B1/C1 variants: neighbor features alongside KNYC features."""

    def test_a1_returns_valid_probs(self, test_db):
        """A1 (raw neighbor + new features) returns valid bracket probs."""
        _seed_phase2b_data(TEST_DB, n_days=200, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=200)
        bt = Backtester(TEST_DB, "NYC")
        result = bt.run(
            wf_phase2b_neighbor_a1,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 10),
            run_hours=[12],
            update_hours_et=[14, 16, 18],
        )
        assert result.mean_brier > 0
        assert result.mean_brier < 2.0  # sanity

    def test_b1_returns_valid_probs(self, test_db):
        """B1 (learned offset + new features) returns valid bracket probs."""
        _seed_phase2b_data(TEST_DB, n_days=200, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=200)
        bt = Backtester(TEST_DB, "NYC")
        result = bt.run(
            wf_phase2b_neighbor_b1,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 10),
            run_hours=[12],
            update_hours_et=[14, 16, 18],
        )
        assert result.mean_brier > 0

    def test_c1_returns_valid_probs(self, test_db):
        """C1 (trend only + new features) returns valid bracket probs."""
        _seed_phase2b_data(TEST_DB, n_days=200, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=200)
        bt = Backtester(TEST_DB, "NYC")
        result = bt.run(
            wf_phase2b_neighbor_c1,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 10),
            run_hours=[12],
            update_hours_et=[14, 16, 18],
        )
        assert result.mean_brier > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_phase2b_models.py::TestNeighborModelsNewFeatures -v`
Expected: FAIL — `ImportError`

**Step 3: Implement model factories**

Add to `services/backtester.py`:

```python
# Phase 3.6 neighbor feature keys (appended after the 4 KNYC features)
_PHASE2B_NEIGHBOR_FEATURE_KEYS = [
    "neighbor_running_max",
    "neighbor_instant",
    "neighbor_cumul",
    "neighbor_slope",
    "peak_passed",              # 1.0 or 0.0
    "minutes_since_peak_update",
    "decline_rate",
    "trend_slope_f_per_hr",
    "trend_accel_f_per_hr2",
]


def _compute_neighbor_features_for_date(
    l1_data, obs_date, cutoff_ts, curve, offset_klga, offset_kewr, variant,
):
    """Compute neighbor-derived features for one date.

    Args:
        l1_data: Level-1 cache data (includes neighbor_obs).
        obs_date: The date being evaluated.
        cutoff_ts: Epoch seconds cutoff for walk-forward safety.
        curve: Forecast curve [(ts, temp_f), ...] for this date.
        offset_klga: KLGA->KNYC offset (0.0 for A variant).
        offset_kewr: KEWR->KNYC offset (0.0 for A variant).
        variant: "A", "B", or "C".

    Returns:
        List of floats (9 features) or None if insufficient data.
    """
    from services.neighbor_obs import (
        compute_neighbor_divergence,
        compute_peak_signal,
        compute_neighbor_trend,
    )

    neighbor_obs = l1_data.get("neighbor_obs", {})
    fcst_ts = [c[0] for c in curve]
    fcst_temps = [c[1] for c in curve]

    # Collect and merge neighbor obs up to cutoff
    all_neighbor_temps = []
    all_neighbor_ts = []
    for station, offset in [("KLGA", offset_klga), ("KEWR", offset_kewr)]:
        day_obs = neighbor_obs.get(station, {}).get(obs_date, [])
        for ts, temp in day_obs:
            if ts <= cutoff_ts:
                all_neighbor_temps.append(temp)
                all_neighbor_ts.append(ts)

    if len(all_neighbor_temps) < 2:
        return None

    # Sort by timestamp
    paired = sorted(zip(all_neighbor_ts, all_neighbor_temps))
    sorted_ts = [p[0] for p in paired]
    sorted_temps = [p[1] for p in paired]

    # Deduplicate by timestamp (keep first)
    deduped_ts = []
    deduped_temps = []
    for i, (t, tmp) in enumerate(zip(sorted_ts, sorted_temps)):
        if i == 0 or t != deduped_ts[-1]:
            deduped_ts.append(t)
            deduped_temps.append(tmp)
    sorted_ts = deduped_ts
    sorted_temps = deduped_temps

    avg_offset = (offset_klga + offset_kewr) / 2.0

    features = [0.0] * 9

    if variant in ("A", "B"):
        div = compute_neighbor_divergence(
            sorted_temps, sorted_ts, fcst_ts, fcst_temps, offset=avg_offset,
        )
        if div:
            features[0] = div["neighbor_running_max"]
            features[1] = div["neighbor_instant"]
            features[2] = div["neighbor_cumul"]
            features[3] = div["neighbor_slope"]

    # Peak signal (always computed — useful for all variants)
    peak = compute_peak_signal(sorted_temps, sorted_ts)
    if peak:
        features[4] = 1.0 if peak["peak_passed"] else 0.0
        features[5] = peak["minutes_since_peak_update"]
        features[6] = peak["decline_rate"]

    if variant == "C":
        trend = compute_neighbor_trend(sorted_temps, sorted_ts)
        if trend:
            features[7] = trend["trend_slope_f_per_hr"]
            features[8] = trend["trend_accel_f_per_hr2"]

    return features


def _make_phase2b_neighbor_model(name, variant, model_name="hrrr"):
    """Factory for Phase 3.6 neighbor models (x1 integration: new features).

    Creates a model that uses KNYC Phase 2B features (4) + neighbor features (9)
    = 13 total regression features.

    Args:
        name: Model name for display.
        variant: "A" (raw), "B" (learned offset), or "C" (trend only).
        model_name: "hrrr", "gfs", or "ecmwf".
    """
    # Feature indices: 0-3 are KNYC, 4-12 are neighbor
    # A/B use neighbor divergence (4-6) + peak (7-9)
    # C uses peak (7-9) + trend (10-11)
    all_indices = list(range(13))

    def _raw(provider, ref_time):
        # ... (follows same pattern as _make_phase2b_model inner function)
        # Get KNYC features from existing Phase 2B
        # Get neighbor features from _compute_neighbor_features_for_date
        # Concatenate into single feature vector
        # Run OLS on extended feature vector
        pass  # Full implementation follows existing _make_phase2b_model pattern

    # ... (wraps _raw same as existing models)
```

Note: The full implementation follows the pattern of the existing `_make_phase2b_model` at lines 943-1040 of backtester.py. The key difference is:
1. After computing KNYC features (4 values), also compute neighbor features (9 values)
2. Concatenate into a 13-element feature vector
3. For training rows: same walk-forward loop, but build 13-element vectors for each historical date
4. OLS regression on the extended vector
5. For variant B: compute walk-forward offset from historical overlapping KNYC/neighbor obs

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_phase2b_models.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add alphatemp/services/backtester.py alphatemp/tests/test_phase2b_models.py
git commit -m "feat(phase3.6): add A1/B1/C1 neighbor model factories"
```

---

## Task 8: Neighbor Model Factories — "Blended Curve" Variants (A2, B2, C2)

Create model factories that build a blended curve and compute standard divergence from it.

**Files:**
- Modify: `services/backtester.py`
- Modify: `tests/test_phase2b_models.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_phase2b_models.py

class TestNeighborModelsBlended:
    """A2/B2/C2 variants: blended curve replaces KNYC-only divergence."""

    def test_a2_returns_valid_probs(self, test_db):
        _seed_phase2b_data(TEST_DB, n_days=200, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=200)
        bt = Backtester(TEST_DB, "NYC")
        result = bt.run(
            wf_phase2b_neighbor_a2,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 10),
            run_hours=[12],
            update_hours_et=[14, 16, 18],
        )
        assert result.mean_brier > 0
        assert result.mean_brier < 2.0

    def test_b2_returns_valid_probs(self, test_db):
        _seed_phase2b_data(TEST_DB, n_days=200, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=200)
        bt = Backtester(TEST_DB, "NYC")
        result = bt.run(
            wf_phase2b_neighbor_b2,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 10),
            run_hours=[12],
            update_hours_et=[14, 16, 18],
        )
        assert result.mean_brier > 0

    def test_c2_returns_valid_probs(self, test_db):
        _seed_phase2b_data(TEST_DB, n_days=200, run_hour=12)
        _seed_neighbor_obs(TEST_DB, n_days=200)
        bt = Backtester(TEST_DB, "NYC")
        result = bt.run(
            wf_phase2b_neighbor_c2,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 10),
            run_hours=[12],
            update_hours_et=[14, 16, 18],
        )
        assert result.mean_brier > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_phase2b_models.py::TestNeighborModelsBlended -v`
Expected: FAIL — `ImportError`

**Step 3: Implement blended curve model factories**

Add to `services/backtester.py`:

```python
def _make_phase2b_blended_model(name, variant, model_name="hrrr"):
    """Factory for Phase 3.6 blended models (x2 integration: blended curve).

    Instead of adding new features, builds a blended KNYC+neighbor temperature
    curve and computes the standard 4 divergence features from it. The blended
    curve has higher temporal resolution, giving running_max and other features
    more data points to work with.

    Uses standard Phase 2B feature indices [0,1,2,3] — same regression,
    just better input data.
    """
    pass  # Implementation follows same _make_phase2b_model pattern
    # Key difference: in _compute_features_for_date, instead of using
    # knyc obs, use build_blended_curve(knyc_obs, neighbor_obs, offset)
    # then compute standard divergence features from the blended curve
```

The key implementation difference from Task 7:
1. Fetch neighbor obs from level-1 cache
2. Call `build_blended_curve()` to merge KNYC + neighbor obs
3. Call existing `compute_divergence_features()` on the blended curve
4. Use standard 4-feature vector — no extended features needed
5. For variant B2: apply offset before blending
6. For variant C2: use neighbor trend to interpolate between KNYC reports

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/alphatemp && venv/bin/python -m pytest alphatemp/tests/test_phase2b_models.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add alphatemp/services/backtester.py alphatemp/tests/test_phase2b_models.py
git commit -m "feat(phase3.6): add A2/B2/C2 blended curve model factories"
```

---

## Task 9: Analysis Script — HRRR Ablation

Run all 6 variants on HRRR across 0-18 ET. Produce crossover chart and improvement curves.

**Files:**
- Create: `scripts/phase36_neighbor_analysis.py`

**Step 1: Write the analysis script**

Follow the pattern from `scripts/phase35b_weights.py`:

```python
#!/usr/bin/env python3
"""Phase 3.6: Neighbor Station Observations — Ablation Analysis.

Runs 6 variants (A1, A2, B1, B2, C1, C2) on HRRR across 0-18 ET.
Compares each variant's Brier score against Phase 2B KNYC-only baseline.

Outputs:
    1. Per-hour Brier scores for each variant
    2. Crossover hour for each variant (when obs go from harmful to helpful)
    3. Improvement curve (delta vs baseline per hour)
    4. Feature importance for x1 (new features) variants
    5. Station contribution analysis (KLGA vs KEWR)
"""
import sys
sys.path.insert(0, ".")

from datetime import date
from services.backtester import (
    Backtester,
    wf_phase2b_full,           # baseline
    wf_phase2b_neighbor_a1,
    wf_phase2b_neighbor_a2,
    wf_phase2b_neighbor_b1,
    wf_phase2b_neighbor_b2,
    wf_phase2b_neighbor_c1,
    wf_phase2b_neighbor_c2,
)

DB_PATH = "data/alphatemp.duckdb"
RUN_HOURS = [0, 6, 12, 18]
UPDATE_HOURS_ET = list(range(0, 19))  # 0-18 ET — full window
START_DATE = date(2021, 6, 1)   # ~90 days after earliest neighbor obs
END_DATE = date(2026, 2, 25)

VARIANTS = {
    "baseline": wf_phase2b_full,
    "A1_raw_features": wf_phase2b_neighbor_a1,
    "A2_raw_blended": wf_phase2b_neighbor_a2,
    "B1_offset_features": wf_phase2b_neighbor_b1,
    "B2_offset_blended": wf_phase2b_neighbor_b2,
    "C1_trend_features": wf_phase2b_neighbor_c1,
    "C2_trend_blended": wf_phase2b_neighbor_c2,
}

GATE_THRESHOLD = 0.02  # 2% improvement


def main():
    bt = Backtester(DB_PATH, "NYC")

    results = {}
    for name, model_fn in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print(f"{'='*60}")
        result = bt.run(
            model_fn,
            start_date=START_DATE,
            end_date=END_DATE,
            run_hours=RUN_HOURS,
            update_hours_et=UPDATE_HOURS_ET,
        )
        results[name] = result
        print(f"  Mean Brier: {result.mean_brier:.4f}")

    # --- Report: Per-update-hour Brier ---
    print(f"\n{'='*60}")
    print("BRIER SCORE BY UPDATE HOUR (ET)")
    print(f"{'='*60}")
    header = f"{'Hour ET':>8}"
    for name in VARIANTS:
        header += f"  {name:>18}"
    print(header)

    baseline_by_hour = {}
    for uhr in UPDATE_HOURS_ET:
        row = f"{uhr:>8}"
        for name in VARIANTS:
            brier = results[name].by_update_hour.get(uhr, {}).get("brier", float("nan"))
            row += f"  {brier:>18.4f}"
            if name == "baseline":
                baseline_by_hour[uhr] = brier
        print(row)

    # --- Report: Improvement vs baseline ---
    print(f"\n{'='*60}")
    print("IMPROVEMENT vs BASELINE (positive = better)")
    print(f"{'='*60}")
    header = f"{'Hour ET':>8}"
    for name in list(VARIANTS.keys())[1:]:
        header += f"  {name:>18}"
    print(header)

    for uhr in UPDATE_HOURS_ET:
        row = f"{uhr:>8}"
        base = baseline_by_hour.get(uhr, float("nan"))
        for name in list(VARIANTS.keys())[1:]:
            variant_brier = results[name].by_update_hour.get(uhr, {}).get("brier", float("nan"))
            if base > 0:
                delta_pct = (base - variant_brier) / base * 100
                row += f"  {delta_pct:>17.1f}%"
            else:
                row += f"  {'N/A':>18}"
        print(row)

    # --- Report: Crossover hours ---
    print(f"\n{'='*60}")
    print("CROSSOVER ANALYSIS")
    print(f"{'='*60}")
    for name in list(VARIANTS.keys())[1:]:
        crossover = None
        for uhr in UPDATE_HOURS_ET:
            base = baseline_by_hour.get(uhr, float("nan"))
            variant = results[name].by_update_hour.get(uhr, {}).get("brier", float("nan"))
            if variant < base:
                crossover = uhr
                break
        if crossover is not None:
            print(f"  {name}: obs help starting at {crossover} ET")
        else:
            print(f"  {name}: obs never help (KILL)")

    # --- Gate check ---
    print(f"\n{'='*60}")
    print("GATE CHECK (>2% improvement at any sustained block)")
    print(f"{'='*60}")
    for name in list(VARIANTS.keys())[1:]:
        best_delta = 0
        best_hour = None
        for uhr in UPDATE_HOURS_ET:
            base = baseline_by_hour.get(uhr, float("nan"))
            variant = results[name].by_update_hour.get(uhr, {}).get("brier", float("nan"))
            if base > 0:
                delta = (base - variant) / base
                if delta > best_delta:
                    best_delta = delta
                    best_hour = uhr
        status = "PASS" if best_delta > GATE_THRESHOLD else "DISCUSS" if best_delta > 0.01 else "KILL"
        print(f"  {name}: best={best_delta*100:.1f}% at {best_hour} ET -> {status}")


if __name__ == "__main__":
    main()
```

**Step 2: Run on HRRR**

Run: `cd ~/Projects/alphatemp/alphatemp && ../venv/bin/python scripts/phase36_neighbor_analysis.py`
Expected: Takes several minutes. Produces per-hour Brier tables, improvement curves, crossover analysis, and gate verdicts.

**Step 3: Commit**

```bash
git add alphatemp/scripts/phase36_neighbor_analysis.py
git commit -m "feat(phase3.6): add HRRR neighbor obs ablation analysis script"
```

---

## Task 10: Extend Winners to GFS/ECMWF and Ensemble

Only run this task for variants that PASS or DISCUSS in Task 9.

**Files:**
- Modify: `scripts/phase36_neighbor_analysis.py` (add multi-model runs)

**Step 1: Add GFS/ECMWF model variants for winners**

For each variant that passed the HRRR gate, create GFS and ECMWF versions:
```python
# Example: if A1 passes
wf_phase2b_neighbor_a1_gfs = _make_phase2b_neighbor_model("a1_gfs", "A", "gfs")
wf_phase2b_neighbor_a1_ecmwf = _make_phase2b_neighbor_model("a1_ecmwf", "A", "ecmwf")
```

**Step 2: Run multi-model analysis**

Extend the analysis script to run winning variants across GFS and ECMWF, then combine through the ensemble with adaptive weights.

**Step 3: Produce final results**

Compare ensemble with neighbor obs vs ensemble without (Phase 3.5 baseline).

**Step 4: Commit**

```bash
git add alphatemp/services/backtester.py alphatemp/scripts/phase36_neighbor_analysis.py
git commit -m "feat(phase3.6): extend winning variants to GFS/ECMWF ensemble"
```

---

## Task 11: Update Progress and Handoff

**Files:**
- Modify: `docs/PROGRESS.md`
- Modify: `HANDOFF.md`

**Step 1: Update PROGRESS.md with Phase 3.6 results**

Add session log entry and update phase gate table with results.

**Step 2: Update HANDOFF.md with current state**

Write handoff with results, which variants passed/failed, and next steps.

**Step 3: Commit**

```bash
git add alphatemp/docs/PROGRESS.md alphatemp/HANDOFF.md
git commit -m "docs: update progress and handoff for Phase 3.6"
```
