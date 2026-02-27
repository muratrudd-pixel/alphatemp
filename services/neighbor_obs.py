# services/neighbor_obs.py
"""Neighbor station observation processing for Phase 3.6.

Provides offset learning, divergence features, peak signal detection,
blended curve construction, and trend extraction from KLGA/KEWR data
to enhance Phase 2B predictions.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.divergence import interpolate_forecast, compute_divergence_features

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

    Returns:
        Dict with neighbor_ prefixed divergence features, or None if < 2 obs.
    """
    if len(obs_temps) < 2:
        return None

    corrected_temps = [t - offset for t in obs_temps]
    fcst_interp = interpolate_forecast(fcst_timestamps, fcst_temps, obs_timestamps)
    t0 = obs_timestamps[0]
    obs_hours = [(t - t0) / 3600.0 for t in obs_timestamps]
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

    return {
        "neighbor_running_max": raw["running_max_divergence"],
        "neighbor_instant": raw["temp_divergence"],
        "neighbor_cumul": raw["cumulative_divergence"],
        "neighbor_slope": raw["slope_divergence"],
        "n_neighbor_obs": raw["n_obs"],
    }


def compute_peak_signal(
    temps: List[float],
    timestamps: List[float],
    decline_threshold: float = 0.0,
) -> Optional[dict]:
    """Detect whether neighbor station temps have started declining.

    Returns dict with peak_passed (bool), minutes_since_peak_update (float),
    decline_rate (float F/min). Returns None if < 3 observations.
    """
    if len(temps) < 3:
        return None

    running_max = temps[0]
    last_peak_idx = 0
    for i in range(1, len(temps)):
        if temps[i] > running_max:
            running_max = temps[i]
            last_peak_idx = i

    latest_idx = len(temps) - 1
    latest_temp = temps[latest_idx]
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


def build_blended_curve(
    knyc_obs: List[Tuple[float, float]],
    neighbor_obs: List[Tuple[float, float]],
    offset: float = 0.0,
) -> List[Tuple[float, float]]:
    """Build blended temperature curve anchored on KNYC.

    At KNYC report times, uses KNYC temperature exactly.
    Between KNYC reports, fills with offset-corrected neighbor data.

    Args:
        knyc_obs: [(epoch_seconds, temp_f), ...] sorted by time.
        neighbor_obs: [(epoch_seconds, temp_f), ...] sorted by time.
        offset: Neighbor-KNYC offset to subtract from neighbor temps.

    Returns:
        [(epoch_seconds, temp_f), ...] sorted by time.
    """
    if not knyc_obs:
        return []
    if not neighbor_obs:
        return list(knyc_obs)

    knyc_times = set(ts for ts, _ in knyc_obs)
    combined = []

    for ts, temp in knyc_obs:
        combined.append((ts, temp))

    first_knyc = knyc_obs[0][0]
    last_knyc = knyc_obs[-1][0]
    for ts, temp in neighbor_obs:
        if ts < first_knyc or ts > last_knyc:
            continue
        is_knyc_time = any(abs(ts - kt) < _MATCH_TOLERANCE_S for kt in knyc_times)
        if is_knyc_time:
            continue
        combined.append((ts, temp - offset))

    combined.sort(key=lambda x: x[0])
    return combined


def compute_neighbor_trend(
    temps: List[float],
    timestamps: List[float],
) -> Optional[dict]:
    """Compute rate-of-change features from neighbor observations.

    For C variants: uses only trajectory, not absolute values.

    Returns dict with trend_slope_f_per_hr and trend_accel_f_per_hr2.
    Returns None if < 3 observations.
    """
    if len(temps) < 3:
        return None

    t0 = timestamps[0]
    hours = np.array([(t - t0) / 3600.0 for t in timestamps])
    y = np.array(temps)

    # Linear fit for slope
    A_lin = np.column_stack([np.ones(len(hours)), hours])
    result_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
    slope = result_lin[1]

    # Quadratic fit for acceleration (need >= 5 obs for meaningful curvature)
    accel = 0.0
    if len(temps) >= 5:
        A_quad = np.column_stack([np.ones(len(hours)), hours, hours ** 2])
        result_quad, _, _, _ = np.linalg.lstsq(A_quad, y, rcond=None)
        accel = 2.0 * result_quad[2]

    return {
        "trend_slope_f_per_hr": float(slope),
        "trend_accel_f_per_hr2": float(accel),
    }
