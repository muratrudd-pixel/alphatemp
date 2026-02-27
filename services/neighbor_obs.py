# services/neighbor_obs.py
"""Neighbor station observation processing for Phase 3.6.

Provides offset learning, divergence features, peak signal detection,
blended curve construction, and trend extraction from KLGA/KEWR data
to enhance Phase 2B predictions.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
