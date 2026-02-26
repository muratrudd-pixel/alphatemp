"""Divergence features — stateless comparison of obs vs raw HRRR curves.

Pure functions, no DB, no classes. Used by Phase 2B backtester and analysis.

All temperatures in °F unless noted. All timestamps in seconds since epoch.
"""

from typing import Dict, List, Optional

from scipy import stats as sp_stats


def interpolate_forecast(
    fc_timestamps,   # type: List[float]
    fc_temps,        # type: List[float]
    target_times,    # type: List[float]
):
    # type: (...) -> List[Optional[float]]
    """Linearly interpolate HRRR hourly forecast curve to target timestamps.

    Same algorithm as BiasEngine._interpolate_forecast_to_obs, but operates
    on pre-extracted numeric timestamps for bulk computation.

    Parameters
    ----------
    fc_timestamps : list of float
        Forecast valid_at times as seconds since epoch, sorted ascending.
    fc_temps : list of float
        Forecast temperatures (°F) at each valid_at.
    target_times : list of float
        Timestamps to interpolate to (e.g., observation times).

    Returns
    -------
    list of Optional[float]
        Interpolated temps. None only if target falls in a gap (shouldn't
        happen with sorted, contiguous hourly data). Clamps to endpoints
        if target is outside forecast range.
    """
    if len(fc_timestamps) < 2:
        return [None] * len(target_times)

    results = []  # type: List[Optional[float]]
    for t in target_times:
        if t <= fc_timestamps[0]:
            results.append(fc_temps[0])
        elif t >= fc_timestamps[-1]:
            results.append(fc_temps[-1])
        else:
            for i in range(len(fc_timestamps) - 1):
                if fc_timestamps[i] <= t <= fc_timestamps[i + 1]:
                    span = fc_timestamps[i + 1] - fc_timestamps[i]
                    if span == 0:
                        results.append(fc_temps[i])
                    else:
                        frac = (t - fc_timestamps[i]) / span
                        interp = fc_temps[i] + frac * (fc_temps[i + 1] - fc_temps[i])
                        results.append(interp)
                    break
            else:
                results.append(None)
    return results


def compute_divergence_features(
    obs_temps,           # type: List[float]
    fcst_interp,         # type: List[Optional[float]]
    obs_hours,           # type: List[float]
    fcst_curve_up_to_t,  # type: List[float]
):
    # type: (...) -> Optional[Dict[str, float]]
    """Compute the 5 divergence features from paired obs/forecast data.

    Parameters
    ----------
    obs_temps : list of float
        Observed temperatures, already truncated to the update window.
    fcst_interp : list of Optional[float]
        Forecast temps interpolated to observation times. May contain None
        for obs outside the forecast curve range (filtered out internally).
    obs_hours : list of float
        Hours since some reference for each observation. Used for slope
        regression — absolute reference doesn't matter, only spacing.
    fcst_curve_up_to_t : list of float
        Raw HRRR forecast curve temperatures for valid_at times up to
        the update time T. Used for running max comparison.

    Returns
    -------
    dict or None
        Keys: temp_divergence, cumulative_divergence, running_max_divergence,
        slope_divergence, n_obs. Returns None if < 2 valid paired obs.
    """
    # Filter to obs where forecast interpolation succeeded
    n = min(len(obs_temps), len(fcst_interp), len(obs_hours))
    p_obs = []   # type: List[float]
    p_fcst = []  # type: List[float]
    p_hrs = []   # type: List[float]
    for i in range(n):
        if fcst_interp[i] is not None:
            p_obs.append(obs_temps[i])
            p_fcst.append(fcst_interp[i])
            p_hrs.append(obs_hours[i])

    if len(p_obs) < 2:
        return None

    # 1. Instantaneous divergence: latest obs vs latest interpolated forecast
    temp_divergence = p_obs[-1] - p_fcst[-1]

    # 2. Cumulative divergence: mean(obs - fcst) across all paired points
    diffs = [o - f for o, f in zip(p_obs, p_fcst)]
    cumulative_divergence = sum(diffs) / len(diffs)

    # 3. Running max divergence: max(obs) - max(raw forecast curve up to T)
    obs_max = max(p_obs)
    if fcst_curve_up_to_t:
        fcst_max = max(fcst_curve_up_to_t)
    else:
        fcst_max = max(p_fcst)
    running_max_divergence = obs_max - fcst_max

    # 4. Slope divergence: linregress slope of divergence time series
    #    Requires >= 3 paired obs for meaningful regression; 0.0 otherwise
    if len(p_hrs) >= 3:
        slope, _, _, _, _ = sp_stats.linregress(p_hrs, diffs)
        slope_divergence = float(slope)
    else:
        slope_divergence = 0.0

    return {
        "temp_divergence": temp_divergence,
        "cumulative_divergence": cumulative_divergence,
        "running_max_divergence": running_max_divergence,
        "slope_divergence": slope_divergence,
        "n_obs": len(p_obs),
    }


def compute_synoptic_max_divergence(
    six_hr_max_c,    # type: Optional[float]
    fc_timestamps,   # type: List[float]
    fc_temps,        # type: List[float]
    window_start,    # type: float
    window_end,      # type: float
):
    # type: (...) -> Optional[float]
    """Compare 6-hour synoptic max against max(raw_fcst) in the same window.

    Parameters
    ----------
    six_hr_max_c : float or None
        6-hour max from METAR synoptic group, in Celsius.
    fc_timestamps : list of float
        Forecast valid_at times as seconds since epoch.
    fc_temps : list of float
        Forecast temperatures (°F) at each valid_at.
    window_start, window_end : float
        Timestamps defining the 6-hour window (e.g., 06z-12z).

    Returns
    -------
    float or None
        Divergence in °F, or None if data unavailable.
    """
    if six_hr_max_c is None:
        return None

    # Convert synoptic max from Celsius to Fahrenheit
    six_hr_max_f = six_hr_max_c * 9.0 / 5.0 + 32.0

    # Filter forecast curve to the 6-hour window
    window_temps = [
        temp for ts, temp in zip(fc_timestamps, fc_temps)
        if window_start <= ts <= window_end
    ]
    if not window_temps:
        return None

    return six_hr_max_f - max(window_temps)
