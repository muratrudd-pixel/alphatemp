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
