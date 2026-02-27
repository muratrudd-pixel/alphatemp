# tests/test_neighbor_obs.py
"""Tests for neighbor station observation processing."""
import pytest
from datetime import datetime, timedelta
from services.neighbor_obs import compute_walk_forward_offset
from services.neighbor_obs import compute_neighbor_divergence


class TestStationOffset:
    """Walk-forward expanding-window offset between neighbor and KNYC."""

    def test_constant_offset(self):
        """If neighbor is always 2F warmer, offset should converge to 2.0."""
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


class TestNeighborDivergence:
    """Divergence features from neighbor obs vs forecast."""

    def test_raw_divergence_no_offset(self):
        """A variant: use raw neighbor temps, no offset correction."""
        base_ts = 1000000.0
        obs_temps = [72.0 + i * 0.05 for i in range(60)]
        obs_timestamps = [base_ts + i * 60 for i in range(60)]
        fcst_timestamps = [base_ts, base_ts + 3600]
        fcst_temps = [70.0, 70.0]
        result = compute_neighbor_divergence(obs_temps, obs_timestamps, fcst_timestamps, fcst_temps, offset=0.0)
        assert result is not None
        assert "neighbor_running_max" in result
        assert "neighbor_instant" in result
        assert "neighbor_cumul" in result
        assert "neighbor_slope" in result
        assert "n_neighbor_obs" in result
        assert abs(result["neighbor_running_max"] - 4.95) < 0.1

    def test_with_offset_correction(self):
        """B variant: apply offset before computing divergence."""
        base_ts = 1000000.0
        obs_temps = [72.0] * 60
        obs_timestamps = [base_ts + i * 60 for i in range(60)]
        fcst_timestamps = [base_ts, base_ts + 3600]
        fcst_temps = [70.0, 70.0]
        result = compute_neighbor_divergence(obs_temps, obs_timestamps, fcst_timestamps, fcst_temps, offset=2.0)
        assert result is not None
        assert abs(result["neighbor_running_max"]) < 0.1

    def test_insufficient_obs_returns_none(self):
        result = compute_neighbor_divergence([72.0], [1000000.0], [1000000.0], [70.0], offset=0.0)
        assert result is None
