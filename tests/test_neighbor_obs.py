# tests/test_neighbor_obs.py
"""Tests for neighbor station observation processing."""
import pytest
from datetime import datetime, timedelta
from services.neighbor_obs import compute_walk_forward_offset


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
