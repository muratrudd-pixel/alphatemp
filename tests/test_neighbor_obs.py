# tests/test_neighbor_obs.py
"""Tests for neighbor station observation processing."""
import pytest
from datetime import datetime, timedelta
from services.neighbor_obs import compute_walk_forward_offset
from services.neighbor_obs import compute_neighbor_divergence
from services.neighbor_obs import compute_peak_signal
from services.neighbor_obs import build_blended_curve


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


class TestPeakSignal:
    """Detect whether neighbor station temps have started declining."""

    def test_still_rising(self):
        temps = [70.0 + i for i in range(5)]
        timestamps = [1000000.0 + i * 300 for i in range(5)]
        result = compute_peak_signal(temps, timestamps)
        assert result["peak_passed"] is False
        assert result["minutes_since_peak_update"] == 0.0

    def test_declining(self):
        temps = [70.0, 72.0, 74.0, 73.0, 71.0]
        timestamps = [1000000.0 + i * 300 for i in range(5)]
        result = compute_peak_signal(temps, timestamps)
        assert result["peak_passed"] is True
        assert abs(result["minutes_since_peak_update"] - 10.0) < 0.1
        assert result["decline_rate"] < 0

    def test_plateau(self):
        temps = [70.0, 72.0, 74.0, 74.0, 74.0]
        timestamps = [1000000.0 + i * 300 for i in range(5)]
        result = compute_peak_signal(temps, timestamps)
        assert result["peak_passed"] is False
        assert result["minutes_since_peak_update"] == 0.0

    def test_insufficient_data(self):
        result = compute_peak_signal([70.0, 71.0], [1000000.0, 1000300.0])
        assert result is None


class TestBlendedCurve:
    """Build blended temperature curve anchored on KNYC."""

    def test_fills_gaps_between_knyc(self):
        base = 1000000.0
        knyc = [(base, 70.0), (base + 3600, 71.0)]
        klga = [(base + i * 60, 72.0 + i * (1.0 / 60)) for i in range(1, 60)]
        result = build_blended_curve(knyc, klga, offset=2.0)
        assert len(result) > 2
        assert result[0] == (base, 70.0)
        assert result[-1] == (base + 3600, 71.0)
        mid = result[len(result) // 2]
        assert mid[0] > base
        assert mid[0] < base + 3600

    def test_knyc_only_when_no_neighbor(self):
        knyc = [(1000000.0, 70.0), (1003600.0, 71.0)]
        result = build_blended_curve(knyc, [], offset=0.0)
        assert len(result) == 2
        assert result[0] == knyc[0]
        assert result[1] == knyc[1]

    def test_anchors_on_knyc_reports(self):
        base = 1000000.0
        knyc = [(base, 70.0), (base + 3600, 71.0)]
        klga = [(base, 75.0), (base + 1800, 76.0), (base + 3600, 76.0)]
        result = build_blended_curve(knyc, klga, offset=0.0)
        assert result[0] == (base, 70.0)
        assert result[-1] == (base + 3600, 71.0)
