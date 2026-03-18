"""Test Herbie timeout wrapper."""

from services.forecast import HRRRFetcher, HERBIE_TIMEOUT_SECONDS


def test_herbie_timeout_constant_exists():
    """Timeout constant should be defined and reasonable."""
    assert HERBIE_TIMEOUT_SECONDS >= 15
    assert HERBIE_TIMEOUT_SECONDS <= 120


def test_herbie_timeout_is_45_seconds():
    """Default timeout should be 45 seconds."""
    assert HERBIE_TIMEOUT_SECONDS == 45
