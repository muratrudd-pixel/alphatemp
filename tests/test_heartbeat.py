"""Tests for the heartbeat registry."""
import pytest
from core.heartbeat import clear_heartbeats, record_heartbeat, get_all_heartbeats


@pytest.fixture(autouse=True)
def _clean_heartbeats():
    clear_heartbeats()
    yield
    clear_heartbeats()


def test_record_and_retrieve_heartbeat():
    record_heartbeat("TestService", duration_ms=150, status="ok")
    beats = get_all_heartbeats()
    assert "TestService" in beats
    assert beats["TestService"]["status"] == "ok"
    assert beats["TestService"]["duration_ms"] == 150
    assert beats["TestService"]["last_run"] is not None
