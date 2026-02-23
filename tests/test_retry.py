"""Tests for the retry decorator."""

import asyncio

import pytest

from core.retry import retry_async


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_success_after_failures(event_loop):
    """Function succeeds on 3rd attempt after 2 failures."""
    call_count = 0

    @retry_async(max_retries=3, base_delay=0.01)
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("boom")
        return "ok"

    result = event_loop.run_until_complete(flaky())
    assert result == "ok"
    assert call_count == 3


def test_exhaustion_raises(event_loop):
    """All retries exhausted — original exception propagates."""
    @retry_async(max_retries=2, base_delay=0.01)
    async def always_fails():
        raise ValueError("permanent failure")

    with pytest.raises(ValueError, match="permanent failure"):
        event_loop.run_until_complete(always_fails())
