"""Retry decorator with exponential backoff for async functions."""

import asyncio
import functools

from loguru import logger


def retry_async(max_retries: int = 3, base_delay: float = 2.0):
    """Decorator: retry an async function with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (total calls = max_retries + 1).
    base_delay : float
        Base delay in seconds. Each retry doubles: base, base*2, base*4, ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"{fn.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {exc}. Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
