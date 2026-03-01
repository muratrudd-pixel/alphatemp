"""Lightweight in-memory heartbeat registry for service health monitoring."""
from datetime import datetime, timezone
from typing import Dict, Optional

_heartbeats = {}  # type: Dict[str, dict]


def record_heartbeat(
    service_name,  # type: str
    duration_ms,  # type: float
    status="ok",  # type: str
    error=None,  # type: Optional[str]
):
    # type: (...) -> None
    """Record a service heartbeat. Called at end of each polling cycle."""
    _heartbeats[service_name] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 1),
        "status": status,
        "error": error,
    }


def get_all_heartbeats():
    # type: () -> Dict[str, dict]
    """Return all heartbeats. Used by /api/health/detailed."""
    return dict(_heartbeats)


def clear_heartbeats():
    # type: () -> None
    """Reset registry. Used in tests."""
    _heartbeats.clear()
