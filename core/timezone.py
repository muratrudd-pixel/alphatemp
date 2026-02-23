"""Timezone utilities — DST-aware Eastern Time ↔ UTC conversions."""

from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def et_day_bounds_utc(date_str: str = None) -> tuple:
    """Return (start_utc, end_utc) for midnight-to-midnight ET on the given date.

    Handles EST (UTC-5) and EDT (UTC-4) automatically via zoneinfo.

    Parameters
    ----------
    date_str : str, optional
        Date as "YYYY-MM-DD". Defaults to today in ET.

    Returns
    -------
    tuple of (datetime, datetime)
        Timezone-naive UTC datetimes for the start and end of the ET day.
    """
    if date_str:
        try:
            naive = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            naive = datetime.now(ET)
    else:
        naive = datetime.now(ET)

    midnight_et = datetime.combine(naive.date(), time.min, tzinfo=ET)
    start_utc = midnight_et.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = start_utc + timedelta(hours=24)
    return start_utc, end_utc


def utc_to_et_hour(dt: datetime) -> int:
    """Convert a UTC datetime to its Eastern Time hour (0-23).

    Handles DST transitions — returns 4pm as 16 regardless of EST/EDT.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).hour
