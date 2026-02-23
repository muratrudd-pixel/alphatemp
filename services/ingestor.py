"""Synoptic API ingestor with METAR T-group parser."""

import re
from typing import Optional

T_GROUP_PATTERN = re.compile(r"\bT(\d)(\d{3})")


def parse_t_group(metar_remarks: str) -> Optional[float]:
    """Extract high-resolution Celsius temperature from METAR T-group.

    The T-group in METAR remarks encodes temperature to 0.1 C precision.
    Format: T[sign][temp_tenths][sign][dewpoint_tenths]
    Sign: 0 = positive, 1 = negative
    Example: T0228 -> +22.8 C, T1005 -> -0.5 C
    """
    match = T_GROUP_PATTERN.search(metar_remarks)
    if not match:
        return None
    sign = -1 if match.group(1) == "1" else 1
    temp = int(match.group(2)) / 10.0
    return sign * temp
